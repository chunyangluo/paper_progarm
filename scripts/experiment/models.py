import os
import sys
import json
import ast
import time
import random
import logging
import hashlib

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
from itertools import product

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from core.models.model_definitions import BERTTextCNN, MultimodalModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


def _build_loader_generator(seed):
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def compute_metrics(y_true, y_pred, y_prob=None):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'precision': float(precision_score(y_true, y_pred, zero_division=0)),
        'recall': float(recall_score(y_true, y_pred, zero_division=0)),
        'f1_score': float(f1_score(y_true, y_pred, zero_division=0)),
        'fpr': float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0,
        'fnr': float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0,
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
    }
    if y_prob is not None:
        from sklearn.metrics import roc_auc_score
        y_true_arr = np.asarray(y_true)
        y_prob_arr = np.asarray(y_prob, dtype=float)
        finite_mask = np.isfinite(y_prob_arr)
        if np.any(finite_mask):
            y_true_valid = y_true_arr[finite_mask]
            y_prob_valid = y_prob_arr[finite_mask]
            # AUC requires both classes; skip when a degenerate prediction collapses to one class.
            if np.unique(y_true_valid).size >= 2:
                metrics['auc'] = float(roc_auc_score(y_true_valid, y_prob_valid))
            else:
                logger.warning("Skipping AUC: y_true has <2 classes after filtering invalid probabilities.")
        else:
            logger.warning("Skipping AUC: all predicted probabilities are NaN/Inf.")
    return metrics


# ============================================================
# Model 1: TF-IDF + Logistic Regression
# ============================================================

class TfidfLRModel:

    def __init__(self):
        self.vectorizer = None
        self.clf = None
        self.name = 'TF-IDF+LR'

    def train(self, texts, labels, val_texts, val_labels, param_grid=None):
        if param_grid is None:
            param_grid = {
                'max_features': [5000, 8000, 10000],
                'C': [0.01, 0.1, 1.0, 10.0],
            }

        best_f1 = -1
        best_params = {}

        for max_feat in param_grid['max_features']:
            for C in param_grid['C']:
                vec = TfidfVectorizer(max_features=max_feat, ngram_range=(1, 2))
                X_train = vec.fit_transform(texts)
                X_val = vec.transform(val_texts)

                clf = LogisticRegression(C=C, max_iter=1000, random_state=SEED)
                clf.fit(X_train, labels)
                val_pred = clf.predict(X_val)
                val_f1 = f1_score(val_labels, val_pred, zero_division=0)

                if val_f1 > best_f1:
                    best_f1 = val_f1
                    best_params = {'max_features': max_feat, 'C': C}
                    self.vectorizer = vec
                    self.clf = clf

        logger.info(f"  [{self.name}] Best params: {best_params}, Val F1: {best_f1:.4f}")
        return best_params, best_f1

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        preds = self.clf.predict(X)
        probs = self.clf.predict_proba(X)[:, 1]
        return preds, probs

    def evaluate(self, texts, labels):
        preds, probs = self.predict(texts)
        return compute_metrics(labels, preds, probs)


# ============================================================
# Model 2: TextCNN
# ============================================================

class TextCNNDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

    def _build_vocab(self, texts):
        word_counts = {}
        for text in texts:
            for ch in str(text):
                word_counts[ch] = word_counts.get(ch, 0) + 1
        vocab = {'<pad>': 0, '<unk>': 1}
        for w, c in sorted(word_counts.items(), key=lambda x: -x[1])[:20000]:
            vocab[w] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        ids = [self.vocab.get(ch, 1) for ch in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class TextCNNModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, filter_sizes=(3, 4, 5), num_filters=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, num_filters, fs) for fs in filter_sizes
        ])
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(num_filters * len(filter_sizes), num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)
        conv_outs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            c = torch.max(c, dim=2)[0]
            conv_outs.append(c)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)


class TextCNNExperiment:

    def __init__(self):
        self.name = 'TextCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, texts, labels, val_texts, val_labels, param_grid=None):
        if param_grid is None:
            param_grid = {
                'filter_sizes': [(3, 4, 5)],
                'num_filters': [32, 64, 128],
                'lr': [1e-4, 5e-4, 1e-3],
            }
        batch_size = int(param_grid.get('batch_size', 64))
        val_batch_size = int(param_grid.get('val_batch_size', batch_size * 2))
        epochs = int(param_grid.get('epochs', 30))
        patience = int(param_grid.get('patience', 5))

        best_f1 = -1
        best_state = None
        best_config = {}

        train_ds = TextCNNDataset(texts, labels)
        vocab = train_ds.vocab
        val_ds = TextCNNDataset(val_texts, val_labels, vocab=vocab)

        for filter_sizes, num_filters, lr in product(
            param_grid['filter_sizes'], param_grid['num_filters'], param_grid['lr']
        ):
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=_build_loader_generator(SEED),
            )
            val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

            model = TextCNNModel(
                vocab_size=len(vocab), embed_dim=128,
                filter_sizes=filter_sizes, num_filters=num_filters
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            patience_counter = 0
            local_best_f1 = -1
            local_best_state = None

            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

                val_f1 = self._eval_f1(model, val_loader)
                if val_f1 > local_best_f1:
                    local_best_f1 = val_f1
                    local_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if local_best_f1 > best_f1:
                best_f1 = local_best_f1
                best_state = local_best_state
                best_config = {
                    'filter_sizes': filter_sizes,
                    'num_filters': num_filters,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': patience,
                }

        self.vocab = vocab
        self.best_config = best_config
        self.model = TextCNNModel(
            vocab_size=len(vocab), embed_dim=128,
            filter_sizes=best_config['filter_sizes'], num_filters=best_config['num_filters']
        ).to(self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()

        logger.info(f"  [{self.name}] Best config: {best_config}, Val F1: {best_f1:.4f}")
        return best_config, best_f1

    def _eval_f1(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                out = model(batch_x)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        return f1_score(all_labels, all_preds, zero_division=0)

    def evaluate(self, texts, labels):
        ds = TextCNNDataset(texts, labels, vocab=self.vocab)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        return compute_metrics(labels, all_preds, all_probs)


# ============================================================
# Model 3: BERT-Base-Chinese
# ============================================================

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        return {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }


class BERTClassifier(nn.Module):
    def __init__(self, bert_model, num_classes=2, freeze_layers=9):
        super().__init__()
        self.bert = bert_model
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_layers:
                for param in layer.parameters():
                    param.requires_grad = False
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_out = out.last_hidden_state[:, 0, :]
        cls_out = self.dropout(cls_out)
        return self.classifier(cls_out)


class BERTExperiment:

    def __init__(self):
        self.name = 'BERT'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-chinese',
            local_files_only=True,
            use_fast=False,
        )

    def train(self, texts, labels, val_texts, val_labels, param_grid=None):
        if param_grid is None:
            param_grid = {
                'bert_lr': [2e-5],  # 减少为1个值
                'cls_lr': [1e-4],   # 减少为1个值
                'batch_size': [4],   # Windows稳定模式：降低内存峰值
            }
        epochs = int(param_grid.get('epochs', 15))
        patience = int(param_grid.get('patience', 5))
        max_length = int(param_grid.get('max_length', 128))
        val_batch_multiplier = int(param_grid.get('val_batch_multiplier', 2))

        best_f1 = -1
        best_state = None
        best_config = {}

        train_ds = BERTDataset(texts, labels, self.tokenizer, max_length=max_length)
        val_ds = BERTDataset(val_texts, val_labels, self.tokenizer, max_length=max_length)

        for bert_lr, cls_lr, batch_size in product(
            param_grid['bert_lr'], param_grid['cls_lr'], param_grid['batch_size']
        ):
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=_build_loader_generator(SEED),
            )
            val_loader = DataLoader(
                val_ds,
                batch_size=batch_size * val_batch_multiplier,
                shuffle=False,
                num_workers=0,
            )

            bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
            model = BERTClassifier(bert, freeze_layers=9).to(self.device)

            optimizer = torch.optim.AdamW([
                {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': bert_lr},
                {'params': model.classifier.parameters(), 'lr': cls_lr},
            ], weight_decay=0.01)

            criterion = nn.CrossEntropyLoss()
            patience_counter = 0
            local_best_f1 = -1
            local_best_state = None

            logger.info(f"  [{self.name}] Testing config: bert_lr={bert_lr}, cls_lr={cls_lr}, batch_size={batch_size}")

            for epoch in range(epochs):
                model.train()
                total_loss = 0
                for batch in train_loader:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    label = batch['label'].to(self.device)
                    optimizer.zero_grad()
                    out = model(input_ids, attention_mask)
                    loss = criterion(out, label)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                val_f1 = self._eval_f1(model, val_loader)
                logger.info(f"  [{self.name}] Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f}")

                if val_f1 > local_best_f1:
                    local_best_f1 = val_f1
                    local_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        logger.info(f"  [{self.name}] Early stopping at epoch {epoch+1}")
                        break

            if local_best_f1 > best_f1:
                best_f1 = local_best_f1
                best_state = local_best_state
                best_config = {
                    'bert_lr': bert_lr,
                    'cls_lr': cls_lr,
                    'batch_size': batch_size,
                    'max_length': max_length,
                    'epochs': epochs,
                    'patience': patience,
                }

            del model, bert
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
        self.model = BERTClassifier(bert, freeze_layers=9).to(self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()

        logger.info(f"  [{self.name}] Best config: {best_config}, Val F1: {best_f1:.4f}")
        return best_config, best_f1

    def _eval_f1(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())
        return f1_score(all_labels, all_preds, zero_division=0)

    def evaluate(self, texts, labels):
        ds = BERTDataset(texts, labels, self.tokenizer)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = self.model(input_ids, attention_mask)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        return compute_metrics(labels, all_preds, all_probs)


# ============================================================
# Model 4: LSTM
# ============================================================

class LSTMDataset(Dataset):
    def __init__(self, texts, labels, vocab=None, max_len=128):
        self.texts = texts
        self.labels = labels
        self.max_len = max_len
        if vocab is None:
            self.vocab = self._build_vocab(texts)
        else:
            self.vocab = vocab

    def _build_vocab(self, texts):
        word_counts = {}
        for text in texts:
            for ch in str(text):
                word_counts[ch] = word_counts.get(ch, 0) + 1
        vocab = {'<pad>': 0, '<unk>': 1}
        for w, c in sorted(word_counts.items(), key=lambda x: -x[1])[:20000]:
            vocab[w] = len(vocab)
        return vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        ids = [self.vocab.get(ch, 1) for ch in text[:self.max_len]]
        ids += [0] * (self.max_len - len(ids))
        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, hidden_dim=128, num_layers=1, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.lstm(x)
        x = out[:, -1, :]
        x = self.dropout(x)
        return self.fc(x)


class LSTMExperiment:

    def __init__(self):
        self.name = 'LSTM'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, texts, labels, val_texts, val_labels, param_grid=None):
        if param_grid is None:
            param_grid = {
                'hidden_dim': [128, 256],
                'num_layers': [1, 2],
                'lr': [1e-4, 5e-4, 1e-3],
            }
        batch_size = int(param_grid.get('batch_size', 64))
        val_batch_size = int(param_grid.get('val_batch_size', batch_size * 2))
        epochs = int(param_grid.get('epochs', 30))
        patience = int(param_grid.get('patience', 5))

        best_f1 = -1
        best_state = None
        best_config = {}

        train_ds = LSTMDataset(texts, labels)
        vocab = train_ds.vocab
        val_ds = LSTMDataset(val_texts, val_labels, vocab=vocab)

        for hidden_dim, num_layers, lr in product(
            param_grid['hidden_dim'], param_grid['num_layers'], param_grid['lr']
        ):
            train_loader = DataLoader(
                train_ds,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                generator=_build_loader_generator(SEED),
            )
            val_loader = DataLoader(val_ds, batch_size=val_batch_size, shuffle=False, num_workers=0)

            model = LSTMClassifier(
                vocab_size=len(vocab), embed_dim=128,
                hidden_dim=hidden_dim, num_layers=num_layers
            ).to(self.device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss()

            patience_counter = 0
            local_best_f1 = -1
            local_best_state = None

            for epoch in range(epochs):
                model.train()
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                    optimizer.zero_grad()
                    out = model(batch_x)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

                val_f1 = self._eval_f1(model, val_loader)
                if val_f1 > local_best_f1:
                    local_best_f1 = val_f1
                    local_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        break

            if local_best_f1 > best_f1:
                best_f1 = local_best_f1
                best_state = local_best_state
                best_config = {
                    'hidden_dim': hidden_dim,
                    'num_layers': num_layers,
                    'lr': lr,
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'patience': patience,
                }

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        self.vocab = vocab
        self.best_config = best_config
        self.model = LSTMClassifier(
            vocab_size=len(vocab), embed_dim=128,
            hidden_dim=best_config['hidden_dim'], num_layers=best_config['num_layers']
        ).to(self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()

        logger.info(f"  [{self.name}] Best config: {best_config}, Val F1: {best_f1:.4f}")
        return best_config, best_f1

    def _eval_f1(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                out = model(batch_x)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.numpy())
        return f1_score(all_labels, all_preds, zero_division=0)

    def evaluate(self, texts, labels):
        ds = LSTMDataset(texts, labels, vocab=self.vocab)
        loader = DataLoader(ds, batch_size=128, shuffle=False)
        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch_x, _ in loader:
                batch_x = batch_x.to(self.device)
                out = self.model(batch_x)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        return compute_metrics(labels, all_preds, all_probs)


# ============================================================
# Model 5: BERT-TextCNN (Hybrid)
# ============================================================

class BERTTextCNNDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128, url_features=None, network_features=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.url_features = url_features
        self.network_features = network_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            str(self.texts[idx]), max_length=self.max_length,
            padding='max_length', truncation=True, return_tensors='pt'
        )
        item = {
            'input_ids': enc['input_ids'].squeeze(0),
            'attention_mask': enc['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long),
        }
        if self.url_features is not None:
            item['url_features'] = torch.tensor(self.url_features[idx], dtype=torch.float)
        if self.network_features is not None:
            item['network_features'] = torch.tensor(self.network_features[idx], dtype=torch.float)
        return item


def generate_url_features_batch(urls):
    features = []
    for url in urls:
        if not url or not isinstance(url, str) or url.strip() == '':
            features.append([0.0] * 16)
            continue
        f = []
        f.append(float(len(url)))
        f.append(1.0 if 'https' in url else 0.0)
        f.append(float(url.count('.')))
        for kw in ['login', 'bank', 'secure', 'verify', 'account',
                   'auth', 'payment', 'password', 'reset', 'confirm', 'update', 'financial', 'phish']:
            f.append(1.0 if kw in url.lower() else 0.0)
        features.append(f[:16])
    return features


def parse_network_feature(value):
    if value is None:
        return None
    if isinstance(value, list):
        arr = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            if text.startswith('['):
                arr = json.loads(text)
            else:
                arr = [float(x.strip()) for x in text.split(',')]
        except Exception:
            try:
                arr = ast.literal_eval(text)
            except Exception:
                return None
    else:
        return None
    try:
        arr = [float(x) for x in arr]
    except Exception:
        return None
    if len(arr) < 8:
        arr = arr + [0.0] * (8 - len(arr))
    return arr[:8]


def build_network_features_batch(urls, network_feature_values=None):
    """Prefer real dataset network features; fallback to lightweight URL-derived features."""
    features = []
    for idx, url in enumerate(urls):
        parsed = None
        if network_feature_values is not None and idx < len(network_feature_values):
            parsed = parse_network_feature(network_feature_values[idx])
        if parsed is not None:
            features.append(parsed)
            continue
        if not url or not isinstance(url, str) or url.strip() == '':
            features.append([0.0] * 8)
            continue
        f = [0.0] * 8
        f[1] = 404.0
        f[5] = 1.0 if url.startswith('https://') else 0.0
        features.append(f[:8])
    return features


class BERTTextCNNExperiment:

    def __init__(self):
        self.name = 'BERT-TextCNN'
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-chinese',
            local_files_only=True,
            use_fast=False,
        )

    def train(self, texts, labels, val_texts, val_labels, train_urls=None, val_urls=None, param_config=None):
        defaults = {
            'bert_lr': 2e-5,
            'cnn_lr': 1e-4,
            'batch_size': 16,
            'max_length': 128,
            'epochs': 15,
            'patience': 5,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'grad_accum_steps': 4,
            'seed': SEED,
        }
        param_config = {**defaults, **(param_config or {})}

        train_ds = BERTTextCNNDataset(texts, labels, self.tokenizer, param_config['max_length'])
        val_ds = BERTTextCNNDataset(val_texts, val_labels, self.tokenizer, param_config['max_length'])

        train_loader = DataLoader(
            train_ds,
            batch_size=param_config['batch_size'],
            shuffle=True,
            num_workers=0,
            generator=_build_loader_generator(param_config['seed']),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=param_config['batch_size'] * 2,
            shuffle=False,
            num_workers=0,
        )

        bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
        for i in range(8):
            for param in bert.encoder.layer[i].parameters():
                param.requires_grad = False

        model = BERTTextCNN(bert).to(self.device)

        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': param_config['bert_lr']},
            {'params': [
                p for n, p in model.named_parameters()
                if not n.startswith('bert.') and p.requires_grad
            ], 'lr': param_config['cnn_lr']},
        ], weight_decay=param_config['weight_decay'])

        total_steps = len(train_loader) * param_config['epochs']
        warmup_steps = int(total_steps * param_config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        best_f1 = -1
        best_state = None
        patience_counter = 0

        for epoch in range(param_config['epochs']):
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(input_ids, attention_mask)
                        loss = criterion(out, label) / param_config['grad_accum_steps']
                    scaler.scale(loss).backward()
                    if (step + 1) % param_config['grad_accum_steps'] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    out = model(input_ids, attention_mask)
                    loss = criterion(out, label) / param_config['grad_accum_steps']
                    loss.backward()
                    if (step + 1) % param_config['grad_accum_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
            if len(train_loader) % param_config['grad_accum_steps'] != 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            val_f1 = self._eval_f1(model, val_loader)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= param_config['patience']:
                    logger.info(f"  [{self.name}] Early stop at epoch {epoch+1}")
                    break

        bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
        for i in range(8):
            for param in bert.encoder.layer[i].parameters():
                param.requires_grad = False
        self.model = BERTTextCNN(bert).to(self.device)
        self.model.load_state_dict(best_state)
        self.model.eval()

        logger.info(f"  [{self.name}] Best Val F1: {best_f1:.4f}")
        return param_config, best_f1

    def _eval_f1(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = model(input_ids, attention_mask)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())
        return f1_score(all_labels, all_preds, zero_division=0)

    def evaluate(self, texts, labels, urls=None):
        ds = BERTTextCNNDataset(texts, labels, self.tokenizer, 128)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                out = self.model(input_ids, attention_mask)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        return compute_metrics(labels, all_preds, all_probs)


class MultimodalAblationExperiment:
    """True multimodal experiment with ablation and checkpoint resume."""

    def __init__(self, ablation_mode='full'):
        self.ablation_mode = ablation_mode
        self.name = f"Multimodal-{ablation_mode}"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-chinese',
            local_files_only=True,
            use_fast=False,
        )
        self.checkpoint_dir = os.path.join(PROJECT_ROOT, 'output', 'experiments', 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_path = None

    def _resolve_checkpoint_path(self, run_context, config):
        payload = {
            "ablation_mode": self.ablation_mode,
            "seed": run_context.get("seed"),
            "dataset_name": run_context.get("dataset_name"),
            "session_dir": os.path.abspath(run_context.get("session_dir") or ""),
            "epochs": config.get("epochs"),
            "batch_size": config.get("batch_size"),
            "max_length": config.get("max_length"),
            "model_version": config.get("model_version"),
        }
        fingerprint = hashlib.sha1(
            json.dumps(payload, ensure_ascii=True, sort_keys=True).encode("utf-8")
        ).hexdigest()[:12]
        return os.path.join(self.checkpoint_dir, f"{self.ablation_mode}_{fingerprint}.pt")

    @staticmethod
    def _calculate_real_network_coverage(network_feature_values):
        """Return ratio of parsable real network feature rows."""
        if not network_feature_values:
            return 0.0, 0, 0
        total = len(network_feature_values)
        valid = 0
        for value in network_feature_values:
            if parse_network_feature(value) is not None:
                valid += 1
        return valid / total if total > 0 else 0.0, valid, total

    def _build_dataset(self, texts, labels, urls, network_values, max_length):
        url_feats = generate_url_features_batch(urls)
        net_feats = build_network_features_batch(urls, network_values)
        if self.ablation_mode == 'text_only':
            url_feats = [[0.0] * 16 for _ in url_feats]
            net_feats = [[0.0] * 8 for _ in net_feats]
        elif self.ablation_mode == 'text_url':
            net_feats = [[0.0] * 8 for _ in net_feats]
        elif self.ablation_mode == 'text_network':
            url_feats = [[0.0] * 16 for _ in url_feats]
        return BERTTextCNNDataset(texts, labels, self.tokenizer, max_length, url_feats, net_feats)

    def train(self, texts, labels, val_texts, val_labels,
              train_urls=None, val_urls=None, train_network_features=None, val_network_features=None,
              param_config=None, run_context=None):
        defaults = {
            'bert_lr': 2e-5,
            'fusion_lr': 1e-4,
            'batch_size': 16,
            'max_length': 128,
            'epochs': 12,
            'patience': 4,
            'warmup_ratio': 0.1,
            'weight_decay': 0.01,
            'grad_accum_steps': 2,
            'resume': True,
            'min_real_network_feature_ratio': 0.70,
            'model_version': 'multimodal_v2',
            'seed': SEED,
        }
        param_config = {**defaults, **(param_config or {})}
        run_context = run_context or {}
        self.checkpoint_path = self._resolve_checkpoint_path(run_context, param_config)

        min_ratio = float(param_config.get('min_real_network_feature_ratio', 0.70))
        train_ratio, train_valid, train_total = self._calculate_real_network_coverage(train_network_features)
        val_ratio, val_valid, val_total = self._calculate_real_network_coverage(val_network_features)
        logger.info(
            f"  [{self.name}] Real network_features coverage | "
            f"train: {train_ratio:.2%} ({train_valid}/{train_total}), "
            f"val: {val_ratio:.2%} ({val_valid}/{val_total}), "
            f"threshold: {min_ratio:.2%}"
        )
        requires_real_network = self.ablation_mode in ('text_network', 'full')
        if requires_real_network and (train_ratio < min_ratio or val_ratio < min_ratio):
            raise ValueError(
                f"[{self.name}] real network_features coverage below threshold "
                f"(train={train_ratio:.2%}, val={val_ratio:.2%}, threshold={min_ratio:.2%}). "
                "Abort training to avoid excessive fallback features."
            )

        train_ds = self._build_dataset(
            texts, labels, train_urls or [''] * len(texts), train_network_features, param_config['max_length']
        )
        val_ds = self._build_dataset(
            val_texts, val_labels, val_urls or [''] * len(val_texts), val_network_features, param_config['max_length']
        )
        train_loader = DataLoader(
            train_ds,
            batch_size=param_config['batch_size'],
            shuffle=True,
            num_workers=0,
            generator=_build_loader_generator(param_config['seed']),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=param_config['batch_size'] * 2,
            shuffle=False,
            num_workers=0,
        )

        bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
        for i in range(8):
            for param in bert.encoder.layer[i].parameters():
                param.requires_grad = False
        model = MultimodalModel(bert).to(self.device)

        optimizer = torch.optim.AdamW([
            {'params': [p for n, p in model.bert.named_parameters() if p.requires_grad], 'lr': param_config['bert_lr']},
            {'params': model.conv1.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.conv2.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.conv3.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.url_encoder.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.network_encoder.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.modal_encoder.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.text_encoder.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.cross_attention.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.fusion.parameters(), 'lr': param_config['fusion_lr']},
            {'params': model.classification.parameters(), 'lr': param_config['fusion_lr']},
            {'params': [model.modal_weights], 'lr': param_config['fusion_lr']},
        ], weight_decay=param_config['weight_decay'])

        total_steps = len(train_loader) * param_config['epochs']
        warmup_steps = int(total_steps * param_config['warmup_ratio'])
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
        criterion = nn.CrossEntropyLoss()
        scaler = torch.amp.GradScaler('cuda') if self.device.type == 'cuda' else None

        start_epoch = 0
        best_f1 = -1.0
        patience_counter = 0
        best_state = None
        if param_config.get('resume') and os.path.exists(self.checkpoint_path):
            ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ckpt['model_state'])
            optimizer.load_state_dict(ckpt['optimizer_state'])
            scheduler.load_state_dict(ckpt['scheduler_state'])
            if scaler and ckpt.get('scaler_state'):
                scaler.load_state_dict(ckpt['scaler_state'])
            start_epoch = int(ckpt.get('epoch', -1)) + 1
            best_f1 = float(ckpt.get('best_f1', -1.0))
            patience_counter = int(ckpt.get('patience_counter', 0))
            logger.info(f"  [{self.name}] Resumed from epoch {start_epoch}")

        for epoch in range(start_epoch, param_config['epochs']):
            model.train()
            optimizer.zero_grad()
            for step, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                label = batch['label'].to(self.device)
                url_features = batch['url_features'].to(self.device)
                network_features = batch['network_features'].to(self.device)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        out = model(input_ids, attention_mask, url_features=url_features, network_features=network_features)
                        loss = criterion(out, label) / param_config['grad_accum_steps']
                    scaler.scale(loss).backward()
                    if (step + 1) % param_config['grad_accum_steps'] == 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                        scheduler.step()
                        optimizer.zero_grad()
                else:
                    out = model(input_ids, attention_mask, url_features=url_features, network_features=network_features)
                    loss = criterion(out, label) / param_config['grad_accum_steps']
                    loss.backward()
                    if (step + 1) % param_config['grad_accum_steps'] == 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
            if len(train_loader) % param_config['grad_accum_steps'] != 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            val_f1 = self._eval_f1(model, val_loader)
            if val_f1 > best_f1:
                best_f1 = val_f1
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'scaler_state': scaler.state_dict() if scaler else None,
                'best_f1': best_f1,
                'patience_counter': patience_counter,
                'ablation_mode': self.ablation_mode,
                'run_context': run_context,
                'config': param_config,
            }, self.checkpoint_path)

            if patience_counter >= param_config['patience']:
                logger.info(f"  [{self.name}] Early stop at epoch {epoch+1}")
                break

        bert = AutoModel.from_pretrained('bert-base-chinese', local_files_only=True)
        for i in range(8):
            for param in bert.encoder.layer[i].parameters():
                param.requires_grad = False
        self.model = MultimodalModel(bert).to(self.device)
        self.model.load_state_dict(best_state if best_state is not None else model.state_dict())
        self.model.eval()
        logger.info(f"  [{self.name}] Best Val F1: {best_f1:.4f}")
        return {**param_config, 'ablation_mode': self.ablation_mode}, best_f1

    def _eval_f1(self, model, loader):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                url_features = batch['url_features'].to(self.device)
                network_features = batch['network_features'].to(self.device)
                out = model(input_ids, attention_mask, url_features=url_features, network_features=network_features)
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch['label'].numpy())
        return f1_score(all_labels, all_preds, zero_division=0)

    def evaluate(self, texts, labels, urls=None, network_features=None):
        ds = self._build_dataset(texts, labels, urls or [''] * len(texts), network_features, 128)
        loader = DataLoader(ds, batch_size=64, shuffle=False)
        self.model.eval()
        all_preds, all_probs = [], []
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                url_features = batch['url_features'].to(self.device)
                network_feats = batch['network_features'].to(self.device)
                out = self.model(input_ids, attention_mask, url_features=url_features, network_features=network_feats)
                probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
                preds = out.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs)
        return compute_metrics(labels, all_preds, all_probs)
