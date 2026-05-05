import os
import sys
import json
import time
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'versions')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
OUTPUT_DIR = os.path.join(PROJECT_ROOT, 'output')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(MODEL_DIR, 'bert_textcnn_training_checkpoint.pt')

sys.path.insert(0, os.path.join(PROJECT_ROOT, 'src'))
from core.models.model_definitions import BERTTextCNN, MultimodalModel


class PhishingDataset(Dataset):
    def __init__(self, texts, labels, urls=None, tokenizer=None, max_length=128):
        self.texts = texts
        self.labels = labels
        self.urls = urls or [''] * len(texts)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True,
            max_length=self.max_length, return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'token_type_ids': encoding.get('token_type_ids', torch.zeros_like(encoding['input_ids'])).squeeze(0),
            'label': torch.tensor(label, dtype=torch.long),
        }


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def _build_loader_generator(seed):
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return generator


def load_and_preprocess_data(dataset_path=None):
    logger.info("=" * 60)
    logger.info("STEP 1: Data Loading and Preprocessing")
    logger.info("=" * 60)

    if dataset_path is None:
        dataset_path = os.path.join(DATA_DIR, 'dataset_20260411_chifraud.csv')

    df = pd.read_csv(dataset_path)
    logger.info(f"Raw dataset: {df.shape[0]} samples, {df.shape[1]} columns")

    df = df.dropna(subset=['text', 'label'])
    df = df[df['text'].str.strip().str.len() >= 2]
    df = df.drop_duplicates(subset=['text'])
    df['label'] = df['label'].astype(int)
    df = df[df['label'].isin([0, 1])]

    label_counts = df['label'].value_counts()
    min_count = label_counts.min()
    df_balanced = pd.concat([
        df[df['label'] == 0].sample(min_count, random_state=42),
        df[df['label'] == 1].sample(min_count, random_state=42)
    ])
    df = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

    logger.info(f"After preprocessing: {df.shape[0]} samples")
    logger.info(f"Label distribution: {df['label'].value_counts().to_dict()}")

    texts = df['text'].tolist()
    labels = df['label'].tolist()
    urls = df.get('url', pd.Series([''] * len(texts))).fillna('').tolist()

    train_texts, temp_texts, train_labels, temp_labels, train_urls, temp_urls = train_test_split(
        texts, labels, urls, test_size=0.3, random_state=42, stratify=labels
    )
    val_texts, test_texts, val_labels, test_labels, val_urls, test_urls = train_test_split(
        temp_texts, temp_labels, temp_urls, test_size=0.5, random_state=42, stratify=temp_labels
    )

    logger.info(f"Train: {len(train_texts)}, Val: {len(val_texts)}, Test: {len(test_texts)}")
    logger.info(f"Train labels: {dict(zip(*np.unique(train_labels, return_counts=True)))}")
    logger.info(f"Val labels: {dict(zip(*np.unique(val_labels, return_counts=True)))}")
    logger.info(f"Test labels: {dict(zip(*np.unique(test_labels, return_counts=True)))}")

    return {
        'train': (train_texts, train_labels, train_urls),
        'val': (val_texts, val_labels, val_urls),
        'test': (test_texts, test_labels, test_urls),
        'total_samples': len(texts),
    }


def train_epoch(model, dataloader, optimizer, scheduler, criterion, device, grad_accum_steps=4):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        labels = batch['label'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        loss = criterion(outputs, labels) / grad_accum_steps
        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        total_loss += loss.item() * grad_accum_steps
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    if len(dataloader) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs)

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else 0.0
    cm = confusion_matrix(all_labels, all_preds)

    return {
        'loss': avg_loss, 'accuracy': acc, 'precision': precision,
        'recall': recall, 'f1': f1, 'auc': auc, 'confusion_matrix': cm,
        'predictions': all_preds, 'labels': all_labels, 'probabilities': all_probs
    }


def train_model(data, config):
    logger.info("=" * 60)
    logger.info("STEP 2-4: Model Training with Monitoring")
    logger.info("=" * 60)

    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese')
    bert_model = AutoModel.from_pretrained('bert-base-chinese')

    for param in bert_model.parameters():
        param.requires_grad = False
    for layer in bert_model.encoder.layer[-4:]:
        for param in layer.parameters():
            param.requires_grad = True

    model = BERTTextCNN(bert_model).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: total={total:,}, trainable={trainable:,} ({trainable/total*100:.1f}%)")

    train_texts, train_labels, train_urls = data['train']
    val_texts, val_labels, val_urls = data['val']
    test_texts, test_labels, test_urls = data['test']

    train_dataset = PhishingDataset(train_texts, train_labels, train_urls, tokenizer, config['max_length'])
    val_dataset = PhishingDataset(val_texts, val_labels, val_urls, tokenizer, config['max_length'])
    test_dataset = PhishingDataset(test_texts, test_labels, test_urls, tokenizer, config['max_length'])

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        generator=_build_loader_generator(config['seed']),
    )
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW([
        {'params': bert_model.encoder.layer[-4:].parameters(), 'lr': config['learning_rate']},
        {'params': model.conv1.parameters(), 'lr': config['learning_rate'] * 10},
        {'params': model.conv2.parameters(), 'lr': config['learning_rate'] * 10},
        {'params': model.conv3.parameters(), 'lr': config['learning_rate'] * 10},
        {'params': model.fc.parameters(), 'lr': config['learning_rate'] * 10},
    ], weight_decay=config['weight_decay'])

    total_steps = len(train_loader) * config['epochs']
    warmup_steps = int(total_steps * config['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    class_weights = torch.tensor([1.0, 1.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {
        'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': [], 'val_auc': [],
        'learning_rates': []
    }

    start_epoch = 0
    best_f1 = 0.0
    best_epoch = 0
    patience_counter = 0
    start_time = time.time()

    if config.get('resume', True) and os.path.exists(CHECKPOINT_PATH):
        logger.info(f"Resuming training from checkpoint: {CHECKPOINT_PATH}")
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if scaler is not None and checkpoint.get('scaler_state_dict') is not None:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        history = checkpoint.get('history', history)
        start_epoch = int(checkpoint.get('epoch', -1)) + 1
        best_f1 = float(checkpoint.get('best_f1', 0.0))
        best_epoch = int(checkpoint.get('best_epoch', 0))
        patience_counter = int(checkpoint.get('patience_counter', 0))
        logger.info(
            f"Checkpoint loaded: start_epoch={start_epoch}, best_f1={best_f1:.4f}, best_epoch={best_epoch}"
        )

    logger.info(f"Config: epochs={config['epochs']}, batch_size={config['batch_size']}, lr={config['learning_rate']}")
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    logger.info("-" * 60)

    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.time()

        model.train()
        epoch_loss = 0
        epoch_preds = []
        epoch_labels = []
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['label'].to(device)

            if scaler and device.type == 'cuda':
                with torch.amp.autocast('cuda'):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                    loss = criterion(outputs, labels) / config['grad_accum_steps']
                scaler.scale(loss).backward()
                if (step + 1) % config['grad_accum_steps'] == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad()
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                loss = criterion(outputs, labels) / config['grad_accum_steps']
                loss.backward()
                if (step + 1) % config['grad_accum_steps'] == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

        if len(train_loader) % config['grad_accum_steps'] != 0:
            if scaler and device.type == 'cuda':
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_loss += loss.item() * config['grad_accum_steps']
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            epoch_preds.extend(preds)
            epoch_labels.extend(labels.cpu().numpy())

        train_loss = epoch_loss / len(train_loader)
        train_acc = accuracy_score(epoch_labels, epoch_preds)

        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1'])
        history['val_auc'].append(val_metrics['auc'])
        history['learning_rates'].append(current_lr)

        logger.info(
            f"Epoch {epoch+1}/{config['epochs']} ({epoch_time:.1f}s) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} "
            f"P: {val_metrics['precision']:.4f} R: {val_metrics['recall']:.4f} "
            f"F1: {val_metrics['f1']:.4f} AUC: {val_metrics['auc']:.4f} | "
            f"LR: {current_lr:.2e}"
        )

        if val_metrics['f1'] > best_f1:
            best_f1 = val_metrics['f1']
            best_epoch = epoch + 1
            patience_counter = 0
            model_path = os.path.join(MODEL_DIR, 'bert_textcnn_best.pth')
            torch.save(model.state_dict(), model_path)
            logger.info(f"  -> Best model saved (F1={best_f1:.4f})")
        else:
            patience_counter += 1

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler is not None else None,
            'history': history,
            'best_f1': best_f1,
            'best_epoch': best_epoch,
            'patience_counter': patience_counter,
            'config': config,
        }, CHECKPOINT_PATH)
        logger.info(f"  -> Checkpoint saved: epoch={epoch+1}")

        if patience_counter >= config['patience']:
            logger.info(f"  -> Early stopping at epoch {epoch+1} (best F1={best_f1:.4f} at epoch {best_epoch})")
            break

    total_time = time.time() - start_time
    logger.info(f"Training completed in {total_time/60:.1f} minutes")
    logger.info(f"Best F1: {best_f1:.4f} at epoch {best_epoch}")

    return model, history, test_loader, device, tokenizer, {
        'best_f1': best_f1, 'best_epoch': best_epoch, 'total_time': total_time,
        'total_steps': total_steps, 'trainable_params': trainable, 'total_params': total
    }


def final_evaluation(model, test_loader, device):
    logger.info("=" * 60)
    logger.info("STEP 5: Final Model Evaluation")
    logger.info("=" * 60)

    criterion = nn.CrossEntropyLoss()
    metrics = evaluate(model, test_loader, criterion, device)

    logger.info(f"Test Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"  AUC Score: {metrics['auc']:.4f}")
    logger.info(f"  Confusion Matrix:\n{metrics['confusion_matrix']}")

    start = time.time()
    for batch in test_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    infer_time = time.time() - start
    infer_speed = len(test_loader.dataset) / infer_time
    logger.info(f"  Inference Speed: {infer_speed:.1f} samples/sec")
    metrics['inference_speed'] = infer_speed

    return metrics


def visualize_results(history, metrics, config, training_info):
    logger.info("=" * 60)
    logger.info("STEP 6: Results Visualization")
    logger.info("=" * 60)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('BERT-TextCNN Training Results', fontsize=16, fontweight='bold')

    epochs = range(1, len(history['train_loss']) + 1)

    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Loss Curve')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].set_title('Accuracy Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(epochs, history['val_precision'], 'g-', label='Precision', linewidth=2)
    axes[0, 2].plot(epochs, history['val_recall'], 'b-', label='Recall', linewidth=2)
    axes[0, 2].plot(epochs, history['val_f1'], 'r-', label='F1', linewidth=2)
    axes[0, 2].plot(epochs, history['val_auc'], 'm--', label='AUC', linewidth=2)
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('Validation Metrics')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(epochs, history['learning_rates'], 'k-', linewidth=2)
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].grid(True, alpha=0.3)

    cm = metrics['confusion_matrix']
    im = axes[1, 1].imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    axes[1, 1].set_title('Confusion Matrix')
    axes[1, 1].set_xlabel('Predicted')
    axes[1, 1].set_ylabel('Actual')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[1, 1].text(j, i, str(cm[i, j]), ha='center', va='center',
                           color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=14)

    fpr, tpr, _ = roc_curve(metrics['labels'], metrics['probabilities'])
    axes[1, 2].plot(fpr, tpr, 'b-', linewidth=2, label=f"AUC={metrics['auc']:.4f}")
    axes[1, 2].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1, 2].set_xlabel('False Positive Rate')
    axes[1, 2].set_ylabel('True Positive Rate')
    axes[1, 2].set_title('ROC Curve')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'training_results.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Training results saved to {fig_path}")

    fig_cm, ax_cm = plt.subplots(figsize=(8, 6))
    cm = metrics['confusion_matrix']
    im = ax_cm.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax_cm.set_title('Confusion Matrix', fontsize=14)
    ax_cm.set_xlabel('Predicted Label', fontsize=12)
    ax_cm.set_ylabel('True Label', fontsize=12)
    ax_cm.set_xticks([0, 1])
    ax_cm.set_yticks([0, 1])
    ax_cm.set_xticklabels(['Normal', 'Phishing'])
    ax_cm.set_yticklabels(['Normal', 'Phishing'])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax_cm.text(j, i, str(cm[i, j]), ha='center', va='center',
                      color='white' if cm[i, j] > cm.max() / 2 else 'black', fontsize=18, fontweight='bold')
    plt.colorbar(im)
    cm_path = os.path.join(OUTPUT_DIR, 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Confusion matrix saved to {cm_path}")


def save_model_and_version(model, metrics, config, training_info, history):
    logger.info("=" * 60)
    logger.info("STEP 7: Model Saving and Version Control")
    logger.info("=" * 60)

    model_path = os.path.join(MODEL_DIR, 'bert_textcnn_best.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f"Model saved to {model_path}")

    final_path = os.path.join(MODEL_DIR, 'bert_textcnn_final.pth')
    torch.save(model.state_dict(), final_path)
    logger.info(f"Model also saved to {final_path}")

    version = f"v2.0.{int(time.time()) % 10000}"
    version_info = {
        'version': version,
        'model_type': 'bert_textcnn',
        'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'training_config': config,
        'training_info': {
            'best_f1': training_info['best_f1'],
            'best_epoch': training_info['best_epoch'],
            'total_time_minutes': round(training_info['total_time'] / 60, 1),
            'trainable_params': training_info['trainable_params'],
            'total_params': training_info['total_params'],
        },
        'test_metrics': {
            'accuracy': float(metrics['accuracy']),
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'f1_score': float(metrics['f1']),
            'auc_score': float(metrics['auc']),
            'confusion_matrix': metrics['confusion_matrix'].tolist(),
            'inference_speed': float(metrics.get('inference_speed', 0)),
        },
        'training_history': {k: [float(v) for v in vals] for k, vals in history.items()},
    }

    version_path = os.path.join(MODEL_DIR, 'model_versions.json')
    versions = {}
    if os.path.exists(version_path):
        try:
            with open(version_path, 'r', encoding='utf-8') as f:
                versions = json.load(f)
        except:
            versions = {}

    for key in versions:
        if isinstance(versions[key], dict):
            versions[key]['is_active'] = False
            versions[key]['is_deployed'] = False

    version_info['is_active'] = True
    version_info['is_deployed'] = True
    version_key = f"{version}_{version_info['model_type']}"
    versions[version_key] = version_info

    with open(version_path, 'w', encoding='utf-8') as f:
        json.dump(versions, f, ensure_ascii=False, indent=2)
    logger.info(f"Version info saved: {version}")

    report_path = os.path.join(OUTPUT_DIR, 'training_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(version_info, f, ensure_ascii=False, indent=2)
    logger.info(f"Training report saved to {report_path}")

    return version_info


def main():
    config = {
        'seed': 42,
        'max_length': 128,
        'batch_size': 32,
        'learning_rate': 2e-5,
        'weight_decay': 0.01,
        'epochs': 10,
        'warmup_ratio': 0.1,
        'patience': 3,
        'grad_accum_steps': 2,
        'dataset': 'dataset_20260421_100k.csv',
        'resume': True,
    }

    logger.info("=" * 60)
    logger.info("  BERT-TextCNN Complete Training Pipeline")
    logger.info("=" * 60)
    logger.info(f"Configuration: {json.dumps(config, indent=2)}")

    data = load_and_preprocess_data(os.path.join(DATA_DIR, config['dataset']))

    model, history, test_loader, device, tokenizer, training_info = train_model(data, config)

    best_model_path = os.path.join(MODEL_DIR, 'bert_textcnn_best.pth')
    bert_model = AutoModel.from_pretrained('bert-base-chinese')
    best_model = BERTTextCNN(bert_model).to(device)
    best_model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))

    metrics = final_evaluation(best_model, test_loader, device)

    visualize_results(history, metrics, config, training_info)

    version_info = save_model_and_version(best_model, metrics, config, training_info, history)

    logger.info("=" * 60)
    logger.info("  TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"  Version: {version_info['version']}")
    logger.info(f"  Best Epoch: {training_info['best_epoch']}")
    logger.info(f"  Training Time: {training_info['total_time']/60:.1f} min")
    logger.info(f"  Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Test F1: {metrics['f1']:.4f}")
    logger.info(f"  Test AUC: {metrics['auc']:.4f}")
    logger.info(f"  Inference Speed: {metrics.get('inference_speed', 0):.1f} samples/sec")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
