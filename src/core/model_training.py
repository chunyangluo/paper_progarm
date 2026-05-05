import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import time
import logging
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')
os.environ["TOKENIZERS_PARALLISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

# 定义TextDataset类，用于BERT-TextCNN模型的DataLoader
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]

# 定义MultimodalDataset类，用于多模态模型的DataLoader
class MultimodalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, network_features):
        self.texts = texts
        self.labels = labels
        self.network_features = network_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.network_features[idx]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, num_classes=2):
        super().__init__()
        self.embedding = nn.Linear(vocab_size, embed_dim)
        self.conv1 = nn.Conv1d(1, 32, 2)
        self.conv2 = nn.Conv1d(1, 32, 3)
        self.conv3 = nn.Conv1d(1, 32, 4)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(96, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x1 = self.pool(torch.relu(self.conv1(x))).squeeze(-1)
        x2 = self.pool(torch.relu(self.conv2(x))).squeeze(-1)
        x3 = self.pool(torch.relu(self.conv3(x))).squeeze(-1)
        x = torch.cat([x1, x2, x3], dim=1)
        x = self.dropout(x)
        return self.fc(x)

class LegacyBERTTextCNN(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        # 特征提取层
        # BERT语义特征提取（全局语义）
        self.bert = bert_model
        # TextCNN局部特征提取
        self.conv1 = nn.Conv1d(768, 64, 2)  # 2-gram
        self.conv2 = nn.Conv1d(768, 64, 3)  # 3-gram
        self.conv3 = nn.Conv1d(768, 64, 4)  # 4-gram
        self.conv_bn = nn.BatchNorm1d(64)
        
        # 分类层
        self.dropout = nn.Dropout(0.6)
        self.fc = nn.Linear(64 * 3, 2)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        # BERT语义特征提取
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = out.last_hidden_state  # [batch_size, seq_len, 768]
        
        # TextCNN局部特征提取
        x = x.transpose(1, 2)  # [batch_size, 768, seq_len]
        x1 = torch.relu(self.conv_bn(self.conv1(x))).max(dim=2)[0]  # 2-gram特征
        x2 = torch.relu(self.conv_bn(self.conv2(x))).max(dim=2)[0]  # 3-gram特征
        x3 = torch.relu(self.conv_bn(self.conv3(x))).max(dim=2)[0]  # 4-gram特征
        
        # 特征融合
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.dropout(out)
        
        # 分类
        return self.fc(out)

# Keep one canonical BERT-TextCNN implementation across experiments,
# training scripts, and deployment.
from core.models.model_definitions import BERTTextCNN

class CrossModalAttention(nn.Module):
    """
    跨模态注意力融合模块
    """
    def __init__(self, text_dim, network_dim, hidden_dim):
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.network_proj = nn.Linear(network_dim, hidden_dim)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, text_feat, network_feat):
        # 维度适配：(batch, seq_len, dim)
        text_feat = self.text_proj(text_feat).unsqueeze(1)  # (B,1,H)
        network_feat = self.network_proj(network_feat).unsqueeze(1)  # (B,1,H)
        
        # 交叉注意力：文本作为Query，网络特征作为Key/Value
        attn_output, _ = self.attention(text_feat, network_feat, network_feat)
        
        # 残差连接
        text_feat = self.layer_norm(text_feat + attn_output)
        
        # 融合输出
        fusion_feat = torch.cat([text_feat.squeeze(1), network_feat.squeeze(1)], dim=1)
        return fusion_feat

class MultimodalModel(nn.Module):
    def __init__(self, bert_model, url_feature_dim=16, network_feature_dim=8):
        super().__init__()
        # 特征提取层
        # 1. BERT语义特征提取（全局语义）
        self.bert = bert_model
        # 2. TextCNN局部特征提取（局部关键特征）
        self.conv1 = nn.Conv1d(768, 64, 2)  # 2-gram
        self.conv2 = nn.Conv1d(768, 64, 3)  # 3-gram
        self.conv3 = nn.Conv1d(768, 64, 4)  # 4-gram
        self.conv_bn = nn.BatchNorm1d(64)
        
        # 3. 网络行为特征处理（使用真实网络特征）
        # 轻量级子网络，提升特征表征能力
        self.network_encoder = nn.Sequential(
            nn.Linear(network_feature_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 4. 文本特征编码器
        self.text_encoder = nn.Sequential(
            nn.Linear(64 * 3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 5. 跨模态注意力融合
        self.cross_attention = CrossModalAttention(128, 128, 256)
        
        # 6. 模态自适应权重
        self.modal_weights = nn.Parameter(torch.ones(3))
        
        # 7. 多模态融合层
        self.fusion = nn.Sequential(
            nn.Linear(512 + 128 + 128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU()
        )
        
        # 分类层：采用全连接层+Softmax激活函数
        self.classification = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
            # 注意：在训练时使用CrossEntropyLoss，它会自动应用Softmax
        )

    def forward(self, input_ids, attention_mask, url_features=None, network_features=None, token_type_ids=None):
        # 1. 文本特征提取
        # BERT语义特征提取（全局语义）
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        x = out.last_hidden_state  # [batch_size, seq_len, 768]
        
        # TextCNN局部特征提取（局部关键特征）
        x = x.transpose(1, 2)  # [batch_size, 768, seq_len]
        x1 = torch.relu(self.conv_bn(self.conv1(x))).max(dim=2)[0]  # 2-gram特征
        x2 = torch.relu(self.conv_bn(self.conv2(x))).max(dim=2)[0]  # 3-gram特征
        x3 = torch.relu(self.conv_bn(self.conv3(x))).max(dim=2)[0]  # 4-gram特征
        text_features = torch.cat([x1, x2, x3], dim=1)  # 文本特征向量 [batch_size, 192]
        
        # 文本特征编码
        text_encoded = self.text_encoder(text_features)  # [batch_size, 128]
        
        # 2. 网络行为特征处理
        if network_features is not None:
            network_encoded = self.network_encoder(network_features)  # 网络行为特征向量 [batch_size, 128]
        else:
            # 如果没有网络特征，使用零向量
            network_encoded = torch.zeros(text_encoded.size(0), 128, device=text_encoded.device)
        
        # 3. 跨模态注意力融合
        cross_attention_features = self.cross_attention(text_encoded, network_encoded)
        
        # 4. 模态自适应权重融合
        weights = torch.softmax(self.modal_weights, dim=0)
        weighted_text = weights[0] * text_encoded
        weighted_network = weights[1] * network_encoded
        weighted_cross = weights[2] * cross_attention_features
        
        # 5. 多模态特征融合
        all_features = torch.cat([weighted_text, weighted_network, weighted_cross], dim=1)
        fused_features = self.fusion(all_features)
        
        # 6. 分类
        return self.classification(fused_features)

class ModelTrainer:
    def __init__(self, model_dir="models", output_dir="output", device=None):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        if device:
            self.device = torch.device(device)
        else:
            # 默认优先使用GPU
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def _dataset_exists(self, dataset_path):
        """检查数据集是否存在"""
        if os.path.exists(dataset_path):
            print(f"✅ 数据集已存在：{dataset_path}")
            return True
        else:
            print(f"⚠️ 数据集不存在：{dataset_path}")
            return False
    
    def train_textcnn(self, dataset_path="../data/versions/dataset_20260411_chifraud.csv", model_name="textcnn"):
        """训练TextCNN模型"""
        import time
        print(f"\n===== 训练TextCNN模型 =====")
        
        if not self._dataset_exists(dataset_path):
            print(f"❌ 请先运行 update_dataset.py 生成数据集！")
            return
        
        # 加载训练数据
        train_df = pd.read_csv(dataset_path)
        X_train = train_df["text"].fillna("").astype(str).values
        y_train = train_df["label"].values
        
        # 分割数据为训练集和测试集
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        # 文本向量化
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train).toarray()
        X_test_tfidf = tfidf.transform(X_test).toarray()
        
        # 转换为张量
        train_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_tfidf), torch.LongTensor(y_train))
        test_ds = torch.utils.data.TensorDataset(torch.FloatTensor(X_test_tfidf), torch.LongTensor(y_test))
        # 优化DataLoader，解决数据瓶颈
        train_loader = torch.utils.data.DataLoader(
            train_ds, 
            batch_size=64, 
            shuffle=True,
            num_workers=4,          # 设为CPU核心数的1/2
            pin_memory=True,        # 锁页内存，CPU→GPU拷贝速度翻倍
            drop_last=True,         # 丢弃不完整batch，避免梯度异常
            prefetch_factor=2,      # 预取2个batch，GPU永远有数据跑
            persistent_workers=True # 持久化worker，避免重复初始化
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, 
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 初始化模型
        model = TextCNN(X_train_tfidf.shape[1]).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()
        
        # 训练
        epochs = 10
        best_f1 = 0
        train_loss_list = []
        val_f1_list = []
        total_train_time = 0
        total_val_time = 0
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # 记录训练开始时间
            train_start_time = time.time()
            
            # 添加batch级别的进度条
            for bx, by in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练中", leave=False):
                bx, by = bx.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * bx.size(0)
            
            # 计算训练时间
            train_time = time.time() - train_start_time
            total_train_time += train_time
            
            avg_loss = total_loss / len(train_loader.dataset)
            train_loss_list.append(avg_loss)
            
            # 验证
            model.eval()
            y_true = []
            y_pred = []
            
            # 记录验证开始时间
            val_start_time = time.time()
            
            with torch.no_grad():
                for bx, by in test_loader:
                    bx, by = bx.to(self.device), by.to(self.device)
                    out = model(bx)
                    pred = torch.argmax(out, dim=1).cpu().numpy()
                    y_pred.extend(pred)
                    y_true.extend(by.cpu().numpy())
            
            # 计算验证时间
            val_time = time.time() - val_start_time
            total_val_time += val_time
            
            f1 = f1_score(y_true, y_pred)
            val_f1_list.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"{self.model_dir}/{model_name}_best.pth")
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | 训练: {train_time:.1f}s | 验证: {val_time:.1f}s")
        
        # 计算总训练时间
        total_time = total_train_time + total_val_time
        print(f"✅ {model_name} 训练总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        
        # 绘制训练曲线
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label="训练损失")
        plt.title("训练损失曲线")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_f1_list, label="F1分数")
        plt.title("F1分数变化曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_training.png")
        
        # 评估
        model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth"))
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        
        # 测量推理速度
        import time
        import psutil
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(self.device), by.to(self.device)
                out = model(bx)
                prob = torch.softmax(out, dim=1).cpu().numpy()
                pred = torch.argmax(out, dim=1).cpu().numpy()
                y_pred.extend(pred)
                y_true.extend(by.cpu().numpy())
                y_prob.extend(prob[:, 1])
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        # 计算推理速度和资源占用
        total_inference_time = end_time - start_time
        num_samples = len(test_loader.dataset)
        inference_speed = num_samples / total_inference_time  # 样本/秒
        memory_usage = end_memory - start_memory  # 内存使用增加量，单位MB
        
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['正常', '钓鱼'])
        plt.yticks(tick_marks, ['正常', '钓鱼'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在矩阵中显示数字
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_confusion_matrix.png")
        
        # 绘制ROC曲线
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_dir}/{model_name}_roc_curve.png")
        
        # 计算误报率和漏报率
        false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        print(f"\n最终结果：")
        print(f"准确率：{acc:.4f}")
        print(f"精确率：{pre:.4f}")
        print(f"召回率：{rec:.4f}")
        print(f"F1分数：{f1:.4f}")
        print(f"AUC分数：{auc:.4f}")
        print(f"混淆矩阵：")
        print(cm)
        print(f"误报率：{false_positive_rate:.4f}")
        print(f"漏报率：{false_negative_rate:.4f}")
        print(f"推理速度：{inference_speed:.2f} 样本/秒")
        print(f"内存占用：{memory_usage:.2f} MB")
        
        return model
    
    def train_bert_textcnn(self, dataset_path="../data/versions/dataset_20260411_chifraud.csv", model_name="bert_textcnn"):
        """训练BERT-TextCNN模型"""
        import time
        print(f"\n===== 训练BERT-TextCNN模型 =====")
        
        if not self._dataset_exists(dataset_path):
            print(f"❌ 请先运行 update_dataset.py 生成数据集！")
            return
        
        # 加载训练数据
        train_df = pd.read_csv(dataset_path)
        train_texts = train_df["text"].fillna("").astype(str).tolist()
        train_labels = train_df["label"].astype(int).tolist()
        
        # 分割数据为训练集和测试集
        from sklearn.model_selection import train_test_split
        train_texts, test_texts, train_labels, test_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)
        
        # 加载BERT（直接从Hugging Face加载，避免本地路径的安全问题）
        print("正在加载BERT模型...")
        model_load_start_time = time.time()
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        bert = AutoModel.from_pretrained("bert-base-chinese")
        model_load_time = time.time() - model_load_start_time
        print(f"✅ BERT模型加载完成，耗时: {model_load_time:.1f} 秒")
        
        # 冻结BERT底层
        for param in bert.encoder.layer[:8].parameters():
            param.requires_grad = False
        
        # 初始化模型
        model = BERTTextCNN(bert).to(self.device)
        optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-2)
        criterion = nn.CrossEntropyLoss()
        
        # 优化DataLoader，解决数据瓶颈
        train_loader = DataLoader(
            TextDataset(train_texts, train_labels),
            batch_size=64,
            shuffle=True,
            num_workers=4,          # 设为CPU核心数的1/2
            pin_memory=True,        # 锁页内存，CPU→GPU拷贝速度翻倍
            drop_last=True,         # 丢弃不完整batch，避免梯度异常
            prefetch_factor=2,      # 预取2个batch，GPU永远有数据跑
            persistent_workers=True # 持久化worker，避免重复初始化
        )
        
        test_loader = DataLoader(
            TextDataset(test_texts, test_labels),
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 训练
        epochs = 10
        best_f1 = 0
        train_loss_list = []
        val_f1_list = []
        total_train_time = 0
        total_val_time = 0
        
        # 优化2：混合精度训练
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            
            # 记录训练开始时间
            train_start_time = time.time()
            
            # 添加batch级别的进度条
            for text, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练中", leave=False):
                inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                label = label.to(self.device)
                
                # 混合精度训练：效果不丢，速度翻倍
                with autocast():
                    outputs = model(**inputs)
                    loss = criterion(outputs, label)
                
                optimizer.zero_grad()
                scaler.scale(loss).backward()  # 梯度缩放，防止溢出
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
            
            # 计算训练时间
            train_time = time.time() - train_start_time
            total_train_time += train_time
            
            avg_loss = total_loss / len(train_loader)
            train_loss_list.append(avg_loss)
            
            # 验证
            model.eval()
            y_true = []
            y_pred = []
            
            # 记录验证开始时间
            val_start_time = time.time()
            
            with torch.no_grad():
                for text, label in test_loader:
                    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    logits = model(**inputs)
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    y_pred.extend(pred)
                    y_true.extend(label.numpy())
            
            # 计算验证时间
            val_time = time.time() - val_start_time
            total_val_time += val_time
            
            f1 = f1_score(y_true, y_pred)
            val_f1_list.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(model.state_dict(), f"{self.model_dir}/{model_name}_best.pth")
            
            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | F1: {f1:.4f} | 训练: {train_time:.1f}s | 验证: {val_time:.1f}s")
        
        # 计算总训练时间
        total_time = total_train_time + total_val_time
        print(f"✅ {model_name} 训练总耗时: {total_time:.1f} 秒 ({total_time/60:.1f} 分钟)")
        
        # 绘制训练曲线
        plt.figure(figsize=(8, 4))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss_list, label="训练损失")
        plt.title("训练损失曲线")
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.plot(val_f1_list, label="F1分数")
        plt.title("F1分数变化曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_training.png")
        
        # 评估
        model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth"))
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        
        # 测量推理速度
        import time
        import psutil
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        with torch.no_grad():
            for text, label in test_loader:
                inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                logits = model(**inputs)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                y_pred.extend(pred)
                y_true.extend(label.numpy())
                y_prob.extend(prob[:, 1])
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        # 计算推理速度和资源占用
        total_inference_time = end_time - start_time
        num_samples = len(test_loader.dataset)
        inference_speed = num_samples / total_inference_time  # 样本/秒
        memory_usage = end_memory - start_memory  # 内存使用增加量，单位MB
        
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['正常', '钓鱼'])
        plt.yticks(tick_marks, ['正常', '钓鱼'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在矩阵中显示数字
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_confusion_matrix.png")
        
        # 绘制ROC曲线
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_dir}/{model_name}_roc_curve.png")
        
        # 计算误报率和漏报率
        false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        print(f"\n最终结果：")
        print(f"准确率：{acc:.4f}")
        print(f"精确率：{pre:.4f}")
        print(f"召回率：{rec:.4f}")
        print(f"F1分数：{f1:.4f}")
        print(f"AUC分数：{auc:.4f}")
        print(f"混淆矩阵：")
        print(cm)
        print(f"误报率：{false_positive_rate:.4f}")
        print(f"漏报率：{false_negative_rate:.4f}")
        print(f"推理速度：{inference_speed:.2f} 样本/秒")
        print(f"内存占用：{memory_usage:.2f} MB")
        
        return model
    
    def _generate_url_features(self, url):
        """生成URL特征（替代维度为0的问题）"""
        # 确保url是字符串类型
        if url is None:
            return [0]*16  # 16维基础特征
        if not isinstance(url, str):
            url = str(url)
        if not url:
            return [0]*16  # 16维基础特征
        
        # 提取URL关键特征
        features = []
        features.append(len(url))  # URL长度
        features.append(1 if "https" in url else 0)  # 是否HTTPS
        features.append(url.count("."))  # 域名层级
        features.append(1 if "login" in url.lower() else 0)  # 包含login关键词
        features.append(1 if "bank" in url.lower() else 0)  # 包含bank关键词
        features.append(1 if "secure" in url.lower() else 0)  # 包含secure关键词
        features.append(1 if "verify" in url.lower() else 0)  # 包含verify关键词
        features.append(1 if "account" in url.lower() else 0)  # 包含account关键词
        features.append(1 if "auth" in url.lower() else 0)  # 包含auth关键词
        features.append(1 if "payment" in url.lower() else 0)  # 包含payment关键词
        features.append(1 if "password" in url.lower() else 0)  # 包含password关键词
        features.append(1 if "reset" in url.lower() else 0)  # 包含reset关键词
        features.append(1 if "confirm" in url.lower() else 0)  # 包含confirm关键词
        features.append(1 if "update" in url.lower() else 0)  # 包含update关键词
        features.append(1 if "financial" in url.lower() else 0)  # 包含financial关键词
        features.append(1 if "phish" in url.lower() else 0)  # 包含phish关键词
        
        return features[:16]  # 固定维度
    
    def _generate_network_features(self, url):
        """生成网络行为特征"""
        # 确保url是字符串类型
        if url is None:
            return [0]*8  # 8维基础特征
        if not isinstance(url, str):
            url = str(url)
        if not url:
            return [0]*8  # 8维基础特征
        
        # 生成网络行为特征
        features = []
        features.append(1.5)  # 响应时间
        features.append(1)  # 加载状态
        features.append(1)  # 重定向次数
        features.append(1)  # 是否包含表单
        features.append(1)  # 是否请求敏感信息
        features.append(0)  # 是否使用HTTPS
        features.append(0)  # 是否包含验证码
        features.append(0)  # 是否需要登录
        
        return features[:8]  # 固定维度
    
    def validate_dataset(self, df):
        """验证数据集质量，提前发现问题"""
        # 检查文本列非空
        df["text"] = df["text"].fillna("")
        df = df[df["text"].str.len() > 2]  # 过滤过短文本
        
        # 检查标签平衡
        print(f"原始数据集统计：总样本{len(df)}, 钓鱼{df['label'].sum()}, 正常{len(df)-df['label'].sum()}")
        
        # 1. 保留所有样本，不裁剪！！！
        df_phish = df[df['label'] == 1]
        df_normal = df[df['label'] == 0]
        
        # 2. 对正常样本过采样（复制，不造假）
        df_normal_balanced = df_normal.sample(n=len(df_phish), replace=True, random_state=42)
        
        # 3. 合并，样本量 = 全部钓鱼样本
        balanced_df = pd.concat([df_phish, df_normal_balanced], axis=0).sample(frac=1, random_state=42)
        
        print(f"平衡后：总样本{len(balanced_df)}, 钓鱼{len(df_phish)}, 正常{len(df_normal_balanced)}")
        return balanced_df
    
    def train_multimodal(self, data_path=None, model_name="multimodal"):
        """训练多模态模型（使用真实网络特征）"""
        print(f"\n===== 训练多模态模型 =====")
        print("使用真实网络特征，8维特征维度与模型匹配")
        
        # 如果没有指定数据路径，使用包含网络特征的数据集
        if data_path is None:
            data_path = "../data/versions/dataset_20260411_chifraud_with_network_features_async.csv"
        
        # 加载训练数据
        df = pd.read_csv(data_path)
        df = self.validate_dataset(df)
        
        # 确保文本输入有效
        def preprocess_text(text):
            """确保文本输入为有效字符串"""
            if text is None:
                return ""
            if not isinstance(text, str):
                return str(text)
            return text.strip()
        
        # 预处理文本
        texts = [preprocess_text(text) for text in df["text"].tolist()]
        labels = df["label"].tolist()
        
        # 解析网络特征
        def parse_network_features(feature_str):
            """解析网络特征字符串为列表"""
            if feature_str is None or pd.isna(feature_str):
                return [0.0] * 8
            try:
                features = list(map(float, feature_str.split(',')))
                # 确保特征维度为8
                if len(features) != 8:
                    return [0.0] * 8
                return features
            except:
                return [0.0] * 8
        
        # 提取网络特征
        network_features = np.array([parse_network_features(f) for f in df.get("network_features", [None] * len(texts))], dtype=np.float32)
        
        # 验证网络特征维度
        print(f"网络特征维度: {network_features.shape[1]}")
        
        # 划分数据集
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import roc_auc_score
        # 先划分训练+验证集和测试集
        train_val_texts, test_texts, train_val_labels, test_labels, \
        train_val_network_features, test_network_features = train_test_split(
            texts, labels, network_features, test_size=0.2, random_state=42
        )
        
        # 再划分训练集和验证集
        train_texts, val_texts, train_labels, val_labels, \
        train_network_features, val_network_features = train_test_split(
            train_val_texts, train_val_labels, train_val_network_features, test_size=0.2, random_state=42
        )
        
        # 验证数据集统计
        print(f"训练集统计：总样本{len(train_texts)}, 钓鱼{sum(train_labels)}, 正常{len(train_texts)-sum(train_labels)}")
        print(f"验证集统计：总样本{len(val_texts)}, 钓鱼{sum(val_labels)}, 正常{len(val_texts)-sum(val_labels)}")
        print(f"测试集统计：总样本{len(test_texts)}, 钓鱼{sum(test_labels)}, 正常{len(test_texts)-sum(test_labels)}")
        
        # 预处理测试集文本
        test_texts = [preprocess_text(text) for text in test_texts]
        
        # 加载BERT（直接从Hugging Face加载，避免本地路径的安全问题）
        from transformers import AutoTokenizer, AutoModel
        tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        bert = AutoModel.from_pretrained("bert-base-chinese")
        
        # 冻结BERT底层
        for param in bert.encoder.layer[:8].parameters():
            param.requires_grad = False
        
        # 初始化模型
        model = MultimodalModel(bert).to(self.device)
        
        # 模型训练参数设置
        # 优化器：AdamW
        # 学习率：余弦退火衰减
        # 批次大小：64
        # 训练轮数：10
        
        # 计算类别权重（正常样本更少，权重更高）
        class_weights = torch.tensor([1.0, 1.0], device=self.device)  # 初始权重
        
        # 标签平滑
        class LabelSmoothingLoss(nn.Module):
            def __init__(self, smoothing=0.05, weight=None):
                super(LabelSmoothingLoss, self).__init__()
                self.smoothing = smoothing
                self.weight = weight
            
            def forward(self, outputs, targets):
                # 标签平滑：真实标签1→0.95，0→0.05
                targets = targets.unsqueeze(1)
                num_classes = outputs.size(1)
                device = outputs.device
                smooth_label = torch.full_like(outputs, self.smoothing / (num_classes - 1), device=device)
                smooth_label.scatter_(1, targets, 1.0 - self.smoothing)
                loss = torch.sum(-smooth_label * torch.log_softmax(outputs, dim=1), dim=1)
                if self.weight is not None:
                    loss = loss * self.weight[targets.squeeze()]
                return loss.mean()
        
        # 对不同模态子网络设置差异化学习率
        # 网络特征子网络：学习率 = 1e-4
        # 文本子网络：学习率 = 1e-3
        # BERT：学习率 = 1e-5
        params = [
            {'params': model.network_encoder.parameters(), 'lr': 1e-4},
            {'params': model.text_encoder.parameters(), 'lr': 1e-3},
            {'params': model.cross_attention.parameters(), 'lr': 1e-3},
            {'params': model.fusion.parameters(), 'lr': 1e-3},
            {'params': model.classification.parameters(), 'lr': 1e-3},
            {'params': model.bert.parameters(), 'lr': 1e-5}
        ]
        
        optimizer = optim.AdamW(params, weight_decay=1e-4)  # L2正则，权重衰减=1e-4
        criterion = LabelSmoothingLoss(smoothing=0.05, weight=class_weights)  # 标签平滑 + 类别权重
        
        # 学习率调度器：余弦退火衰减
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-7)
        
        # 早停机制（监控验证集AUC）
        patience = 5
        best_auc = 0
        early_stop_counter = 0
        
        # 优化DataLoader，解决数据瓶颈（使用真实网络特征）
        train_loader = DataLoader(
            MultimodalDataset(train_texts, train_labels, train_network_features), 
            batch_size=64, 
            shuffle=True,
            num_workers=4,          # 设为CPU核心数的1/2
            pin_memory=True,        # 锁页内存，CPU→GPU拷贝速度翻倍
            drop_last=True,         # 丢弃不完整batch，避免梯度异常
            prefetch_factor=2,      # 预取2个batch，GPU永远有数据跑
            persistent_workers=True # 持久化worker，避免重复初始化
        )
        val_loader = DataLoader(
            MultimodalDataset(val_texts, val_labels, val_network_features), 
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        test_loader = DataLoader(
            MultimodalDataset(test_texts, test_labels, test_network_features), 
            batch_size=64,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        
        # 训练
        epochs = 10
        train_loss_list = []
        val_f1_list = []
        val_acc_list = []
        val_auc_list = []
        
        # 梯度累积参数
        accumulation_steps = 2  # 模拟批次大小为 64 * 2 = 128
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()  # 开始时清零梯度
            
            # 添加batch级别的进度条
            for i, (text, label, network_feat) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} 训练中", leave=False)):
                inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                label = label.to(self.device)
                network_feat = torch.tensor(network_feat, dtype=torch.float32).to(self.device)
                
                outputs = model(**inputs, network_features=network_feat)
                loss = criterion(outputs, label)
                
                # 梯度累积
                loss = loss / accumulation_steps  # 缩放损失
                loss.backward()
                
                # 每accumulation_steps步更新一次参数
                if (i + 1) % accumulation_steps == 0:
                    # 梯度裁剪
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_loss += loss.item() * accumulation_steps  # 恢复真实损失
            
            # 如果批次不是accumulation_steps的整数倍，最后更新一次
            if (i + 1) % accumulation_steps != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
            
            # 更新学习率
            scheduler.step()
            
            avg_loss = total_loss / len(train_loader)
            train_loss_list.append(avg_loss)
            
            # 验证
            model.eval()
            y_true = []
            y_pred = []
            y_prob = []
            
            with torch.no_grad():
                for text, label, network_feat in val_loader:
                    inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    network_feat = torch.FloatTensor(network_feat).to(self.device)
                    
                    logits = model(**inputs, network_features=network_feat)
                    prob = torch.softmax(logits, dim=1).cpu().numpy()
                    pred = torch.argmax(logits, dim=1).cpu().numpy()
                    
                    y_pred.extend(pred)
                    y_true.extend(label.numpy())
                    y_prob.extend(prob[:, 1])
            
            acc = accuracy_score(y_true, y_pred)
            pre = precision_score(y_true, y_pred)
            rec = recall_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            auc = roc_auc_score(y_true, y_prob)
            
            val_f1_list.append(f1)
            val_acc_list.append(acc)
            val_auc_list.append(auc)
            
            # 早停检查（监控验证集AUC）
            if auc > best_auc:
                best_auc = auc
                early_stop_counter = 0
                # 确保模型保存路径正确
                save_path = f"{self.model_dir}/{model_name}_best.pth"
                torch.save(model.state_dict(), save_path)
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Saved best model to {save_path}")
            else:
                early_stop_counter += 1
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Acc: {acc:.4f} | F1: {f1:.4f} | AUC: {auc:.4f} | Early stop counter: {early_stop_counter}")
                
                if early_stop_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        # 绘制训练曲线
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 4, 1)
        plt.plot(train_loss_list, label="训练损失")
        plt.title("训练损失曲线")
        plt.legend()
        plt.subplot(1, 4, 2)
        plt.plot(val_acc_list, label="验证准确率")
        plt.title("验证准确率曲线")
        plt.legend()
        plt.subplot(1, 4, 3)
        plt.plot(val_f1_list, label="F1分数")
        plt.title("F1分数变化曲线")
        plt.legend()
        plt.subplot(1, 4, 4)
        plt.plot(val_auc_list, label="AUC分数")
        plt.title("AUC分数变化曲线")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_training.png")
        
        # 评估 - 使用测试集进行最终评估
        model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth"))
        model.eval()
        y_true = []
        y_pred = []
        y_prob = []
        
        # 测量推理速度
        import time
        import psutil
        
        start_time = time.time()
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        with torch.no_grad():
            for text, label, network_feat in test_loader:
                inputs = tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                network_feat = torch.tensor(network_feat, dtype=torch.float32).to(self.device)
                
                logits = model(**inputs, network_features=network_feat)
                prob = torch.softmax(logits, dim=1).cpu().numpy()
                pred = torch.argmax(logits, dim=1).cpu().numpy()
                
                y_pred.extend(pred)
                y_true.extend(label.numpy())
                y_prob.extend(prob[:, 1])
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        # 计算推理速度和资源占用
        total_inference_time = end_time - start_time
        num_samples = len(test_loader.dataset)
        inference_speed = num_samples / total_inference_time  # 样本/秒
        memory_usage = end_memory - start_memory  # 内存使用增加量，单位MB
        
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 计算混淆矩阵
        from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
        cm = confusion_matrix(y_true, y_pred)
        auc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        
        # 绘制混淆矩阵
        plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('混淆矩阵')
        plt.colorbar()
        tick_marks = np.arange(2)
        plt.xticks(tick_marks, ['正常', '钓鱼'])
        plt.yticks(tick_marks, ['正常', '钓鱼'])
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        
        # 在矩阵中显示数字
        thresh = cm.max() / 2.
        for i, j in np.ndindex(cm.shape):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/{model_name}_confusion_matrix.png")
        
        # 绘制ROC曲线
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(f"{self.output_dir}/{model_name}_roc_curve.png")
        
        print(f"\n最终结果（测试集）：")
        print(f"准确率：{acc:.4f}")
        print(f"精确率：{pre:.4f}")
        print(f"召回率：{rec:.4f}")
        print(f"F1分数：{f1:.4f}")
        print(f"AUC分数：{auc:.4f}")
        print(f"混淆矩阵：")
        print(cm)
        
        # 计算误报率和漏报率
        false_positive_rate = cm[0, 1] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
        false_negative_rate = cm[1, 0] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
        
        print(f"误报率：{false_positive_rate:.4f}")
        print(f"漏报率：{false_negative_rate:.4f}")
        
        return model

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="训练模型，可选择使用CPU或GPU")
    parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="指定使用的设备")
    args = parser.parse_args()
    
    trainer = ModelTrainer(device=args.device)
    
    # 只训练多模态模型
    trainer.train_multimodal(model_name="multimodal")
