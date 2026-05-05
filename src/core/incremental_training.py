import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import BertTokenizer, BertModel
from model_training import MultimodalModel, BERTTextCNN, ModelTrainer
from data_preprocessing import URLFeatureExtractor, NetworkBehaviorExtractor

class IncrementalTrainer:
    """增量训练模块，支持模型的持续学习和更新"""
    
    def __init__(self, model_dir="../models", data_dir="../data"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.url_extractor = URLFeatureExtractor()
        self.network_extractor = NetworkBehaviorExtractor()
        self.tokenizer = None
        self.bert = None
        self._load_bert()
    
    def _load_bert(self):
        """加载BERT模型"""
        from transformers import AutoTokenizer, AutoModel
        
        # 使用AutoModel和AutoTokenizer，它们会自动处理安全加载
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.bert = AutoModel.from_pretrained("bert-base-chinese")
        self.bert = self.bert.to(self.device)
    
    def collect_data(self, new_samples):
        """收集新的样本数据"""
        # 将新样本保存到文件
        os.makedirs(f"{self.data_dir}/incremental", exist_ok=True)
        
        # 加载现有的增量数据
        incremental_data_path = f"{self.data_dir}/incremental/incremental_data.csv"
        if os.path.exists(incremental_data_path):
            existing_data = pd.read_csv(incremental_data_path)
        else:
            existing_data = pd.DataFrame(columns=['text', 'url', 'label', 'scenario'])
        
        # 添加新样本
        new_data = pd.DataFrame(new_samples)
        combined_data = pd.concat([existing_data, new_data], ignore_index=True)
        
        # 去重
        combined_data = combined_data.drop_duplicates(subset=['text', 'url'])
        
        # 保存数据
        combined_data.to_csv(incremental_data_path, index=False, encoding='utf-8-sig')
        
        print(f"收集了 {len(new_samples)} 个新样本，总样本数：{len(combined_data)}")
        
        return combined_data
    
    def label_data(self, data, auto_label=False):
        """标注数据"""
        if auto_label:
            # 自动标注逻辑
            # 这里可以使用现有的模型进行预测作为自动标注
            from inference import PhishingDetector
            detector = PhishingDetector()
            
            labels = []
            for _, row in data.iterrows():
                result = detector.detect(row['text'], row['url'])
                labels.append(1 if result['prediction'] == '钓鱼' else 0)
            
            data['label'] = labels
        
        # 人工标注可以通过其他方式实现，这里返回数据
        return data
    
    def extract_features(self, data):
        """提取特征"""
        # 提取URL特征
        url_features = data.apply(lambda row: self.url_extractor.extract_features(row['url']), axis=1).apply(pd.Series)
        
        # 提取网络行为特征
        network_features = data.apply(lambda row: self.network_extractor.extract_features(row['url']), axis=1).apply(pd.Series)
        
        # 合并特征
        data_with_features = pd.concat([data, url_features, network_features], axis=1)
        
        return data_with_features
    
    def train_model(self, data, model_type="multimodal", epochs=10):
        """训练模型"""
        # 提取特征
        data_with_features = self.extract_features(data)
        
        # 保存数据
        incremental_features_path = f"{self.data_dir}/incremental/incremental_features.csv"
        data_with_features.to_csv(incremental_features_path, index=False, encoding='utf-8-sig')
        
        # 训练模型
        trainer = ModelTrainer(model_dir=self.model_dir)
        
        if model_type == "multimodal":
            model = trainer.train_multimodal(incremental_features_path, f"{model_type}_incremental")
        elif model_type == "bert_textcnn":
            # 这里需要实现BERT-TextCNN的增量训练
            # 暂时使用现有的训练方法
            model = trainer.train_bert_textcnn(incremental_features_path, f"{model_type}_incremental")
        else:
            # TextCNN模型
            model = trainer.train_textcnn(incremental_features_path, f"{model_type}_incremental")
        
        return model
    
    def evaluate_model(self, model, test_data):
        """评估模型"""
        # 提取特征
        test_data_with_features = self.extract_features(test_data)
        
        # 准备测试数据
        texts = test_data_with_features["text"].astype(str).tolist()
        labels = test_data_with_features["label"].tolist()
        
        # 提取URL特征和网络行为特征
        url_feature_cols = ['domain_length', 'is_https', 'has_suspicious_keywords', 
                           'subdomain_count', 'path_depth', 'param_count', 
                           'has_ip_address']
        network_feature_cols = ['response_time', 'load_status', 'redirect_count', 
                               'has_form', 'requests_sensitive_info']
        
        url_features = test_data_with_features[url_feature_cols].values.astype(float)
        network_features = test_data_with_features[network_feature_cols].values.astype(float)
        
        # 模型评估
        model.eval()
        y_true = []
        y_pred = []
        
        with torch.no_grad():
            for text, label, url_feat, network_feat in zip(texts, labels, url_features, network_features):
                inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                url_feat = torch.tensor(url_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
                network_feat = torch.tensor(network_feat, dtype=torch.float32).unsqueeze(0).to(self.device)
                
                outputs = model(**inputs, url_features=url_feat, network_features=network_feat)
                pred = torch.argmax(outputs, dim=1).cpu().numpy()[0]
                
                y_pred.append(pred)
                y_true.append(label)
        
        # 计算评估指标
        acc = accuracy_score(y_true, y_pred)
        pre = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        print(f"评估结果：")
        print(f"准确率：{acc:.4f}")
        print(f"精确率：{pre:.4f}")
        print(f"召回率：{rec:.4f}")
        print(f"F1分数：{f1:.4f}")
        
        return {
            "accuracy": acc,
            "precision": pre,
            "recall": rec,
            "f1": f1
        }
    
    def update_model(self, new_samples, model_type="multimodal", epochs=10):
        """更新模型"""
        # 收集数据
        collected_data = self.collect_data(new_samples)
        
        # 标注数据
        labeled_data = self.label_data(collected_data, auto_label=True)
        
        # 划分训练集和测试集
        train_data, test_data = train_test_split(labeled_data, test_size=0.2, random_state=42, stratify=labeled_data['label'])
        
        # 训练模型
        model = self.train_model(train_data, model_type, epochs)
        
        # 评估模型
        evaluation_results = self.evaluate_model(model, test_data)
        
        # 部署模型
        self.deploy_model(model, model_type)
        
        return {
            "model_type": model_type,
            "training_samples": len(train_data),
            "test_samples": len(test_data),
            "evaluation_results": evaluation_results
        }
    
    def deploy_model(self, model, model_type):
        """部署模型"""
        # 保存模型
        model_path = f"{self.model_dir}/{model_type}_best.pth"
        torch.save(model.state_dict(), model_path)
        
        print(f"模型已部署到：{model_path}")
        
        # 可以在这里添加模型部署到生产环境的逻辑
        # 例如，复制模型到指定位置，更新模型版本等
