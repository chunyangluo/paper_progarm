import os
import pandas as pd
import numpy as np
import re
import jieba
import requests
from urllib.parse import urlparse
from transformers import BertTokenizer, BertModel
import torch
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class FeatureExtractor:
    """多模态特征提取与预处理模块"""
    
    def __init__(self, data_dir="../data"):
        self.data_dir = data_dir
        self.tokenizer = None
        self.bert_model = None
        self._load_bert()
        self._create_directories()
    
    def _create_directories(self):
        """创建数据目录"""
        directories = [
            f"{self.data_dir}/processed/features",
            f"{self.data_dir}/processed/datasets"
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def _load_bert(self):
        """加载BERT模型"""
        print("正在加载BERT模型...")
        try:
            local_bert_path = r"C:\Users\chuny\.cache\huggingface\hub\models--bert-base-chinese"
            self.tokenizer = BertTokenizer.from_pretrained(local_bert_path)
            self.bert_model = BertModel.from_pretrained(local_bert_path)
            print("BERT模型加载成功！")
        except Exception as e:
            print(f"BERT模型加载失败: {e}")
            # 如果加载失败，使用默认值
            self.tokenizer = None
            self.bert_model = None
    
    def extract_text_features(self, text):
        """提取文本特征"""
        if not text or not self.tokenizer or not self.bert_model:
            return np.zeros(768)
        
        # 分词
        tokens = list(jieba.cut(text))
        
        # 去停用词
        stop_words = self._load_stop_words()
        tokens = [token for token in tokens if token not in stop_words]
        
        # BERT词嵌入
        inputs = self.tokenizer(' '.join(tokens), return_tensors="pt", max_length=128, truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.bert_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
        
        return embedding
    
    def _load_stop_words(self):
        """加载停用词"""
        stop_words = set()
        try:
            with open(f"{self.data_dir}/stopwords.txt", "r", encoding="utf-8") as f:
                for line in f:
                    stop_words.add(line.strip())
        except:
            # 如果文件不存在，使用默认停用词
            stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有", "看", "好", "自己", "这"}
        return stop_words
    
    def extract_url_features(self, url):
        """提取URL特征"""
        if not url:
            return np.zeros(8)
        
        # URL标准化
        url = self._normalize_url(url)
        
        # 字符特征提取
        features = []
        
        # 域名长度
        domain = self._extract_domain(url)
        features.append(len(domain))
        
        # 是否使用HTTPS
        features.append(1 if url.startswith('https://') else 0)
        
        # 是否包含可疑关键词
        suspicious_keywords = ['verify', 'secure', 'login', 'account', 'auth', 'payment', 'bank', 'financial', 'update', 'confirm', 'reset', 'password']
        features.append(1 if any(keyword in url.lower() for keyword in suspicious_keywords) else 0)
        
        # 子域名数量
        features.append(len(domain.split('.')) - 2 if domain else 0)
        
        # 路径深度
        path = urlparse(url).path
        features.append(len([p for p in path.split('/') if p]))
        
        # 参数数量
        query = urlparse(url).query
        features.append(len(query.split('&')) if query else 0)
        
        # 是否包含IP地址
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        features.append(1 if re.search(ip_pattern, url) else 0)
        
        # 顶级域名类型
        tld = domain.split('.')[-1] if domain else ''
        tld_map = {'com': 0, 'cn': 1, 'net': 2, 'org': 3, 'gov': 4, 'edu': 5, '其他': 6}
        features.append(tld_map.get(tld, 6))
        
        return np.array(features)
    
    def _normalize_url(self, url):
        """URL标准化"""
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'https://' + url
        return url
    
    def _extract_domain(self, url):
        """提取域名"""
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ''
    
    def extract_network_features(self, url):
        """提取网络行为特征"""
        if not url:
            return np.zeros(8)
        
        features = []
        
        try:
            # 响应时间（模拟）
            response = requests.get(url, timeout=5)
            features.append(response.elapsed.total_seconds())
            
            # 状态码
            features.append(response.status_code)
            
            # 重定向次数
            features.append(len(response.history))
            
            # 内容长度
            content_length = len(response.content)
            features.append(content_length)
            
            # 是否包含表单
            has_form = 1 if '<form' in response.text.lower() else 0
            features.append(has_form)
            
            # 是否包含脚本
            has_script = 1 if '<script' in response.text.lower() else 0
            features.append(has_script)
            
            # 是否包含iframe
            has_iframe = 1 if '<iframe' in response.text.lower() else 0
            features.append(has_iframe)
            
            # 页面标题长度
            title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE)
            title_length = len(title_match.group(1)) if title_match else 0
            features.append(title_length)
        except Exception as e:
            # 如果请求失败，使用默认值
            features = [3.0, 404, 0, 0, 0, 0, 0, 0]
        
        return np.array(features)
    
    def preprocess_features(self, features):
        """预处理特征"""
        # 标准化数值特征
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
        return features
    
    def process_all(self):
        """处理所有样本的特征"""
        print("开始提取多模态特征...")
        
        scenarios = ["sms", "email", "link"]
        processed_data = {}
        
        for scenario in scenarios:
            print(f"\n处理 {scenario} 场景...")
            
            # 加载处理后的数据
            phishing_df = pd.read_csv(f"{self.data_dir}/processed/phishing_{scenario}.csv")
            normal_df = pd.read_csv(f"{self.data_dir}/processed/normal_{scenario}.csv")
            
            # 提取特征
            phishing_features = self._extract_features_from_df(phishing_df)
            normal_features = self._extract_features_from_df(normal_df)
            
            # 合并数据
            phishing_features['label'] = 1
            normal_features['label'] = 0
            
            combined_df = pd.concat([phishing_features, normal_features], ignore_index=True)
            
            # 保存特征数据
            combined_df.to_csv(f"{self.data_dir}/processed/features/{scenario}_features.csv", index=False, encoding="utf-8-sig")
            
            processed_data[scenario] = combined_df
        
        print("\n特征提取完成！")
        return processed_data
    
    def _extract_features_from_df(self, df):
        """从DataFrame中提取特征"""
        features = []
        
        for _, row in df.iterrows():
            # 提取文本特征
            text = row.get('text', '')
            text_feature = self.extract_text_features(text)
            
            # 提取URL特征
            url = row.get('url', '')
            url_feature = self.extract_url_features(url)
            
            # 提取网络行为特征
            network_feature = self.extract_network_features(url)
            
            # 合并特征
            combined_feature = np.concatenate([text_feature, url_feature, network_feature])
            features.append(combined_feature)
        
        # 转换为DataFrame
        feature_columns = [f'text_feat_{i}' for i in range(768)] + \
                        [f'url_feat_{i}' for i in range(8)] + \
                        [f'network_feat_{i}' for i in range(8)]
        
        features_df = pd.DataFrame(features, columns=feature_columns)
        
        # 添加原始数据
        if 'text' in df.columns:
            features_df['text'] = df['text'].values
        if 'url' in df.columns:
            features_df['url'] = df['url'].values
        if 'scenario' in df.columns:
            features_df['scenario'] = df['scenario'].values
        
        return features_df

if __name__ == "__main__":
    extractor = FeatureExtractor()
    extractor.process_all()
