from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch
import logging

logger = logging.getLogger(__name__)

def extract_text_features(texts, method='tfidf', max_features=5000, ngram_range=(1, 2)):
    """
    提取文本特征
    
    参数:
        texts: 文本列表
        method: 特征提取方法，可选 'tfidf' 或 'bert'
        max_features: 最大特征数（仅用于tfidf）
        ngram_range: n-gram范围（仅用于tfidf）
    
    返回:
        特征矩阵或特征向量
    """
    try:
        logger.info(f"提取文本特征，方法: {method}")
        
        if method == 'tfidf':
            # 使用TF-IDF提取特征
            vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=ngram_range)
            features = vectorizer.fit_transform(texts)
            logger.info(f"TF-IDF特征提取完成，特征维度: {features.shape}")
            return features, vectorizer
        elif method == 'bert':
            # 使用BERT提取特征
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            model = AutoModel.from_pretrained("bert-base-chinese")
            
            # 处理文本
            inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
            
            # 获取BERT特征
            with torch.no_grad():
                outputs = model(**inputs)
                features = outputs.last_hidden_state[:, 0, :].numpy()  # 使用[CLS] token的特征
            
            logger.info(f"BERT特征提取完成，特征维度: {features.shape}")
            return features, tokenizer
        else:
            raise ValueError(f"不支持的特征提取方法: {method}")
    except Exception as e:
        logger.error(f"提取文本特征失败: {e}")
        raise
