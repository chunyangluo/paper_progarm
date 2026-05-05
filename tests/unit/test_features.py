import unittest
from src.core.features.text_features import extract_text_features
from src.core.features.url_features import extract_url_features
from src.core.features.network_features import extract_network_features, calculate_estimated_domain_age
import asyncio

class TestFeaturesModule(unittest.TestCase):
    def test_extract_text_features_tfidf(self):
        """测试TF-IDF文本特征提取"""
        texts = ['这是一个测试文本', '另一个测试文本']
        features, vectorizer = extract_text_features(texts, method='tfidf', max_features=10)
        
        # 验证特征维度
        self.assertEqual(features.shape[0], 2)
        self.assertLessEqual(features.shape[1], 10)
    
    def test_extract_text_features_bert(self):
        """测试BERT文本特征提取"""
        texts = ['这是一个测试文本']
        features, tokenizer = extract_text_features(texts, method='bert')
        
        # 验证特征维度
        self.assertEqual(features.shape[0], 1)
        self.assertEqual(features.shape[1], 768)  # BERT-base-chinese的特征维度
    
    def test_extract_url_features(self):
        """测试URL特征提取"""
        url = "https://www.example.com/login?user=test&pass=123"
        features = extract_url_features(url)
        
        # 验证特征数量
        self.assertEqual(len(features), 16)
        
        # 验证特征值
        self.assertGreater(features[0], 0)  # URL长度
        self.assertEqual(features[1], 1)  # 是否使用HTTPS
        self.assertGreater(features[6], 0)  # 是否包含login关键词
    
    def test_calculate_estimated_domain_age(self):
        """测试域名年龄估算"""
        # 测试短域名
        age1 = calculate_estimated_domain_age("example.com")
        self.assertGreater(age1, 0)
        
        # 测试长域名
        age2 = calculate_estimated_domain_age("thisisalongdomainnameexample.com")
        self.assertGreater(age2, 0)
        
        # 测试空域名
        age3 = calculate_estimated_domain_age("")
        self.assertEqual(age3, 0)
    
    async def test_extract_network_features(self):
        """测试网络行为特征提取"""
        url = "https://www.example.com"
        features = await extract_network_features(url)
        
        # 验证特征数量
        self.assertEqual(len(features), 8)
        
        # 验证特征值
        self.assertGreaterEqual(features[0], 0)  # 响应时间
        self.assertIn(features[1], [0, 1])  # 加载状态
        self.assertGreaterEqual(features[2], 0)  # 重定向次数
        self.assertIn(features[3], [0, 1])  # 是否包含表单
        self.assertIn(features[4], [0, 1])  # 是否请求敏感信息
        self.assertEqual(features[5], 1)  # 是否使用HTTPS
        self.assertGreaterEqual(features[6], 0)  # 页面大小
        self.assertGreaterEqual(features[7], 0)  # 域名年龄
    
    def test_extract_network_features_sync(self):
        """同步测试网络行为特征提取"""
        url = "https://www.example.com"
        features = asyncio.run(extract_network_features(url))
        
        # 验证特征数量
        self.assertEqual(len(features), 8)

if __name__ == '__main__':
    unittest.main()
