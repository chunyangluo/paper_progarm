import unittest
import pandas as pd
import tempfile
import os
from src.core.data.data_loader import load_dataset, preprocess_text
from src.core.data.data_splitter import split_dataset
from src.core.features.text_features import extract_text_features
from src.core.inference.inference_engine import InferenceEngine

class TestSystemIntegration(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.test_data = {
            'text': ['这是一个正常文本', '钓鱼网站警告：您的账户已被盗用，请立即登录验证', '这是另一个正常文本'],
            'label': [0, 1, 0],
            'url': ['http://example.com', 'http://phishing.com', 'http://test.com']
        }
        self.df = pd.DataFrame(self.test_data)
        
        # 创建临时文件
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.df.to_csv(self.temp_file, index=False)
        self.temp_file.close()
        
        # 模型路径
        self.bert_textcnn_model_path = "models/bert_textcnn_best.pth"
    
    def tearDown(self):
        # 删除临时文件
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
    
    def test_full_integration(self):
        """测试完整的系统集成流程"""
        if not os.path.exists(self.bert_textcnn_model_path):
            self.skipTest(f"模型文件不存在: {self.bert_textcnn_model_path}")
        
        # 1. 加载数据集
        df = load_dataset(self.temp_file.name)
        self.assertEqual(len(df), 3)
        
        # 2. 预处理文本
        df['text'] = df['text'].apply(preprocess_text)
        
        # 3. 分割数据集
        X = df['text'].values
        y = df['label'].values
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.33, val_size=0.5)
        
        # 4. 提取文本特征
        features, vectorizer = extract_text_features(X_train, method='tfidf')
        self.assertEqual(features.shape[0], len(X_train))
        
        # 5. 模型推理
        engine = InferenceEngine(self.bert_textcnn_model_path, 'bert_textcnn')
        
        # 测试单次推理
        result = engine.infer(X_test[0])
        self.assertIn('probabilities', result)
        self.assertIn('prediction', result)
        self.assertIn('confidence', result)
        
        # 测试批量推理
        results = engine.batch_infer(X_test)
        self.assertEqual(len(results), len(X_test))
        for result in results:
            self.assertIn('probabilities', result)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)

if __name__ == '__main__':
    unittest.main()
