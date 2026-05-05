import unittest
import pandas as pd
import numpy as np
from src.core.data.data_loader import load_dataset, preprocess_text
from src.core.data.data_splitter import split_dataset
from src.core.data.data_augmenter import augment_data
import tempfile
import os

class TestDataModule(unittest.TestCase):
    def setUp(self):
        # 创建测试数据
        self.test_data = {
            'text': ['这是一个测试文本', '另一个测试文本', '钓鱼文本测试'],
            'label': [0, 0, 1],
            'url': ['http://example.com', 'http://test.com', 'http://phishing.com']
        }
        self.df = pd.DataFrame(self.test_data)
        
        # 创建临时文件
        self.temp_file = tempfile.NamedTemporaryFile(suffix='.csv', delete=False)
        self.df.to_csv(self.temp_file, index=False)
        self.temp_file.close()
    
    def tearDown(self):
        # 删除临时文件
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)
    
    def test_load_dataset(self):
        """测试加载数据集"""
        df = load_dataset(self.temp_file.name)
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df.columns), ['text', 'label', 'url'])
    
    def test_preprocess_text(self):
        """测试文本预处理"""
        # 测试正常文本
        text = "  这是一个  测试文本  "
        result = preprocess_text(text)
        self.assertEqual(result, "这是一个 测试文本")
        
        # 测试None值
        result = preprocess_text(None)
        self.assertEqual(result, "")
        
        # 测试非字符串值
        result = preprocess_text(123)
        self.assertEqual(result, "123")
    
    def test_split_dataset(self):
        """测试数据集分割"""
        X = self.df['text'].values
        y = self.df['label'].values
        
        X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, test_size=0.33, val_size=0.5)
        
        # 验证分割后的数据集大小
        self.assertEqual(len(X_train), 1)
        self.assertEqual(len(X_val), 1)
        self.assertEqual(len(X_test), 1)
        
        # 验证标签分割正确
        self.assertEqual(len(y_train), 1)
        self.assertEqual(len(y_val), 1)
        self.assertEqual(len(y_test), 1)
    
    def test_augment_data(self):
        """测试数据增强"""
        augmented_df = augment_data(self.df, augment_factor=1)
        
        # 验证增强后的数据大小
        self.assertEqual(len(augmented_df), 6)
        
        # 验证增强后的数据包含原始数据
        original_texts = set(self.df['text'])
        augmented_texts = set(augmented_df['text'])
        for text in original_texts:
            self.assertIn(text, augmented_texts)

if __name__ == '__main__':
    unittest.main()
