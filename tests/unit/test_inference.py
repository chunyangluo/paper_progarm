import unittest
import torch
from src.core.inference.inference_engine import InferenceEngine
import os

class TestInferenceModule(unittest.TestCase):
    def setUp(self):
        # 检查模型文件是否存在
        self.bert_textcnn_model_path = "models/bert_textcnn_best.pth"
        self.multimodal_model_path = "models/multimodal_best.pth"
        self.textcnn_model_path = "models/textcnn_best.pth"
        
        # 测试数据
        self.test_text = "这是一个测试文本"
        self.test_texts = ["这是一个测试文本", "另一个测试文本"]
        self.test_network_features = [0.5] * 8
        self.test_network_features_list = [[0.5] * 8, [0.5] * 8]
    
    def test_bert_textcnn_inference(self):
        """测试BERT-TextCNN模型推理"""
        if os.path.exists(self.bert_textcnn_model_path):
            engine = InferenceEngine(self.bert_textcnn_model_path, 'bert_textcnn')
            
            # 测试单次推理
            result = engine.infer(self.test_text)
            self.assertIn('probabilities', result)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertEqual(len(result['probabilities']), 2)
            
            # 测试批量推理
            results = engine.batch_infer(self.test_texts)
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn('probabilities', result)
                self.assertIn('prediction', result)
                self.assertIn('confidence', result)
        else:
            self.skipTest(f"模型文件不存在: {self.bert_textcnn_model_path}")
    
    def test_multimodal_inference(self):
        """测试多模态模型推理"""
        if os.path.exists(self.multimodal_model_path):
            engine = InferenceEngine(self.multimodal_model_path, 'multimodal')
            
            # 测试单次推理
            result = engine.infer(self.test_text, self.test_network_features)
            self.assertIn('probabilities', result)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertEqual(len(result['probabilities']), 2)
            
            # 测试批量推理
            results = engine.batch_infer(self.test_texts, self.test_network_features_list)
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn('probabilities', result)
                self.assertIn('prediction', result)
                self.assertIn('confidence', result)
        else:
            self.skipTest(f"模型文件不存在: {self.multimodal_model_path}")
    
    def test_textcnn_inference(self):
        """测试TextCNN模型推理"""
        if os.path.exists(self.textcnn_model_path):
            engine = InferenceEngine(self.textcnn_model_path, 'textcnn')
            
            # 测试单次推理
            test_feature = [0.0] * 5000  # 模拟TF-IDF特征
            result = engine.infer(test_feature)
            self.assertIn('probabilities', result)
            self.assertIn('prediction', result)
            self.assertIn('confidence', result)
            self.assertEqual(len(result['probabilities']), 2)
            
            # 测试批量推理
            test_features = [[0.0] * 5000, [0.0] * 5000]
            results = engine.batch_infer(test_features)
            self.assertEqual(len(results), 2)
            for result in results:
                self.assertIn('probabilities', result)
                self.assertIn('prediction', result)
                self.assertIn('confidence', result)
        else:
            self.skipTest(f"模型文件不存在: {self.textcnn_model_path}")

if __name__ == '__main__':
    unittest.main()
