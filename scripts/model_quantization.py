import torch
import torch.nn as nn
import time
import psutil
import numpy as np
from transformers import AutoTokenizer, AutoModel

class ModelQuantizer:
    def __init__(self, model_dir="models", output_dir="output"):
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
    
    def load_model(self, model_name):
        """加载模型"""
        from src.core.model_training import BERTTextCNN, MultimodalModel, TextCNN
        
        if model_name == "bert_textcnn":
            # 加载BERT模型
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            bert = AutoModel.from_pretrained("bert-base-chinese")
            model = BERTTextCNN(bert).to(self.device)
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device))
            return model, tokenizer
        elif model_name == "multimodal":
            # 加载BERT模型
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            bert = AutoModel.from_pretrained("bert-base-chinese")
            model = MultimodalModel(bert).to(self.device)
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device))
            return model, tokenizer
        elif model_name == "textcnn":
            # TextCNN模型需要特殊处理
            # 这里需要根据实际情况加载
            model = TextCNN(5000).to(self.device)
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device))
            return model, None
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
    
    def dynamic_quantization(self, model, model_name):
        """动态量化模型"""
        print(f"\n===== 动态量化 {model_name} 模型 =====")
        
        # 动态量化
        if isinstance(model, nn.Module):
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv1d, nn.Conv2d},
                dtype=torch.qint8
            )
            
            # 保存量化后的模型
            torch.save(quantized_model.state_dict(), f"{self.model_dir}/{model_name}_quantized.pth")
            print(f"✅ 动态量化完成，模型已保存到 {self.model_dir}/{model_name}_quantized.pth")
            return quantized_model
        else:
            print("❌ 模型类型不支持动态量化")
            return None
    
    def static_quantization(self, model, model_name, calibration_data):
        """静态量化模型"""
        print(f"\n===== 静态量化 {model_name} 模型 =====")
        
        # 准备模型进行静态量化
        model.eval()
        model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        model_prepared = torch.quantization.prepare(model)
        
        # 校准模型
        print("正在校准模型...")
        with torch.no_grad():
            for data in calibration_data:
                if isinstance(data, tuple) and len(data) > 0:
                    # 对于文本数据，需要特殊处理
                    if model_name in ["bert_textcnn", "multimodal"]:
                        # 这里需要根据实际情况进行处理
                        pass
                    else:
                        model_prepared(data)
        
        # 量化模型
        quantized_model = torch.quantization.convert(model_prepared)
        
        # 保存量化后的模型
        torch.save(quantized_model.state_dict(), f"{self.model_dir}/{model_name}_quantized_static.pth")
        print(f"✅ 静态量化完成，模型已保存到 {self.model_dir}/{model_name}_quantized_static.pth")
        return quantized_model
    
    def benchmark_model(self, model, model_name, test_data, tokenizer=None):
        """基准测试模型性能"""
        print(f"\n===== 测试 {model_name} 模型性能 =====")
        
        model.eval()
        total_time = 0
        total_memory = 0
        num_samples = len(test_data)
        
        start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        with torch.no_grad():
            start_time = time.time()
            for data in test_data:
                if model_name in ["bert_textcnn", "multimodal"] and tokenizer:
                    # 处理文本数据
                    if isinstance(data, str):
                        inputs = tokenizer(data, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                        if model_name == "multimodal":
                            # 对于多模态模型，需要添加网络特征
                            network_features = torch.zeros(1, 8, device=self.device)
                            outputs = model(**inputs, network_features=network_features)
                        else:
                            outputs = model(**inputs)
                else:
                    # 处理其他类型数据
                    if isinstance(data, torch.Tensor):
                        data = data.to(self.device)
                        outputs = model(data)
        
        end_time = time.time()
        end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # 内存使用，单位MB
        
        total_time = end_time - start_time
        total_memory = end_memory - start_memory
        inference_speed = num_samples / total_time
        
        print(f"推理速度: {inference_speed:.2f} 样本/秒")
        print(f"内存占用: {total_memory:.2f} MB")
        print(f"总推理时间: {total_time:.2f} 秒")
        
        return {
            "inference_speed": inference_speed,
            "memory_usage": total_memory,
            "total_time": total_time
        }
    
    def compare_models(self, original_model, quantized_model, model_name, test_data, tokenizer=None):
        """比较原始模型和量化模型的性能"""
        print(f"\n===== 比较 {model_name} 原始模型和量化模型 =====")
        
        # 测试原始模型
        print("测试原始模型:")
        original_performance = self.benchmark_model(original_model, f"{model_name}_original", test_data, tokenizer)
        
        # 测试量化模型
        print("\n测试量化模型:")
        quantized_performance = self.benchmark_model(quantized_model, f"{model_name}_quantized", test_data, tokenizer)
        
        # 计算性能提升
        speed_up = (quantized_performance["inference_speed"] - original_performance["inference_speed"]) / original_performance["inference_speed"] * 100
        memory_reduction = (original_performance["memory_usage"] - quantized_performance["memory_usage"]) / original_performance["memory_usage"] * 100
        
        print(f"\n性能对比:")
        print(f"推理速度提升: {speed_up:.2f}%")
        print(f"内存占用减少: {memory_reduction:.2f}%")
        
        return {
            "original": original_performance,
            "quantized": quantized_performance,
            "speed_up": speed_up,
            "memory_reduction": memory_reduction
        }

if __name__ == "__main__":
    import argparse
    import pandas as pd
    
    parser = argparse.ArgumentParser(description="模型量化工具")
    parser.add_argument("--model", type=str, default="bert_textcnn", choices=["bert_textcnn", "multimodal", "textcnn"], help="要量化的模型")
    parser.add_argument("--quantization_type", type=str, default="dynamic", choices=["dynamic", "static"], help="量化类型")
    args = parser.parse_args()
    
    quantizer = ModelQuantizer()
    
    # 加载模型
    model, tokenizer = quantizer.load_model(args.model)
    
    # 准备测试数据
    if args.model in ["bert_textcnn", "multimodal"]:
        # 加载一些测试文本
        test_df = pd.read_csv("../data/versions/dataset_20260411_chifraud.csv")
        test_texts = test_df["text"].fillna("").astype(str).tolist()[:100]  # 使用前100个样本
        test_data = test_texts
    else:
        # 对于TextCNN，需要准备TF-IDF特征
        from sklearn.feature_extraction.text import TfidfVectorizer
        test_df = pd.read_csv("../data/versions/dataset_20260411_chifraud.csv")
        test_texts = test_df["text"].fillna("").astype(str).tolist()[:100]
        
        # 拟合TF-IDF
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        tfidf.fit(test_texts)
        test_features = tfidf.transform(test_texts).toarray()
        test_data = [torch.FloatTensor(feat) for feat in test_features]
    
    # 量化模型
    if args.quantization_type == "dynamic":
        quantized_model = quantizer.dynamic_quantization(model, args.model)
    else:
        # 对于静态量化，需要校准数据
        calibration_data = test_data[:50]  # 使用前50个样本作为校准数据
        quantized_model = quantizer.static_quantization(model, args.model, calibration_data)
    
    # 比较模型性能
    if quantized_model:
        comparison = quantizer.compare_models(model, quantized_model, args.model, test_data, tokenizer)
        
        # 保存性能对比结果
        import json
        with open(f"{quantizer.output_dir}/{args.model}_quantization_comparison.json", "w", encoding="utf-8") as f:
            json.dump(comparison, f, ensure_ascii=False, indent=2)
        
        print(f"\n性能对比结果已保存到 {quantizer.output_dir}/{args.model}_quantization_comparison.json")
