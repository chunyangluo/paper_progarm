#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合性能测试脚本

此脚本提供全面的性能测试功能，包括：
- 自动化性能基准测试
- 关键指标实时监控（吞吐量、延迟、CPU/GPU利用率等）
- 性能数据记录与可视化
- 多场景测试用例（不同输入规模、硬件配置下的性能表现）
- 生成标准化性能评估报告
"""

import torch
import numpy as np
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModel
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('performance_tester.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PerformanceTester:
    def __init__(self, model_dir="models", output_dir="output"):
        """
        初始化性能测试器
        
        参数:
            model_dir: 模型目录
            output_dir: 输出目录
        """
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # 测试配置
        self.test_configs = {
            "batch_sizes": [1, 8, 16, 32, 64, 128],
            "input_sizes": [10, 100, 1000],
            "models": ["bert_textcnn", "multimodal", "textcnn"]
        }
        
        # 性能数据
        self.performance_data = []
        
        # 设备信息
        self.device_info = {
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "cpu_count": psutil.cpu_count(),
            "memory_total": psutil.virtual_memory().total / (1024 * 1024 * 1024),  # GB
        }
        
        if torch.cuda.is_available():
            self.device_info["gpu_name"] = torch.cuda.get_device_name(0)
            self.device_info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024 * 1024)  # GB
        
        logger.info(f"性能测试器初始化完成")
        logger.info(f"设备信息: {self.device_info}")
    
    def load_model(self, model_name):
        """
        加载模型
        
        参数:
            model_name: 模型名称
        
        返回:
            model, tokenizer
        """
        from src.core.model_training import BERTTextCNN, MultimodalModel, TextCNN
        
        if model_name == "bert_textcnn":
            # 加载BERT模型
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            bert = AutoModel.from_pretrained("bert-base-chinese")
            model = BERTTextCNN(bert).to(self.device_info["device"])
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device_info["device"]))
            return model, tokenizer
        elif model_name == "multimodal":
            # 加载BERT模型
            tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            bert = AutoModel.from_pretrained("bert-base-chinese")
            model = MultimodalModel(bert).to(self.device_info["device"])
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device_info["device"]))
            return model, tokenizer
        elif model_name == "textcnn":
            # TextCNN模型需要特殊处理
            model = TextCNN(5000).to(self.device_info["device"])
            model.load_state_dict(torch.load(f"{self.model_dir}/{model_name}_best.pth", map_location=self.device_info["device"]))
            return model, None
        else:
            raise ValueError(f"不支持的模型类型: {model_name}")
    
    def get_system_metrics(self):
        """
        获取系统指标
        
        返回:
            系统指标字典
        """
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
        memory_percent = psutil.virtual_memory().percent
        
        metrics = {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "memory_percent": memory_percent
        }
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
            gpu_memory_percent = gpu_memory / torch.cuda.get_device_properties(0).total_memory * 100
            metrics["gpu_memory"] = gpu_memory
            metrics["gpu_memory_percent"] = gpu_memory_percent
        
        return metrics
    
    def test_model_performance(self, model, tokenizer, model_name, input_size, batch_size):
        """
        测试模型性能
        
        参数:
            model: 模型
            tokenizer: 分词器
            model_name: 模型名称
            input_size: 输入规模
            batch_size: 批处理大小
        
        返回:
            性能数据
        """
        logger.info(f"测试 {model_name} 模型，输入规模: {input_size}，批处理大小: {batch_size}")
        
        # 准备测试数据
        if model_name in ["bert_textcnn", "multimodal"]:
            test_texts = ["这是一个测试文本，用于性能测试" for _ in range(input_size)]
            if model_name == "multimodal":
                test_data = [(text, [0.5] * 8) for text in test_texts]
            else:
                test_data = test_texts
        else:
            # 对于TextCNN，准备随机特征
            test_data = [np.random.rand(5000).astype(np.float32) for _ in range(input_size)]
        
        # 测试前的系统状态
        pre_metrics = self.get_system_metrics()
        
        # 模型推理
        model.eval()
        start_time = time.time()
        total_inference_time = 0
        
        with torch.no_grad():
            for i in range(0, input_size, batch_size):
                batch_end = min(i + batch_size, input_size)
                batch_data = test_data[i:batch_end]
                
                if model_name in ["bert_textcnn", "multimodal"]:
                    if model_name == "multimodal":
                        texts, network_features = zip(*batch_data)
                        inputs = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device_info["device"])
                        network_features = torch.tensor(network_features, dtype=torch.float32).to(self.device_info["device"])
                        batch_start = time.time()
                        outputs = model(**inputs, network_features=network_features)
                    else:
                        inputs = tokenizer(batch_data, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device_info["device"])
                        batch_start = time.time()
                        outputs = model(**inputs)
                else:
                    batch_tensor = torch.tensor(batch_data).to(self.device_info["device"])
                    batch_start = time.time()
                    outputs = model(batch_tensor)
                
                batch_end_time = time.time()
                total_inference_time += (batch_end_time - batch_start)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # 测试后的系统状态
        post_metrics = self.get_system_metrics()
        
        # 计算性能指标
        throughput = input_size / total_time  # 样本/秒
        latency = total_time / input_size * 1000  # 毫秒/样本
        
        performance = {
            "model": model_name,
            "input_size": input_size,
            "batch_size": batch_size,
            "throughput": throughput,
            "latency": latency,
            "total_time": total_time,
            "inference_time": total_inference_time,
            "pre_system_metrics": pre_metrics,
            "post_system_metrics": post_metrics,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"测试完成: 吞吐量 = {throughput:.2f} 样本/秒, 延迟 = {latency:.2f} 毫秒/样本")
        
        return performance
    
    def run_comprehensive_test(self):
        """
        运行综合性能测试
        """
        logger.info("开始综合性能测试")
        
        for model_name in self.test_configs["models"]:
            logger.info(f"测试模型: {model_name}")
            
            try:
                # 加载模型
                model, tokenizer = self.load_model(model_name)
                
                for input_size in self.test_configs["input_sizes"]:
                    for batch_size in self.test_configs["batch_sizes"]:
                        # 测试性能
                        performance = self.test_model_performance(
                            model, tokenizer, model_name, input_size, batch_size
                        )
                        self.performance_data.append(performance)
                
                # 释放模型占用的内存
                del model
                if tokenizer:
                    del tokenizer
                torch.cuda.empty_cache()
                
            except Exception as e:
                logger.error(f"测试 {model_name} 模型失败: {e}")
                import traceback
                traceback.print_exc()
        
        # 保存性能数据
        self.save_performance_data()
        
        # 生成性能报告
        self.generate_performance_report()
        
        logger.info("综合性能测试完成")
    
    def save_performance_data(self):
        """
        保存性能数据
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_file = f"{self.output_dir}/performance_data_{timestamp}.json"
        
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump({
                "device_info": self.device_info,
                "test_configs": self.test_configs,
                "performance_data": self.performance_data,
                "timestamp": datetime.now().isoformat()
            }, f, ensure_ascii=False, indent=2)
        
        logger.info(f"性能数据已保存到: {data_file}")
    
    def generate_performance_report(self):
        """
        生成性能报告
        """
        # 转换为DataFrame
        df = pd.DataFrame(self.performance_data)
        
        # 生成报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"{self.output_dir}/performance_report_{timestamp}.html"
        
        # 生成HTML报告
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>性能测试报告</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1, h2 {{ color: #333; }}
                table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                tr:nth-child(even) {{ background-color: #f9f9f9; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; border: 1px solid #ddd; border-radius: 5px; }}
                .chart {{ margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>性能测试报告</h1>
            <p>生成时间: {datetime.now().isoformat()}</p>
            
            <h2>设备信息</h2>
            <div class="metrics">
                <div class="metric">
                    <strong>设备:</strong> {self.device_info['device']}<br>
                    <strong>CPU核心数:</strong> {self.device_info['cpu_count']}<br>
                    <strong>总内存:</strong> {self.device_info['memory_total']:.2f} GB
                </div>
                {f"<div class='metric'><strong>GPU:</strong> {self.device_info['gpu_name']}<br><strong>GPU内存:</strong> {self.device_info['gpu_memory']:.2f} GB</div>" if 'gpu_name' in self.device_info else ''}
            </div>
            
            <h2>测试配置</h2>
            <p>批处理大小: {self.test_configs['batch_sizes']}</p>
            <p>输入规模: {self.test_configs['input_sizes']}</p>
            <p>测试模型: {self.test_configs['models']}</p>
            
            <h2>性能数据</h2>
            <table>
                <tr>
                    <th>模型</th>
                    <th>输入规模</th>
                    <th>批处理大小</th>
                    <th>吞吐量 (样本/秒)</th>
                    <th>延迟 (毫秒/样本)</th>
                    <th>总时间 (秒)</th>
                </tr>
        """
        
        # 添加表格数据
        for _, row in df.iterrows():
            html_content += f"""
                <tr>
                    <td>{row['model']}</td>
                    <td>{row['input_size']}</td>
                    <td>{row['batch_size']}</td>
                    <td>{row['throughput']:.2f}</td>
                    <td>{row['latency']:.2f}</td>
                    <td>{row['total_time']:.2f}</td>
                </tr>
            """
        
        # 结束HTML
        html_content += f"""
            </table>
            
            <h2>性能分析</h2>
            <p>测试完成时间: {datetime.now().isoformat()}</p>
            <p>共测试 {len(self.performance_data)} 个场景</p>
        </body>
        </html>
        """
        
        # 保存HTML报告
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"性能报告已生成: {report_file}")
        
        # 生成性能图表
        self.generate_performance_charts()
    
    def generate_performance_charts(self):
        """
        生成性能图表
        """
        # 转换为DataFrame
        df = pd.DataFrame(self.performance_data)
        
        # 按模型分组
        models = df['model'].unique()
        
        # 生成吞吐量图表
        plt.figure(figsize=(12, 8))
        
        for model in models:
            model_data = df[df['model'] == model]
            for input_size in self.test_configs['input_sizes']:
                input_data = model_data[model_data['input_size'] == input_size]
                plt.plot(input_data['batch_size'], input_data['throughput'], marker='o', label=f"{model} - 输入规模: {input_size}")
        
        plt.title('不同模型的吞吐量对比')
        plt.xlabel('批处理大小')
        plt.ylabel('吞吐量 (样本/秒)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/throughput_comparison.png")
        
        # 生成延迟图表
        plt.figure(figsize=(12, 8))
        
        for model in models:
            model_data = df[df['model'] == model]
            for input_size in self.test_configs['input_sizes']:
                input_data = model_data[model_data['input_size'] == input_size]
                plt.plot(input_data['batch_size'], input_data['latency'], marker='o', label=f"{model} - 输入规模: {input_size}")
        
        plt.title('不同模型的延迟对比')
        plt.xlabel('批处理大小')
        plt.ylabel('延迟 (毫秒/样本)')
        plt.legend()
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/latency_comparison.png")
        
        logger.info("性能图表已生成")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="综合性能测试脚本")
    parser.add_argument("--model_dir", type=str, default="models", help="模型目录")
    parser.add_argument("--output_dir", type=str, default="output", help="输出目录")
    parser.add_argument("--models", type=str, nargs='+', default=["bert_textcnn", "multimodal"], help="要测试的模型")
    parser.add_argument("--batch_sizes", type=int, nargs='+', default=[1, 8, 16, 32, 64], help="批处理大小")
    parser.add_argument("--input_sizes", type=int, nargs='+', default=[10, 100, 1000], help="输入规模")
    
    args = parser.parse_args()
    
    # 初始化测试器
    tester = PerformanceTester(model_dir=args.model_dir, output_dir=args.output_dir)
    
    # 更新测试配置
    tester.test_configs["models"] = args.models
    tester.test_configs["batch_sizes"] = args.batch_sizes
    tester.test_configs["input_sizes"] = args.input_sizes
    
    # 运行测试
    tester.run_comprehensive_test()
