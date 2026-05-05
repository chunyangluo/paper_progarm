#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批处理优化工具

此脚本实现了高效的批处理机制，优化批量数据处理流程。
支持动态批处理大小调整，根据输入数据特征和系统资源状况自动优化批处理参数，
提高批量检测任务的吞吐量和资源利用率。

功能特性：
- 动态批处理大小调整
- 基于系统资源的自适应调整
- 批量数据预处理优化
- 性能监控和分析
- 支持多模型批处理
"""

import torch
import numpy as np
import time
import psutil
import threading
import queue
from transformers import AutoTokenizer
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('batch_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BatchProcessor:
    def __init__(self, model, tokenizer=None, device=None):
        """
        初始化批处理器
        
        参数:
            model: 要使用的模型
            tokenizer: 文本分词器（用于NLP任务）
            device: 运行设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        
        # 系统资源监控
        self.cpu_count = psutil.cpu_count()
        self.memory_total = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
        
        # 批处理参数
        self.min_batch_size = 1
        self.max_batch_size = 128
        self.optimal_batch_size = 32
        
        # 性能统计
        self.performance_stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_time": 0,
            "avg_batch_time": 0,
            "avg_sample_time": 0,
            "throughput": 0
        }
        
        # 动态调整参数
        self.adjustment_interval = 10  # 每处理10个批次调整一次
        self.batch_history = []
        
        logger.info(f"批处理器初始化完成")
        logger.info(f"设备: {self.device}")
        logger.info(f"CPU核心数: {self.cpu_count}")
        logger.info(f"总内存: {self.memory_total:.2f} GB")
        logger.info(f"初始批处理大小: {self.optimal_batch_size}")
    
    def get_system_resources(self):
        """
        获取当前系统资源使用情况
        """
        cpu_usage = psutil.cpu_percent(interval=0.1)
        memory_usage = psutil.virtual_memory().used / (1024 * 1024 * 1024)  # GB
        memory_percent = psutil.virtual_memory().percent
        
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024 * 1024)  # GB
            gpu_memory_percent = gpu_memory / torch.cuda.get_device_properties(0).total_memory * 100
        else:
            gpu_memory = 0
            gpu_memory_percent = 0
        
        return {
            "cpu_usage": cpu_usage,
            "memory_usage": memory_usage,
            "memory_percent": memory_percent,
            "gpu_memory": gpu_memory,
            "gpu_memory_percent": gpu_memory_percent
        }
    
    def adjust_batch_size(self):
        """
        根据系统资源和历史性能调整批处理大小
        """
        if len(self.batch_history) < self.adjustment_interval:
            return
        
        # 分析历史批次性能
        avg_batch_time = np.mean([item["time"] for item in self.batch_history[-self.adjustment_interval:]])
        avg_batch_size = np.mean([item["size"] for item in self.batch_history[-self.adjustment_interval:]])
        
        # 获取当前系统资源
        resources = self.get_system_resources()
        
        # 基于资源使用情况调整批处理大小
        new_batch_size = self.optimal_batch_size
        
        # 如果CPU或内存使用过高，减小批处理大小
        if resources["cpu_usage"] > 80 or resources["memory_percent"] > 80:
            new_batch_size = max(self.min_batch_size, self.optimal_batch_size // 2)
            logger.info(f"系统资源使用过高，调整批处理大小为: {new_batch_size}")
        # 如果GPU内存使用过高，减小批处理大小
        elif resources["gpu_memory_percent"] > 80:
            new_batch_size = max(self.min_batch_size, self.optimal_batch_size // 2)
            logger.info(f"GPU内存使用过高，调整批处理大小为: {new_batch_size}")
        # 如果处理时间较短且资源充足，增加批处理大小
        elif avg_batch_time < 0.5 and resources["cpu_usage"] < 50 and resources["memory_percent"] < 50:
            new_batch_size = min(self.max_batch_size, self.optimal_batch_size * 2)
            logger.info(f"系统资源充足，调整批处理大小为: {new_batch_size}")
        
        # 更新最优批处理大小
        if new_batch_size != self.optimal_batch_size:
            self.optimal_batch_size = new_batch_size
            logger.info(f"更新最优批处理大小为: {self.optimal_batch_size}")
    
    def process_batch(self, batch_data):
        """
        处理单个批次的数据
        
        参数:
            batch_data: 批次数据
        
        返回:
            处理结果
        """
        start_time = time.time()
        
        try:
            if self.tokenizer:
                # NLP任务处理
                inputs = self.tokenizer(batch_data, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model(**inputs)
                results = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                # 其他任务处理
                if isinstance(batch_data, list):
                    batch_data = torch.tensor(batch_data).to(self.device)
                with torch.no_grad():
                    outputs = self.model(batch_data)
                results = torch.softmax(outputs, dim=1).cpu().numpy()
            
            batch_time = time.time() - start_time
            batch_size = len(batch_data)
            
            # 记录批次性能
            self.batch_history.append({"size": batch_size, "time": batch_time})
            
            # 更新性能统计
            self.performance_stats["total_batches"] += 1
            self.performance_stats["total_samples"] += batch_size
            self.performance_stats["total_time"] += batch_time
            self.performance_stats["avg_batch_time"] = self.performance_stats["total_time"] / self.performance_stats["total_batches"]
            self.performance_stats["avg_sample_time"] = self.performance_stats["total_time"] / self.performance_stats["total_samples"]
            self.performance_stats["throughput"] = self.performance_stats["total_samples"] / self.performance_stats["total_time"]
            
            # 调整批处理大小
            if self.performance_stats["total_batches"] % self.adjustment_interval == 0:
                self.adjust_batch_size()
            
            return results
            
        except Exception as e:
            logger.error(f"处理批次失败: {e}")
            return None
    
    def process(self, data, batch_size=None):
        """
        处理数据
        
        参数:
            data: 输入数据
            batch_size: 批处理大小（如果为None，则使用自动调整的批处理大小）
        
        返回:
            处理结果
        """
        start_time = time.time()
        logger.info(f"开始处理 {len(data)} 个样本")
        
        # 使用指定的批处理大小或自动调整的批处理大小
        current_batch_size = batch_size or self.optimal_batch_size
        
        # 分批处理
        results = []
        for i in range(0, len(data), current_batch_size):
            batch_end = min(i + current_batch_size, len(data))
            batch_data = data[i:batch_end]
            
            # 处理批次
            batch_results = self.process_batch(batch_data)
            if batch_results is not None:
                results.extend(batch_results)
            
            # 打印进度
            progress = (batch_end / len(data)) * 100
            if progress % 10 == 0:
                logger.info(f"处理进度: {progress:.1f}%")
        
        total_time = time.time() - start_time
        logger.info(f"处理完成，总耗时: {total_time:.2f} 秒")
        logger.info(f"吞吐量: {len(data) / total_time:.2f} 样本/秒")
        
        return results
    
    def process_with_multithreading(self, data, num_threads=4, batch_size=None):
        """
        使用多线程处理数据
        
        参数:
            data: 输入数据
            num_threads: 线程数
            batch_size: 批处理大小
        
        返回:
            处理结果
        """
        start_time = time.time()
        logger.info(f"开始使用 {num_threads} 线程处理 {len(data)} 个样本")
        
        # 分割数据
        data_chunks = np.array_split(data, num_threads)
        
        # 创建结果队列
        result_queue = queue.Queue()
        
        def worker(data_chunk):
            """工作线程函数"""
            chunk_results = self.process(data_chunk, batch_size)
            result_queue.put(chunk_results)
        
        # 创建并启动线程
        threads = []
        for i, chunk in enumerate(data_chunks):
            thread = threading.Thread(target=worker, args=(chunk.tolist(),))
            threads.append(thread)
            thread.start()
            logger.info(f"启动线程 {i+1}/{num_threads}")
        
        # 等待所有线程完成
        for thread in threads:
            thread.join()
        
        # 收集结果
        results = []
        while not result_queue.empty():
            results.extend(result_queue.get())
        
        total_time = time.time() - start_time
        logger.info(f"多线程处理完成，总耗时: {total_time:.2f} 秒")
        logger.info(f"吞吐量: {len(data) / total_time:.2f} 样本/秒")
        
        return results
    
    def get_performance_stats(self):
        """
        获取性能统计信息
        """
        return self.performance_stats
    
    def reset_performance_stats(self):
        """
        重置性能统计信息
        """
        self.performance_stats = {
            "total_batches": 0,
            "total_samples": 0,
            "total_time": 0,
            "avg_batch_time": 0,
            "avg_sample_time": 0,
            "throughput": 0
        }
        self.batch_history = []
        logger.info("性能统计信息已重置")

class MultimodalBatchProcessor(BatchProcessor):
    """
    多模态批处理器
    """
    def __init__(self, model, tokenizer=None, device=None):
        super().__init__(model, tokenizer, device)
    
    def process_batch(self, batch_data):
        """
        处理多模态批次数据
        
        参数:
            batch_data: 批次数据，格式为 (texts, network_features)
        
        返回:
            处理结果
        """
        start_time = time.time()
        
        try:
            texts, network_features = zip(*batch_data)
            
            # 处理文本
            inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            
            # 处理网络特征
            network_features = torch.tensor(network_features, dtype=torch.float32).to(self.device)
            
            # 模型推理
            with torch.no_grad():
                outputs = self.model(**inputs, network_features=network_features)
            
            results = torch.softmax(outputs, dim=1).cpu().numpy()
            
            batch_time = time.time() - start_time
            batch_size = len(batch_data)
            
            # 记录批次性能
            self.batch_history.append({"size": batch_size, "time": batch_time})
            
            # 更新性能统计
            self.performance_stats["total_batches"] += 1
            self.performance_stats["total_samples"] += batch_size
            self.performance_stats["total_time"] += batch_time
            self.performance_stats["avg_batch_time"] = self.performance_stats["total_time"] / self.performance_stats["total_batches"]
            self.performance_stats["avg_sample_time"] = self.performance_stats["total_time"] / self.performance_stats["total_samples"]
            self.performance_stats["throughput"] = self.performance_stats["total_samples"] / self.performance_stats["total_time"]
            
            # 调整批处理大小
            if self.performance_stats["total_batches"] % self.adjustment_interval == 0:
                self.adjust_batch_size()
            
            return results
            
        except Exception as e:
            logger.error(f"处理多模态批次失败: {e}")
            return None

if __name__ == "__main__":
    import argparse
    from src.core.model_training import BERTTextCNN, MultimodalModel
    from transformers import AutoModel
    
    parser = argparse.ArgumentParser(description="批处理优化工具")
    parser.add_argument("--model_type", type=str, default="bert_textcnn", choices=["bert_textcnn", "multimodal"], help="模型类型")
    parser.add_argument("--batch_size", type=int, default=None, help="批处理大小")
    parser.add_argument("--num_threads", type=int, default=1, help="线程数")
    parser.add_argument("--test_size", type=int, default=100, help="测试样本数")
    
    args = parser.parse_args()
    
    # 加载模型
    logger.info(f"加载 {args.model_type} 模型...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
    bert = AutoModel.from_pretrained("bert-base-chinese")
    
    if args.model_type == "bert_textcnn":
        model = BERTTextCNN(bert)
        model.load_state_dict(torch.load("models/bert_textcnn_best.pth", map_location="cpu"))
        processor = BatchProcessor(model, tokenizer)
    else:
        model = MultimodalModel(bert)
        model.load_state_dict(torch.load("models/multimodal_best.pth", map_location="cpu"))
        processor = MultimodalBatchProcessor(model, tokenizer)
    
    # 准备测试数据
    logger.info(f"准备 {args.test_size} 个测试样本...")
    test_texts = ["这是一个测试文本" for _ in range(args.test_size)]
    
    if args.model_type == "multimodal":
        # 为多模态模型准备网络特征
        test_data = [(text, [0.5] * 8) for text in test_texts]
    else:
        test_data = test_texts
    
    # 处理数据
    if args.num_threads > 1:
        results = processor.process_with_multithreading(test_data, args.num_threads, args.batch_size)
    else:
        results = processor.process(test_data, args.batch_size)
    
    # 打印性能统计
    stats = processor.get_performance_stats()
    logger.info("=" * 80)
    logger.info("性能统计")
    logger.info("=" * 80)
    for key, value in stats.items():
        logger.info(f"{key}: {value:.4f}")
    logger.info("=" * 80)
