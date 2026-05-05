#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文钓鱼文本识别项目主脚本

本脚本用于运行整个项目的完整流程，包括：
1. 数据预处理（构建数据集）
2. 模型训练（TextCNN和BERT-TextCNN）
3. 模型评估
"""

import os
import sys

# 添加scripts目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))

from data_preprocessing import DataPreprocessor
from data_collection import DataCollector
from model_training import ModelTrainer

class PhishingDetector:
    def __init__(self, data_dir="data", model_dir="models", output_dir="output"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.output_dir = output_dir
        self.preprocessor = DataPreprocessor(data_dir)
        self.trainer = ModelTrainer(model_dir, output_dir)
    
    def run_full_pipeline(self):
        """运行完整的项目流程"""
        print("=" * 80)
        print("中文钓鱼文本识别项目")
        print("=" * 80)
        
        # 1. 数据预处理
        print("\n1. 开始数据预处理...")
        
        # 下载和处理CHIFRAUD数据集
        print("正在下载和处理CHIFRAUD数据集...")
        collector = DataCollector(self.data_dir)
        collector.collect_chifraud_data()
        collector.process_chifraud_data()
        
        # 构建其他数据集
        self.preprocessor.build_synthetic_dataset()
        self.preprocessor.build_real_dataset()
        self.preprocessor.generate_random_dataset()
        print("数据预处理完成！")
        
        # 2. 模型训练
        print("\n2. 开始模型训练...")
        # 先运行update_dataset.py生成版本数据集
        print("正在运行update_dataset.py生成版本数据集...")
        import subprocess
        subprocess.run(["python", "update_dataset.py"], check=True)
        
        # 使用生成的版本数据集进行训练
        import os
        # 获取绝对路径
        version_dataset_path = os.path.join(os.path.dirname(__file__), "../data/versions/dataset_20260411_chifraud.csv")
        self.trainer.train_textcnn(version_dataset_path, "textcnn")
        self.trainer.train_bert_textcnn(version_dataset_path, "bert_textcnn")
        self.trainer.train_multimodal(model_name="multimodal")
        print("模型训练完成！")
        
        # 3. 模型评估
        print("\n3. 模型评估完成！")
        print("\n所有任务已完成！")
        print("\n生成的文件：")
        print(f"- 数据集：{self.data_dir}/")
        print(f"- 模型：{self.model_dir}/")
        print(f"- 输出：{self.output_dir}/")
    
    def run_data_preprocessing(self):
        """仅运行数据预处理"""
        print("开始数据预处理...")
        
        # 下载和处理CHIFRAUD数据集
        print("正在下载和处理CHIFRAUD数据集...")
        collector = DataCollector(self.data_dir)
        collector.collect_chifraud_data()
        collector.process_chifraud_data()
        
        # 构建其他数据集
        self.preprocessor.build_synthetic_dataset()
        self.preprocessor.build_real_dataset()
        self.preprocessor.generate_random_dataset()
        
        print("数据预处理完成！")
    
    def run_model_training(self):
        """仅运行模型训练"""
        print("开始模型训练...")
        # 使用版本数据集进行训练
        import os
        # 获取绝对路径
        version_dataset_path = os.path.join(os.path.dirname(__file__), "../data/versions/dataset_20260411_chifraud.csv")
        self.trainer.train_textcnn(version_dataset_path, "textcnn")
        self.trainer.train_bert_textcnn(version_dataset_path, "bert_textcnn")
        self.trainer.train_multimodal(model_name="multimodal")
        print("模型训练完成！")

if __name__ == "__main__":
    import argparse
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="中文钓鱼文本识别项目")
    parser.add_argument("--fast", action="store_true", help="快速模式，跳过数据采集，直接训练")
    args = parser.parse_args()
    
    detector = PhishingDetector()
    
    if args.fast:
        # 快速模式：直接训练
        print("🚀 快速模式：跳过数据采集，直接训练")
        detector.run_model_training()
    else:
        # 完整模式：采集数据 + 训练
        print("🔄 完整模式：采集数据 + 训练")
        detector.run_full_pipeline()
