#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只训练BERT-TextCNN模型的脚本
"""

import sys
import os
import argparse
from model_training import ModelTrainer

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="训练 BERT-TextCNN 模型")
        parser.add_argument("--epochs", type=int, default=10, help="训练轮数（当前脚本暂未透传）")
        parser.add_argument("--batch-size", type=int, default=16, help="批大小（当前脚本暂未透传）")
        parser.add_argument("--learning-rate", type=float, default=1e-5, help="学习率（当前脚本暂未透传）")
        parser.add_argument("--dataset", type=str, default="", help="数据集路径")
        args = parser.parse_args()

        # 初始化ModelTrainer
        trainer = ModelTrainer("../../models", "../../output")
        
        # 版本数据集路径
        version_dataset_path = args.dataset or "../../data/versions/dataset_20260411_chifraud.csv"
        
        # 检查数据集是否存在
        if not os.path.exists(version_dataset_path):
            print(f"错误: 数据集不存在：{version_dataset_path}")
            print("请先运行 update_dataset.py 生成数据集！")
            sys.exit(1)
        
        # 训练BERT-TextCNN模型
        print("\n开始训练BERT-TextCNN模型...")
        print(
            f"接收到参数: epochs={args.epochs}, batch_size={args.batch_size}, "
            f"learning_rate={args.learning_rate}。当前训练实现使用脚本内默认训练参数。"
        )
        trainer.train_bert_textcnn(version_dataset_path, "bert_textcnn")
        
        print("\nBERT-TextCNN模型训练完成！")
        print("\n训练结果：")
        print("   - BERT-TextCNN模型已训练完成")
        print("   - 模型已保存到 models/ 目录")
        print("   - 训练结果已保存到 output/ 目录")
    except Exception as e:
        print(f"执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)