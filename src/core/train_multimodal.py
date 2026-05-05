#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
只训练多模态模型的脚本，用于测试过拟合问题的修复效果
"""

import sys
import os
from model_training import ModelTrainer

if __name__ == "__main__":
    try:
        # 初始化ModelTrainer
        trainer = ModelTrainer("../models", "../output")
        
        # 版本数据集路径
        version_dataset_path = "../data/versions/dataset_20260411_chifraud_with_network_features_async.csv"
        
        # 检查数据集是否存在
        if not os.path.exists(version_dataset_path):
            print(f"❌ 数据集不存在：{version_dataset_path}")
            print("请先运行 update_dataset.py 生成数据集！")
            sys.exit(1)
        
        # 训练多模态模型
        print("\n开始训练多模态模型...")
        trainer.train_multimodal(version_dataset_path, "multimodal")
        
        print("\n✅ 多模态模型训练完成！")
        print("\n📊 训练结果：")
        print("   - 多模态模型已训练完成")
        print("   - 模型已保存到 models/ 目录")
        print("   - 训练结果已保存到 output/ 目录")
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
