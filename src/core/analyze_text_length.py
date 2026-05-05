#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析文本长度分布，确定最佳的max_seq_len
"""

import pandas as pd
import numpy as np
from transformers import BertTokenizer

if __name__ == "__main__":
    try:
        # 1. 加载数据集
        dataset_path = "../data/versions/dataset_20260411_chifraud.csv"
        df = pd.read_csv(dataset_path)
        
        # 2. 提取所有文本数据
        all_texts = df['text'].dropna().tolist()
        print(f"总样本数: {len(all_texts)}")
        
        # 3. 用BERT tokenizer计算每条文本的token长度
        print("正在计算token长度...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
        
        # 批量计算token长度，避免循环太慢
        tokenized = tokenizer(all_texts, truncation=False, padding=False, return_length=True)
        lengths = tokenized["length"]  # 每条文本的token长度
        
        # 4. 统计关键指标
        max_len = max(lengths)
        min_len = min(lengths)
        avg_len = np.mean(lengths)
        p95_len = np.percentile(lengths, 95)  # 95%分位：95%的样本长度≤这个值
        p99_len = np.percentile(lengths, 99)  # 99%分位：99%的样本长度≤这个值
        
        print(f"\n长度统计:")
        print(f"最大长度: {max_len}")
        print(f"最小长度: {min_len}")
        print(f"平均长度: {avg_len:.2f}")
        print(f"95%分位长度: {p95_len:.0f}")
        print(f"99%分位长度: {p99_len:.0f}")
        
        # 5. 统计不同长度的占比
        len_leq_64 = sum(1 for l in lengths if l <= 64) / len(lengths)
        len_leq_128 = sum(1 for l in lengths if l <= 128) / len(lengths)
        len_leq_256 = sum(1 for l in lengths if l <= 256) / len(lengths)
        
        print(f"\n长度占比:")
        print(f"长度≤64的样本占比: {len_leq_64:.2%}")
        print(f"长度≤128的样本占比: {len_leq_128:.2%}")
        print(f"长度≤256的样本占比: {len_leq_256:.2%}")
        
        # 6. 建议
        print(f"\n建议:")
        if len_leq_64 >= 0.99:
            print("✅ 99%+ 的样本长度≤64：直接改max_seq_len=64，完全没问题，截断率不到 1%，几乎不影响性能")
        elif len_leq_64 >= 0.95:
            print("✅ 95%≤64，99%≤80：可以改max_seq_len=64，损失极小，论文完全能用")
        elif len_leq_64 >= 0.90:
            print("⚠️ 只有 90%≤64：建议改max_seq_len=80，平衡速度和截断率")
        else:
            print("❌ 大部分样本长度超过64：建议保持max_seq_len=128")
            
    except Exception as e:
        print(f"❌ 执行过程中出错: {e}")
