#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
验证生成的数据集
"""

import pandas as pd

# 读取版本数据集
df = pd.read_csv("../data/versions/dataset_20260411_chifraud.csv", encoding="utf-8-sig")

# 打印基本信息
print("数据集形状:", df.shape)
print("\n样本分布:")
print(df["label"].value_counts())  # 0=正常，1=钓鱼
print("\n数据来源分布:")
print(df["source"].value_counts())
print("\n前5条数据:")
print(df.head())
