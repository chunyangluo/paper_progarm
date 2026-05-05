#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
下载和处理CHIFRAUD中文欺诈文本基准集
"""

from data_collection import DataCollector

if __name__ == "__main__":
    # 初始化DataCollector
    collector = DataCollector("../data")
    
    # 下载CHIFRAUD数据集
    print("开始下载CHIFRAUD中文欺诈文本基准集...")
    collector.collect_chifraud_data()
    
    # 处理CHIFRAUD数据集
    print("开始处理CHIFRAUD数据集...")
    collector.process_chifraud_data()
    
    print("CHIFRAUD数据集处理完成！")
