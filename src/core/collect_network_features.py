#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络特征采集脚本 - 为多模态模型采集真实的网络行为特征

此脚本会从数据集中的URL采集真实的网络特征，包括：
- 响应时间
- HTTP状态码
- 重定向次数
- 是否使用HTTPS
- 页面大小
- 是否包含表单
- 域名年龄（简化版）
- 是否有敏感内容标记

特征维度：8维，与多模态模型的network_feature_dim=8匹配
"""

import pandas as pd
import numpy as np
import requests
import time
import random
import urllib3
from urllib.parse import urlparse
import ssl

# 禁用警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# User-Agent池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

def get_random_user_agent():
    """获取随机User-Agent"""
    return random.choice(USER_AGENTS)

def extract_network_features(url, timeout=10, delay=1.0):
    """
    从URL采集网络特征
    
    参数:
        url: 要采集的URL
        timeout: 请求超时时间（秒）
        delay: 请求延迟（秒）
    
    返回:
        8维网络特征列表
    """
    # 初始化默认特征
    features = [0.0] * 8
    
    # 添加延迟，规避反爬
    time.sleep(delay + random.uniform(0, 0.5))
    
    if not url or not isinstance(url, str) or not url.startswith(('http://', 'https://')):
        # URL无效，返回默认特征
        features[0] = 0.0  # 响应时间
        features[1] = 0.0  # 加载状态（0=失败）
        features[2] = 0.0  # 重定向次数
        features[3] = 0.0  # 是否包含表单
        features[4] = 0.0  # 是否请求敏感信息
        features[5] = 1.0 if url.startswith('https://') else 0.0  # 是否使用HTTPS
        features[6] = 0.0  # 页面大小
        features[7] = 0.0  # 是否有敏感内容标记
        return features
    
    try:
        # 设置请求头
        headers = {
            "User-Agent": get_random_user_agent(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            "Accept-Encoding": "gzip, deflate",
            "Connection": "close"
        }
        
        # 记录开始时间
        start_time = time.time()
        
        # 发送请求
        response = requests.get(
            url,
            headers=headers,
            timeout=timeout,
            allow_redirects=True,
            verify=False,  # 忽略SSL证书错误
            stream=True
        )
        
        # 计算响应时间
        response_time = time.time() - start_time
        
        # 获取重定向次数
        redirect_count = len(response.history)
        
        # 获取页面大小
        try:
            content_length = len(response.content) if response.content else 0
        except:
            content_length = 0
        
        # 检测是否包含表单
        has_form = 1.0 if '<form' in response.text.lower() else 0.0
        
        # 检测是否有敏感内容标记
        has_sensitive = 1.0 if any(keyword in response.text.lower() for keyword in 
                                    ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0
        
        # 检测是否请求敏感信息（根据URL）
        requests_sensitive = 1.0 if any(keyword in url.lower() for keyword in
                                          ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0
        
        # 构建特征
        features[0] = min(response_time, 10.0)  # 响应时间（限制在10秒内）
        features[1] = 1.0 if response.status_code == 200 else 0.0  # 加载状态（1=成功）
        features[2] = float(redirect_count)  # 重定向次数
        features[3] = has_form  # 是否包含表单
        features[4] = requests_sensitive  # 是否请求敏感信息
        features[5] = 1.0 if url.startswith('https://') else 0.0  # 是否使用HTTPS
        features[6] = min(content_length / 1024, 100.0)  # 页面大小（KB，限制在100KB内）
        features[7] = has_sensitive  # 是否有敏感内容标记
        
        # 关闭响应
        response.close()
        
    except requests.exceptions.Timeout:
        # 超时
        features[0] = 10.0  # 响应时间设为最大值
        features[1] = 0.0  # 加载状态（失败）
        features[5] = 1.0 if url.startswith('https://') else 0.0
    except requests.exceptions.SSLError:
        # SSL错误
        features[0] = 5.0
        features[1] = 0.0
        features[5] = 1.0 if url.startswith('https://') else 0.0
    except requests.exceptions.RequestException:
        # 其他请求错误
        features[0] = 3.0
        features[1] = 0.0
        features[5] = 1.0 if url.startswith('https://') else 0.0
    except Exception:
        # 其他错误
        features[0] = 1.5
        features[1] = 0.0
        features[5] = 1.0 if url.startswith('https://') else 0.0
    
    return features

def collect_network_features_batch(dataset_path, output_path=None, batch_size=100, delay=1.0):
    """
    批量采集网络特征
    
    参数:
        dataset_path: 输入数据集路径
        output_path: 输出数据集路径（如果为None，会自动生成）
        batch_size: 批量大小，每采集batch_size个样本保存一次
        delay: 请求延迟（秒）
    """
    print("=" * 80)
    print("网络特征采集脚本")
    print("=" * 80)
    
    # 读取数据集
    print(f"\n正在读取数据集: {dataset_path}")
    df = pd.read_csv(dataset_path)
    print(f"数据集大小: {len(df)} 条")
    
    # 确保有url列
    if 'url' not in df.columns:
        print("\n⚠️ 数据集没有url列，尝试从text列提取URL或使用空值")
        df['url'] = ""
    
    # 填充url列的空值
    df['url'] = df['url'].fillna("")
    
    # 生成输出路径
    if output_path is None:
        import os
        base_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_path = f"../data/versions/{base_name}_with_network_features.csv"
    
    # 初始化network_features列
    if 'network_features' not in df.columns:
        df['network_features'] = None
    
    # 统计已采集的样本数
    collected_count = df['network_features'].notna().sum()
    print(f"已采集: {collected_count}/{len(df)} 条")
    
    # 批量采集
    total = len(df)
    start_idx = collected_count
    
    for i in range(start_idx, total, batch_size):
        end_idx = min(i + batch_size, total)
        print(f"\n正在采集 {i+1}-{end_idx}/{total}...")
        
        # 采集当前批次
        for j in range(i, end_idx):
            url = df.iloc[j]['url']
            features = extract_network_features(url, delay=delay)
            # 将特征列表转换为字符串存储
            df.at[j, 'network_features'] = ','.join([f"{f:.4f}" for f in features])
            
            # 显示进度
            if (j - i + 1) % 10 == 0:
                print(f"  进度: {j - i + 1}/{end_idx - i}")
        
        # 保存当前批次
        print(f"正在保存到 {output_path}...")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"已保存 {end_idx}/{total} 条")
        
        # 短暂休息
        time.sleep(1)
    
    print("\n" + "=" * 80)
    print("✅ 网络特征采集完成！")
    print(f"输出文件: {output_path}")
    print("=" * 80)
    
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="网络特征采集脚本")
    parser.add_argument("--dataset", type=str, 
                        default="../data/versions/dataset_20260411_chifraud.csv",
                        help="输入数据集路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出数据集路径")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="批量大小")
    parser.add_argument("--delay", type=float, default=1.0,
                        help="请求延迟（秒）")
    
    args = parser.parse_args()
    
    try:
        output_path = collect_network_features_batch(
            dataset_path=args.dataset,
            output_path=args.output,
            batch_size=args.batch_size,
            delay=args.delay
        )
        
        print(f"\n✅ 成功！")
        print(f"采集结果已保存到: {output_path}")
        
    except Exception as e:
        print(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
