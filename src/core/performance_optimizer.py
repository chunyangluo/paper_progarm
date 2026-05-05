#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化脚本
专注于提高数据采集速度和系统稳定性
"""
import sys
import os
import time
import asyncio
import aiohttp
import concurrent.futures
from urllib.parse import urlparse

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataCollector

def optimize_network_features_collection():
    """优化网络特征采集"""
    print("\n===== 优化网络特征采集 =====")
    
    # 读取数据集
    import pandas as pd
    try:
        df = pd.read_csv("../data/real_multimodal_dataset.csv")
        urls = df['url'].tolist()[:100]  # 测试前100个URL
        print(f"测试样本数: {len(urls)}")
    except Exception as e:
        print(f"读取数据集失败: {e}")
        return
    
    # 测试不同并发数的性能
    concurrency_levels = [25, 50, 75, 100]
    
    for concurrency in concurrency_levels:
        print(f"\n测试并发数: {concurrency}")
        start_time = time.time()
        
        async def collect_features(url):
            """异步采集单个URL的网络特征"""
            try:
                # 解析域名
                parsed_url = urlparse(url)
                domain = parsed_url.netloc
                
                # 构建会话
                async with aiohttp.ClientSession() as session:
                    # 设置请求头
                    headers = {
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                        "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                        "Accept-Encoding": "gzip, deflate",
                        "Connection": "close"
                    }
                    
                    # 动态超时管理
                    base_timeout = 5
                    max_retries = 3
                    
                    for attempt in range(max_retries):
                        current_timeout = base_timeout * (1.5 ** attempt)
                        try:
                            async with session.get(
                                url,
                                headers=headers,
                                timeout=current_timeout,
                                allow_redirects=True,
                                ssl=False
                            ) as response:
                                # 读取响应内容
                                content = await response.text()
                                # 计算响应时间
                                response_time = time.time() - start_time
                                # 构建特征
                                features = [
                                    min(response_time, 30.0),
                                    1.0 if response.status in range(200, 400) else 0.0,
                                    float(len(response.history)),
                                    1.0 if '<form' in content.lower() else 0.0,
                                    1.0 if any(keyword in url.lower() for keyword in ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0,
                                    1.0 if url.startswith('https://') else 0.0,
                                    min(len(content) / 1024, 1000.0),
                                    0.0  # 域名年龄，暂时设为0
                                ]
                                return features
                        except asyncio.TimeoutError:
                            if attempt == max_retries - 1:
                                return [30.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
                            await asyncio.sleep(1 * (attempt + 1))
                        except Exception:
                            if attempt == max_retries - 1:
                                return [10.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
                            await asyncio.sleep(1 * (attempt + 1))
            except Exception:
                return [5.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
        
        async def main():
            """异步主函数"""
            tasks = []
            for url in urls:
                task = asyncio.create_task(collect_features(url))
                tasks.append(task)
            
            # 分批执行任务
            results = []
            for i in range(0, len(tasks), concurrency):
                batch_tasks = tasks[i:i+concurrency]
                batch_results = await asyncio.gather(*batch_tasks)
                results.extend(batch_results)
            
            return results
        
        # 运行异步任务
        results = asyncio.run(main())
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"完成时间: {elapsed:.2f}秒")
        print(f"平均响应时间: {elapsed / len(urls):.4f}秒/URL")
        
        # 计算成功率
        success_count = sum(1 for features in results if features[1] == 1.0)
        success_rate = success_count / len(results) * 100
        print(f"成功率: {success_rate:.2f}%")

def optimize_data_collection():
    """优化数据采集"""
    print("\n===== 优化数据采集 =====")
    
    collector = DataCollector()
    
    # 测试不同数据源的采集速度
    data_sources = [
        ("OpenPhish", lambda: collector.collect_openphish_data(count=100)),
        ("PhishTank", lambda: collector.collect_phishTank_data(count=100)),
        ("Majestic", lambda: collector.collect_majestic_data(count=100)),
    ]
    
    for source_name, collect_func in data_sources:
        print(f"\n测试{source_name}采集速度...")
        start_time = time.time()
        
        try:
            data = collect_func()
            end_time = time.time()
            elapsed = end_time - start_time
            print(f"采集成功，获取 {len(data)} 条数据，用时 {elapsed:.2f}秒")
        except Exception as e:
            print(f"采集失败: {e}")

def optimize_cache_strategy():
    """优化缓存策略"""
    print("\n===== 优化缓存策略 =====")
    
    # 测试不同缓存策略的性能
    import tempfile
    import pickle
    
    # 生成测试数据
    test_data = {f"url_{i}": [i * 0.1 for _ in range(8)] for i in range(1000)}
    
    # 测试内存缓存
    print("测试内存缓存...")
    start_time = time.time()
    memory_cache = test_data.copy()
    end_time = time.time()
    print(f"内存缓存设置时间: {end_time - start_time:.4f}秒")
    
    # 测试文件缓存
    print("测试文件缓存...")
    start_time = time.time()
    with tempfile.NamedTemporaryFile(delete=False) as f:
        pickle.dump(test_data, f)
    end_time = time.time()
    print(f"文件缓存设置时间: {end_time - start_time:.4f}秒")
    
    # 清理临时文件
    os.unlink(f.name)

def main():
    """主函数"""
    print("开始性能优化...")
    
    # 优化网络特征采集
    optimize_network_features_collection()
    
    # 优化数据采集
    optimize_data_collection()
    
    # 优化缓存策略
    optimize_cache_strategy()
    
    print("\n性能优化完成！")

if __name__ == "__main__":
    main()
