#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
综合优化脚本
专注于提高系统性能和稳定性
"""
import sys
import os
import time
import asyncio
import aiohttp
import pickle
import hashlib

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataCollector

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.collector = DataCollector()
        self.cache_dir = "../data/cache"
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def optimize_network_features_collection(self):
        """优化网络特征采集"""
        print("\n===== 优化网络特征采集 =====")
        
        # 读取测试数据
        import pandas as pd
        try:
            df = pd.read_csv("../data/real_multimodal_dataset.csv")
            urls = df['url'].tolist()[:50]  # 测试前50个URL
            print(f"测试样本数: {len(urls)}")
        except Exception as e:
            print(f"读取数据集失败: {e}")
            return
        
        # 优化配置
        optimal_config = {
            "concurrency": 50,
            "timeout": 5,
            "max_retries": 3,
            "delay": 0.1
        }
        
        print(f"使用优化配置: {optimal_config}")
        
        # 测试优化后的性能
        start_time = time.time()
        results = asyncio.run(self._collect_features_batch(urls, optimal_config))
        end_time = time.time()
        
        elapsed = end_time - start_time
        success_count = sum(1 for features in results if features[1] == 1.0)
        success_rate = success_count / len(results) * 100
        
        print(f"完成时间: {elapsed:.2f}秒")
        print(f"平均响应时间: {elapsed / len(urls):.4f}秒/URL")
        print(f"成功率: {success_rate:.2f}%")
        
        return optimal_config
    
    async def _collect_features_batch(self, urls, config):
        """批量采集网络特征"""
        tasks = []
        for url in urls:
            task = asyncio.create_task(self._collect_features(url, config))
            tasks.append(task)
        
        # 分批执行任务
        results = []
        for i in range(0, len(tasks), config["concurrency"]):
            batch_tasks = tasks[i:i+config["concurrency"]]
            batch_results = await asyncio.gather(*batch_tasks)
            results.extend(batch_results)
            # 添加小延迟，避免过度请求
            await asyncio.sleep(config["delay"])
        
        return results
    
    async def _collect_features(self, url, config):
        """采集单个URL的网络特征"""
        # 检查缓存
        cache_key = self._get_cache_key(url)
        cached_features = self._get_from_cache(cache_key)
        if cached_features:
            return cached_features
        
        try:
            async with aiohttp.ClientSession() as session:
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
                    "Accept-Encoding": "gzip, deflate",
                    "Connection": "close"
                }
                
                for attempt in range(config["max_retries"]):
                    current_timeout = config["timeout"] * (1.5 ** attempt)
                    try:
                        # 记录开始时间
                        request_start_time = time.time()
                        async with session.get(
                            url,
                            headers=headers,
                            timeout=current_timeout,
                            allow_redirects=True,
                            ssl=False
                        ) as response:
                            content = await response.text()
                            response_time = time.time() - request_start_time
                            features = [
                                min(response_time, 30.0),
                                1.0 if response.status in range(200, 400) else 0.0,
                                float(len(response.history)),
                                1.0 if '<form' in content.lower() else 0.0,
                                1.0 if any(keyword in url.lower() for keyword in ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0,
                                1.0 if url.startswith('https://') else 0.0,
                                min(len(content) / 1024, 1000.0),
                                0.0  # 域名年龄
                            ]
                            # 保存到缓存
                            self._save_to_cache(cache_key, features)
                            return features
                    except asyncio.TimeoutError:
                        if attempt == config["max_retries"] - 1:
                            features = [30.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
                            self._save_to_cache(cache_key, features)
                            return features
                        await asyncio.sleep(1 * (attempt + 1))
                    except Exception:
                        if attempt == config["max_retries"] - 1:
                            features = [10.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
                            self._save_to_cache(cache_key, features)
                            return features
                        await asyncio.sleep(1 * (attempt + 1))
        except Exception:
            features = [5.0, 0.0, 0.0, 0.0, 0.0, 1.0 if url.startswith('https://') else 0.0, 0.0, 0.0]
            self._save_to_cache(cache_key, features)
            return features
    
    def optimize_data_collection(self):
        """优化数据采集"""
        print("\n===== 优化数据采集 =====")
        
        # 测试不同数据源的采集速度
        data_sources = [
            ("OpenPhish", lambda: self.collector.collect_openphish_data(count=100)),
            ("Phishing Army", lambda: self.collector.collect_additional_data(count=100)),
            ("Majestic", lambda: self.collector.collect_majestic_data(count=100)),
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
    
    def optimize_cache_strategy(self):
        """优化缓存策略"""
        print("\n===== 优化缓存策略 =====")
        
        # 测试缓存性能
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
        cache_file = os.path.join(self.cache_dir, "test_cache.pkl")
        with open(cache_file, 'wb') as f:
            pickle.dump(test_data, f)
        end_time = time.time()
        print(f"文件缓存设置时间: {end_time - start_time:.4f}秒")
        
        # 测试缓存读取
        start_time = time.time()
        with open(cache_file, 'rb') as f:
            loaded_data = pickle.load(f)
        end_time = time.time()
        print(f"文件缓存读取时间: {end_time - start_time:.4f}秒")
        
        # 清理测试文件
        if os.path.exists(cache_file):
            os.unlink(cache_file)
    
    def _get_cache_key(self, url):
        """生成缓存键"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _save_to_cache(self, key, data):
        """保存到缓存"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
        except:
            pass
    
    def _get_from_cache(self, key):
        """从缓存获取"""
        cache_file = os.path.join(self.cache_dir, f"{key}.pkl")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except:
                pass
        return None
    
    def run_optimization(self):
        """运行所有优化"""
        print("开始综合性能优化...")
        
        # 优化网络特征采集
        network_config = self.optimize_network_features_collection()
        
        # 优化数据采集
        self.optimize_data_collection()
        
        # 优化缓存策略
        self.optimize_cache_strategy()
        
        print("\n综合性能优化完成！")
        print(f"推荐配置: {network_config}")

def main():
    """主函数"""
    optimizer = PerformanceOptimizer()
    optimizer.run_optimization()

if __name__ == "__main__":
    main()
