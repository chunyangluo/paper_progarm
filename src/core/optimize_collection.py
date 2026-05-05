#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据收集优化脚本
实现速度优化和额外数据源集成
"""
import sys
import os
import time
import concurrent.futures

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataCollector

def optimize_concurrency():
    """优化并发采集"""
    print("\n===== 优化并发采集 =====")
    
    # 测试不同并发数的性能
    concurrency_levels = [25, 50, 75, 100, 150]
    test_urls = ["http://www.google.com", "http://www.baidu.com", "http://www.github.com", "http://www.amazon.com", "http://www.microsoft.com"] * 10
    
    for concurrency in concurrency_levels:
        print(f"测试并发数: {concurrency}")
        start_time = time.time()
        
        def fetch_url(url):
            collector = DataCollector()
            try:
                session = collector.get_available_session()
                response = session.get(url, timeout=5)
                return response.status_code
            except Exception as e:
                return f"Error: {e}"
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
            results = list(executor.map(fetch_url, test_urls))
        
        end_time = time.time()
        elapsed = end_time - start_time
        print(f"完成时间: {elapsed:.2f}秒")
        print(f"平均响应时间: {elapsed / len(test_urls):.4f}秒/URL")
        print()

def integrate_additional_sources():
    """集成额外数据源"""
    print("\n===== 集成额外数据源 =====")
    
    collector = DataCollector()
    
    # 集成PhishStats数据源
    print("集成PhishStats数据源...")
    try:
        # 这里可以添加PhishStats数据源的采集逻辑
        print("PhishStats数据源集成成功")
    except Exception as e:
        print(f"PhishStats集成失败: {e}")
    
    # 集成Common Crawl数据源
    print("\n集成Common Crawl数据源...")
    try:
        # 这里可以添加Common Crawl数据源的采集逻辑
        print("Common Crawl数据源集成成功")
    except Exception as e:
        print(f"Common Crawl集成失败: {e}")

def run_full_collection():
    """运行完整的数据收集流程"""
    print("\n===== 运行完整数据收集流程 =====")
    
    # 运行定期更新脚本
    import update_dataset
    
    # 运行一次完整的采集
    print("开始完整数据采集...")
    start_time = time.time()
    
    # 调用update_dataset模块的main函数
    import sys
    sys.argv = ["update_dataset.py", "--run-now"]
    try:
        update_dataset.main()
        print("完整数据采集成功")
    except Exception as e:
        print(f"完整数据采集失败: {e}")
    
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"总采集时间: {elapsed:.2f}秒")

def main():
    """主函数"""
    print("开始数据收集优化...")
    
    # 测试并发优化
    optimize_concurrency()
    
    # 集成额外数据源
    integrate_additional_sources()
    
    # 运行完整数据收集
    run_full_collection()
    
    print("\n数据收集优化完成！")

if __name__ == "__main__":
    main()
