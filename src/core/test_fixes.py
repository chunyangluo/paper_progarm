#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试修复效果的脚本
验证代理检测、自动切换和网络特征采集功能
"""
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataCollector

def test_proxy_detection():
    """测试代理检测功能"""
    print("\n===== 测试代理检测功能 =====")
    collector = DataCollector()
    
    # 测试check_proxy方法
    proxy_available = collector.check_proxy()
    print(f"代理状态: {'可用' if proxy_available else '不可用'}")
    
    # 测试get_available_session方法
    session = collector.get_available_session()
    print("会话获取成功")
    
    # 测试会话是否可用
    try:
        # 测试一个简单的HTTP请求
        test_url = "http://www.google.com"
        response = session.get(test_url, timeout=5)
        print(f"会话测试成功，状态码: {response.status_code}")
        return True
    except Exception as e:
        print(f"会话测试失败: {e}")
        return False

def test_data_collection():
    """测试数据采集功能"""
    print("\n===== 测试数据采集功能 =====")
    collector = DataCollector()
    
    # 测试OpenPhish数据采集
    print("测试OpenPhish数据采集...")
    try:
        openphish_data = collector.collect_openphish_data(count=10)
        print(f"OpenPhish采集成功，获取 {len(openphish_data)} 条数据")
    except Exception as e:
        print(f"OpenPhish采集失败: {e}")
    
    # 测试Majestic数据采集
    print("\n测试Majestic数据采集...")
    try:
        majestic_data = collector.collect_majestic_data(count=10)
        print(f"Majestic采集成功，获取 {len(majestic_data)} 条数据")
    except Exception as e:
        print(f"Majestic采集失败: {e}")

def main():
    """主函数"""
    print("开始测试修复效果...")
    
    # 测试代理检测
    proxy_test_result = test_proxy_detection()
    print(f"代理检测测试: {'通过' if proxy_test_result else '失败'}")
    
    # 测试数据采集
    test_data_collection()
    
    print("\n测试完成！")

if __name__ == "__main__":
    main()
