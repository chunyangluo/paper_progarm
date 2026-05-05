#!/usr/bin/env python3
"""
测试网络请求与资源管理优化措施
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import time
from core.inference import PhishingDetector, Config


def test_global_config():
    """测试全局配置"""
    print("===== 测试全局配置 =====")
    print(f"网络超时: {Config.NETWORK_TIMEOUT}秒")
    print(f"最大重试次数: {Config.MAX_RETRIES}")
    print(f"重试延迟: {Config.RETRY_DELAY}秒")
    print(f"缓存过期时间: {Config.CACHE_TTL}秒")
    print(f"哈希算法: {Config.HASH_ALGORITHM.__name__}")
    print(f"默认模型目录: {Config.DEFAULT_MODEL_DIR}")
    print(f"默认BERT模型: {Config.DEFAULT_BERT_MODEL}")
    print(f"API配置: {Config.API_CONFIG}")
    print("✓ 全局配置测试通过")


def test_url_validation():
    """测试URL验证"""
    print("\n===== 测试URL验证 =====")
    detector = PhishingDetector(model_dir="models")
    
    # 测试None URL
    result = detector.detect("测试文本", None)
    assert 'error' in result, "None URL 应该返回错误"
    print("✓ None URL 测试通过")
    
    # 测试空URL
    result = detector.detect("测试文本", "")
    assert 'error' in result, "空URL 应该返回错误"
    print("✓ 空URL 测试通过")
    
    # 测试非字符串URL
    result = detector.detect("测试文本", 123)
    assert 'error' not in result, "非字符串URL 应该被转换"
    print("✓ 非字符串URL 测试通过")


def test_network_timeout():
    """测试网络超时控制"""
    print("\n===== 测试网络超时控制 =====")
    detector = PhishingDetector(model_dir="models")
    
    # 测试超时URL（使用不存在的域名）
    start_time = time.time()
    result = detector.detect("测试文本", "https://this-domain-does-not-exist-1234567890.com")
    elapsed_time = time.time() - start_time
    
    # 应该在合理时间内返回（小于15秒）
    assert elapsed_time < 15, f"网络请求超时控制失败，耗时: {elapsed_time}秒"
    print(f"✓ 网络超时控制测试通过，耗时: {elapsed_time:.2f}秒")


def test_cache_optimization():
    """测试缓存优化"""
    print("\n===== 测试缓存优化 =====")
    detector = PhishingDetector(model_dir="models", cache_ttl=1)
    
    # 第一次检测（应该缓存）
    text = "测试缓存文本"
    url = "https://example.com"
    
    start_time = time.time()
    result1 = detector.detect(text, url)
    time1 = time.time() - start_time
    print(f"第一次检测耗时: {time1:.4f}秒")
    
    # 第二次检测（应该从缓存获取）
    start_time = time.time()
    result2 = detector.detect(text, url)
    time2 = time.time() - start_time
    print(f"第二次检测耗时: {time2:.4f}秒")
    
    # 缓存应该更快
    assert time2 < time1 * 0.5, "缓存优化失败"
    print("✓ 缓存优化测试通过")
    
    # 测试缓存过期
    time.sleep(1.1)  # 等待缓存过期
    start_time = time.time()
    result3 = detector.detect(text, url)
    time3 = time.time() - start_time
    print(f"缓存过期后检测耗时: {time3:.4f}秒")
    assert time3 > time2, "缓存过期测试失败"
    print("✓ 缓存过期测试通过")


def test_model_fallback():
    """测试模型加载失败降级策略"""
    print("\n===== 测试模型加载失败降级策略 =====")
    
    # 使用不存在的模型目录
    detector = PhishingDetector(model_dir="non_existent_directory")
    
    # 应该降级到规则引擎
    result = detector.detect("【支付宝】你的账户存在安全风险", "https://alipay-veri.com")
    assert result['model'] == 'rule_based', "模型降级策略失败"
    print("✓ 模型加载失败降级策略测试通过")


def test_batch_processing():
    """测试批量处理"""
    print("\n===== 测试批量处理 =====")
    detector = PhishingDetector(model_dir="models")
    
    texts = [
        "【支付宝】你的账户存在安全风险",
        "【微信团队】你的微信账号异地登录",
        "【支付宝】你的余额宝收益已到账",
        "【微信团队】你的微信支付分已更新"
    ]
    
    urls = [
        "https://alipay-veri.com",
        "https://wx-safe.cn",
        "https://www.alipay.com",
        "https://weixin.qq.com"
    ]
    
    # 测试普通批量处理
    start_time = time.time()
    results1 = detector.batch_detect(texts, urls)
    time1 = time.time() - start_time
    print(f"普通批量处理耗时: {time1:.4f}秒")
    assert len(results1) == 4, "普通批量处理失败"
    print("✓ 普通批量处理测试通过")
    
    # 测试基于文本长度的负载均衡
    start_time = time.time()
    results2 = detector.batch_detect(texts, urls, use_length_balancing=True)
    time2 = time.time() - start_time
    print(f"负载均衡批量处理耗时: {time2:.4f}秒")
    assert len(results2) == 4, "负载均衡批量处理失败"
    print("✓ 负载均衡批量处理测试通过")


def test_edge_cases():
    """测试边界情况"""
    print("\n===== 测试边界情况 =====")
    detector = PhishingDetector(model_dir="models")
    
    # 测试长文本
    long_text = "a" * 1000
    result = detector.detect(long_text, "https://example.com")
    assert 'error' not in result, "长文本测试失败"
    print("✓ 长文本测试通过")
    
    # 测试长URL
    long_url = "https://example.com/" + "a" * 1000
    result = detector.detect("测试文本", long_url)
    assert 'error' not in result, "长URL测试失败"
    print("✓ 长URL测试通过")
    
    # 测试特殊字符URL
    special_url = "https://example.com/path?param1=value1&param2=value2#fragment"
    result = detector.detect("测试文本", special_url)
    assert 'error' not in result, "特殊字符URL测试失败"
    print("✓ 特殊字符URL测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始测试网络请求与资源管理优化措施...\n")
    
    try:
        test_global_config()
        test_url_validation()
        test_network_timeout()
        test_cache_optimization()
        test_model_fallback()
        test_batch_processing()
        test_edge_cases()
        
        print("\n🎉 所有测试通过！")
        print("网络请求与资源管理优化措施已成功实施。")
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    run_all_tests()
