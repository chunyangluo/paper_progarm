#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
中文钓鱼文本识别系统

本系统包含以下功能：
1. 多模态钓鱼检测（文本/URL/网络行为）
2. 实时预警
3. 可视化溯源
4. 增量训练
5. 支持多输入方式
6. 支持多检测场景
"""

import os
import sys
import time
import json
from datetime import datetime
from core.inference import PhishingDetector
from core.visualization import VisualizationModule
from core.incremental_training import IncrementalTrainer
from core.scenario_processor import ScenarioProcessor

class PhishingDetectionSystem:
    """完整的钓鱼检测系统"""
    
    def __init__(self, model_dir="models", data_dir="data", output_dir="output"):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.detector = PhishingDetector(model_dir=model_dir)
        self.visualizer = VisualizationModule()
        self.incremental_trainer = IncrementalTrainer(model_dir=model_dir, data_dir=data_dir)
        self.scenario_processor = ScenarioProcessor()
        self.alert_history = []
        self.detection_history = []
        
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/alerts", exist_ok=True)
        os.makedirs(f"{output_dir}/reports", exist_ok=True)
    
    def detect(self, text, url, scenario="general", use_cache=True):
        """检测钓鱼文本"""
        start_time = time.time()
        
        # 处理场景
        processed_text, processed_url = self.scenario_processor.process(scenario, text, url)
        
        # 检测
        result = self.detector.detect(processed_text, processed_url, use_cache)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        result['processing_time'] = processing_time
        
        # 提取特征
        features = self._extract_features(processed_url)
        
        # 生成报告
        sample = {
            'id': f"sample_{int(time.time())}",
            'text': text,
            'url': url,
            'scenario': scenario
        }
        
        report = self.visualizer.generate_report(sample, result, features)
        
        # 记录检测历史
        self.detection_history.append({
            'timestamp': datetime.now().isoformat(),
            'text': text,
            'url': url,
            'scenario': scenario,
            'result': result,
            'processing_time': processing_time
        })
        
        # 实时预警
        if result.get('prediction') == '钓鱼' and result.get('confidence', 0) > 0.7:
            alert = self._generate_alert(sample, result, features)
            self.alert_history.append(alert)
            self._save_alert(alert)
            print(f"🚨 预警：检测到钓鱼攻击！置信度：{result['confidence']:.4f}")
        
        # 保存报告
        self._save_report(report)
        
        return result, report
    
    def batch_detect(self, samples, batch_size=16):
        """批量检测钓鱼文本"""
        start_time = time.time()
        
        texts = [sample['text'] for sample in samples]
        urls = [sample['url'] for sample in samples]
        scenarios = [sample.get('scenario', 'general') for sample in samples]
        
        # 处理场景
        processed_texts = []
        processed_urls = []
        for text, url, scenario in zip(texts, urls, scenarios):
            processed_text, processed_url = self.scenario_processor.process(scenario, text, url)
            processed_texts.append(processed_text)
            processed_urls.append(processed_url)
        
        # 批量检测
        results = self.detector.batch_detect(processed_texts, processed_urls, batch_size)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 生成报告和预警
        reports = []
        for i, (result, sample) in enumerate(zip(results, samples)):
            result['processing_time'] = processing_time / len(samples)
            
            # 提取特征
            features = self._extract_features(processed_urls[i])
            
            # 生成报告
            report_sample = {
                'id': f"sample_{int(time.time())}_{i}",
                'text': sample['text'],
                'url': sample['url'],
                'scenario': sample.get('scenario', 'general')
            }
            
            report = self.visualizer.generate_report(report_sample, result, features)
            reports.append(report)
            
            # 记录检测历史
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'text': sample['text'],
                'url': sample['url'],
                'scenario': sample.get('scenario', 'general'),
                'result': result,
                'processing_time': result['processing_time']
            })
            
            # 实时预警
            if result.get('prediction') == '钓鱼' and result.get('confidence', 0) > 0.7:
                alert = self._generate_alert(report_sample, result, features)
                self.alert_history.append(alert)
                self._save_alert(alert)
                print(f"🚨 预警：检测到钓鱼攻击！置信度：{result['confidence']:.4f}")
            
            # 保存报告
            self._save_report(report)
        
        return results, reports
    
    def incremental_train(self, new_samples, model_type="multimodal", epochs=10):
        """增量训练模型"""
        print(f"开始增量训练 {model_type} 模型...")
        
        # 准备样本数据
        samples = []
        for sample in new_samples:
            samples.append({
                'text': sample['text'],
                'url': sample['url'],
                'label': 1 if sample.get('label', 'normal') == 'phishing' else 0,
                'scenario': sample.get('scenario', 'general')
            })
        
        # 执行增量训练
        result = self.incremental_trainer.update_model(samples, model_type, epochs)
        
        # 重新加载模型
        self.detector = PhishingDetector(model_dir=self.model_dir)
        
        print(f"增量训练完成！")
        print(f"模型类型：{result['model_type']}")
        print(f"训练样本数：{result['training_samples']}")
        print(f"测试样本数：{result['test_samples']}")
        print(f"评估结果：")
        for key, value in result['evaluation_results'].items():
            print(f"  {key}: {value:.4f}")
        
        return result
    
    def visualize(self, sample, result, features):
        """可视化分析"""
        # 生成攻击路径
        attack_path = self.visualizer.generate_attack_path(sample)
        
        # 可视化特征重要性
        feature_importance = self.visualizer.visualize_features(features)
        
        # 分析趋势
        trend_analysis = self.visualizer.analyze_trends(self.detection_history[-100:])  # 最近100条记录
        
        return {
            'attack_path': attack_path,
            'feature_importance': feature_importance,
            'trend_analysis': trend_analysis
        }
    
    def get_stats(self):
        """获取系统统计信息"""
        total_detections = len(self.detection_history)
        phishing_detections = sum(1 for item in self.detection_history if item['result'].get('prediction') == '钓鱼')
        normal_detections = total_detections - phishing_detections
        
        # 计算平均处理时间
        avg_processing_time = 0
        if total_detections > 0:
            avg_processing_time = sum(item['processing_time'] for item in self.detection_history) / total_detections
        
        return {
            'total_detections': total_detections,
            'phishing_detections': phishing_detections,
            'normal_detections': normal_detections,
            'alert_count': len(self.alert_history),
            'avg_processing_time': avg_processing_time
        }
    
    def _extract_features(self, url):
        """提取特征"""
        from data_preprocessing import URLFeatureExtractor, NetworkBehaviorExtractor
        
        url_extractor = URLFeatureExtractor()
        network_extractor = NetworkBehaviorExtractor()
        
        url_features = url_extractor.extract_features(url)
        network_features = network_extractor.extract_features(url)
        
        return {
            **url_features,
            **network_features
        }
    
    def _generate_alert(self, sample, result, features):
        """生成预警"""
        return {
            'alert_id': f"alert_{int(time.time())}",
            'timestamp': datetime.now().isoformat(),
            'sample_id': sample['id'],
            'text': sample['text'],
            'url': sample['url'],
            'scenario': sample['scenario'],
            'prediction': result['prediction'],
            'confidence': result['confidence'],
            'features': features,
            'severity': 'high' if result['confidence'] > 0.9 else 'medium'
        }
    
    def _save_alert(self, alert):
        """保存预警"""
        alert_path = f"{self.output_dir}/alerts/alert_{alert['alert_id']}.json"
        with open(alert_path, 'w', encoding='utf-8') as f:
            json.dump(alert, f, ensure_ascii=False, indent=2)
    
    def _save_report(self, report):
        """保存报告"""
        report_id = report['sample_info']['timestamp'].replace(':', '-').replace('.', '-')
        report_path = f"{self.output_dir}/reports/report_{report_id}.json"
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

def main():
    """主函数"""
    system = PhishingDetectionSystem()
    
    print("=" * 80)
    print("中文钓鱼文本识别系统")
    print("=" * 80)
    print("功能：")
    print("1. 多模态钓鱼检测（文本/URL/网络行为）")
    print("2. 实时预警")
    print("3. 可视化溯源")
    print("4. 增量训练")
    print("5. 支持多输入方式")
    print("6. 支持多检测场景")
    print("=" * 80)
    
    # 测试样本
    test_samples = [
        {
            'text': "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户",
            'url': "https://alipay-veri.com",
            'scenario': "sms"
        },
        {
            'text': "【微信团队】你的微信账号异地登录，点击 https://wx-safe.cn 验证手机号，否则24小时封禁",
            'url': "https://wx-safe.cn",
            'scenario': "sms"
        },
        {
            'text': "【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情，官方网址 https://www.alipay.com",
            'url': "https://www.alipay.com",
            'scenario': "sms"
        },
        {
            'text': "【微信团队】你的微信支付分已更新，打开微信APP-我-服务可查询，官方网址 https://weixin.qq.com",
            'url': "https://weixin.qq.com",
            'scenario': "sms"
        }
    ]
    
    print("\n开始测试多模态钓鱼检测...")
    for i, sample in enumerate(test_samples):
        print(f"\n测试样本 {i+1}:")
        print(f"文本: {sample['text']}")
        print(f"URL: {sample['url']}")
        print(f"场景: {sample['scenario']}")
        
        result, report = system.detect(sample['text'], sample['url'], sample['scenario'])
        
        print(f"\n检测结果:")
        print(f"模型: {result['model']}")
        print(f"预测: {result['prediction']}")
        print(f"置信度: {result['confidence']:.4f}")
        print(f"处理时间: {result['processing_time']:.4f} 秒")
        
        if 'details' in result:
            print(f"\n特征详情:")
            print(f"URL特征: {result['details']['url_features']}")
            print(f"网络行为特征: {result['details']['network_features']}")
    
    print("\n" + "=" * 80)
    print("批量检测测试...")
    results, reports = system.batch_detect(test_samples)
    print(f"批量检测完成，处理了 {len(results)} 个样本")
    
    print("\n" + "=" * 80)
    print("系统统计信息:")
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 80)
    print("系统测试完成！")
    print("您可以通过以下方式使用系统：")
    print("1. 单次检测: system.detect(text, url, scenario)")
    print("2. 批量检测: system.batch_detect(samples)")
    print("3. 增量训练: system.incremental_train(new_samples)")
    print("4. 可视化分析: system.visualize(sample, result, features)")
    print("5. 获取统计信息: system.get_stats()")

if __name__ == "__main__":
    main()
