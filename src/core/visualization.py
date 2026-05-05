import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import networkx as nx
from datetime import datetime
import base64
from io import BytesIO

class VisualizationModule:
    """可视化溯源模块，提供钓鱼攻击的可视化分析和溯源功能"""
    
    def __init__(self):
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    def generate_attack_path(self, sample, related_samples=None):
        """生成钓鱼攻击路径图"""
        # 创建有向图
        G = nx.DiGraph()
        
        # 添加节点
        G.add_node(sample['id'], label=sample['text'][:50] + '...' if len(sample['text']) > 50 else sample['text'])
        
        # 添加相关样本节点
        if related_samples:
            for i, related in enumerate(related_samples):
                related_id = f"related_{i}"
                G.add_node(related_id, label=related['text'][:50] + '...' if len(related['text']) > 50 else related['text'])
                G.add_edge(related_id, sample['id'], label="关联")
        
        # 绘制图形
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, k=0.3)
        
        # 绘制节点
        nx.draw_networkx_nodes(G, pos, node_size=3000, node_color='lightblue')
        
        # 绘制边
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True)
        
        # 绘制标签
        labels = nx.get_node_attributes(G, 'label')
        nx.draw_networkx_labels(G, pos, labels, font_size=10)
        
        # 添加边标签
        edge_labels = nx.get_edge_attributes(G, 'label')
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title('钓鱼攻击路径分析')
        plt.axis('off')
        
        # 保存为Base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def visualize_features(self, features):
        """可视化特征重要性"""
        # 提取特征重要性
        feature_names = list(features.keys())
        feature_values = list(features.values())
        
        # 绘制条形图
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature_values, y=feature_names)
        plt.title('特征重要性分析')
        plt.xlabel('特征值')
        plt.ylabel('特征名称')
        plt.tight_layout()
        
        # 保存为Base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def analyze_trends(self, historical_data):
        """分析钓鱼攻击趋势"""
        # 按日期分组统计
        date_counts = {}
        for item in historical_data:
            date = item.get('date', datetime.now().strftime('%Y-%m-%d'))
            if date not in date_counts:
                date_counts[date] = 0
            date_counts[date] += 1
        
        # 排序日期
        sorted_dates = sorted(date_counts.keys())
        counts = [date_counts[date] for date in sorted_dates]
        
        # 绘制趋势图
        plt.figure(figsize=(12, 6))
        plt.plot(sorted_dates, counts, marker='o')
        plt.title('钓鱼攻击趋势分析')
        plt.xlabel('日期')
        plt.ylabel('攻击数量')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # 保存为Base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
    
    def generate_report(self, sample, detection_result, features):
        """生成详细的分析报告"""
        report = {
            "sample_info": {
                "text": sample.get('text', ''),
                "url": sample.get('url', ''),
                "scenario": sample.get('scenario', 'general'),
                "timestamp": datetime.now().isoformat()
            },
            "detection_result": {
                "model": detection_result.get('model', 'unknown'),
                "prediction": detection_result.get('prediction', 'unknown'),
                "confidence": detection_result.get('confidence', 0.0),
                "processing_time": detection_result.get('processing_time', 0.0)
            },
            "features": features,
            "visualizations": {
                "attack_path": self.generate_attack_path(sample),
                "feature_importance": self.visualize_features(features)
            },
            "recommendations": self._generate_recommendations(detection_result, features)
        }
        
        return report
    
    def _generate_recommendations(self, detection_result, features):
        """生成安全建议"""
        recommendations = []
        
        if detection_result.get('prediction') == '钓鱼':
            recommendations.append('不要点击邮件中的链接')
            recommendations.append('不要提供个人敏感信息')
            recommendations.append('报告此钓鱼邮件给相关部门')
        
        # 根据特征生成具体建议
        if features.get('has_suspicious_keywords'):
            recommendations.append('邮件中包含可疑关键词，请注意核实')
        
        if features.get('has_suspicious_url'):
            recommendations.append('链接看起来可疑，建议不要访问')
        
        return recommendations
    
    def visualize_confusion_matrix(self, cm):
        """可视化混淆矩阵"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['正常', '钓鱼'], 
                    yticklabels=['正常', '钓鱼'])
        plt.title('混淆矩阵')
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.tight_layout()
        
        # 保存为Base64编码
        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close()
        
        return image_base64
