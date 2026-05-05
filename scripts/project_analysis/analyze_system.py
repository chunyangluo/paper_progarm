#!/usr/bin/env python3
"""
系统分析脚本
自动分析系统核心文件并导出关键信息到分析报告
"""

import os
import re
import ast
from datetime import datetime

_REPO_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
_REPORTS_DIR = os.path.join(_REPO_ROOT, "docs", "reports")

# 要分析的核心文件
CORE_FILES = [
    'core/system.py',
    'core/inference.py',
    'core/data_preprocessing.py',
    'core/model_training.py',
    'core/incremental_training.py',
    'core/scenario_processor.py',
    'core/visualization.py'
]

# 辅助文件
AUX_FILES = [
    'requirements.txt',
    'Dockerfile',
    'deploy.sh'
]

def read_file(file_path):
    """读取文件内容"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def extract_class_info(content):
    """提取类信息"""
    tree = ast.parse(content)
    classes = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_info = {
                'name': node.name,
                'methods': [],
                'docstring': ast.get_docstring(node) or ''
            }
            
            # 提取方法
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    method_info = {
                        'name': item.name,
                        'docstring': ast.get_docstring(item) or '',
                        'args': [arg.arg for arg in item.args.args]
                    }
                    class_info['methods'].append(method_info)
            
            classes.append(class_info)
    
    return classes

def extract_function_info(content):
    """提取函数信息"""
    tree = ast.parse(content)
    functions = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and not isinstance(node.parent, ast.ClassDef):
            func_info = {
                'name': node.name,
                'docstring': ast.get_docstring(node) or '',
                'args': [arg.arg for arg in node.args.args]
            }
            functions.append(func_info)
    
    return functions

def analyze_system_py(content):
    """分析 system.py"""
    info = {
        '初始化逻辑': '',
        '资源管理': '',
        '异常处理': '',
        '目录创建/权限': ''
    }
    
    # 提取初始化逻辑
    init_pattern = r'def __init__\([^)]*\):[^\n]*\n[\s\S]*?(?=def |class |$)'
    init_matches = re.findall(init_pattern, content, re.MULTILINE)
    if init_matches:
        info['初始化逻辑'] = '\n'.join(init_matches[:2])
    
    # 提取资源管理
    resource_pattern = r'(load|release|model|resource)[^\n]*\n[\s\S]*?(?=def |class |$)'
    resource_matches = re.findall(resource_pattern, content, re.IGNORECASE | re.MULTILINE)
    if resource_matches:
        info['资源管理'] = '\n'.join(resource_matches[:3])
    
    # 提取异常处理
    exception_pattern = r'try:\s*\n[\s\S]*?except[^\n]*:\s*\n[\s\S]*?(?=finally:|$)'
    exception_matches = re.findall(exception_pattern, content, re.MULTILINE)
    if exception_matches:
        info['异常处理'] = '\n'.join(exception_matches[:2])
    
    # 提取目录创建/权限
    dir_pattern = r'(os\.makedirs|mkdir|chmod|permission)[^\n]*\n[\s\S]*?(?=def |class |$)'
    dir_matches = re.findall(dir_pattern, content, re.IGNORECASE | re.MULTILINE)
    if dir_matches:
        info['目录创建/权限'] = '\n'.join(dir_matches[:2])
    
    return info

def analyze_inference_py(content):
    """分析 inference.py"""
    info = {
        '模型推理流程': '',
        '多模态特征融合逻辑': '',
        '置信度计算': '',
        '效率优化': ''
    }
    
    # 提取模型推理流程
    detect_pattern = r'def detect\([^)]*\):[^\n]*\n[\s\S]*?(?=def |class |$)'
    detect_matches = re.findall(detect_pattern, content, re.MULTILINE)
    if detect_matches:
        info['模型推理流程'] = detect_matches[0][:500] + '...' if len(detect_matches[0]) > 500 else detect_matches[0]
    
    # 提取多模态特征融合逻辑
    fusion_pattern = r'(multimodal|feature.*fusion|url_features|network_features)[^\n]*\n[\s\S]*?(?=def |class |$)'
    fusion_matches = re.findall(fusion_pattern, content, re.IGNORECASE | re.MULTILINE)
    if fusion_matches:
        info['多模态特征融合逻辑'] = '\n'.join(fusion_matches[:3])
    
    # 提取置信度计算
    confidence_pattern = r'(softmax|confidence|prob)[^\n]*\n[\s\S]*?(?=def |class |$)'
    confidence_matches = re.findall(confidence_pattern, content, re.IGNORECASE | re.MULTILINE)
    if confidence_matches:
        info['置信度计算'] = '\n'.join(confidence_matches[:2])
    
    # 提取效率优化
    optimize_pattern = r'(cache|batch|efficiency|optimize)[^\n]*\n[\s\S]*?(?=def |class |$)'
    optimize_matches = re.findall(optimize_pattern, content, re.IGNORECASE | re.MULTILINE)
    if optimize_matches:
        info['效率优化'] = '\n'.join(optimize_matches[:3])
    
    return info

def analyze_data_preprocessing_py(content):
    """分析 data_preprocessing.py"""
    info = {
        '文本清洗': '',
        'URL特征提取规则': '',
        '网络行为特征采集逻辑': '',
        '数据归一化/标准化': ''
    }
    
    # 提取文本清洗
    text_clean_pattern = r'(clean|preprocess|text)[^\n]*\n[\s\S]*?(?=def |class |$)'
    text_clean_matches = re.findall(text_clean_pattern, content, re.IGNORECASE | re.MULTILINE)
    if text_clean_matches:
        info['文本清洗'] = '\n'.join(text_clean_matches[:2])
    
    # 提取URL特征提取规则
    url_pattern = r'(url.*feature|extract.*url)[^\n]*\n[\s\S]*?(?=def |class |$)'
    url_matches = re.findall(url_pattern, content, re.IGNORECASE | re.MULTILINE)
    if url_matches:
        info['URL特征提取规则'] = '\n'.join(url_matches[:3])
    
    # 提取网络行为特征采集逻辑
    network_pattern = r'(network.*feature|extract.*network)[^\n]*\n[\s\S]*?(?=def |class |$)'
    network_matches = re.findall(network_pattern, content, re.IGNORECASE | re.MULTILINE)
    if network_matches:
        info['网络行为特征采集逻辑'] = '\n'.join(network_matches[:2])
    
    # 提取数据归一化/标准化
    normalize_pattern = r'(normalize|standardize|scale)[^\n]*\n[\s\S]*?(?=def |class |$)'
    normalize_matches = re.findall(normalize_pattern, content, re.IGNORECASE | re.MULTILINE)
    if normalize_matches:
        info['数据归一化/标准化'] = '\n'.join(normalize_matches[:2])
    
    return info

def analyze_model_training_py(content):
    """分析 model_training.py"""
    info = {
        'BERT-TextCNN混合模型定义': '',
        '损失函数设计': '',
        '训练策略': '',
        '验证集划分': ''
    }
    
    # 提取BERT-TextCNN混合模型定义
    model_pattern = r'class BERTTextCNN\([^)]*\):[^\n]*\n[\s\S]*?(?=class |def |$)'
    model_matches = re.findall(model_pattern, content, re.MULTILINE)
    if model_matches:
        info['BERT-TextCNN混合模型定义'] = model_matches[0][:500] + '...' if len(model_matches[0]) > 500 else model_matches[0]
    
    # 提取损失函数设计
    loss_pattern = r'(loss|criterion)[^\n]*\n[\s\S]*?(?=def |class |$)'
    loss_matches = re.findall(loss_pattern, content, re.IGNORECASE | re.MULTILINE)
    if loss_matches:
        info['损失函数设计'] = '\n'.join(loss_matches[:2])
    
    # 提取训练策略
    train_pattern = r'(optimizer|scheduler|lr|learning.*rate)[^\n]*\n[\s\S]*?(?=def |class |$)'
    train_matches = re.findall(train_pattern, content, re.IGNORECASE | re.MULTILINE)
    if train_matches:
        info['训练策略'] = '\n'.join(train_matches[:3])
    
    # 提取验证集划分
    val_pattern = r'(train_test_split|validation|val.*set)[^\n]*\n[\s\S]*?(?=def |class |$)'
    val_matches = re.findall(val_pattern, content, re.IGNORECASE | re.MULTILINE)
    if val_matches:
        info['验证集划分'] = '\n'.join(val_matches[:2])
    
    return info

def analyze_incremental_training_py(content):
    """分析 incremental_training.py"""
    info = {
        '增量数据融合方式': '',
        '模型参数更新策略': '',
        '过拟合防控': '',
        '评估指标完整性': ''
    }
    
    # 提取增量数据融合方式
    fusion_pattern = r'(merge|fusion|combine|incremental.*data)[^\n]*\n[\s\S]*?(?=def |class |$)'
    fusion_matches = re.findall(fusion_pattern, content, re.IGNORECASE | re.MULTILINE)
    if fusion_matches:
        info['增量数据融合方式'] = '\n'.join(fusion_matches[:2])
    
    # 提取模型参数更新策略
    update_pattern = r'(update|fine.*tune|parameter.*update)[^\n]*\n[\s\S]*?(?=def |class |$)'
    update_matches = re.findall(update_pattern, content, re.IGNORECASE | re.MULTILINE)
    if update_matches:
        info['模型参数更新策略'] = '\n'.join(update_matches[:2])
    
    # 提取过拟合防控
    overfit_pattern = r'(dropout|regularization|overfit)[^\n]*\n[\s\S]*?(?=def |class |$)'
    overfit_matches = re.findall(overfit_pattern, content, re.IGNORECASE | re.MULTILINE)
    if overfit_matches:
        info['过拟合防控'] = '\n'.join(overfit_matches[:2])
    
    # 提取评估指标完整性
    metric_pattern = r'(metric|evaluation|accuracy|precision|recall|f1)[^\n]*\n[\s\S]*?(?=def |class |$)'
    metric_matches = re.findall(metric_pattern, content, re.IGNORECASE | re.MULTILINE)
    if metric_matches:
        info['评估指标完整性'] = '\n'.join(metric_matches[:3])
    
    return info

def analyze_scenario_processor_py(content):
    """分析 scenario_processor.py"""
    info = {
        '不同场景的差异化处理逻辑': '',
        '场景特征权重设计': ''
    }
    
    # 提取不同场景的差异化处理逻辑
    scenario_pattern = r'(process.*sms|process.*email|process.*link)[^\n]*\n[\s\S]*?(?=def |class |$)'
    scenario_matches = re.findall(scenario_pattern, content, re.IGNORECASE | re.MULTILINE)
    if scenario_matches:
        info['不同场景的差异化处理逻辑'] = '\n'.join(scenario_matches[:3])
    
    # 提取场景特征权重设计
    weight_pattern = r'(weight|feature.*importance|scenario.*feature)[^\n]*\n[\s\S]*?(?=def |class |$)'
    weight_matches = re.findall(weight_pattern, content, re.IGNORECASE | re.MULTILINE)
    if weight_matches:
        info['场景特征权重设计'] = '\n'.join(weight_matches[:2])
    
    return info

def analyze_visualization_py(content):
    """分析 visualization.py"""
    info = {
        '可视化图表的准确性': '',
        '特征重要性计算逻辑': '',
        '攻击路径分析的合理性': ''
    }
    
    # 提取可视化图表的准确性
    chart_pattern = r'(plot|chart|visualize)[^\n]*\n[\s\S]*?(?=def |class |$)'
    chart_matches = re.findall(chart_pattern, content, re.IGNORECASE | re.MULTILINE)
    if chart_matches:
        info['可视化图表的准确性'] = '\n'.join(chart_matches[:2])
    
    # 提取特征重要性计算逻辑
    importance_pattern = r'(importance|feature.*importance|weight.*feature)[^\n]*\n[\s\S]*?(?=def |class |$)'
    importance_matches = re.findall(importance_pattern, content, re.IGNORECASE | re.MULTILINE)
    if importance_matches:
        info['特征重要性计算逻辑'] = '\n'.join(importance_matches[:2])
    
    # 提取攻击路径分析的合理性
    path_pattern = r'(path|attack.*path|trace)[^\n]*\n[\s\S]*?(?=def |class |$)'
    path_matches = re.findall(path_pattern, content, re.IGNORECASE | re.MULTILINE)
    if path_matches:
        info['攻击路径分析的合理性'] = '\n'.join(path_matches[:2])
    
    return info

def analyze_requirements_txt(content):
    """分析 requirements.txt"""
    lines = content.strip().split('\n')
    dependencies = [line for line in lines if line and not line.startswith('#')]
    return {
        '依赖总数': len(dependencies),
        '依赖列表': dependencies
    }

def generate_report():
    """生成分析报告"""
    report = f"# 系统分析报告\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    # 分析核心文件
    for file_path in CORE_FILES:
        if os.path.exists(file_path):
            content = read_file(file_path)
            report += f"## {os.path.basename(file_path)}\n"
            
            if file_path == 'core/system.py':
                info = analyze_system_py(content)
            elif file_path == 'core/inference.py':
                info = analyze_inference_py(content)
            elif file_path == 'core/data_preprocessing.py':
                info = analyze_data_preprocessing_py(content)
            elif file_path == 'core/model_training.py':
                info = analyze_model_training_py(content)
            elif file_path == 'core/incremental_training.py':
                info = analyze_incremental_training_py(content)
            elif file_path == 'core/scenario_processor.py':
                info = analyze_scenario_processor_py(content)
            elif file_path == 'core/visualization.py':
                info = analyze_visualization_py(content)
            
            for key, value in info.items():
                report += f"### {key}\n"
                report += f"```python\n{value}\n```\n\n"
        else:
            report += f"## {os.path.basename(file_path)}\n"
            report += "文件不存在\n\n"
    
    # 分析辅助文件
    report += "## 辅助文件分析\n"
    for file_path in AUX_FILES:
        if os.path.exists(file_path):
            content = read_file(file_path)
            report += f"### {os.path.basename(file_path)}\n"
            
            if file_path == 'requirements.txt':
                info = analyze_requirements_txt(content)
                report += f"依赖总数: {info['依赖总数']}\n"
                report += "依赖列表:\n"
                for dep in info['依赖列表']:
                    report += f"- {dep}\n"
            else:
                report += f"```\n{content[:500]}...\n```\n"
            report += "\n"
        else:
            report += f"### {os.path.basename(file_path)}\n"
            report += "文件不存在\n\n"
    
    # 保存报告
    os.makedirs(_REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(_REPORTS_DIR, "system_analysis_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"分析报告已生成: {report_path}")
    return report_path

if __name__ == "__main__":
    generate_report()
