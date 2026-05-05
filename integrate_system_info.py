#!/usr/bin/env python3
"""
系统信息整合脚本
自动整合系统核心文件的原始内容到一个文件中
"""

import os
from datetime import datetime

# 要整合的核心文件
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

def integrate_info():
    """整合系统信息"""
    report = f"# 系统核心文件整合\n"
    report += f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    report += f"整合文件数量: {len(CORE_FILES) + len(AUX_FILES)}\n\n"
    
    # 整合核心文件
    report += "## 核心文件\n\n"
    for file_path in CORE_FILES:
        if os.path.exists(file_path):
            content = read_file(file_path)
            report += f"### {file_path}\n"
            report += f"```python\n{content}\n```\n\n"
        else:
            report += f"### {file_path}\n"
            report += "文件不存在\n\n"
    
    # 整合辅助文件
    report += "## 辅助文件\n\n"
    for file_path in AUX_FILES:
        if os.path.exists(file_path):
            content = read_file(file_path)
            report += f"### {file_path}\n"
            if file_path.endswith('.py'):
                report += f"```python\n{content}\n```\n\n"
            else:
                report += f"```\n{content}\n```\n\n"
        else:
            report += f"### {file_path}\n"
            report += "文件不存在\n\n"
    
    # 保存报告
    report_path = 'system_integration_report.txt'
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"信息整合完成: {report_path}")
    return report_path

if __name__ == "__main__":
    integrate_info()
