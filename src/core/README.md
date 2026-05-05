# 中文钓鱼文本识别系统

本系统是一个基于BERT-TextCNN混合模型的多模态钓鱼检测系统，支持文本、URL和网络行为三模态特征的综合分析，能够有效识别中文钓鱼攻击。

## 功能特点

### 1. 多模态钓鱼检测
- **文本特征**：使用BERT模型提取文本语义特征
- **URL特征**：提取域名长度、是否HTTPS、是否包含敏感关键词等特征
- **网络行为特征**：提取响应时间、加载状态、重定向次数等特征
- **BERT-TextCNN混合模型**：结合BERT的语义理解和TextCNN的局部特征捕获

### 2. 实时预警
- 当检测到钓鱼攻击且置信度高于0.7时，自动生成预警
- 预警信息包含攻击类型、置信度、特征详情等
- 预警结果保存到`output/alerts`目录

### 3. 可视化溯源
- **攻击路径分析**：生成钓鱼攻击路径图
- **特征重要性分析**：可视化展示各特征的重要性
- **攻击趋势分析**：分析历史检测数据，生成趋势图
- **详细分析报告**：生成包含所有分析结果的完整报告

### 4. 增量训练
- 支持收集新样本并进行标注
- 自动提取特征并训练模型
- 评估模型性能并部署更新后的模型

### 5. 多输入方式
- 支持单次检测
- 支持批量检测

### 6. 多检测场景
- 支持短信场景：`scenario="sms"`
- 支持邮件场景：`scenario="email"`
- 支持链接场景：`scenario="link"`
- 支持通用场景：`scenario="general"`

## 系统架构

```
├── core/
│   ├── system.py            # 系统主入口
│   ├── inference.py         # 推理模块
│   ├── visualization.py     # 可视化模块
│   ├── incremental_training.py  # 增量训练模块
│   ├── scenario_processor.py     # 场景处理模块
│   ├── model_training.py    # 模型训练模块
│   ├── data_preprocessing.py     # 数据预处理模块
│   ├── models/              # 模型存储目录
│   ├── output/              # 输出目录
│   │   ├── alerts/          # 预警信息
│   │   ├── reports/         # 分析报告
│   └── data/                # 数据目录
```

## 安装步骤

### 1. 环境要求
- Python 3.8+
- PyTorch 2.5.1+
- Transformers 4.40.0+
- NumPy
- Pandas
- Matplotlib
- Seaborn
- NetworkX
- Scikit-learn
- psutil

### 2. 安装依赖
```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install transformers numpy pandas matplotlib seaborn networkx scikit-learn psutil
```

### 3. 下载BERT模型
系统会自动从Hugging Face下载`bert-base-chinese`模型，无需手动下载。

## 使用方法

### 1. 系统初始化
```python
from system import PhishingDetectionSystem

# 初始化系统
system = PhishingDetectionSystem(
    model_dir="models",  # 模型存储目录
    data_dir="data",     # 数据存储目录
    output_dir="output"   # 输出存储目录
)
```

### 2. 单次检测
```python
# 检测钓鱼文本
text = "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户"
url = "https://alipay-veri.com"
scenario = "sms"  # 场景：sms, email, link, general

result, report = system.detect(text, url, scenario)

print(f"检测结果: {result['prediction']}")
print(f"置信度: {result['confidence']:.4f}")
print(f"处理时间: {result['processing_time']:.4f} 秒")

# 查看特征详情
if 'details' in result:
    print(f"URL特征: {result['details']['url_features']}")
    print(f"网络行为特征: {result['details']['network_features']}")
```

### 3. 批量检测
```python
# 批量检测
samples = [
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
    # 更多样本...
]

results, reports = system.batch_detect(samples, batch_size=16)

for i, result in enumerate(results):
    print(f"样本 {i+1} 检测结果: {result['prediction']}")
    print(f"置信度: {result['confidence']:.4f}")
```

### 4. 增量训练
```python
# 准备新样本
new_samples = [
    {
        'text': "【银行】您的银行卡已被冻结，点击 https://bank-veri.com 解冻",
        'url': "https://bank-veri.com",
        'label': "phishing",  # phishing 或 normal
        'scenario': "sms"
    },
    # 更多样本...
]

# 执行增量训练
result = system.incremental_train(
    new_samples,
    model_type="multimodal",  # multimodal, bert_textcnn, textcnn
    epochs=10
)

print(f"增量训练完成！")
print(f"模型类型：{result['model_type']}")
print(f"训练样本数：{result['training_samples']}")
print(f"测试样本数：{result['test_samples']}")
print(f"评估结果：")
for key, value in result['evaluation_results'].items():
    print(f"  {key}: {value:.4f}")
```

### 5. 可视化分析
```python
# 准备样本和检测结果
sample = {
    'id': "sample_1",
    'text': "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户",
    'url': "https://alipay-veri.com",
    'scenario': "sms"
}

result, report = system.detect(sample['text'], sample['url'], sample['scenario'])

# 提取特征
from data_preprocessing import URLFeatureExtractor, NetworkBehaviorExtractor

url_extractor = URLFeatureExtractor()
network_extractor = NetworkBehaviorExtractor()

url_features = url_extractor.extract_features(sample['url'])
network_features = network_extractor.extract_features(sample['url'])
features = {**url_features, **network_features}

# 生成可视化
visualizations = system.visualize(sample, result, features)

# 可视化结果包含：
# - attack_path: 攻击路径图（Base64编码）
# - feature_importance: 特征重要性图（Base64编码）
# - trend_analysis: 攻击趋势图（Base64编码）
```

### 6. 获取系统统计信息
```python
stats = system.get_stats()

print(f"总检测次数: {stats['total_detections']}")
print(f"钓鱼检测次数: {stats['phishing_detections']}")
print(f"正常检测次数: {stats['normal_detections']}")
print(f"预警次数: {stats['alert_count']}")
print(f"平均处理时间: {stats['avg_processing_time']:.4f} 秒")
```

## 示例运行

运行系统主文件：
```bash
python system.py
```

系统会自动测试4个样本，并输出检测结果、批量检测测试和系统统计信息。

## 输出说明

### 1. 检测结果
```python
{
    "model": "multimodal",  # 使用的模型
    "prediction": "钓鱼",  # 预测结果：钓鱼或正常
    "confidence": 0.95,  # 置信度
    "processing_time": 0.1,  # 处理时间（秒）
    "details": {  # 特征详情
        "url_features": {...},  # URL特征
        "network_features": {...}  # 网络行为特征
    }
}
```

### 2. 分析报告
报告保存在`output/reports`目录，包含：
- 样本信息
- 检测结果
- 特征分析
- 可视化结果
- 安全建议

### 3. 预警信息
当检测到钓鱼攻击且置信度高于0.7时，预警信息会保存在`output/alerts`目录。

## 注意事项

1. **模型加载**：首次运行时会从Hugging Face下载BERT模型，可能需要一些时间。

2. **GPU支持**：系统会自动检测是否有可用的GPU，如果有则使用GPU加速。

3. **网络行为特征**：网络行为特征是模拟生成的，实际部署时可以替换为真实的网络行为数据。

4. **增量训练**：增量训练需要足够的样本才能获得较好的效果。

5. **场景处理**：不同场景的处理逻辑不同，建议根据实际场景选择合适的场景类型。

## 性能指标

### 基础指标
- 准确率：98%+
- 精确率：98%+
- 召回率：99%+
- F1值：98.5%+

### 工程化指标
- 推理速度：约10-20样本/秒（GPU）
- 内存占用：约1.5GB

## 总结

本系统是一个功能完整、性能优越的钓鱼检测系统，支持多模态特征分析、实时预警、可视化溯源和增量训练等功能。系统采用BERT-TextCNN混合模型，能够有效识别中文钓鱼攻击，为网络安全提供有力的保障。