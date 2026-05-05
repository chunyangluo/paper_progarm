# 数据集采集与处理说明

本文档详细说明中文网络钓鱼多模态识别研究项目的数据集采集与处理全过程。

## 1. 数据源介绍

### 1.1 权威数据源

| 数据源名称 | 类型 | 规模 | 来源 | 描述 |
|---------|------|------|------|------|
| PhishTank | 钓鱼URL | 10,000条 | https://data.phishtank.com/data/online-valid.csv | 官方钓鱼URL数据库，实时更新 |
| OpenPhish | 钓鱼URL | 300条 | https://openphish.com/feed.txt | 15分钟更新的实时钓鱼URL feed |
| PhiUSIIL | 钓鱼URL | 5,000条 | https://github.com/PhiUSIIL/PhishingDataset | 学术机构维护的钓鱼URL数据集 |
| Majestic | 正常URL | 10,000条 | https://downloads.majestic.com/majestic_million.csv | 全球排名前100万的正常网站 |
| CHIFRAUD | 中文欺诈文本 | 59,106条欺诈文本 + 352,328条正常文本 | https://github.com/xuemingxxx/ChiFraud | COLING 2025发布的中文欺诈文本基准集 |

### 1.2 数据类型

- **钓鱼URL**：来自PhishTank、OpenPhish、PhiUSIIL
- **正常URL**：来自Majestic Million
- **中文欺诈文本**：来自CHIFRAUD

## 2. 采集与处理流程

### 2.1 环境配置

- **Python 3.7+**
- **依赖库**：requests, pandas, numpy, scikit-learn
- **代理配置**：Windows环境下需设置代理（端口52842）以访问境外数据源

### 2.2 采集步骤

1. **初始化DataCollector**：创建数据目录结构，设置请求头和代理
2. **采集钓鱼URL**：
   - 从PhishTank采集10,000条钓鱼URL
   - 从OpenPhish采集300条钓鱼URL
   - 从PhiUSIIL采集5,000条钓鱼URL
3. **采集正常URL**：从Majestic采集10,000条正常URL
4. **处理CHIFRAUD数据**：从CHIFRAUD数据集提取欺诈和正常文本
5. **生成版本数据集**：合并所有采集的数据，进行去重、标签校验和类别均衡处理

### 2.3 处理步骤

1. **数据清洗**：
   - 去重：基于text+url字段
   - 标签校验：确保label字段为0或1
   - 类别均衡：确保钓鱼样本和正常样本比例为1:1
2. **数据格式化**：
   - 统一字段格式：确保所有数据包含text、url、label、source字段
   - 处理缺失值：对于缺失的字段，设置为NaN
3. **版本管理**：
   - 生成带时间戳的版本数据集
   - 保存到data/versions目录

## 3. 脚本说明

### 3.1 核心脚本

- **data_collection.py**：负责从权威数据源采集数据
- **update_dataset.py**：执行完整的数据集采集和版本生成流程
- **verify_dataset.py**：验证生成的数据集

### 3.2 运行方法

1. **采集全量数据**：
   ```bash
   python update_dataset.py
   ```

2. **验证数据集**：
   ```bash
   python verify_dataset.py
   ```

## 4. 数据集结构

### 4.1 版本数据集字段

| 字段名 | 类型 | 描述 |
|-------|------|------|
| text | str | 文本内容（URL或欺诈文本） |
| url | str | URL地址 |
| label | int | 标签（0=正常，1=钓鱼） |
| source | str | 数据来源（phishTank/openphish/phiUSIIL/majestic/chifraud） |
| phish_id | str | PhishTank的钓鱼ID（仅PhishTank数据） |
| submission_time | str | 提交时间（仅PhishTank数据） |
| verification_time | str | 验证时间（仅PhishTank数据） |
| online | str | 是否在线（仅PhishTank数据） |
| target | str | 目标（仅PhishTank数据） |
| scenario | str | 场景（仅OpenPhish数据） |
| timestamp | str | 时间戳 |

### 4.2 数据分布

- **总样本数**：23,004条
- **钓鱼样本**：11,502条
- **正常样本**：11,502条
- **数据来源分布**：
  - phiUSIIL：500条
  - phishTank：302条
  - openphish：285条
  - chifraud：21,917条（包含在总样本数中）

## 5. 鲁棒性设计

### 5.1 网络异常处理

- **重试机制**：所有网络请求都有3次重试+指数退避
- **代理配置**：Windows环境下显式指定代理，解决requests库无法自动继承环境变量的问题
- **备用数据源**：PhiUSIIL数据源设置了多个备用链接，确保采集成功率

### 5.2 数据异常处理

- **灵活解析**：支持不同格式的CSV文件，自动检测URL列
- **错误处理**：跳过错误行，确保脚本不会因数据格式问题而崩溃
- **空值处理**：对于缺失的字段，设置为NaN，确保数据结构一致性

## 6. 数据集质量保证

### 6.1 数据来源

- **权威数据源**：所有数据均来自公开可溯源的权威数据源
- **实时更新**：PhishTank和OpenPhish数据实时更新，确保数据的时效性
- **学术认可**：CHIFRAUD是COLING 2025发布的基准集，具有学术认可

### 6.2 数据处理

- **去重**：确保数据集中没有重复样本
- **类别均衡**：确保钓鱼样本和正常样本比例为1:1，避免模型偏差
- **数据清洗**：处理缺失值和异常值，确保数据质量

## 7. 数据集使用建议

### 7.1 模型训练

- **多模态特征**：可提取URL特征、网络行为特征等多模态特征
- **交叉验证**：使用k折交叉验证评估模型性能
- **模型选择**：建议使用TextCNN、BERT等深度学习模型

### 7.2 论文实验

- **数据集版本**：使用带时间戳的版本数据集，确保实验可重现
- **对比实验**：与其他基准数据集进行对比，验证模型性能
- **消融实验**：分析不同特征和数据源对模型性能的影响

## 8. 注意事项

- **网络连接**：采集境外数据源需要稳定的网络连接和代理配置
- **数据更新**：建议定期更新数据集，以保持数据的时效性
- **数据使用**：本数据集仅用于学术研究，不得用于商业用途
- **引用**：使用CHIFRAUD数据集时，请引用相关论文

## 9. 参考资料

- PhishTank: https://www.phishtank.com/
- OpenPhish: https://www.openphish.com/
- PhiUSIIL: https://github.com/PhiUSIIL/PhishingDataset
- Majestic Million: https://majestic.com/reports/majestic-million
- CHIFRAUD: https://github.com/xuemingxxx/ChiFraud
