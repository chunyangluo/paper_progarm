# 多模态钓鱼智能识别与预警系统用户指南

## 1. 系统概述

多模态钓鱼智能识别与预警系统是一个基于深度学习的智能系统，旨在实时检测和预警钓鱼攻击，支持短信、邮件、链接等多场景的识别。系统采用BERT-TextCNN混合模型，结合多模态特征融合技术，实现对钓鱼攻击的精准识别和溯源分析。

### 1.1 核心功能

- **实时检测**：支持实时检测钓鱼文本，响应速度快
- **多场景支持**：支持短信、邮件、链接等多种场景的钓鱼识别
- **可视化溯源**：提供钓鱼攻击的可视化分析和溯源功能
- **增量训练**：支持模型的持续学习和更新
- **工程化部署**：支持Docker容器化部署，便于集成到现有系统

### 1.2 技术栈

- **后端**：Python 3.12, FastAPI, Redis, Celery
- **深度学习**：PyTorch, Transformers
- **数据库**：MongoDB, PostgreSQL
- **容器化**：Docker, Kubernetes
- **前端**：React, D3.js
- **监控**：Prometheus, Grafana

## 2. 安装指南

### 2.1 环境要求

- Docker 20.10+ 
- Docker Compose 1.29+
- 至少4GB内存
- 至少10GB磁盘空间

### 2.2 部署步骤

1. **克隆项目**
   ```bash
   git clone https://github.com/your-repo/phishing-detection-system.git
   cd phishing-detection-system
   ```

2. **构建和启动容器**
   ```bash
   # 赋予部署脚本执行权限
   chmod +x deploy.sh
   
   # 运行部署脚本
   ./deploy.sh
   ```

3. **验证服务**
   部署脚本会自动验证服务是否正常运行。如果服务启动成功，会显示以下信息：
   ```
   ✅ 服务启动成功！
   API接口地址: http://localhost:8000
   API文档地址: http://localhost:8000/docs
   ```

### 2.3 手动部署

如果不想使用部署脚本，可以手动执行以下步骤：

1. **构建Docker镜像**
   ```bash
   docker-compose build
   ```

2. **启动容器**
   ```bash
   docker-compose up -d
   ```

3. **检查服务状态**
   ```bash
   docker-compose ps
   ```

## 3. API接口文档

### 3.1 健康检查

- **URL**: `/health`
- **方法**: GET
- **描述**: 检查服务是否正常运行
- **响应**: `{"status": "healthy"}`

### 3.2 实时检测

- **URL**: `/detect`
- **方法**: POST
- **描述**: 实时检测钓鱼文本
- **请求体**:
  ```json
  {
    "text": "请点击链接验证您的账户：https://alipay-veri.com",
    "url": "https://alipay-veri.com",
    "scenario": "general"
  }
  ```
- **响应**:
  ```json
  {
    "model": "multimodal",
    "prediction": "钓鱼",
    "confidence": 0.9999,
    "details": {
      "url_features": {...},
      "network_features": {...}
    },
    "processing_time": 0.123
  }
  ```

### 3.3 批量检测

- **URL**: `/batch_detect`
- **方法**: POST
- **描述**: 批量检测钓鱼文本
- **请求体**:
  ```json
  {
    "items": [
      {
        "text": "请点击链接验证您的账户：https://alipay-veri.com",
        "url": "https://alipay-veri.com"
      },
      {
        "text": "您的支付宝余额宝收益已到账，可前往支付宝APP查看详情",
        "url": "https://www.alipay.com"
      }
    ]
  }
  ```
- **响应**:
  ```json
  {
    "results": [
      {
        "model": "multimodal",
        "prediction": "钓鱼",
        "confidence": 0.9999,
        "details": {...},
        "processing_time": 0.0615
      },
      {
        "model": "multimodal",
        "prediction": "正常",
        "confidence": 0.9999,
        "details": {...},
        "processing_time": 0.0615
      }
    ],
    "total_processing_time": 0.123
  }
  ```

### 3.4 场景特定检测

- **URL**: `/detect_sms`
- **方法**: POST
- **描述**: 检测钓鱼短信
- **请求体**: 与`/detect`相同

- **URL**: `/detect_email`
- **方法**: POST
- **描述**: 检测钓鱼邮件
- **请求体**: 与`/detect`相同

- **URL**: `/detect_link`
- **方法**: POST
- **描述**: 检测钓鱼链接
- **请求体**: 与`/detect`相同

### 3.5 钓鱼攻击溯源

- **URL**: `/trace`
- **方法**: POST
- **描述**: 钓鱼攻击溯源分析
- **请求体**:
  ```json
  {
    "text": "请点击链接验证您的账户：https://alipay-veri.com",
    "url": "https://alipay-veri.com",
    "scenario": "sms"
  }
  ```
- **响应**:
  ```json
  {
    "report": {
      "sample_info": {...},
      "detection_result": {...},
      "features": {...},
      "visualizations": {
        "attack_path": "base64编码的图片",
        "feature_importance": "base64编码的图片"
      },
      "recommendations": [...]
    }
  }
  ```

### 3.6 趋势分析

- **URL**: `/analyze_trends`
- **方法**: POST
- **描述**: 分析钓鱼攻击趋势
- **请求体**:
  ```json
  [
    {"date": "2024-01-01", "text": "...", "url": "..."},
    {"date": "2024-01-02", "text": "...", "url": "..."}
  ]
  ```
- **响应**:
  ```json
  {
    "trend_image": "base64编码的图片"
  }
  ```

### 3.7 增量训练

- **URL**: `/incremental_train`
- **方法**: POST
- **描述**: 增量训练模型
- **请求体**:
  ```json
  {
    "samples": [
      {"text": "...", "url": "...", "label": 1, "scenario": "sms"},
      {"text": "...", "url": "...", "label": 0, "scenario": "email"}
    ],
    "model_type": "multimodal",
    "epochs": 10
  }
  ```
- **响应**:
  ```json
  {
    "model_type": "multimodal",
    "training_samples": 80,
    "test_samples": 20,
    "evaluation_results": {
      "accuracy": 0.99,
      "precision": 0.99,
      "recall": 0.99,
      "f1": 0.99
    }
  }
  ```

### 3.8 数据收集

- **URL**: `/collect_data`
- **方法**: POST
- **描述**: 收集新的样本数据
- **请求体**:
  ```json
  [
    {"text": "...", "url": "...", "label": 1, "scenario": "sms"},
    {"text": "...", "url": "...", "label": 0, "scenario": "email"}
  ]
  ```
- **响应**:
  ```json
  {
    "collected_samples": 2,
    "total_samples": 100
  }
  ```

## 4. 使用示例

### 4.1 实时检测示例

#### Python示例
```python
import requests

url = "http://localhost:8000/detect"

# 钓鱼文本示例
data = {
    "text": "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户",
    "url": "https://alipay-veri.com",
    "scenario": "sms"
}

response = requests.post(url, json=data)
print(response.json())

# 正常文本示例
data = {
    "text": "【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情",
    "url": "https://www.alipay.com",
    "scenario": "sms"
}

response = requests.post(url, json=data)
print(response.json())
```

#### curl示例
```bash
# 检测钓鱼文本
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户", "url": "https://alipay-veri.com", "scenario": "sms"}'

# 检测正常文本
curl -X POST http://localhost:8000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情", "url": "https://www.alipay.com", "scenario": "sms"}'
```

### 4.2 批量检测示例

```python
import requests

url = "http://localhost:8000/batch_detect"

data = {
    "items": [
        {
            "text": "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户",
            "url": "https://alipay-veri.com"
        },
        {
            "text": "【微信团队】你的微信账号异地登录，点击 https://wx-safe.cn 验证手机号，否则24小时封禁",
            "url": "https://wx-safe.cn"
        },
        {
            "text": "【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情",
            "url": "https://www.alipay.com"
        },
        {
            "text": "【微信团队】你的微信支付分已更新，打开微信APP-我-服务可查询",
            "url": "https://weixin.qq.com"
        }
    ]
}

response = requests.post(url, json=data)
print(response.json())
```

### 4.3 钓鱼攻击溯源示例

```python
import requests
import base64
from PIL import Image
from io import BytesIO

url = "http://localhost:8000/trace"

data = {
    "text": "【支付宝】你的账户存在安全风险，点击 https://alipay-veri.com 解冻，逾期将注销账户",
    "url": "https://alipay-veri.com",
    "scenario": "sms"
}

response = requests.post(url, json=data)
result = response.json()

# 保存攻击路径图
attack_path_image = result['report']['visualizations']['attack_path']
image_data = base64.b64decode(attack_path_image)
image = Image.open(BytesIO(image_data))
image.save('attack_path.png')

# 保存特征重要性图
feature_importance_image = result['report']['visualizations']['feature_importance']
image_data = base64.b64decode(feature_importance_image)
image = Image.open(BytesIO(image_data))
image.save('feature_importance.png')

print("溯源报告生成成功！")
print(f"预测结果: {result['report']['detection_result']['prediction']}")
print(f"置信度: {result['report']['detection_result']['confidence']}")
print(f"安全建议: {result['report']['recommendations']}")
```

## 5. 系统管理

### 5.1 查看日志

```bash
docker-compose logs api
```

### 5.2 停止服务

```bash
docker-compose down
```

### 5.3 重启服务

```bash
docker-compose restart
```

### 5.4 更新模型

1. **收集新样本**：使用`/collect_data`接口收集新样本
2. **增量训练**：使用`/incremental_train`接口训练模型
3. **验证模型**：使用`/detect`接口验证模型性能

## 6. 故障排除

### 6.1 服务启动失败

- 检查Docker是否正常运行
- 检查端口8000是否被占用
- 查看服务日志：`docker-compose logs api`

### 6.2 模型加载失败

- 检查BERT模型是否下载成功
- 检查模型文件是否存在于`models`目录
- 查看服务日志获取详细错误信息

### 6.3 检测结果不准确

- 检查输入数据格式是否正确
- 考虑使用`/incremental_train`接口更新模型
- 检查模型是否为最新版本

## 7. 性能优化

### 7.1 提高检测速度

- 使用批处理接口`/batch_detect`处理大量请求
- 启用Redis缓存以提高重复查询的响应速度
- 考虑使用GPU加速模型推理

### 7.2 提高检测准确率

- 定期使用新样本更新模型
- 调整模型参数和训练策略
- 增加更多的特征类型和数据源

## 8. 安全考虑

### 8.1 数据安全

- 对敏感数据进行加密存储和传输
- 实现基于角色的访问控制
- 定期清理过期数据

### 8.2 模型安全

- 保护模型不被恶意攻击和窃取
- 定期更新模型以应对新的钓鱼攻击手法
- 监控模型性能和异常行为

## 9. 未来扩展

- **多语言支持**：扩展系统支持多语言钓鱼检测
- **多渠道集成**：集成更多渠道的钓鱼检测，如社交媒体、即时通讯工具等
- **威胁情报**：集成威胁情报，提高检测准确率
- **自动化响应**：实现自动化的钓鱼攻击响应机制
- **AI对抗**：增强模型对对抗样本的鲁棒性

## 10. 联系与支持

- **项目地址**：https://github.com/your-repo/phishing-detection-system
- **文档地址**：http://localhost:8000/docs
- **技术支持**：support@phishing-detection-system.com

---

**© 2024 多模态钓鱼智能识别与预警系统**
