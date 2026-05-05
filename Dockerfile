# 使用Python 3.12作为基础镜像
FROM python:3.12-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements.txt文件
COPY api/requirements.txt /app/requirements.txt

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . /app

# 创建必要的目录
RUN mkdir -p /app/data/synthetic \
    && mkdir -p /app/data/real \
    && mkdir -p /app/data/multimodal \
    && mkdir -p /app/data/incremental \
    && mkdir -p /app/models \
    && mkdir -p /app/output

# 下载BERT模型
RUN python -c "from transformers import BertTokenizer, BertModel; \
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese'); \
    model = BertModel.from_pretrained('bert-base-chinese')"

# 暴露端口
EXPOSE 8000

# 启动API服务
CMD ["python", "api/main.py"]
