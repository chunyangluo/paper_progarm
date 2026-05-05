#!/bin/bash

# 部署脚本

echo "===== 多模态钓鱼智能识别与预警系统部署 ====="

# 检查Docker是否安装
if ! command -v docker &> /dev/null; then
    echo "错误: Docker未安装，请先安装Docker"
    exit 1
fi

# 检查Docker Compose是否安装
if ! command -v docker-compose &> /dev/null; then
    echo "错误: Docker Compose未安装，请先安装Docker Compose"
    exit 1
fi

# 构建Docker镜像
echo "正在构建Docker镜像..."
docker-compose build

# 启动容器
echo "正在启动容器..."
docker-compose up -d

# 等待服务启动
echo "等待服务启动..."
sleep 10

# 验证服务是否正常运行
echo "验证服务是否正常运行..."
response=$(curl -s http://localhost:8000/health)

if [[ $response == *"healthy"* ]]; then
    echo "✅ 服务启动成功！"
    echo "API接口地址: http://localhost:8000"
    echo "API文档地址: http://localhost:8000/docs"
else
    echo "❌ 服务启动失败，请检查日志"
    docker-compose logs api
    exit 1
fi

echo "===== 部署完成 ====="
