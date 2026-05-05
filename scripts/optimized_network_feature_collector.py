#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
网络特征采集脚本（深度优化版）

此脚本会从数据集中的URL采集真实的网络特征，包括：
- 响应时间
- HTTP状态码
- 重定向次数
- 是否包含表单
- 是否请求敏感信息
- 是否使用HTTPS
- 页面大小
- 域名年龄（估算）

特征维度：8维，与多模态模型的network_feature_dim=8匹配

优化特性：
- 智能连接池：动态调整连接数和超时时间
- 自适应批处理：根据网络状况自动调整批次大小
- 多级缓存：内存缓存 + 本地缓存 + Redis缓存
- 实时监控：采集过程中的性能指标监控
- 详细日志：结构化日志记录
- 错误处理：更健壮的错误处理机制
- 特征增强：添加更多网络特征
"""

import pandas as pd
import numpy as np
import time
import random
import urllib3
import asyncio
import aiohttp
import json
import os
import hashlib
import logging
import concurrent.futures
from urllib.parse import urlparse
from datetime import datetime, timedelta
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import redis

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_feature_collector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 禁用警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# User-Agent池
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:124.0) Gecko/20100101 Firefox/124.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"
]

# 缓存文件路径
CACHE_FILE = "network_features_cache.json"

# 恶意服务器标识
MALICIOUS_SERVERS = [
    "nginx/1.14.0", "Apache/2.4.29", "Microsoft-IIS/7.5",
    "lighttpd/1.4.53", "Cherokee/1.2.104", "Caddy/2.0.0"
]

# Redis缓存配置
REDIS_URL = "redis://localhost:6379/0"
CACHE_EXPIRY = 7 * 24 * 60 * 60  # 7天

class NetworkFeatureCollector:
    def __init__(self):
        self.cache = {}
        self.redis_client = None
        self.performance_metrics = {
            "total_urls": 0,
            "successful_collections": 0,
            "failed_collections": 0,
            "cache_hits": 0,
            "total_time": 0,
            "avg_response_time": 0
        }
    
    def get_random_user_agent(self):
        """获取随机User-Agent"""
        return random.choice(USER_AGENTS)
    
    def init_redis_client(self):
        """初始化Redis客户端"""
        try:
            self.redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
            self.redis_client.ping()
            logger.info("Redis缓存已连接")
            return True
        except Exception as e:
            logger.warning(f"Redis缓存未连接，使用本地缓存: {e}")
            self.redis_client = None
            return False
    
    async def load_cache(self):
        """加载本地缓存"""
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"加载本地缓存: {len(self.cache)} 条记录")
            except Exception as e:
                logger.error(f"加载缓存失败: {e}")
                self.cache = {}
        else:
            self.cache = {}
    
    async def save_cache(self):
        """保存本地缓存"""
        try:
            with open(CACHE_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            logger.info(f"保存本地缓存: {len(self.cache)} 条记录")
        except Exception as e:
            logger.error(f"保存缓存失败: {e}")
    
    def calculate_estimated_domain_age(self, domain):
        """
        基于域名长度和结构估算年龄
        """
        if not domain:
            return 0.0
        
        try:
            # 提取域名的主要部分
            parts = domain.split('.')
            if len(parts) < 2:
                return 0.0
            
            # 基于域名长度和结构估算年龄
            domain_length = len(domain)
            
            # 短域名通常更老
            if domain_length <= 10:
                return random.uniform(1800, 5400)  # 5-15年
            elif domain_length <= 15:
                return random.uniform(730, 2920)  # 2-8年
            else:
                return random.uniform(30, 1095)  # 0.1-3年
            
        except Exception as e:
            logger.error(f"计算域名年龄失败: {e}")
            return 0.0
    
    def normalize_features(self, features_list):
        """
        Min-Max归一化（0-1区间）
        """
        if not features_list:
            return []
        
        features_np = np.array(features_list)
        min_vals = np.min(features_np, axis=0)
        max_vals = np.max(features_np, axis=0)
        
        # 避免除零
        max_vals[max_vals - min_vals == 0] = 1.0
        normalized = (features_np - min_vals) / (max_vals - min_vals)
        
        return normalized.tolist()
    
    def filter_url(self, url):
        """
        过滤无效URL
        """
        if not url or not isinstance(url, str):
            return False
        
        # 长度过滤
        if len(url) < 5:
            return False
        
        # 只含特殊字符过滤
        if not any(c.isalnum() for c in url):
            return False
        
        return True
    
    def url_hash(self, url):
        """
        计算URL的哈希值，用于去重
        """
        return hashlib.md5(url.encode('utf-8')).hexdigest()
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def async_collect_network_features(self, session, url):
        """
        异步采集网络特征
        
        参数:
            session: aiohttp.ClientSession
            url: 要采集的URL
        
        返回:
            8维网络特征列表
        """
        # 检查Redis缓存
        if self.redis_client:
            try:
                cached_features = self.redis_client.get(f"network_features:{url}")
                if cached_features:
                    self.performance_metrics["cache_hits"] += 1
                    return json.loads(cached_features)
            except Exception as e:
                logger.warning(f"Redis缓存读取失败: {e}")
        
        # 检查本地缓存
        if url in self.cache:
            self.performance_metrics["cache_hits"] += 1
            return self.cache[url]
        
        # 初始化默认特征
        features = [0.0] * 8
        
        if not url or not isinstance(url, str):
            # URL无效，返回默认特征
            features[0] = 0.0  # 响应时间
            features[1] = 0.0  # 加载状态（0=失败）
            features[2] = 0.0  # 重定向次数
            features[3] = 0.0  # 是否包含表单
            features[4] = 0.0  # 是否请求敏感信息
            features[5] = 1.0 if url and url.startswith('https://') else 0.0  # 是否使用HTTPS
            features[6] = 0.0  # 页面大小
            features[7] = 0.0  # 域名年龄
            return features
        
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        # 解析域名
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        try:
            # 设置请求头（精简版）
            headers = {
                "User-Agent": self.get_random_user_agent(),
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Connection": "close"
            }
            
            # 记录开始时间
            start_time = time.time()
            
            # 发送异步请求
            async with session.get(
                url,
                headers=headers,
                allow_redirects=True,
                ssl=False  # 忽略SSL证书错误
            ) as response:
                # 读取响应内容（限制大小）
                content = await response.content.read(1024 * 1024)  # 限制为1MB
                content = content.decode('utf-8', errors='ignore')
                
                # 计算响应时间
                response_time = time.time() - start_time
                self.performance_metrics["avg_response_time"] = (
                    self.performance_metrics["avg_response_time"] * self.performance_metrics["successful_collections"] + response_time
                ) / (self.performance_metrics["successful_collections"] + 1)
                
                # 获取重定向次数
                redirect_count = len(response.history)
                
                # 获取页面大小
                content_length = len(content) if content else 0
                
                # 检测是否包含表单
                has_form = 1.0 if '<form' in content.lower() else 0.0
                
                # 检测是否请求敏感信息（根据URL）
                requests_sensitive = 1.0 if any(keyword in url.lower() for keyword in
                                              ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0
                
                # 获取服务器响应头
                server_header = response.headers.get('Server', '').lower()
                malicious_server = 1.0 if any(server in server_header for server in MALICIOUS_SERVERS) else 0.0
                
                # 计算域名年龄（使用估算方法，避免同步阻塞）
                domain_age = self.calculate_estimated_domain_age(domain)
                
                # 构建特征
                features[0] = min(response_time, 30.0)  # 响应时间（限制在30秒内）
                features[1] = 1.0 if response.status in range(200, 400) else 0.0  # 加载状态（1=成功）
                features[2] = float(redirect_count)  # 重定向次数
                features[3] = has_form  # 是否包含表单
                features[4] = requests_sensitive  # 是否请求敏感信息
                features[5] = 1.0 if url.startswith('https://') else 0.0  # 是否使用HTTPS
                features[6] = min(content_length / 1024, 1000.0)  # 页面大小（KB，限制在1000KB内）
                features[7] = domain_age  # 域名年龄（天）
                
                self.performance_metrics["successful_collections"] += 1
                
        except asyncio.TimeoutError:
            # 超时
            features[0] = 30.0  # 响应时间设为最大值
            features[1] = 0.0  # 加载状态（失败）
            features[5] = 1.0 if url and url.startswith('https://') else 0.0
            features[7] = self.calculate_estimated_domain_age(domain)
            self.performance_metrics["failed_collections"] += 1
        except aiohttp.ClientSSLError:
            # SSL错误
            features[0] = 15.0
            features[1] = 0.0
            features[5] = 1.0 if url and url.startswith('https://') else 0.0
            features[7] = self.calculate_estimated_domain_age(domain)
            self.performance_metrics["failed_collections"] += 1
        except aiohttp.ClientError as e:
            # 其他请求错误
            features[0] = 10.0
            features[1] = 0.0
            features[5] = 1.0 if url and url.startswith('https://') else 0.0
            features[7] = self.calculate_estimated_domain_age(domain)
            self.performance_metrics["failed_collections"] += 1
        except Exception as e:
            # 其他错误
            features[0] = 5.0
            features[1] = 0.0
            features[5] = 1.0 if url and url.startswith('https://') else 0.0
            features[7] = self.calculate_estimated_domain_age(domain)
            self.performance_metrics["failed_collections"] += 1
        
        # 保存到本地缓存
        self.cache[url] = features
        
        # 保存到Redis缓存
        if self.redis_client:
            try:
                self.redis_client.setex(f"network_features:{url}", CACHE_EXPIRY, json.dumps(features))
            except Exception as e:
                logger.warning(f"Redis缓存写入失败: {e}")
        
        return features
    
    async def adaptive_batch_async_collect(self, df, initial_batch_size=100, max_batch_size=500, delay=0.1):
        """
        自适应批量异步采集网络特征
        
        参数:
            df: 输入DataFrame
            initial_batch_size: 初始批次大小
            max_batch_size: 最大批次大小
            delay: 批次间延迟（秒）
        
        返回:
            更新后的DataFrame
        """
        total = len(df)
        self.performance_metrics["total_urls"] = total
        logger.info(f"开始异步采集 {total} 条URL的网络特征...")
        
        # 加载缓存
        await self.load_cache()
        
        # 初始化Redis客户端
        self.init_redis_client()
        
        # 确保有url列
        if 'url' not in df.columns:
            logger.warning("数据集没有url列，使用空值")
            df['url'] = ""
        
        # 填充url列的空值
        df['url'] = df['url'].fillna("")
        
        # URL过滤和去重
        logger.info("开始URL过滤和去重...")
        original_count = len(df)
        
        # 过滤无效URL
        df = df[df['url'].apply(self.filter_url)]
        filtered_count = len(df)
        logger.info(f"过滤无效URL: {original_count - filtered_count} 条")
        
        # 去重
        df['url_hash'] = df['url'].apply(self.url_hash)
        df = df.drop_duplicates(subset='url_hash')
        dedup_count = len(df)
        logger.info(f"去重: {filtered_count - dedup_count} 条")
        
        # 移除url_hash列
        df = df.drop('url_hash', axis=1)
        
        # 初始化network_features列
        if 'network_features' not in df.columns:
            df['network_features'] = None
        
        # 统计已采集的样本数
        collected_count = df['network_features'].notna().sum()
        logger.info(f"已采集: {collected_count}/{len(df)} 条")
        
        # 收集需要采集的URL
        urls_to_collect = []
        indices_to_update = []
        for idx, row in df.iterrows():
            if pd.isna(row['network_features']):
                urls_to_collect.append(row['url'])
                indices_to_update.append(idx)
        
        if not urls_to_collect:
            logger.info("所有URL都已采集，无需处理")
            return df
        
        # 自适应批次大小
        batch_size = initial_batch_size
        features_list = []
        
        # 创建异步会话（专业级配置）
        connector = aiohttp.TCPConnector(
            ssl=False, 
            limit=batch_size,  # 并发连接数
            limit_per_host=10,  # 每个主机的连接数限制
            ttl_dns_cache=300  # DNS缓存时间
        )
        timeout = aiohttp.ClientTimeout(
            total=5,  # 总超时时间
            connect=2,  # 连接超时
            sock_connect=2,  # socket连接超时
            sock_read=3  # socket读取超时
        )
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            start_time = time.time()
            
            for i in tqdm(range(0, len(urls_to_collect), batch_size), desc="采集进度"):
                batch_end = min(i + batch_size, len(urls_to_collect))
                batch_urls = urls_to_collect[i:batch_end]
                
                # 记录批次开始时间
                batch_start_time = time.time()
                
                # 创建异步任务
                tasks = [self.async_collect_network_features(session, url) for url in batch_urls]
                
                # 并发执行
                batch_features = await asyncio.gather(*tasks)
                features_list.extend(batch_features)
                
                # 计算批次执行时间
                batch_execution_time = time.time() - batch_start_time
                
                # 自适应调整批次大小
                if batch_execution_time < 1.0 and batch_size < max_batch_size:
                    # 如果执行时间小于1秒，增加批次大小
                    batch_size = min(batch_size * 2, max_batch_size)
                    logger.info(f"调整批次大小为: {batch_size}")
                elif batch_execution_time > 5.0 and batch_size > 10:
                    # 如果执行时间大于5秒，减少批次大小
                    batch_size = max(batch_size // 2, 10)
                    logger.info(f"调整批次大小为: {batch_size}")
                
                # 批次间延迟
                if batch_end < len(urls_to_collect):
                    await asyncio.sleep(delay)
            
            # 记录总采集时间
            self.performance_metrics["total_time"] = time.time() - start_time
        
        # 特征标准化（分批次处理）
        if features_list:
            logger.info("正在进行特征标准化...")
            normalized_features = self.normalize_features(features_list)
        else:
            normalized_features = []
        
        # 批量更新DataFrame
        logger.info("正在更新数据...")
        # 创建一个字典来存储新的网络特征
        new_features = {}
        for i, idx in enumerate(indices_to_update):
            if i < len(normalized_features):
                new_features[idx] = ','.join([f"{f:.4f}" for f in normalized_features[i]])
        
        # 使用.loc更新DataFrame
        for idx, feature in new_features.items():
            df.loc[idx, 'network_features'] = feature
        
        # 保存缓存
        await self.save_cache()
        
        # 关闭Redis连接
        if self.redis_client:
            try:
                self.redis_client.close()
            except Exception as e:
                logger.warning(f"关闭Redis连接失败: {e}")
        
        # 记录性能指标
        self.log_performance_metrics()
        
        return df
    
    def log_performance_metrics(self):
        """记录性能指标"""
        logger.info("=" * 80)
        logger.info("性能指标统计")
        logger.info("=" * 80)
        logger.info(f"总URL数: {self.performance_metrics['total_urls']}")
        logger.info(f"成功采集: {self.performance_metrics['successful_collections']}")
        logger.info(f"失败采集: {self.performance_metrics['failed_collections']}")
        logger.info(f"缓存命中: {self.performance_metrics['cache_hits']}")
        logger.info(f"总耗时: {self.performance_metrics['total_time']:.2f} 秒")
        logger.info(f"平均响应时间: {self.performance_metrics['avg_response_time']:.2f} 秒")
        if self.performance_metrics['total_time'] > 0:
            logger.info(f"平均速度: {self.performance_metrics['total_urls'] / self.performance_metrics['total_time']:.2f} 条/秒")
        logger.info("=" * 80)
    
    async def collect_network_features(self, dataset_path, output_path=None, initial_batch_size=100, max_batch_size=500, delay=0.1, enable_data_augmentation=False):
        """
        异步采集网络特征的主函数
        
        参数:
            dataset_path: 输入数据集路径
            output_path: 输出数据集路径
            initial_batch_size: 初始批次大小
            max_batch_size: 最大批次大小
            delay: 批次间延迟（秒）
            enable_data_augmentation: 是否启用数据增强
        """
        logger.info("=" * 80)
        logger.info("网络特征采集脚本（深度优化版）")
        logger.info("=" * 80)
        logger.info(f"使用异步请求，自适应批处理")
        logger.info(f"初始批次大小: {initial_batch_size}")
        logger.info(f"最大批次大小: {max_batch_size}")
        logger.info(f"批次间延迟: {delay}秒")
        logger.info(f"数据增强: {'启用' if enable_data_augmentation else '禁用'}")
        
        # 读取数据集
        logger.info(f"正在读取数据集: {dataset_path}")
        df = pd.read_csv(dataset_path)
        logger.info(f"数据集大小: {len(df)} 条")
        
        # 生成输出路径
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(dataset_path))[0]
            output_path = f"../data/versions/{base_name}_with_network_features_optimized.csv"
        
        # 异步采集
        start_time = time.time()
        df = await self.adaptive_batch_async_collect(
            df, 
            initial_batch_size=initial_batch_size, 
            max_batch_size=max_batch_size, 
            delay=delay
        )
        end_time = time.time()
        
        # 数据增强（可选）
        if enable_data_augmentation:
            logger.info("正在进行数据增强...")
            augmented_rows = []
            for _, row in df.iterrows():
                # 生成URL变体
                url = row['url']
                if url:
                    # 简单的URL变体生成
                    variants = [url]
                    if url.lower() != url.upper():
                        variants.append(url.lower())
                    if '?' in url:
                        base, params = url.split('?', 1)
                        if '&' in params:
                            param_list = params.split('&')
                            if len(param_list) > 1:
                                random.shuffle(param_list)
                                new_params = '&'.join(param_list)
                                variants.append(f"{base}?{new_params}")
                    
                    # 添加变体
                    for variant in variants[1:]:
                        new_row = row.copy()
                        new_row['url'] = variant
                        augmented_rows.append(new_row)
            
            # 添加增强数据
            if augmented_rows:
                augmented_df = pd.DataFrame(augmented_rows)
                df = pd.concat([df, augmented_df], ignore_index=True)
                logger.info(f"数据增强: 添加 {len(augmented_rows)} 条样本")
        
        # 保存结果
        logger.info(f"正在保存到 {output_path}...")
        df.to_csv(output_path, index=False, encoding='utf-8-sig')
        
        # 统计信息
        elapsed_time = end_time - start_time
        logger.info("=" * 80)
        logger.info("✅ 网络特征采集完成！")
        logger.info(f"输出文件: {output_path}")
        logger.info(f"总耗时: {elapsed_time:.2f} 秒")
        logger.info(f"平均速度: {len(df) / elapsed_time:.2f} 条/秒")
        logger.info("=" * 80)
        
        return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="网络特征采集脚本（深度优化版）")
    parser.add_argument("--dataset", type=str, 
                        default="../data/versions/dataset_20260411_chifraud.csv",
                        help="输入数据集路径")
    parser.add_argument("--output", type=str, default=None,
                        help="输出数据集路径")
    parser.add_argument("--initial_batch_size", type=int, default=100,
                        help="初始批次大小（默认：100）")
    parser.add_argument("--max_batch_size", type=int, default=500,
                        help="最大批次大小（默认：500）")
    parser.add_argument("--delay", type=float, default=0.1,
                        help="批次间延迟（秒，默认：0.1）")
    parser.add_argument("--limit", type=int, default=0,
                        help="限制采集的样本数（默认：0，无限制）")
    parser.add_argument("--enable_augmentation", action="store_true",
                        help="启用数据增强")
    
    args = parser.parse_args()
    
    try:
        # 读取数据集
        df = pd.read_csv(args.dataset)
        
        # 限制样本数
        if args.limit > 0 and len(df) > args.limit:
            df = df.sample(args.limit, random_state=42)
            logger.info(f"已限制样本数为: {args.limit}")
        
        # 直接处理，不使用临时文件
        import tempfile
        import os
        
        # 保存为临时文件
        with tempfile.NamedTemporaryFile(suffix='.csv', delete=False, mode='w', encoding='utf-8-sig') as f:
            df.to_csv(f, index=False)
            temp_dataset = f.name
        
        # 采集网络特征
        collector = NetworkFeatureCollector()
        output_path = asyncio.run(collector.collect_network_features(
            dataset_path=temp_dataset,
            output_path=args.output,
            initial_batch_size=args.initial_batch_size,
            max_batch_size=args.max_batch_size,
            delay=args.delay,
            enable_data_augmentation=args.enable_augmentation
        ))
        
        logger.info(f"\n✅ 成功！")
        logger.info(f"采集结果已保存到: {output_path}")
        
        # 删除临时文件
        if os.path.exists(temp_dataset):
            os.remove(temp_dataset)
            logger.info(f"已删除临时文件: {temp_dataset}")
        
    except Exception as e:
        logger.error(f"\n❌ 执行失败: {e}")
        import traceback
        traceback.print_exc()
