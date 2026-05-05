import asyncio
import aiohttp
import time
from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

async def extract_network_features(url, timeout=5):
    """
    提取网络行为特征
    
    参数:
        url: URL字符串
        timeout: 请求超时时间（秒）
    
    返回:
        list: 网络行为特征列表
    """
    try:
        if not url or not isinstance(url, str):
            return [0.0] * 8
        
        # 确保URL以http或https开头
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        # 解析域名
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        
        # 初始化特征
        features = [0.0] * 8
        
        # 发送请求
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, timeout=timeout, allow_redirects=True, ssl=False) as response:
                    # 计算响应时间
                    response_time = time.time() - start_time
                    
                    # 读取响应内容（限制大小）
                    content = await response.content.read(1024 * 1024)  # 限制为1MB
                    content = content.decode('utf-8', errors='ignore')
                    
                    # 获取重定向次数
                    redirect_count = len(response.history)
                    
                    # 获取页面大小
                    content_length = len(content) if content else 0
                    
                    # 检测是否包含表单
                    has_form = 1.0 if '<form' in content.lower() else 0.0
                    
                    # 检测是否请求敏感信息（根据URL）
                    requests_sensitive = 1.0 if any(keyword in url.lower() for keyword in
                                                  ['login', 'password', 'account', 'bank', 'credit', 'verify', 'confirm']) else 0.0
                    
                    # 构建特征
                    features[0] = min(response_time, 30.0)  # 响应时间（限制在30秒内）
                    features[1] = 1.0 if response.status in range(200, 400) else 0.0  # 加载状态（1=成功）
                    features[2] = float(redirect_count)  # 重定向次数
                    features[3] = has_form  # 是否包含表单
                    features[4] = requests_sensitive  # 是否请求敏感信息
                    features[5] = 1.0 if url.startswith('https://') else 0.0  # 是否使用HTTPS
                    features[6] = min(content_length / 1024, 1000.0)  # 页面大小（KB，限制在1000KB内）
                    features[7] = calculate_estimated_domain_age(domain)  # 域名年龄（天）
                    
            except asyncio.TimeoutError:
                # 超时
                features[0] = 30.0  # 响应时间设为最大值
                features[1] = 0.0  # 加载状态（失败）
                features[5] = 1.0 if url.startswith('https://') else 0.0
                features[7] = calculate_estimated_domain_age(domain)
            except aiohttp.ClientError:
                # 其他请求错误
                features[0] = 10.0
                features[1] = 0.0
                features[5] = 1.0 if url.startswith('https://') else 0.0
                features[7] = calculate_estimated_domain_age(domain)
    except Exception as e:
        logger.error(f"提取网络行为特征失败: {e}")
        features = [0.0] * 8
    
    logger.debug(f"网络行为特征提取完成: {features}")
    return features

def calculate_estimated_domain_age(domain):
    """
    基于域名长度和结构估算年龄
    
    参数:
        domain: 域名
    
    返回:
        float: 估算的域名年龄（天）
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
            return 1800.0  # 约5年
        elif domain_length <= 15:
            return 730.0  # 约2年
        else:
            return 365.0  # 约1年
    except Exception as e:
        logger.error(f"计算域名年龄失败: {e}")
        return 0.0
