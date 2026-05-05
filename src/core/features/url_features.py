from urllib.parse import urlparse
import logging

logger = logging.getLogger(__name__)

def extract_url_features(url):
    """
    提取URL特征
    
    参数:
        url: URL字符串
    
    返回:
        list: URL特征列表
    """
    try:
        if not url or not isinstance(url, str):
            return [0] * 16
        
        # 确保URL以http或https开头
        if not url.startswith(('http://', 'https://')):
            url = f'http://{url}'
        
        # 解析URL
        parsed_url = urlparse(url)
        domain = parsed_url.netloc
        path = parsed_url.path
        query = parsed_url.query
        
        # 提取特征
        features = []
        
        # 1. URL长度
        features.append(len(url))
        
        # 2. 是否使用HTTPS
        features.append(1 if url.startswith('https://') else 0)
        
        # 3. 域名长度
        features.append(len(domain))
        
        # 4. 域名层级
        features.append(domain.count('.'))
        
        # 5. 路径长度
        features.append(len(path))
        
        # 6. 查询参数长度
        features.append(len(query))
        
        # 7. 是否包含login关键词
        features.append(1 if 'login' in url.lower() else 0)
        
        # 8. 是否包含bank关键词
        features.append(1 if 'bank' in url.lower() else 0)
        
        # 9. 是否包含secure关键词
        features.append(1 if 'secure' in url.lower() else 0)
        
        # 10. 是否包含verify关键词
        features.append(1 if 'verify' in url.lower() else 0)
        
        # 11. 是否包含account关键词
        features.append(1 if 'account' in url.lower() else 0)
        
        # 12. 是否包含auth关键词
        features.append(1 if 'auth' in url.lower() else 0)
        
        # 13. 是否包含payment关键词
        features.append(1 if 'payment' in url.lower() else 0)
        
        # 14. 是否包含password关键词
        features.append(1 if 'password' in url.lower() else 0)
        
        # 15. 是否包含reset关键词
        features.append(1 if 'reset' in url.lower() else 0)
        
        # 16. 是否包含confirm关键词
        features.append(1 if 'confirm' in url.lower() else 0)
        
        logger.debug(f"URL特征提取完成: {features}")
        return features
    except Exception as e:
        logger.error(f"提取URL特征失败: {e}")
        return [0] * 16
