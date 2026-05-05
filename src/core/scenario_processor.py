import re
import html
from urllib.parse import urlparse

class ScenarioProcessor:
    """场景处理器，处理不同场景的钓鱼检测"""
    
    def __init__(self):
        # 短信相关的关键词和模式
        self.sms_keywords = [
            '验证码', '密码', '账户', '安全', '风险', '解冻', '封禁', 
            '充值', '提现', '转账', '汇款', '支付', '领奖', '中奖',
            '银行', '支付宝', '微信', '京东', '淘宝', '顺丰', '快递'
        ]
        
        # 邮件相关的关键词和模式
        self.email_keywords = [
            '重置密码', '账户验证', '安全警报', '登录异常', '账单', 
            '发票', '付款', '订单', '配送', '退款', '优惠', '促销'
        ]
        
        # 链接相关的可疑模式
        self.suspicious_url_patterns = [
            r'\b(verify|secure|login|account|auth|payment)\b',
            r'\b(bank|financial|update|confirm|reset|password)\b',
            r'\b(verification|validate|authentication|signin|sign-up)\b',
            r'\b(phish|钓鱼|诈骗|欺诈)\b'
        ]
    
    def process(self, scenario, text, url):
        """根据场景类型处理文本和URL"""
        # 场景处理
        if scenario == "sms":
            # 短信场景处理
            pass
        elif scenario == "email":
            # 邮件场景处理
            pass
        elif scenario == "link":
            # 链接场景处理
            pass
        else:
            # 通用场景处理
            pass
        
        # 返回处理后的文本和URL
        return text, url
    
    def process_sms(self, text, url):
        """处理短信场景"""
        # 短信通常更短，需要特殊处理
        features = {
            'sms_length': len(text),
            'has_suspicious_keywords': self._contains_keywords(text, self.sms_keywords),
            'has_url': 1 if url else 0,
            'url_length': len(url) if url else 0,
            'has_suspicious_url': self._has_suspicious_url(url)
        }
        return features
    
    def process_email(self, text, url):
        """处理邮件场景"""
        # 邮件可能包含HTML内容和附件
        features = {
            'email_length': len(text),
            'has_html': 1 if '<html>' in text.lower() or '</html>' in text.lower() else 0,
            'has_attachments': 1 if 'attachment' in text.lower() else 0,
            'has_suspicious_keywords': self._contains_keywords(text, self.email_keywords),
            'has_url': 1 if url else 0,
            'url_length': len(url) if url else 0,
            'has_suspicious_url': self._has_suspicious_url(url)
        }
        return features
    
    def process_link(self, text, url):
        """处理链接场景"""
        # 链接场景需要更多的URL特征分析
        features = {
            'url_length': len(url) if url else 0,
            'has_suspicious_url': self._has_suspicious_url(url),
            'url_domain': self._extract_domain(url),
            'has_ip_address': self._has_ip_address(url),
            'subdomain_count': self._count_subdomains(url),
            'path_depth': self._calculate_path_depth(url),
            'param_count': self._count_parameters(url)
        }
        return features
    
    def _contains_keywords(self, text, keywords):
        """检查文本是否包含关键词"""
        text_lower = text.lower()
        return 1 if any(keyword in text_lower for keyword in keywords) else 0
    
    def _has_suspicious_url(self, url):
        """检查URL是否包含可疑模式"""
        if not url:
            return 0
        url_lower = url.lower()
        for pattern in self.suspicious_url_patterns:
            if re.search(pattern, url_lower):
                return 1
        return 0
    
    def _extract_domain(self, url):
        """提取域名"""
        if not url:
            return ''
        try:
            parsed = urlparse(url)
            return parsed.netloc
        except:
            return ''
    
    def _has_ip_address(self, url):
        """检查URL是否包含IP地址"""
        if not url:
            return 0
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        return 1 if re.search(ip_pattern, url) else 0
    
    def _count_subdomains(self, url):
        """计算子域名数量"""
        if not url:
            return 0
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            if domain:
                return len(domain.split('.')) - 2
            return 0
        except:
            return 0
    
    def _calculate_path_depth(self, url):
        """计算路径深度"""
        if not url:
            return 0
        try:
            parsed = urlparse(url)
            path = parsed.path
            if path:
                return len([p for p in path.split('/') if p])
            return 0
        except:
            return 0
    
    def _count_parameters(self, url):
        """计算参数数量"""
        if not url:
            return 0
        try:
            parsed = urlparse(url)
            query = parsed.query
            if query:
                return len(query.split('&'))
            return 0
        except:
            return 0
