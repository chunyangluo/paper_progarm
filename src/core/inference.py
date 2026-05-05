import torch
import pandas as pd
import numpy as np
import time
import requests
import hashlib
import ipaddress
from urllib.parse import urlparse
from transformers import BertTokenizer, BertModel
from core.model_training import TextCNN
from core.models.model_definitions import instantiate_bert_textcnn_for_state_dict


# 全局配置
class Config:
    """全局配置类"""
    # 网络请求配置
    NETWORK_TIMEOUT = 3  # 超时时间（秒）
    MAX_RETRIES = 3       # 最大重试次数
    RETRY_DELAY = 1       # 重试延迟（秒）
    
    # 缓存配置
    CACHE_TTL = 3600      # 缓存过期时间（秒）
    HASH_ALGORITHM = hashlib.md5  # 缓存键哈希算法
    
    # 模型配置
    DEFAULT_MODEL_DIR = "../models"
    DEFAULT_BERT_MODEL = "bert-base-chinese"
    ENABLE_ONLINE_NETWORK_PROBE = False
    
    # API路径配置
    API_CONFIG = {
        'bert_base_chinese': 'bert-base-chinese',
        'bert_base_uncased': 'bert-base-uncased'
    }


def generate_url_features(url):
    """生成URL特征（与训练时一致，16维）"""
    if url is None:
        return [0]*16
    if not isinstance(url, str):
        url = str(url)
    if not url:
        return [0]*16
    
    features = []
    features.append(len(url))
    features.append(1 if "https" in url else 0)
    features.append(url.count("."))
    features.append(1 if "login" in url.lower() else 0)
    features.append(1 if "bank" in url.lower() else 0)
    features.append(1 if "secure" in url.lower() else 0)
    features.append(1 if "verify" in url.lower() else 0)
    features.append(1 if "account" in url.lower() else 0)
    features.append(1 if "auth" in url.lower() else 0)
    features.append(1 if "payment" in url.lower() else 0)
    features.append(1 if "password" in url.lower() else 0)
    features.append(1 if "reset" in url.lower() else 0)
    features.append(1 if "confirm" in url.lower() else 0)
    features.append(1 if "update" in url.lower() else 0)
    features.append(1 if "financial" in url.lower() else 0)
    features.append(1 if "phish" in url.lower() else 0)
    
    return features[:16]


def generate_network_features(url):
    """生成网络行为特征（动态计算，8维）"""
    if url is None:
        return [3.0, 404, 0, 0, 0, 0, 0, 0]
    if not isinstance(url, str):
        url = str(url)
    if not url:
        return [3.0, 404, 0, 0, 0, 0, 0, 0]
    if not Config.ENABLE_ONLINE_NETWORK_PROBE or not is_safe_public_url(url):
        return [
            0.0,
            404,
            0,
            0,
            0,
            1 if url.startswith('https://') else 0,
            0,
            1 if 'login' in url.lower() else 0,
        ]
    
    features = []
    
    for attempt in range(Config.MAX_RETRIES):
        try:
            # 响应时间
            start_time = time.time()
            response = requests.get(url, timeout=Config.NETWORK_TIMEOUT)
            response_time = time.time() - start_time
            features.append(response_time)
            
            # 状态码
            features.append(response.status_code)
            
            # 重定向次数
            features.append(len(response.history))
            
            # 是否包含表单
            has_form = 1 if '<form' in response.text.lower() else 0
            features.append(has_form)
            
            # 是否请求敏感信息
            sensitive_patterns = ['password', 'credit card', 'bank account', 'ssn', '身份证', '银行卡']
            has_sensitive = 1 if any(pattern in response.text.lower() for pattern in sensitive_patterns) else 0
            features.append(has_sensitive)
            
            # 是否使用HTTPS
            features.append(1 if url.startswith('https://') else 0)
            
            # 是否包含验证码
            has_captcha = 1 if 'captcha' in response.text.lower() else 0
            features.append(has_captcha)
            
            # 是否需要登录
            has_login = 1 if 'login' in response.text.lower() else 0
            features.append(has_login)
            
            break  # 成功获取，跳出重试循环
            
        except Exception as e:
            print(f"网络请求失败 (尝试 {attempt+1}/{Config.MAX_RETRIES}): {e}")
            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)
            else:
                # 所有重试都失败，使用默认值
                features = [3.0, 404, 0, 0, 0, 0, 0, 0]
    
    return features[:8]


def is_safe_public_url(url):
    """Allow only http/https public hosts to reduce SSRF risk in legacy scripts."""
    try:
        parsed = urlparse(str(url).strip())
        if parsed.scheme not in ("http", "https") or not parsed.hostname:
            return False
        host = parsed.hostname
        if host == "localhost":
            return False
        try:
            ip_obj = ipaddress.ip_address(host)
            return not (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_multicast
                or ip_obj.is_reserved
                or ip_obj.is_unspecified
            )
        except ValueError:
            return True
    except Exception:
        return False


def generate_cache_key(text, url):
    """生成缓存键（使用哈希算法）"""
    data = f"{text}:{url}"
    return Config.HASH_ALGORITHM(data.encode()).hexdigest()


class RuleBasedDetector:
    """基于规则的备用检测器"""
    
    def __init__(self):
        self.suspicious_keywords = [
            '支付宝', '微信', '银行', '账户', '安全', '风险', '解冻', '封禁',
            '验证码', '密码', '登录', '验证', '支付', '转账', '汇款',
            '钓鱼', '诈骗', '欺诈', 'verify', 'secure', 'login', 'account',
            'auth', 'payment', 'bank', 'financial', 'update', 'confirm',
            'reset', 'password'
        ]
        
        self.suspicious_url_patterns = [
            r'\b(verify|secure|login|account|auth|payment)\b',
            r'\b(bank|financial|update|confirm|reset|password)\b',
            r'\b(verification|validate|authentication|signin|sign-up)\b',
            r'\b(phish|钓鱼|诈骗|欺诈)\b'
        ]
    
    def detect(self, text, url):
        """基于规则检测钓鱼"""
        score = 0
        
        # 文本规则
        text_lower = text.lower()
        for keyword in self.suspicious_keywords:
            if keyword in text_lower:
                score += 1
        
        # URL规则
        if url:
            url_lower = url.lower()
            for pattern in self.suspicious_url_patterns:
                import re
                if re.search(pattern, url_lower):
                    score += 1
            
            # URL长度检查
            if len(url) > 50:
                score += 1
            
            # 是否使用HTTPS
            if not url.startswith('https://'):
                score += 1
        
        # 基于分数判断
        if score >= 3:
            return {
                "model": "rule_based",
                "prediction": "钓鱼",
                "confidence": min(score / 10.0, 0.99),
                "details": {
                    "rule_score": score,
                    "text_keywords": [k for k in self.suspicious_keywords if k in text_lower],
                    "url_patterns": [p for p in self.suspicious_url_patterns if re.search(p, url.lower())] if url else []
                }
            }
        else:
            return {
                "model": "rule_based",
                "prediction": "正常",
                "confidence": max(1.0 - score / 10.0, 0.5),
                "details": {
                    "rule_score": score
                }
            }


class PhishingDetector:
    def __init__(self, model_dir=Config.DEFAULT_MODEL_DIR, use_cuda=True, bert_model=Config.DEFAULT_BERT_MODEL, cache_ttl=Config.CACHE_TTL):
        self.model_dir = model_dir
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.bert_model_name = bert_model
        self.cache_ttl = cache_ttl
        self._load_models()
        self.cache = {}  # 格式: {cache_key: (result, timestamp)}
        self.rule_detector = RuleBasedDetector()
    
    def _load_models(self):
        """加载训练好的模型"""
        from transformers import AutoTokenizer, AutoModel
        
        try:
            # 加载BERT模型
            bert_model_path = Config.API_CONFIG.get(self.bert_model_name, self.bert_model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(bert_model_path)
            self.bert = AutoModel.from_pretrained(bert_model_path)
            self.bert = self.bert.to(self.device)
            print("✅ BERT模型加载成功")
        except Exception as e:
            print(f"⚠️ BERT模型加载失败: {e}")
            self.tokenizer = None
            self.bert = None
        
        # 加载BERT-TextCNN模型
        try:
            model_path = f"{self.model_dir}/bert_textcnn_best.pth"
            if self.bert:
                state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                self.bert_textcnn_model = instantiate_bert_textcnn_for_state_dict(self.bert, state_dict).to(self.device)
                self.bert_textcnn_model.load_state_dict(state_dict, strict=True)
                self.bert_textcnn_model.eval()
                print("✅ BERT-TextCNN模型加载成功")
            else:
                self.bert_textcnn_model = None
                print("⚠️ BERT-TextCNN模型加载失败: BERT模型不可用")
        except Exception as e:
            print(f"⚠️ BERT-TextCNN模型加载失败: {e}")
            self.bert_textcnn_model = None
        
        # 检查是否所有模型都加载失败
        if not self.bert_textcnn_model:
            print("⚠️ 所有模型加载失败，将使用基于规则的检测器")
    
    def _is_cache_valid(self, timestamp):
        """检查缓存是否有效"""
        return time.time() - timestamp < self.cache_ttl
    
    def detect(self, text, url, use_cache=True):
        """检测钓鱼文本"""
        # URL验证
        if url is None:
            return {"error": "URL cannot be None", "details": {"reason": "URL is None"}}
        if not isinstance(url, str):
            url = str(url)
        if not url:
            return {"error": "URL cannot be empty", "details": {"reason": "URL is empty"}}
        
        # 生成缓存键
        cache_key = generate_cache_key(text, url)
        
        # 检查缓存
        if use_cache and cache_key in self.cache:
            result, timestamp = self.cache[cache_key]
            if self._is_cache_valid(timestamp):
                return result
        
        # 提取特征
        url_features = generate_url_features(url)
        network_features = generate_network_features(url)
        
        # 多源信息用于记录和辅助分析，核心判别统一使用BERT-TextCNN。
        if self.bert_textcnn_model:
            result = self._fallback_to_bert_textcnn(text)
        
        else:
            # 所有模型都失败，使用基于规则的检测器
            result = self.rule_detector.detect(text, url)

        result.setdefault("details", {})
        result["details"].update({
            "url_features": url_features,
            "network_features": network_features,
            "multimodal_role": "多源信息用于输入解析、记录和辅助分析；核心分类由BERT-TextCNN完成。",
        })
        
        # 缓存结果
        if use_cache:
            self.cache[cache_key] = (result, time.time())
        
        return result
    
    def _fallback_to_bert_textcnn(self, text):
        """降级到BERT-TextCNN模型"""
        try:
            inputs = self.tokenizer(text, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
            with torch.no_grad():
                torch.set_grad_enabled(False)
                outputs = self.bert_textcnn_model(**inputs)
                prob = torch.softmax(outputs, dim=1).cpu().numpy()[0]
                pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])
            
            return {
                "model": "bert_textcnn",
                "prediction": "钓鱼" if pred == 1 else "正常",
                "confidence": float(prob[pred])
            }
        except Exception as e:
            print(f"BERT-TextCNN模型推理失败: {e}")
            # 降级到规则引擎
            return self.rule_detector.detect(text, "")
    
    def batch_detect(self, texts, urls, batch_size=16, use_length_balancing=False):
        """批量检测钓鱼文本"""
        if use_length_balancing:
            # 基于文本长度进行负载均衡
            indexed_texts = list(enumerate(texts))
            indexed_urls = list(enumerate(urls))
            
            # 按文本长度排序
            sorted_items = sorted(zip(indexed_texts, indexed_urls), key=lambda x: len(x[0][1]))
            sorted_texts = [item[0][1] for item in sorted_items]
            sorted_urls = [item[1][1] for item in sorted_items]
            original_indices = [item[0][0] for item in sorted_items]
            
            # 处理排序后的文本
            sorted_results = []
            for i in range(0, len(sorted_texts), batch_size):
                batch_texts = sorted_texts[i:i+batch_size]
                batch_urls = sorted_urls[i:i+batch_size]
                batch_results = self._batch_process(batch_texts, batch_urls)
                sorted_results.extend(batch_results)
            
            # 恢复原始顺序
            results = [None] * len(texts)
            for i, (orig_idx, result) in enumerate(zip(original_indices, sorted_results)):
                results[orig_idx] = result
            return results
        else:
            # 常规批处理
            results = []
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                batch_urls = urls[i:i+batch_size]
                batch_results = self._batch_process(batch_texts, batch_urls)
                results.extend(batch_results)
            return results
    
    def _batch_process(self, texts, urls):
        """批量处理文本和URL"""
        results = []
        
        cache_hits = []
        cache_results = []
        valid_indices = []
        valid_texts = []
        valid_urls = []
        
        for i, (text, url) in enumerate(zip(texts, urls)):
            # URL验证
            if url is None or (isinstance(url, str) and not url):
                # URL无效，直接返回错误
                results.append({"error": "Invalid URL", "details": {"reason": "URL is None or empty"}})
                cache_hits.append(True)  # 标记为已处理
                cache_results.append(results[-1])
                continue
            
            if not isinstance(url, str):
                url = str(url)
            
            # 生成缓存键
            cache_key = generate_cache_key(text, url)
            
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                if self._is_cache_valid(timestamp):
                    cache_hits.append(True)
                    cache_results.append(result)
                    results.append(result)
                    continue
            
            # 缓存未命中或已过期
            cache_hits.append(False)
            valid_indices.append(i)
            valid_texts.append(text)
            valid_urls.append(url)
            
        # 处理有效样本
        if valid_texts:
            if self.bert_textcnn_model:
                try:
                    inputs = self.tokenizer(valid_texts, padding=True, truncation=True, max_length=128, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        torch.set_grad_enabled(False)
                        outputs = self.bert_textcnn_model(**inputs)
                        probs = torch.softmax(outputs, dim=1).cpu().numpy()
                        preds = torch.argmax(outputs, dim=1).cpu().numpy()
                    
                    for i, (text, url, pred, prob) in enumerate(zip(valid_texts, valid_urls, preds, probs)):
                        result = {
                            "model": "bert_textcnn",
                            "prediction": "钓鱼" if pred == 1 else "正常",
                            "confidence": float(prob[pred]),
                            "details": {
                                "url_features": generate_url_features(url),
                                "network_features": generate_network_features(url),
                                "multimodal_role": "多源信息用于输入解析、记录和辅助分析；核心分类由BERT-TextCNN完成。",
                            }
                        }
                        results.append(result)
                        
                        # 缓存结果
                        cache_key = generate_cache_key(text, url)
                        self.cache[cache_key] = (result, time.time())
                except Exception as e:
                    print(f"BERT-TextCNN模型批量推理失败: {e}")
                    # 降级到单个处理
                    for text, url in zip(valid_texts, valid_urls):
                        result = self.detect(text, url, use_cache=False)
                        results.append(result)
            
            else:
                # 所有模型都失败，使用基于规则的检测器
                for text, url in zip(valid_texts, valid_urls):
                    result = self.rule_detector.detect(text, url)
                    results.append(result)
                    
                    # 缓存结果
                    cache_key = generate_cache_key(text, url)
                    self.cache[cache_key] = (result, time.time())
        
        return results


if __name__ == "__main__":
    detector = PhishingDetector(model_dir="models")
    
    test_samples = [
        ("【支付宝】你的账户存在安全风险，点击 `https://alipay-veri.com` 解冻，逾期将注销账户", "https://alipay-veri.com"),
        ("【微信团队】你的微信账号异地登录，点击 `https://wx-safe.cn` 验证手机号，否则24小时封禁", "https://wx-safe.cn"),
        ("【支付宝】你的余额宝收益已到账，可前往支付宝APP查看详情，官方网址 `https://www.alipay.com`", "https://www.alipay.com"),
        ("【微信团队】你的微信支付分已更新，打开微信APP-我-服务可查询，官方网址 `https://weixin.qq.com`", "https://weixin.qq.com"),
        ("测试空URL", ""),
        ("测试None URL", None)
    ]
    
    print("\n===== 钓鱼检测测试 =====")
    for text, url in test_samples:
        result = detector.detect(text, url)
        print(f"\n文本: {text}")
        print(f"URL: {url}")
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"预测结果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"使用模型: {result['model']}")
            if 'details' in result:
                print(f"URL特征维度: {len(result['details'].get('url_features', []))}")
                print(f"网络行为特征维度: {len(result['details'].get('network_features', []))}")
    
    print("\n===== 批量检测测试（普通批处理）=====")
    texts = [sample[0] for sample in test_samples[:4]]  # 只取前4个有效样本
    urls = [sample[1] for sample in test_samples[:4]]
    batch_results = detector.batch_detect(texts, urls)
    for i, result in enumerate(batch_results):
        print(f"\n样本 {i+1}:")
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"预测结果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"使用模型: {result['model']}")
    
    print("\n===== 批量检测测试（基于文本长度负载均衡）=====")
    batch_results_balanced = detector.batch_detect(texts, urls, use_length_balancing=True)
    for i, result in enumerate(batch_results_balanced):
        print(f"\n样本 {i+1}:")
        if 'error' in result:
            print(f"错误: {result['error']}")
        else:
            print(f"预测结果: {result['prediction']}")
            print(f"置信度: {result['confidence']:.4f}")
            print(f"使用模型: {result['model']}")