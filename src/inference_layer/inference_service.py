import os
import sys
import time
import json
import hashlib
import logging
import threading
import ipaddress
from urllib.parse import urlparse
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from collections import OrderedDict

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.models.model_definitions import BERTTextCNN, TextCNN, instantiate_bert_textcnn_for_state_dict

logger = logging.getLogger(__name__)


class LRUCache:
    def __init__(self, max_size: int = 1000):
        self._cache: OrderedDict = OrderedDict()
        self._max_size = max_size
        self._lock = threading.Lock()

    def get(self, key: str):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key]
            return None

    def put(self, key: str, value: Any):
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            self._cache[key] = value
            if len(self._cache) > self._max_size:
                self._cache.popitem(last=False)

    def clear(self):
        with self._lock:
            self._cache.clear()

    def __len__(self):
        return len(self._cache)


class PriorityTaskQueue:
    def __init__(self):
        self._queue: List[Dict] = []
        self._lock = threading.Lock()

    def put(self, task: Dict, priority: int = 0):
        with self._lock:
            task["_priority"] = priority
            self._queue.append(task)
            self._queue.sort(key=lambda x: x.get("_priority", 0), reverse=True)

    def get(self) -> Optional[Dict]:
        with self._lock:
            if self._queue:
                task = self._queue.pop(0)
                task.pop("_priority", None)
                return task
            return None

    def __len__(self):
        return len(self._queue)


class InferenceService:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_dir: str = None, device: str = None, cache_size: int = 1000):
        if self._initialized:
            return
        self._initialized = True

        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'
            )
        self.model_dir = model_dir
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self._models: Dict[str, Any] = {}
        self._tokenizer = None
        self._bert = None
        self._cache = LRUCache(max_size=cache_size)
        self._feature_cache = LRUCache(max_size=max(500, cache_size))
        self._task_queue = PriorityTaskQueue()
        self._model_lock = threading.RLock()
        self._active_model_type = "bert_textcnn"
        # Keep inference path non-blocking by default.
        self._enable_online_network_probe = False

        self._load_tokenizer()
        self._load_all_models()

    def _load_tokenizer(self):
        try:
            self._tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
            logger.info("Tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load tokenizer: {e}")

    def _load_all_models(self):
        self._load_bert_textcnn()

    def _load_bert_textcnn(self):
        model_path = os.path.join(self.model_dir, "bert_textcnn_best.pth")
        if not os.path.exists(model_path):
            logger.warning(f"BERT-TextCNN model not found: {model_path}")
            return
        try:
            if self._bert is None:
                self._bert = AutoModel.from_pretrained("bert-base-chinese")
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            model = instantiate_bert_textcnn_for_state_dict(self._bert, state_dict).to(self.device)
            model.load_state_dict(state_dict, strict=True)
            model.eval()
            with self._model_lock:
                self._models["bert_textcnn"] = model
            logger.info("BERT-TextCNN model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT-TextCNN model: {e}")

    def _generate_cache_key(self, text: str, url: str = "", model_type: str = "") -> str:
        data = f"{model_type}:{text}:{url}"
        return hashlib.md5(data.encode()).hexdigest()

    @staticmethod
    def _normalize_model_type(model_type: Optional[str]) -> str:
        # The deployed recognizer always uses BERT-TextCNN; multisource data is
        # retained as analysis context instead of being fed into a multimodal net.
        if model_type in (None, "", "bert_textcnn", "multimodal", "textcnn"):
            return "bert_textcnn"
        return "bert_textcnn"

    @staticmethod
    def generate_url_features(url: str) -> List[float]:
        if not url or not isinstance(url, str):
            return [0.0] * 16
        features = []
        features.append(float(len(url)))
        features.append(1.0 if "https" in url else 0.0)
        features.append(float(url.count(".")))
        for kw in ["login", "bank", "secure", "verify", "account",
                    "auth", "payment", "password", "reset", "confirm", "update", "financial", "phish"]:
            features.append(1.0 if kw in url.lower() else 0.0)
        return features[:16]

    @staticmethod
    def generate_network_features(url: str, enable_online_probe: bool = False) -> List[float]:
        if not url or not isinstance(url, str):
            return [3.0, 404.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if not InferenceService._is_safe_public_url(url):
            logger.warning("Skipped unsafe URL during network feature generation")
            return [3.0, 404.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        if not enable_online_probe:
            return [
                0.0,
                404.0,
                0.0,
                0.0,
                0.0,
                1.0 if url.startswith('https://') else 0.0,
                0.0,
                1.0 if 'login' in url.lower() else 0.0,
            ]
        features = [3.0, 404.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        try:
            import requests
            start_time = time.time()
            response = requests.get(url, timeout=3, allow_redirects=True)
            response_time = time.time() - start_time
            features = [
                response_time,
                float(response.status_code),
                float(len(response.history)),
                1.0 if '<form' in response.text.lower() else 0.0,
                1.0 if any(p in response.text.lower() for p in ['password', 'credit card', '身份证', '银行卡']) else 0.0,
                1.0 if url.startswith('https://') else 0.0,
                1.0 if 'captcha' in response.text.lower() else 0.0,
                1.0 if 'login' in response.text.lower() else 0.0,
            ]
        except Exception:
            pass
        return features[:8]

    def _get_multisource_features(self, url: str) -> Tuple[List[float], List[float]]:
        cache_key = self._generate_cache_key("", url, "feature_pack")
        cached = self._feature_cache.get(cache_key)
        if cached is not None:
            return cached["url_features"], cached["network_features"]
        url_features = self.generate_url_features(url)
        network_features = self.generate_network_features(
            url, enable_online_probe=self._enable_online_network_probe
        )
        packed = {
            "url_features": url_features,
            "network_features": network_features,
        }
        self._feature_cache.put(cache_key, packed)
        return url_features, network_features

    def _build_multimodal_context(self, url: str, scenario: str = "general") -> Dict[str, Any]:
        url_features, network_features = self._get_multisource_features(url)
        return {
            "url_features": url_features,
            "network_features": network_features,
            "feature_summary": {
                "has_url": bool(url),
                "scenario": scenario,
                "url_length": int(url_features[0]) if url_features else 0,
                "uses_https": bool(url_features[1]) if len(url_features) > 1 else False,
                "domain_dot_count": int(url_features[2]) if len(url_features) > 2 else 0,
                "online_network_probe": self._enable_online_network_probe,
            },
            "multimodal_role": (
                "文本、URL和网络行为特征用于系统输入解析、记录、预警和辅助分析；"
                "核心分类结果由BERT-TextCNN混合模型输出。"
            ),
        }

    @staticmethod
    def _is_safe_public_url(url: str) -> bool:
        """Allow only http/https public hosts to reduce SSRF risk."""
        try:
            parsed = urlparse(url.strip())
            if parsed.scheme not in ("http", "https"):
                return False
            if not parsed.hostname:
                return False
            host = parsed.hostname
            # Block obvious localhost names.
            if host in ("localhost",):
                return False
            # Block direct private/link-local/loopback IP targets.
            try:
                ip_obj = ipaddress.ip_address(host)
                if (
                    ip_obj.is_private
                    or ip_obj.is_loopback
                    or ip_obj.is_link_local
                    or ip_obj.is_multicast
                    or ip_obj.is_reserved
                    or ip_obj.is_unspecified
                ):
                    return False
            except ValueError:
                # Host is a domain; keep allowed.
                pass
            return True
        except Exception:
            return False

    def predict(self, text: str, url: str = "", model_type: str = None,
                scenario: str = "general", use_cache: bool = True) -> Dict:
        model_type = self._normalize_model_type(model_type or self._active_model_type)
        multimodal_context = self._build_multimodal_context(url, scenario=scenario)

        if not text or len(text.strip()) < 2:
            return {
                "prediction": "正常",
                "confidence": 0.5,
                "is_phishing": False,
                "processing_time": 0.0,
                "model": "rule_based",
                "from_cache": False,
                "details": multimodal_context,
            }

        cache_key = self._generate_cache_key(text, url, model_type)
        if use_cache:
            cached = self._cache.get(cache_key)
            if cached is not None:
                cached["from_cache"] = True
                return cached

        start_time = time.time()
        try:
            if "bert_textcnn" in self._models and self._tokenizer is not None:
                result = self._predict_bert_textcnn(text, url)
            else:
                result = self._predict_rule_based(text, url)

            result["details"] = multimodal_context
            result["processing_time"] = time.time() - start_time
            result["model"] = model_type if model_type in self._models else "rule_based"
            result["from_cache"] = False

            if use_cache:
                self._cache.put(cache_key, result)
            return result
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return {
                "prediction": "error",
                "confidence": 0.0,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "model": model_type,
                "from_cache": False,
                "details": multimodal_context,
            }

    @staticmethod
    def _rule_based_score(text: str, url: str = "") -> float:
        suspicious_keywords = [
            '支付宝', '微信', '银行', '账户', '安全', '风险', '解冻', '封禁',
            '验证码', '密码', '登录', '验证', '支付', '转账', '冻结', '异常',
            '中奖', '领取', '优惠', '退款', '过期', '紧急', '立即', '点击',
            '逾期', '征信', '社保', '退税', '税务局', '社保局', '信用卡',
            '永久', '作废', '保管费', '异地', '领取', '补录',
            'verify', 'secure', 'login', 'account', 'auth', 'payment', 'bank',
            'password', 'reset', 'confirm', 'update', 'financial', 'free', 'claim',
            'prize', 'win', 'reward', 'urgent', 'expire', 'suspend', 'freeze'
        ]
        score = sum(1 for kw in suspicious_keywords if kw in text.lower())
        if url:
            score += sum(1 for kw in suspicious_keywords if kw in url.lower())
            if not url.startswith('https://'):
                score += 1
            if len(url) > 50:
                score += 1
        urgency_patterns = ['请立即', '请点击', '否则将', '逾期', '24小时', '仅限今日', '逾期作废']
        score += sum(1 for p in urgency_patterns if p in text)
        return score

    def _predict_bert_textcnn(self, text: str, url: str = "") -> Dict:
        with self._model_lock:
            model = self._models.get("bert_textcnn")
            if model is None:
                raise ValueError("BERT-TextCNN model not loaded")

        inputs = self._tokenizer(
            text, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)

        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            probabilities = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            pred = int(np.argmax(probabilities))

        model_confidence = float(probabilities[pred])
        rule_score = self._rule_based_score(text, url)

        if model_confidence < 0.95 and rule_score >= 3:
            pred = 1
            model_confidence = min(0.5 + rule_score * 0.05, 0.99)

        return {
            "prediction": "钓鱼" if pred == 1 else "正常",
            "confidence": model_confidence,
            "probabilities": probabilities.tolist(),
            "is_phishing": pred == 1,
        }

    def _predict_bert_textcnn_batch(self, texts: List[str], urls: List[str]) -> List[Dict]:
        with self._model_lock:
            model = self._models.get("bert_textcnn")
            if model is None:
                raise ValueError("BERT-TextCNN model not loaded")
        inputs = self._tokenizer(
            texts, padding=True, truncation=True, max_length=128, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    outputs = model(**inputs)
            else:
                outputs = model(**inputs)
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
        results = []
        for text, url, prob_vec, pred in zip(texts, urls, probs, preds):
            pred = int(pred)
            model_confidence = float(prob_vec[pred])
            rule_score = self._rule_based_score(text, url)
            if model_confidence < 0.95 and rule_score >= 3:
                pred = 1
                model_confidence = min(0.5 + rule_score * 0.05, 0.99)
            results.append({
                "prediction": "钓鱼" if pred == 1 else "正常",
                "confidence": model_confidence,
                "probabilities": prob_vec.tolist(),
                "is_phishing": pred == 1,
            })
        return results

    def _predict_rule_based(self, text: str, url: str) -> Dict:
        suspicious_keywords = [
            '支付宝', '微信', '银行', '账户', '安全', '风险', '解冻', '封禁',
            '验证码', '密码', '登录', '验证', '支付', '转账', 'verify', 'secure',
            'login', 'account', 'auth', 'payment', 'bank'
        ]
        score = sum(1 for kw in suspicious_keywords if kw in text.lower())
        if url:
            score += sum(1 for kw in suspicious_keywords if kw in url.lower())
            if not url.startswith('https://'):
                score += 1
            if len(url) > 50:
                score += 1

        is_phishing = score >= 3
        confidence = min(score / 10.0, 0.99) if is_phishing else max(1.0 - score / 10.0, 0.5)
        return {
            "prediction": "钓鱼" if is_phishing else "正常",
            "confidence": confidence,
            "is_phishing": is_phishing,
            "model": "rule_based"
        }

    def batch_predict(self, texts: List[str], urls: List[str] = None,
                      model_type: str = None, scenario: str = "general") -> List[Dict]:
        model_type = self._normalize_model_type(model_type or self._active_model_type)
        if urls is None:
            urls = [""] * len(texts)
        results: List[Optional[Dict]] = [None] * len(texts)
        pending_indices = []
        pending_texts = []
        pending_urls = []
        now = time.time()

        for idx, (text, url) in enumerate(zip(texts, urls)):
            if not text or len(text.strip()) < 2:
                results[idx] = {
                    "prediction": "正常",
                    "confidence": 0.5,
                    "is_phishing": False,
                    "processing_time": 0.0,
                    "model": "rule_based",
                    "from_cache": False,
                    "details": self._build_multimodal_context(url, scenario=scenario),
                }
                continue
            cache_key = self._generate_cache_key(text, url, model_type)
            cached = self._cache.get(cache_key)
            if cached is not None:
                item = dict(cached)
                item["from_cache"] = True
                results[idx] = item
                continue
            pending_indices.append(idx)
            pending_texts.append(text)
            pending_urls.append(url)

        if pending_indices:
            if "bert_textcnn" in self._models and self._tokenizer is not None:
                pending_results = self._predict_bert_textcnn_batch(pending_texts, pending_urls)
            else:
                pending_results = [self._predict_rule_based(t, u) for t, u in zip(pending_texts, pending_urls)]

            for idx, result in zip(pending_indices, pending_results):
                item = dict(result)
                item["details"] = self._build_multimodal_context(urls[idx], scenario=scenario)
                item["processing_time"] = max(time.time() - now, 0.0) / max(len(pending_indices), 1)
                item["model"] = model_type if model_type in self._models else "rule_based"
                item["from_cache"] = False
                results[idx] = item
                cache_key = self._generate_cache_key(texts[idx], urls[idx], model_type)
                self._cache.put(cache_key, dict(item))

        return [r for r in results if r is not None]

    def reload_model(self, model_type: str, model_path: str = None) -> Dict:
        model_type = self._normalize_model_type(model_type)
        if model_path is None:
            if model_type == "bert_textcnn":
                model_path = os.path.join(self.model_dir, "bert_textcnn_best.pth")
            else:
                return {"success": False, "error": f"Unknown model type: {model_type}"}

        if not os.path.exists(model_path):
            return {"success": False, "error": f"Model file not found: {model_path}"}

        try:
            with self._model_lock:
                if model_type == "bert_textcnn":
                    if self._bert is None:
                        self._bert = AutoModel.from_pretrained("bert-base-chinese")
                    state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
                    model = instantiate_bert_textcnn_for_state_dict(self._bert, state_dict).to(self.device)
                    model.load_state_dict(state_dict, strict=True)
                    model.eval()
                else:
                    return {"success": False, "error": f"Unknown model type: {model_type}"}

                self._models[model_type] = model

            self._cache.clear()
            self._feature_cache.clear()
            logger.info(f"Model {model_type} reloaded from {model_path}")
            return {"success": True, "model_type": model_type, "model_path": model_path}
        except Exception as e:
            logger.error(f"Failed to reload model {model_type}: {e}")
            return {"success": False, "error": str(e)}

    def set_active_model(self, model_type: str) -> Dict:
        model_type = self._normalize_model_type(model_type)
        if model_type not in self._models:
            return {"success": False, "error": f"Model {model_type} not loaded"}
        self._active_model_type = model_type
        self._cache.clear()
        self._feature_cache.clear()
        return {"success": True, "active_model": model_type}

    def get_model_info(self) -> Dict:
        return {
            "loaded_models": list(self._models.keys()),
            "active_model": self._active_model_type,
            "device": str(self.device),
            "cache_size": len(self._cache),
            "multimodal_handling": "input_analysis_only",
            "recognition_model": "bert_textcnn",
        }

    def get_model_performance(self, model_type: str = None) -> Dict:
        if model_type is None:
            model_type = self._active_model_type
        model_type = self._normalize_model_type(model_type)
        return {
            "model_type": model_type,
            "device": str(self.device),
            "is_loaded": model_type in self._models,
            "multimodal_handling": "URL/network features are extracted for analysis and record keeping, not for neural fusion inference.",
        }
