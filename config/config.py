import yaml
import os
import logging
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

class ConfigManager:
    """
    配置管理类
    """
    def __init__(self, config_file='config/config.yaml'):
        """
        初始化配置管理器
        
        参数:
            config_file: 配置文件路径
        """
        self.config_file = config_file
        self.config = self._load_config()
    
    def _load_config(self):
        """
        加载配置文件
        
        返回:
            dict: 配置字典
        """
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info(f"配置文件加载成功: {self.config_file}")
                return config
            else:
                logger.warning(f"配置文件不存在，使用默认配置: {self.config_file}")
                return self._get_default_config()
        except Exception as e:
            logger.error(f"加载配置文件失败: {e}")
            return self._get_default_config()
    
    def _get_default_config(self):
        """
        获取默认配置
        
        返回:
            dict: 默认配置字典
        """
        return {
            "model": {
                "bert_textcnn": {
                    "model_path": "models/bert_textcnn_best.pth",
                    "max_length": 128,
                    "batch_size": 64,
                    "learning_rate": 1e-5,
                    "epochs": 10
                },
                "multimodal": {
                    "model_path": "models/multimodal_best.pth",
                    "max_length": 128,
                    "batch_size": 64,
                    "learning_rate": 1e-5,
                    "epochs": 10
                },
                "textcnn": {
                    "model_path": "models/textcnn_best.pth",
                    "max_features": 5000,
                    "ngram_range": [1, 2],
                    "batch_size": 64,
                    "learning_rate": 1e-3,
                    "epochs": 10
                }
            },
            "data": {
                "dataset_path": "data/versions/dataset_20260411_chifraud.csv",
                "test_size": 0.2,
                "val_size": 0.2,
                "random_state": 42
            },
            "training": {
                "device": "cuda",
                "num_workers": 4,
                "pin_memory": true,
                "drop_last": true,
                "prefetch_factor": 2,
                "persistent_workers": true
            },
            "inference": {
                "batch_size": 32,
                "timeout": 5
            },
            "network": {
                "timeout": 5,
                "max_retries": 3,
                "batch_size": 100,
                "delay": 0.1
            },
            "logging": {
                "level": "INFO",
                "log_dir": "logs"
            }
        }
    
    def get(self, key, default=None):
        """
        获取配置值
        
        参数:
            key: 配置键，支持点号分隔的路径，如 "model.bert_textcnn.batch_size"
            default: 默认值
        
        返回:
            配置值
        """
        try:
            keys = key.split('.')
            value = self.config
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            logger.warning(f"配置键不存在: {key}，使用默认值: {default}")
            return default
    
    def set(self, key, value):
        """
        设置配置值
        
        参数:
            key: 配置键，支持点号分隔的路径
            value: 配置值
        """
        try:
            keys = key.split('.')
            config = self.config
            for k in keys[:-1]:
                if k not in config:
                    config[k] = {}
                config = config[k]
            config[keys[-1]] = value
            logger.info(f"配置更新: {key} = {value}")
        except Exception as e:
            logger.error(f"设置配置失败: {e}")
    
    def save(self, output_file=None):
        """
        保存配置到文件
        
        参数:
            output_file: 输出文件路径，默认使用初始化时的配置文件
        """
        try:
            output_file = output_file or self.config_file
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True)
            logger.info(f"配置保存成功: {output_file}")
        except Exception as e:
            logger.error(f"保存配置失败: {e}")

# 创建全局配置实例
global_config = ConfigManager()

def get_config():
    """
    获取全局配置实例
    
    返回:
        ConfigManager: 配置管理实例
    """
    return global_config
