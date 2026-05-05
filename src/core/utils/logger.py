import logging
import os
from logging.handlers import RotatingFileHandler
from datetime import datetime

class Logger:
    """
    日志管理类
    """
    def __init__(self, name='root', log_dir='logs', level=logging.INFO):
        """
        初始化日志管理器
        
        参数:
            name: 日志名称
            log_dir: 日志目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        # 创建日志目录
        os.makedirs(log_dir, exist_ok=True)
        
        # 创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # 清除已有的处理器
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # 创建格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        )
        
        # 创建控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # 创建文件处理器（带轮转）
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5  # 保留5个备份
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        
    def debug(self, message, **kwargs):
        """
        记录调试级别的日志
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.debug(message, extra=extra)
    
    def info(self, message, **kwargs):
        """
        记录信息级别的日志
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.info(message, extra=extra)
    
    def warning(self, message, **kwargs):
        """
        记录警告级别的日志
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.warning(message, extra=extra)
    
    def error(self, message, **kwargs):
        """
        记录错误级别的日志
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.error(message, extra=extra)
    
    def critical(self, message, **kwargs):
        """
        记录严重错误级别的日志
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.critical(message, extra=extra)
    
    def exception(self, message, **kwargs):
        """
        记录异常信息
        
        参数:
            message: 日志消息
            **kwargs: 额外的日志数据
        """
        extra = kwargs if kwargs else {}
        self.logger.exception(message, extra=extra)

# 创建全局日志实例
global_logger = Logger('phishing_detection')

def get_logger(name=None):
    """
    获取日志实例
    
    参数:
        name: 日志名称
    
    返回:
        Logger: 日志实例
    """
    if name:
        return Logger(name)
    return global_logger
