import logging
import os
import json
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Dict, Any


class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.now().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "line": record.lineno,
            "message": record.getMessage(),
        }
        if hasattr(record, "extra_data"):
            log_data["extra"] = record.extra_data
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        return json.dumps(log_data, ensure_ascii=False)


class ContextLogger:
    def __init__(self, name: str, log_dir: str = None, level: int = logging.INFO):
        self.name = name
        if log_dir is None:
            log_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'logs'
            )
        os.makedirs(log_dir, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)

        if self.logger.handlers:
            self.logger.handlers.clear()

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_fmt = logging.Formatter(
            '%(asctime)s [%(levelname)s] %(name)s - %(module)s:%(lineno)d - %(message)s'
        )
        console_handler.setFormatter(console_fmt)
        self.logger.addHandler(console_handler)

        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d")}.log')
        file_handler = RotatingFileHandler(
            log_file, maxBytes=50 * 1024 * 1024, backupCount=10, encoding='utf-8'
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(file_handler)

        error_file = os.path.join(log_dir, f'{name}_error_{datetime.now().strftime("%Y%m%d")}.log')
        error_handler = RotatingFileHandler(
            error_file, maxBytes=50 * 1024 * 1024, backupCount=10, encoding='utf-8'
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(StructuredFormatter())
        self.logger.addHandler(error_handler)

    def _log(self, level, message, **kwargs):
        extra = {"extra_data": kwargs} if kwargs else {}
        self.logger.log(level, message, extra=extra)

    def debug(self, message, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def warning(self, message, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def error(self, message, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def critical(self, message, **kwargs):
        self._log(logging.CRITICAL, message, **kwargs)

    def exception(self, message, **kwargs):
        extra = {"extra_data": kwargs} if kwargs else {}
        self.logger.exception(message, extra=extra)


_loggers: Dict[str, ContextLogger] = {}


def get_context_logger(name: str = "system", log_dir: str = None, level: int = logging.INFO) -> ContextLogger:
    if name not in _loggers:
        _loggers[name] = ContextLogger(name, log_dir, level)
    return _loggers[name]


system_logger = get_context_logger("phishing_system")
api_logger = get_context_logger("phishing_api")
inference_logger = get_context_logger("phishing_inference")
data_logger = get_context_logger("phishing_data")
