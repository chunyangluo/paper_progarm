import os
import time
import threading
import logging
from typing import Dict, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ResourceMonitor:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, interval: int = 60):
        if self._initialized:
            return
        self._initialized = True
        self._interval = interval
        self._metrics_history: list = []
        self._max_history = 1440
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None

    def start_monitoring(self):
        if self._monitoring:
            return
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Resource monitoring started")

    def stop_monitoring(self):
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
        logger.info("Resource monitoring stopped")

    def _monitor_loop(self):
        while self._monitoring:
            try:
                metrics = self.collect_metrics()
                self._metrics_history.append(metrics)
                if len(self._metrics_history) > self._max_history:
                    self._metrics_history = self._metrics_history[-self._max_history:]
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            time.sleep(self._interval)

    def collect_metrics(self) -> Dict:
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "cpu": self._get_cpu_usage(),
            "memory": self._get_memory_usage(),
            "disk": self._get_disk_usage(),
            "gpu": self._get_gpu_usage(),
        }
        return metrics

    def _get_cpu_usage(self) -> Dict:
        try:
            import psutil
            return {
                "percent": psutil.cpu_percent(interval=1),
                "count": psutil.cpu_count(),
                "load_avg": os.getloadavg() if hasattr(os, 'getloadavg') else None,
            }
        except ImportError:
            return {"percent": 0.0, "count": os.cpu_count() or 0}
        except Exception:
            return {"percent": 0.0, "count": 0}

    def _get_memory_usage(self) -> Dict:
        try:
            import psutil
            mem = psutil.virtual_memory()
            return {
                "total_gb": round(mem.total / (1024 ** 3), 2),
                "used_gb": round(mem.used / (1024 ** 3), 2),
                "available_gb": round(mem.available / (1024 ** 3), 2),
                "percent": mem.percent,
            }
        except ImportError:
            return {"total_gb": 0.0, "used_gb": 0.0, "available_gb": 0.0, "percent": 0.0}
        except Exception:
            return {"total_gb": 0.0, "used_gb": 0.0, "available_gb": 0.0, "percent": 0.0}

    def _get_disk_usage(self) -> Dict:
        try:
            import psutil
            disk = psutil.disk_usage('/')
            return {
                "total_gb": round(disk.total / (1024 ** 3), 2),
                "used_gb": round(disk.used / (1024 ** 3), 2),
                "free_gb": round(disk.free / (1024 ** 3), 2),
                "percent": disk.percent,
            }
        except ImportError:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}
        except Exception:
            return {"total_gb": 0.0, "used_gb": 0.0, "free_gb": 0.0, "percent": 0.0}

    def _get_gpu_usage(self) -> Dict:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_id = 0
                total_mem = torch.cuda.get_device_properties(gpu_id).total_mem / (1024 ** 3)
                used_mem = torch.cuda.memory_allocated(gpu_id) / (1024 ** 3)
                return {
                    "available": True,
                    "device_name": torch.cuda.get_device_name(gpu_id),
                    "total_memory_gb": round(total_mem, 2),
                    "used_memory_gb": round(used_mem, 2),
                    "utilization_percent": round(used_mem / total_mem * 100, 2) if total_mem > 0 else 0,
                }
        except Exception:
            pass
        return {"available": False}

    def get_current_metrics(self) -> Dict:
        return self.collect_metrics()

    def get_metrics_history(self, minutes: int = 60) -> list:
        count = max(1, minutes // (self._interval // 60)) if self._interval >= 60 else minutes
        return self._metrics_history[-count:]

    def get_summary(self) -> Dict:
        current = self.collect_metrics()
        return {
            "current": current,
            "history_size": len(self._metrics_history),
            "monitoring": self._monitoring,
            "interval_seconds": self._interval,
        }
