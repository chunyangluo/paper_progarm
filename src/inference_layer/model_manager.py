import os
import json
import shutil
import logging
import threading
from typing import Dict, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)


class ModelManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, model_dir: str = None, version_file: str = None):
        if self._initialized:
            return
        self._initialized = True

        if model_dir is None:
            model_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'models'
            )
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)

        if version_file is None:
            self.version_file = os.path.join(model_dir, "model_versions.json")
        else:
            self.version_file = version_file

        self._versions: Dict = {}
        self._load_versions()

    def _load_versions(self):
        if os.path.exists(self.version_file):
            try:
                with open(self.version_file, 'r', encoding='utf-8') as f:
                    self._versions = json.load(f)
                logger.info(f"Loaded {len(self._versions)} model versions")
            except Exception as e:
                logger.error(f"Failed to load model versions: {e}")
                self._versions = {}
        else:
            self._versions = self._init_default_versions()
            self._save_versions()

    def _init_default_versions(self) -> Dict:
        versions = {}
        model_files = {
            "bert_textcnn_best.pth": {
                "version": "v1.0.0",
                "model_type": "bert_textcnn",
                "description": "BERT-TextCNN core recognition model",
                "metrics": {
                    "accuracy": 0.9934,
                    "precision": 0.9925,
                    "recall": 0.9942,
                    "f1_score": 0.9934,
                    "auc_score": 0.9991,
                }
            }
        }
        for filename, info in model_files.items():
            filepath = os.path.join(self.model_dir, filename)
            if os.path.exists(filepath):
                versions[info["version"] + "_" + info["model_type"]] = {
                    **info,
                    "filename": filename,
                    "model_path": filepath,
                    "is_active": True,
                    "is_deployed": True,
                    "created_at": datetime.now().isoformat(),
                    "file_size": os.path.getsize(filepath),
                }
        return versions

    def _save_versions(self):
        try:
            with open(self.version_file, 'w', encoding='utf-8') as f:
                json.dump(self._versions, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save model versions: {e}")

    def register_version(self, model_type: str, version: str, model_path: str,
                         description: str = "", metrics: Dict = None) -> Dict:
        if model_type != "bert_textcnn":
            return {
                "success": False,
                "error": "Only BERT-TextCNN is supported as the deployed recognition model.",
            }
        version_key = f"{version}_{model_type}"
        if version_key in self._versions:
            logger.warning(f"Version {version_key} already exists, updating...")

        self._versions[version_key] = {
            "version": version,
            "model_type": model_type,
            "model_path": model_path,
            "description": description,
            "metrics": metrics or {},
            "is_active": False,
            "is_deployed": False,
            "created_at": datetime.now().isoformat(),
            "file_size": os.path.getsize(model_path) if os.path.exists(model_path) else 0,
        }
        self._save_versions()
        return self._versions[version_key]

    def get_version(self, version: str, model_type: str) -> Optional[Dict]:
        if model_type != "bert_textcnn":
            return None
        version_key = f"{version}_{model_type}"
        return self._versions.get(version_key)

    def list_versions(self, model_type: str = None) -> List[Dict]:
        versions = [v for v in self._versions.values() if v.get("model_type") == "bert_textcnn"]
        if model_type:
            versions = [v for v in versions if v["model_type"] == model_type]
        return sorted(versions, key=lambda x: x.get("created_at", ""), reverse=True)

    def activate_version(self, version: str, model_type: str) -> Dict:
        if model_type != "bert_textcnn":
            return {
                "success": False,
                "error": "Only BERT-TextCNN can be activated for recognition.",
            }
        version_key = f"{version}_{model_type}"
        if version_key not in self._versions:
            return {"success": False, "error": f"Version {version_key} not found"}

        for key in self._versions:
            if self._versions[key]["model_type"] == model_type:
                self._versions[key]["is_active"] = False

        self._versions[version_key]["is_active"] = True
        self._save_versions()

        from .inference_service import InferenceService
        service = InferenceService()
        reload_result = service.reload_model(model_type, self._versions[version_key]["model_path"])

        return {
            "success": reload_result.get("success", False),
            "version": version,
            "model_type": model_type,
            "reload_result": reload_result
        }

    def deploy_version(self, version: str, model_type: str) -> Dict:
        if model_type != "bert_textcnn":
            return {
                "success": False,
                "error": "Only BERT-TextCNN can be deployed for recognition.",
            }
        version_key = f"{version}_{model_type}"
        if version_key not in self._versions:
            return {"success": False, "error": f"Version {version_key} not found"}

        for key in self._versions:
            if self._versions[key]["model_type"] == model_type:
                self._versions[key]["is_deployed"] = False
                self._versions[key]["is_active"] = False

        self._versions[version_key]["is_deployed"] = True
        self._versions[version_key]["is_active"] = True
        self._save_versions()

        return self.activate_version(version, model_type)

    def get_active_version(self, model_type: str) -> Optional[Dict]:
        for v in self._versions.values():
            if v["model_type"] == model_type and v.get("is_active"):
                return v
        return None

    def delete_version(self, version: str, model_type: str) -> Dict:
        version_key = f"{version}_{model_type}"
        if version_key not in self._versions:
            return {"success": False, "error": f"Version {version_key} not found"}

        version_info = self._versions[version_key]
        if version_info.get("is_active") or version_info.get("is_deployed"):
            return {"success": False, "error": "Cannot delete active or deployed version"}

        del self._versions[version_key]
        self._save_versions()
        return {"success": True, "deleted_version": version_key}

    def compare_versions(self, version1: str, model_type1: str,
                         version2: str, model_type2: str) -> Dict:
        v1 = self.get_version(version1, model_type1)
        v2 = self.get_version(version2, model_type2)
        if not v1 or not v2:
            return {"success": False, "error": "One or both versions not found"}

        metrics1 = v1.get("metrics", {})
        metrics2 = v2.get("metrics", {})
        comparison = {}
        all_keys = set(list(metrics1.keys()) + list(metrics2.keys()))
        for key in all_keys:
            val1 = metrics1.get(key, 0)
            val2 = metrics2.get(key, 0)
            comparison[key] = {
                "version1": val1,
                "version2": val2,
                "diff": val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
            }
        return {
            "success": True,
            "version1": f"{version1}_{model_type1}",
            "version2": f"{version2}_{model_type2}",
            "comparison": comparison
        }
