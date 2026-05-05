import os
import shutil
import json
from datetime import datetime
from typing import Optional


class BackupManager:
    def __init__(self, db_path: str = None, backup_dir: str = None):
        if db_path is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
            self.db_path = os.path.join(db_dir, 'phishing_detection.db')
        else:
            self.db_path = db_path

        if backup_dir is None:
            self.backup_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'data', 'backups'
            )
        else:
            self.backup_dir = backup_dir
        os.makedirs(self.backup_dir, exist_ok=True)

    def create_backup(self, label: str = None) -> dict:
        if not os.path.exists(self.db_path):
            return {"success": False, "error": "Database file not found"}

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        label_suffix = f"_{label}" if label else ""
        backup_filename = f"phishing_detection_{timestamp}{label_suffix}.db"
        backup_path = os.path.join(self.backup_dir, backup_filename)

        try:
            shutil.copy2(self.db_path, backup_path)
            meta = {
                "filename": backup_filename,
                "timestamp": timestamp,
                "label": label,
                "original_size": os.path.getsize(self.db_path),
                "backup_size": os.path.getsize(backup_path),
            }
            meta_path = os.path.join(self.backup_dir, f"{backup_filename}.meta")
            with open(meta_path, 'w', encoding='utf-8') as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

            self._cleanup_old_backups(max_backups=10)
            return {"success": True, "backup_path": backup_path, "meta": meta}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def restore_backup(self, backup_filename: str) -> dict:
        backup_path = os.path.join(self.backup_dir, backup_filename)
        if not os.path.exists(backup_path):
            return {"success": False, "error": "Backup file not found"}

        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            current_backup = f"pre_restore_{timestamp}.db"
            shutil.copy2(self.db_path, os.path.join(self.backup_dir, current_backup))

            shutil.copy2(backup_path, self.db_path)
            return {"success": True, "restored_from": backup_filename}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def list_backups(self) -> list:
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.db') and not filename.endswith('.meta'):
                meta_path = os.path.join(self.backup_dir, f"{filename}.meta")
                meta = {}
                if os.path.exists(meta_path):
                    with open(meta_path, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                backups.append({
                    "filename": filename,
                    "size": os.path.getsize(os.path.join(self.backup_dir, filename)),
                    "created_at": meta.get("timestamp", ""),
                    "label": meta.get("label", ""),
                })
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)

    def _cleanup_old_backups(self, max_backups: int = 10):
        backups = []
        for filename in os.listdir(self.backup_dir):
            if filename.endswith('.db'):
                filepath = os.path.join(self.backup_dir, filename)
                backups.append((os.path.getmtime(filepath), filename, filepath))

        backups.sort(reverse=True)
        for _, filename, filepath in backups[max_backups:]:
            try:
                os.remove(filepath)
                meta_path = f"{filepath}.meta"
                if os.path.exists(meta_path):
                    os.remove(meta_path)
            except OSError:
                pass
