import hashlib
import base64
import os
import json
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC


class EncryptionManager:
    _instance = None
    _fernet = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, secret_key: str = None):
        if self._fernet is not None:
            return
        if secret_key is None:
            secret_key = os.environ.get("ENCRYPTION_KEY", "phishing_detection_default_key_2024")
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'phishing_detection_salt',
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(secret_key.encode()))
        self._fernet = Fernet(key)

    def encrypt(self, data: str) -> str:
        if not data:
            return data
        return self._fernet.encrypt(data.encode()).decode()

    def decrypt(self, encrypted_data: str) -> str:
        if not encrypted_data:
            return encrypted_data
        try:
            return self._fernet.decrypt(encrypted_data.encode()).decode()
        except Exception:
            return encrypted_data

    def encrypt_dict(self, data: dict) -> str:
        if not data:
            return ""
        json_str = json.dumps(data, ensure_ascii=False)
        return self.encrypt(json_str)

    def decrypt_dict(self, encrypted_data: str) -> dict:
        if not encrypted_data:
            return {}
        try:
            json_str = self.decrypt(encrypted_data)
            return json.loads(json_str)
        except Exception:
            return {}

    @staticmethod
    def hash_data(data: str) -> str:
        if not data:
            return ""
        return hashlib.sha256(data.encode()).hexdigest()

    @staticmethod
    def mask_sensitive(text: str) -> str:
        if not text:
            return text
        if len(text) <= 8:
            return "*" * len(text)
        return text[:3] + "*" * (len(text) - 6) + text[-3:]


_encryption_manager: EncryptionManager = None


def get_encryption_manager() -> EncryptionManager:
    global _encryption_manager
    if _encryption_manager is None:
        _encryption_manager = EncryptionManager()
    return _encryption_manager
