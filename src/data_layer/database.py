import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator

from .models import Base


class DatabaseManager:
    _instance = None
    _engine = None
    _session_factory = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, db_url: str = None):
        if self._engine is not None:
            return
        if db_url is None:
            db_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'db')
            os.makedirs(db_dir, exist_ok=True)
            db_url = f"sqlite:///{os.path.join(db_dir, 'phishing_detection.db')}"
        self._engine = create_engine(
            db_url,
            echo=False,
            pool_pre_ping=True,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        self._session_factory = sessionmaker(bind=self._engine, autocommit=False, autoflush=False)
        Base.metadata.create_all(bind=self._engine)

    @property
    def engine(self):
        return self._engine

    def get_session(self) -> Session:
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self):
        if self._engine:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None
            DatabaseManager._instance = None


_db_manager: DatabaseManager = None


def init_db(db_url: str = None):
    global _db_manager
    _db_manager = DatabaseManager(db_url)
    return _db_manager


def get_db_manager() -> DatabaseManager:
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
    return _db_manager


def get_db_session() -> Generator[Session, None, None]:
    manager = get_db_manager()
    with manager.session_scope() as session:
        yield session
