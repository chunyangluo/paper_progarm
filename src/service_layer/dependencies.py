import os
import sys
import time
from functools import lru_cache
from typing import Generator

from fastapi import Depends, Request
from sqlalchemy.orm import Session

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_layer.database import get_db_manager, DatabaseManager
from data_layer.repository import (
    DetectionRepository, AlertRepository, SampleRepository,
    ModelRepository, TrainingRepository
)
from inference_layer.inference_service import InferenceService
from inference_layer.model_manager import ModelManager
from inference_layer.resource_monitor import ResourceMonitor


def get_db() -> Generator[Session, None, None]:
    manager = get_db_manager()
    with manager.session_scope() as session:
        yield session


def get_detection_repo(session: Session = Depends(get_db)) -> DetectionRepository:
    return DetectionRepository(session)


def get_alert_repo(session: Session = Depends(get_db)) -> AlertRepository:
    return AlertRepository(session)


def get_sample_repo(session: Session = Depends(get_db)) -> SampleRepository:
    return SampleRepository(session)


def get_model_repo(session: Session = Depends(get_db)) -> ModelRepository:
    return ModelRepository(session)


def get_training_repo(session: Session = Depends(get_db)) -> TrainingRepository:
    return TrainingRepository(session)


@lru_cache()
def get_inference_service() -> InferenceService:
    return InferenceService()


@lru_cache()
def get_model_manager() -> ModelManager:
    return ModelManager()


@lru_cache()
def get_resource_monitor() -> ResourceMonitor:
    return ResourceMonitor()


_app_start_time = time.time()


def get_app_uptime() -> float:
    return time.time() - _app_start_time
