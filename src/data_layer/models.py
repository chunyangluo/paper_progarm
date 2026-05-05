from sqlalchemy import Column, Integer, String, Float, DateTime, Text, Boolean, JSON
from sqlalchemy.sql import func
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()


class DetectionRecord(Base):
    __tablename__ = "detection_records"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(64), index=True, nullable=False)
    text = Column(Text, nullable=True)
    url = Column(String(2048), nullable=True)
    scenario = Column(String(32), default="general")
    prediction = Column(String(16), nullable=False)
    confidence = Column(Float, nullable=False)
    model_name = Column(String(64), nullable=False)
    url_features = Column(JSON, nullable=True)
    network_features = Column(JSON, nullable=True)
    text_features = Column(JSON, nullable=True)
    processing_time = Column(Float, default=0.0)
    is_phishing = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "text": self.text,
            "url": self.url,
            "scenario": self.scenario,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "url_features": self.url_features,
            "network_features": self.network_features,
            "text_features": self.text_features,
            "processing_time": self.processing_time,
            "is_phishing": self.is_phishing,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class AlertInfo(Base):
    __tablename__ = "alert_infos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    alert_id = Column(String(64), unique=True, index=True, nullable=False)
    sample_id = Column(String(64), index=True, nullable=False)
    severity = Column(String(16), default="medium")
    alert_type = Column(String(32), default="phishing")
    content = Column(Text, nullable=True)
    text = Column(Text, nullable=True)
    url = Column(String(2048), nullable=True)
    scenario = Column(String(32), default="general")
    confidence = Column(Float, default=0.0)
    is_handled = Column(Boolean, default=False)
    handler = Column(String(64), nullable=True)
    handle_note = Column(Text, nullable=True)
    handled_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "alert_id": self.alert_id,
            "sample_id": self.sample_id,
            "severity": self.severity,
            "alert_type": self.alert_type,
            "content": self.content,
            "text": self.text,
            "url": self.url,
            "scenario": self.scenario,
            "confidence": self.confidence,
            "is_handled": self.is_handled,
            "handler": self.handler,
            "handle_note": self.handle_note,
            "handled_at": self.handled_at.isoformat() if self.handled_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class SampleData(Base):
    __tablename__ = "sample_datas"

    id = Column(Integer, primary_key=True, autoincrement=True)
    sample_id = Column(String(64), unique=True, index=True, nullable=False)
    text = Column(Text, nullable=True)
    url = Column(String(2048), nullable=True)
    scenario = Column(String(32), default="general")
    label = Column(Integer, nullable=True)
    source = Column(String(64), nullable=True)
    is_processed = Column(Boolean, default=False)
    processed_text = Column(Text, nullable=True)
    url_features = Column(JSON, nullable=True)
    network_features = Column(JSON, nullable=True)
    text_features = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    def to_dict(self):
        return {
            "id": self.id,
            "sample_id": self.sample_id,
            "text": self.text,
            "url": self.url,
            "scenario": self.scenario,
            "label": self.label,
            "source": self.source,
            "is_processed": self.is_processed,
            "processed_text": self.processed_text,
            "url_features": self.url_features,
            "network_features": self.network_features,
            "text_features": self.text_features,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None
        }


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    version = Column(String(32), unique=True, index=True, nullable=False)
    model_type = Column(String(32), nullable=False)
    model_path = Column(String(512), nullable=False)
    description = Column(Text, nullable=True)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    auc_score = Column(Float, nullable=True)
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    training_time = Column(Float, nullable=True)
    is_active = Column(Boolean, default=False)
    is_deployed = Column(Boolean, default=False)
    config = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "version": self.version,
            "model_type": self.model_type,
            "model_path": self.model_path,
            "description": self.description,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "auc_score": self.auc_score,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "training_time": self.training_time,
            "is_active": self.is_active,
            "is_deployed": self.is_deployed,
            "config": self.config,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }


class TrainingLog(Base):
    __tablename__ = "training_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_id = Column(String(64), unique=True, index=True, nullable=False)
    model_type = Column(String(32), nullable=False)
    model_version = Column(String(32), nullable=True)
    status = Column(String(16), default="pending")
    epochs = Column(Integer, nullable=True)
    current_epoch = Column(Integer, default=0)
    learning_rate = Column(Float, nullable=True)
    batch_size = Column(Integer, nullable=True)
    training_samples = Column(Integer, nullable=True)
    test_samples = Column(Integer, nullable=True)
    best_accuracy = Column(Float, nullable=True)
    best_f1_score = Column(Float, nullable=True)
    loss_history = Column(JSON, nullable=True)
    metrics_history = Column(JSON, nullable=True)
    error_message = Column(Text, nullable=True)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now(), index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "status": self.status,
            "epochs": self.epochs,
            "current_epoch": self.current_epoch,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "training_samples": self.training_samples,
            "test_samples": self.test_samples,
            "best_accuracy": self.best_accuracy,
            "best_f1_score": self.best_f1_score,
            "loss_history": self.loss_history,
            "metrics_history": self.metrics_history,
            "error_message": self.error_message,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "created_at": self.created_at.isoformat() if self.created_at else None
        }
