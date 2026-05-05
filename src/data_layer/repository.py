from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_, Integer

from .models import DetectionRecord, AlertInfo, SampleData, ModelVersion, TrainingLog


class BaseRepository:
    model = None

    def __init__(self, session: Session):
        self.session = session

    def get_by_id(self, record_id: int):
        return self.session.query(self.model).filter(self.model.id == record_id).first()

    def get_all(self, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).offset(skip).limit(limit).all()

    def count(self):
        return self.session.query(self.model).count()

    def create(self, obj_dict: Dict[str, Any]):
        obj = self.model(**obj_dict)
        self.session.add(obj)
        self.session.flush()
        return obj

    def update(self, record_id: int, update_dict: Dict[str, Any]):
        obj = self.get_by_id(record_id)
        if obj:
            for key, value in update_dict.items():
                if hasattr(obj, key):
                    setattr(obj, key, value)
            self.session.flush()
        return obj

    def delete(self, record_id: int):
        obj = self.get_by_id(record_id)
        if obj:
            self.session.delete(obj)
            self.session.flush()
        return obj


class DetectionRepository(BaseRepository):
    model = DetectionRecord

    def get_by_sample_id(self, sample_id: str):
        return self.session.query(self.model).filter(self.model.sample_id == sample_id).first()

    def get_recent(self, limit: int = 50):
        return self.session.query(self.model).order_by(desc(self.model.created_at)).limit(limit).all()

    def get_by_prediction(self, prediction: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.prediction == prediction
        ).offset(skip).limit(limit).all()

    def get_by_date_range(self, start_date: datetime, end_date: datetime):
        return self.session.query(self.model).filter(
            and_(self.model.created_at >= start_date, self.model.created_at <= end_date)
        ).all()

    def get_by_scenario(self, scenario: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.scenario == scenario
        ).offset(skip).limit(limit).all()

    def get_stats(self):
        total = self.session.query(self.model).count()
        phishing = self.session.query(self.model).filter(self.model.prediction == "钓鱼").count()
        normal = self.session.query(self.model).filter(self.model.prediction == "正常").count()
        error = total - phishing - normal
        avg_confidence = 0.0
        if total > 0:
            from sqlalchemy import func as sql_func
            result = self.session.query(sql_func.avg(self.model.confidence)).scalar()
            avg_confidence = float(result) if result else 0.0
        return {
            "total": total,
            "phishing": phishing,
            "normal": normal,
            "error": error,
            "avg_confidence": avg_confidence
        }

    def get_trend_data(self, days: int = 30):
        from sqlalchemy import func as sql_func
        results = self.session.query(
            sql_func.date(self.model.created_at).label("date"),
            sql_func.count().label("total"),
            sql_func.sum(sql_func.cast(self.model.is_phishing, Integer)).label("phishing")
        ).group_by(
            sql_func.date(self.model.created_at)
        ).order_by(
            sql_func.date(self.model.created_at).desc()
        ).limit(days).all()
        return [{"date": str(r.date), "total": r.total, "phishing": r.phishing or 0} for r in results]


class AlertRepository(BaseRepository):
    model = AlertInfo

    def get_by_alert_id(self, alert_id: str):
        return self.session.query(self.model).filter(self.model.alert_id == alert_id).first()

    def get_unhandled(self, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.is_handled == False
        ).order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()

    def get_by_severity(self, severity: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.severity == severity
        ).offset(skip).limit(limit).all()

    def handle_alert(self, alert_id: str, handler: str, note: str = ""):
        alert = self.get_by_alert_id(alert_id)
        if alert:
            alert.is_handled = True
            alert.handler = handler
            alert.handle_note = note
            alert.handled_at = datetime.now()
            self.session.flush()
        return alert

    def count_unhandled(self):
        return self.session.query(self.model).filter(self.model.is_handled == False).count()


class SampleRepository(BaseRepository):
    model = SampleData

    def get_by_sample_id(self, sample_id: str):
        return self.session.query(self.model).filter(self.model.sample_id == sample_id).first()

    def get_unprocessed(self, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.is_processed == False
        ).offset(skip).limit(limit).all()

    def get_by_label(self, label: int, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.label == label
        ).offset(skip).limit(limit).all()

    def get_by_source(self, source: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.source == source
        ).offset(skip).limit(limit).all()

    def bulk_create(self, samples: List[Dict[str, Any]]):
        objects = [self.model(**s) for s in samples]
        self.session.bulk_save_objects(objects)
        self.session.flush()
        return len(objects)

    def mark_processed(self, sample_id: str, processed_text: str = None, features: Dict = None):
        sample = self.get_by_sample_id(sample_id)
        if sample:
            sample.is_processed = True
            if processed_text:
                sample.processed_text = processed_text
            if features:
                if "url_features" in features:
                    sample.url_features = features["url_features"]
                if "network_features" in features:
                    sample.network_features = features["network_features"]
                if "text_features" in features:
                    sample.text_features = features["text_features"]
            self.session.flush()
        return sample


class ModelRepository(BaseRepository):
    model = ModelVersion

    def get_by_version(self, version: str):
        return self.session.query(self.model).filter(self.model.version == version).first()

    def get_active(self):
        return self.session.query(self.model).filter(self.model.is_active == True).first()

    def get_deployed(self):
        return self.session.query(self.model).filter(self.model.is_deployed == True).first()

    def get_by_type(self, model_type: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.model_type == model_type
        ).order_by(desc(self.model.created_at)).offset(skip).limit(limit).all()

    def activate_version(self, version: str):
        self.session.query(self.model).update({self.model.is_active: False})
        model = self.get_by_version(version)
        if model:
            model.is_active = True
            self.session.flush()
        return model

    def deploy_version(self, version: str):
        self.session.query(self.model).update({self.model.is_deployed: False})
        model = self.get_by_version(version)
        if model:
            model.is_deployed = True
            model.is_active = True
            self.session.flush()
        return model


class TrainingRepository(BaseRepository):
    model = TrainingLog

    def get_by_task_id(self, task_id: str):
        return self.session.query(self.model).filter(self.model.task_id == task_id).first()

    def get_by_status(self, status: str, skip: int = 0, limit: int = 100):
        return self.session.query(self.model).filter(
            self.model.status == status
        ).offset(skip).limit(limit).all()

    def get_recent(self, limit: int = 20):
        return self.session.query(self.model).order_by(
            desc(self.model.created_at)
        ).limit(limit).all()

    def update_progress(self, task_id: str, epoch: int, metrics: Dict = None):
        log = self.get_by_task_id(task_id)
        if log:
            log.current_epoch = epoch
            if metrics:
                if log.metrics_history is None:
                    log.metrics_history = []
                log.metrics_history.append(metrics)
                if metrics.get("accuracy") and (log.best_accuracy is None or metrics["accuracy"] > log.best_accuracy):
                    log.best_accuracy = metrics["accuracy"]
                if metrics.get("f1_score") and (log.best_f1_score is None or metrics["f1_score"] > log.best_f1_score):
                    log.best_f1_score = metrics["f1_score"]
            self.session.flush()
        return log

    def update_status(self, task_id: str, status: str, error_message: str = None):
        log = self.get_by_task_id(task_id)
        if log:
            log.status = status
            if error_message:
                log.error_message = error_message
            if status == "running" and log.started_at is None:
                log.started_at = datetime.now()
            if status in ("completed", "failed"):
                log.completed_at = datetime.now()
            self.session.flush()
        return log

