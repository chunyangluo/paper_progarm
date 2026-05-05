import logging
from typing import List

from fastapi import APIRouter, Depends

from ..schemas import StatsResponse, TrendDataPoint, HealthResponse
from ..dependencies import (
    get_detection_repo, get_alert_repo, get_inference_service,
    get_resource_monitor, get_app_uptime
)
from data_layer.repository import DetectionRepository, AlertRepository
from inference_layer.inference_service import InferenceService
from inference_layer.resource_monitor import ResourceMonitor

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/system", tags=["system"])


@router.get("/health", response_model=HealthResponse)
async def health_check(
    service: InferenceService = Depends(get_inference_service),
    monitor: ResourceMonitor = Depends(get_resource_monitor),
):
    model_info = service.get_model_info()
    resource_info = monitor.get_current_metrics()
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime=get_app_uptime(),
        model_info=model_info,
        resource_info=resource_info,
    )


@router.get("/stats", response_model=StatsResponse)
async def get_stats(
    detection_repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
    service: InferenceService = Depends(get_inference_service),
):
    try:
        stats = detection_repo.get_stats()
        unhandled = alert_repo.count_unhandled()
        model_info = service.get_model_info()

        return StatsResponse(
            total_detections=stats["total"],
            phishing_count=stats["phishing"],
            normal_count=stats["normal"],
            avg_confidence=stats["avg_confidence"],
            unhandled_alerts=unhandled,
            active_model=model_info["active_model"],
            loaded_models=model_info["loaded_models"],
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        return StatsResponse(
            total_detections=0,
            phishing_count=0,
            normal_count=0,
            avg_confidence=0.0,
            unhandled_alerts=0,
            active_model="unknown",
            loaded_models=[],
        )


@router.get("/trends", response_model=List[TrendDataPoint])
async def get_trends(
    days: int = 30,
    repo: DetectionRepository = Depends(get_detection_repo),
):
    try:
        return repo.get_trend_data(days=days)
    except Exception as e:
        logger.error(f"Failed to get trends: {e}")
        return []


@router.get("/resources")
async def get_resources(
    monitor: ResourceMonitor = Depends(get_resource_monitor),
):
    return monitor.get_summary()


@router.post("/backup")
async def create_backup():
    from data_layer.backup import BackupManager
    manager = BackupManager()
    result = manager.create_backup()
    result["security_note"] = "Backup endpoints are intended for trusted administrative use only."
    return result


@router.get("/backups")
async def list_backups():
    from data_layer.backup import BackupManager
    manager = BackupManager()
    return {
        "security_note": "Backup listings are intended for trusted administrative use only.",
        "items": manager.list_backups(),
    }
