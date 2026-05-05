import time
import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import AlertResponse, AlertHandleRequest, SeverityEnum
from ..dependencies import get_alert_repo, get_detection_repo
from ..exceptions import NotFoundException
from data_layer.repository import AlertRepository, DetectionRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/alerts", tags=["alerts"])


@router.get("", response_model=List[AlertResponse])
async def list_alerts(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    severity: Optional[SeverityEnum] = None,
    unhandled_only: bool = False,
    repo: AlertRepository = Depends(get_alert_repo),
):
    try:
        if unhandled_only:
            alerts = repo.get_unhandled(skip=skip, limit=limit)
        elif severity:
            alerts = repo.get_by_severity(severity.value, skip=skip, limit=limit)
        else:
            alerts = repo.get_all(skip=skip, limit=limit)
        return [AlertResponse(**a.to_dict()) for a in alerts]
    except Exception as e:
        logger.error(f"Failed to list alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/unhandled-count")
async def get_unhandled_count(repo: AlertRepository = Depends(get_alert_repo)):
    try:
        count = repo.count_unhandled()
        return {"unhandled_count": count}
    except Exception as e:
        logger.error(f"Failed to get unhandled count: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{alert_id}", response_model=AlertResponse)
async def get_alert(alert_id: str, repo: AlertRepository = Depends(get_alert_repo)):
    alert = repo.get_by_alert_id(alert_id)
    if not alert:
        raise NotFoundException(f"Alert {alert_id} not found")
    return AlertResponse(**alert.to_dict())


@router.post("/{alert_id}/handle", response_model=AlertResponse)
async def handle_alert(
    alert_id: str,
    request: AlertHandleRequest,
    repo: AlertRepository = Depends(get_alert_repo),
):
    try:
        alert = repo.handle_alert(alert_id, request.handler, request.note)
        if not alert:
            raise NotFoundException(f"Alert {alert_id} not found")
        return AlertResponse(**alert.to_dict())
    except NotFoundException:
        raise
    except Exception as e:
        logger.error(f"Failed to handle alert: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def create_alert_for_detection(detection_result: dict, sample_id: str, repo: AlertRepository):
    if detection_result.get("is_phishing") and detection_result.get("confidence", 0) > 0.7:
        confidence = detection_result.get("confidence", 0)
        severity = "critical" if confidence > 0.9 else "high" if confidence > 0.8 else "medium"
        alert_id = f"alert_{int(time.time())}_{uuid.uuid4().hex[:6]}"
        try:
            repo.create({
                "alert_id": alert_id,
                "sample_id": sample_id,
                "severity": severity,
                "alert_type": "phishing",
                "content": f"Phishing detected with confidence {confidence:.2%}",
                "text": detection_result.get("text", ""),
                "url": detection_result.get("url", ""),
                "scenario": detection_result.get("scenario", "general"),
                "confidence": confidence,
            })
        except Exception as e:
            logger.warning(f"Failed to create alert: {e}")
