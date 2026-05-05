import time
import uuid
import logging
from typing import List

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import (
    DetectionRequest, DetectionResponse, BatchDetectionRequest,
    BatchDetectionResponse, ScenarioEnum
)
from ..dependencies import get_inference_service, get_detection_repo, get_alert_repo
from ..exceptions import InferenceException
from .alerts import create_alert_for_detection
from data_layer.repository import DetectionRepository, AlertRepository
from inference_layer.inference_service import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/detection", tags=["detection"])


@router.post("/single", response_model=DetectionResponse)
async def detect_single(
    request: DetectionRequest,
    service: InferenceService = Depends(get_inference_service),
    repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
):
    sample_id = f"s_{int(time.time())}_{uuid.uuid4().hex[:8]}"
    try:
        model_type = request.model_type.value if request.model_type else None
        result = service.predict(
            text=request.text,
            url=request.url,
            model_type=model_type,
            scenario=request.scenario.value,
            use_cache=request.use_cache,
        )

        if "error" in result and result.get("prediction") == "error":
            raise InferenceException(result.get("error", "Unknown inference error"))

        try:
            repo.create({
                "sample_id": sample_id,
                "text": request.text,
                "url": request.url,
                "scenario": request.scenario.value,
                "prediction": result.get("prediction", "unknown"),
                "confidence": result.get("confidence", 0.0),
                "model_name": result.get("model", "unknown"),
                "url_features": result.get("details", {}).get("url_features") if result.get("details") else None,
                "network_features": result.get("details", {}).get("network_features") if result.get("details") else None,
                "processing_time": result.get("processing_time", 0.0),
                "is_phishing": result.get("is_phishing", False),
            })
        except Exception as e:
            logger.warning(f"Failed to save detection record: {e}")
        try:
            create_alert_for_detection(
                detection_result={
                    **result,
                    "text": request.text,
                    "url": request.url,
                    "scenario": request.scenario.value,
                },
                sample_id=sample_id,
                repo=alert_repo,
            )
        except Exception as e:
            logger.warning(f"Failed to create alert for detection: {e}")

        return DetectionResponse(
            sample_id=sample_id,
            prediction=result.get("prediction", "unknown"),
            confidence=result.get("confidence", 0.0),
            is_phishing=result.get("is_phishing", False),
            model=result.get("model", "unknown"),
            scenario=request.scenario.value,
            processing_time=result.get("processing_time", 0.0),
            from_cache=result.get("from_cache", False),
            details=result.get("details"),
        )
    except InferenceException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    request: BatchDetectionRequest,
    service: InferenceService = Depends(get_inference_service),
    repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
):
    start_time = time.time()
    results: List[DetectionResponse] = []
    phishing_count = 0
    normal_count = 0

    model_type = request.model_type.value if request.model_type else None

    for item in request.items:
        sample_id = f"s_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        try:
            result = service.predict(
                text=item.text,
                url=item.url,
                model_type=model_type,
                scenario=item.scenario.value,
                use_cache=item.use_cache,
            )

            if result.get("is_phishing"):
                phishing_count += 1
            else:
                normal_count += 1

            try:
                repo.create({
                    "sample_id": sample_id,
                    "text": item.text,
                    "url": item.url,
                    "scenario": item.scenario.value,
                    "prediction": result.get("prediction", "unknown"),
                    "confidence": result.get("confidence", 0.0),
                    "model_name": result.get("model", "unknown"),
                    "url_features": result.get("details", {}).get("url_features") if result.get("details") else None,
                    "network_features": result.get("details", {}).get("network_features") if result.get("details") else None,
                    "processing_time": result.get("processing_time", 0.0),
                    "is_phishing": result.get("is_phishing", False),
                })
            except Exception as e:
                logger.warning(f"Failed to save detection record: {e}")
            try:
                create_alert_for_detection(
                    detection_result={
                        **result,
                        "text": item.text,
                        "url": item.url,
                        "scenario": item.scenario.value,
                    },
                    sample_id=sample_id,
                    repo=alert_repo,
                )
            except Exception as e:
                logger.warning(f"Failed to create alert for batch item: {e}")

            results.append(DetectionResponse(
                sample_id=sample_id,
                prediction=result.get("prediction", "unknown"),
                confidence=result.get("confidence", 0.0),
                is_phishing=result.get("is_phishing", False),
                model=result.get("model", "unknown"),
                scenario=item.scenario.value,
                processing_time=result.get("processing_time", 0.0),
                from_cache=result.get("from_cache", False),
                details=result.get("details"),
            ))
        except Exception as e:
            logger.error(f"Batch detection item failed: {e}")
            results.append(DetectionResponse(
                sample_id=sample_id,
                prediction="error",
                confidence=0.0,
                is_phishing=False,
                model="unknown",
                scenario=item.scenario.value,
                processing_time=0.0,
                error=str(e),
            ))

    total_time = time.time() - start_time
    return BatchDetectionResponse(
        results=results,
        total=len(results),
        phishing_count=phishing_count,
        normal_count=normal_count,
        total_processing_time=total_time,
        avg_processing_time=total_time / len(results) if results else 0.0,
    )


@router.post("/sms", response_model=DetectionResponse)
async def detect_sms(
    request: DetectionRequest,
    service: InferenceService = Depends(get_inference_service),
    repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
):
    request.scenario = ScenarioEnum.sms
    return await detect_single(request, service, repo, alert_repo)


@router.post("/email", response_model=DetectionResponse)
async def detect_email(
    request: DetectionRequest,
    service: InferenceService = Depends(get_inference_service),
    repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
):
    request.scenario = ScenarioEnum.email
    return await detect_single(request, service, repo, alert_repo)


@router.post("/link", response_model=DetectionResponse)
async def detect_link(
    request: DetectionRequest,
    service: InferenceService = Depends(get_inference_service),
    repo: DetectionRepository = Depends(get_detection_repo),
    alert_repo: AlertRepository = Depends(get_alert_repo),
):
    request.scenario = ScenarioEnum.link
    return await detect_single(request, service, repo, alert_repo)
