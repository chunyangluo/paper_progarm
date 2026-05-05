import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException

from ..schemas import ModelVersionResponse, ModelActivateRequest, ModelTypeEnum
from ..dependencies import get_model_repo, get_model_manager, get_inference_service
from ..exceptions import NotFoundException
from data_layer.repository import ModelRepository
from inference_layer.model_manager import ModelManager
from inference_layer.inference_service import InferenceService

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/models", tags=["models"])


@router.get("", response_model=List[ModelVersionResponse])
async def list_models(
    model_type: Optional[ModelTypeEnum] = None,
    repo: ModelRepository = Depends(get_model_repo),
):
    try:
        if model_type:
            models = repo.get_by_type(model_type.value)
        else:
            models = repo.get_by_type(ModelTypeEnum.bert_textcnn.value)
        return [ModelVersionResponse(**m.to_dict()) for m in models]
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/active", response_model=ModelVersionResponse)
async def get_active_model(
    model_type: ModelTypeEnum = ModelTypeEnum.bert_textcnn,
    repo: ModelRepository = Depends(get_model_repo),
):
    active_candidates = [m for m in repo.get_by_type(model_type.value) if m.is_active]
    model = active_candidates[0] if active_candidates else None
    if not model:
        raise NotFoundException("No active model found")
    return ModelVersionResponse(**model.to_dict())


@router.get("/info")
async def get_model_info(
    service: InferenceService = Depends(get_inference_service),
    manager: ModelManager = Depends(get_model_manager),
):
    try:
        info = service.get_model_info()
        versions = manager.list_versions()
        return {
            "inference_service": info,
            "available_versions": len(versions),
            "versions": versions[:10],
        }
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/activate")
async def activate_model(
    request: ModelActivateRequest,
    manager: ModelManager = Depends(get_model_manager),
):
    try:
        result = manager.activate_version(request.version, request.model_type.value)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Activation failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to activate model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reload/{model_type}")
async def reload_model(
    model_type: ModelTypeEnum,
    service: InferenceService = Depends(get_inference_service),
):
    try:
        result = service.reload_model(model_type.value)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Reload failed"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/set-active/{model_type}")
async def set_active_model(
    model_type: ModelTypeEnum,
    service: InferenceService = Depends(get_inference_service),
):
    try:
        result = service.set_active_model(model_type.value)
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("error", "Failed to set active model"))
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to set active model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/performance")
async def get_model_performance(
    model_type: Optional[ModelTypeEnum] = None,
    service: InferenceService = Depends(get_inference_service),
):
    try:
        return service.get_model_performance(model_type.value if model_type else None)
    except Exception as e:
        logger.error(f"Failed to get model performance: {e}")
        raise HTTPException(status_code=500, detail=str(e))
