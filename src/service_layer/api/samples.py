import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query

from ..schemas import SampleCreateRequest, SampleBatchCreateRequest, SampleResponse
from ..dependencies import get_sample_repo
from ..exceptions import NotFoundException
from data_layer.repository import SampleRepository

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/samples", tags=["samples"])


@router.get("", response_model=List[SampleResponse])
async def list_samples(
    skip: int = Query(default=0, ge=0),
    limit: int = Query(default=50, ge=1, le=200),
    label: Optional[int] = None,
    source: Optional[str] = None,
    unprocessed_only: bool = False,
    repo: SampleRepository = Depends(get_sample_repo),
):
    try:
        if unprocessed_only:
            samples = repo.get_unprocessed(skip=skip, limit=limit)
        elif label is not None:
            samples = repo.get_by_label(label, skip=skip, limit=limit)
        elif source:
            samples = repo.get_by_source(source, skip=skip, limit=limit)
        else:
            samples = repo.get_all(skip=skip, limit=limit)
        return [SampleResponse(**s.to_dict()) for s in samples]
    except Exception as e:
        logger.error(f"Failed to list samples: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("", response_model=SampleResponse)
async def create_sample(
    request: SampleCreateRequest,
    repo: SampleRepository = Depends(get_sample_repo),
):
    try:
        sample_id = f"sample_{uuid.uuid4().hex[:12]}"
        sample = repo.create({
            "sample_id": sample_id,
            "text": request.text,
            "url": request.url,
            "scenario": request.scenario.value,
            "label": request.label,
            "source": request.source,
        })
        return SampleResponse(**sample.to_dict())
    except Exception as e:
        logger.error(f"Failed to create sample: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/batch", response_model=dict)
async def create_samples_batch(
    request: SampleBatchCreateRequest,
    repo: SampleRepository = Depends(get_sample_repo),
):
    try:
        samples_data = []
        for item in request.samples:
            sample_id = f"sample_{uuid.uuid4().hex[:12]}"
            samples_data.append({
                "sample_id": sample_id,
                "text": item.text,
                "url": item.url,
                "scenario": item.scenario.value,
                "label": item.label,
                "source": item.source,
            })
        count = repo.bulk_create(samples_data)
        return {"created_count": count}
    except Exception as e:
        logger.error(f"Failed to create samples batch: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{sample_id}", response_model=SampleResponse)
async def get_sample(sample_id: str, repo: SampleRepository = Depends(get_sample_repo)):
    sample = repo.get_by_sample_id(sample_id)
    if not sample:
        raise NotFoundException(f"Sample {sample_id} not found")
    return SampleResponse(**sample.to_dict())


@router.delete("/{sample_id}")
async def delete_sample(sample_id: str, repo: SampleRepository = Depends(get_sample_repo)):
    sample = repo.get_by_sample_id(sample_id)
    if not sample:
        raise NotFoundException(f"Sample {sample_id} not found")
    repo.delete(sample.id)
    return {"success": True, "deleted_sample_id": sample_id}
