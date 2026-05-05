import uuid
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks

from ..schemas import TrainingStartRequest, TrainingLogResponse, TaskStatusEnum
from ..dependencies import get_training_repo
from ..exceptions import NotFoundException
from data_layer.repository import TrainingRepository
logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/training", tags=["training"])


@router.post("/start")
async def start_training(
    request: TrainingStartRequest,
    background_tasks: BackgroundTasks,
    repo: TrainingRepository = Depends(get_training_repo),
):
    task_id = f"train_{uuid.uuid4().hex[:12]}"
    try:
        log = repo.create({
            "task_id": task_id,
            "model_type": request.model_type.value,
            "status": "pending",
            "epochs": request.epochs,
            "learning_rate": request.learning_rate,
            "batch_size": request.batch_size,
        })

        background_tasks.add_task(
            _run_training_task,
            task_id=task_id,
            model_type=request.model_type.value,
            epochs=request.epochs,
            learning_rate=request.learning_rate,
            batch_size=request.batch_size,
            dataset_path=request.dataset_path,
        )

        return {
            "task_id": task_id,
            "status": "pending",
            "message": "Training task created and will start in background"
        }
    except Exception as e:
        logger.error(f"Failed to start training: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks", response_model=List[TrainingLogResponse])
async def list_training_tasks(
    skip: int = 0,
    limit: int = 20,
    status: Optional[TaskStatusEnum] = None,
    repo: TrainingRepository = Depends(get_training_repo),
):
    try:
        if status:
            tasks = repo.get_by_status(status.value, skip=skip, limit=limit)
        else:
            tasks = repo.get_recent(limit=limit)
        return [TrainingLogResponse(**t.to_dict()) for t in tasks]
    except Exception as e:
        logger.error(f"Failed to list training tasks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tasks/{task_id}", response_model=TrainingLogResponse)
async def get_training_task(task_id: str, repo: TrainingRepository = Depends(get_training_repo)):
    task = repo.get_by_task_id(task_id)
    if not task:
        raise NotFoundException(f"Training task {task_id} not found")
    return TrainingLogResponse(**task.to_dict())


@router.post("/tasks/{task_id}/cancel")
async def cancel_training_task(
    task_id: str,
    repo: TrainingRepository = Depends(get_training_repo),
):
    task = repo.get_by_task_id(task_id)
    if not task:
        raise NotFoundException(f"Training task {task_id} not found")
    if task.status not in ("pending", "running"):
        raise HTTPException(status_code=400, detail="Task cannot be cancelled in current status")

    repo.update_status(task_id, "failed", "Cancelled by user")
    return {"task_id": task_id, "status": "cancelled"}


def _run_training_task(task_id: str, model_type: str, epochs: int,
                       learning_rate: float, batch_size: int,
                       dataset_path: str = None):
    import os
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

    from data_layer.database import get_db_manager

    db_manager = get_db_manager()
    with db_manager.session_scope() as session:
        from data_layer.repository import TrainingRepository
        repo = TrainingRepository(session)

        try:
            repo.update_status(task_id, "running")

            import subprocess
            cmd = [
                sys.executable,
                os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                             'core', 'train_bert_textcnn.py'),
                '--epochs', str(epochs),
                '--batch-size', str(batch_size),
                '--learning-rate', str(learning_rate),
            ]
            if dataset_path:
                cmd.extend(['--dataset', dataset_path])

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600,
                cwd=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            )

            if result.returncode == 0:
                repo.update_status(task_id, "completed")
            else:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown training error"
                repo.update_status(task_id, "failed", error_msg)

        except subprocess.TimeoutExpired:
            repo.update_status(task_id, "failed", "Training timeout (1 hour)")
        except Exception as e:
            repo.update_status(task_id, "failed", str(e))
