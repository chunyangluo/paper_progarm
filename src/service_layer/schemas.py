from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class ScenarioEnum(str, Enum):
    general = "general"
    sms = "sms"
    email = "email"
    link = "link"


class SeverityEnum(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


class ModelTypeEnum(str, Enum):
    bert_textcnn = "bert_textcnn"


class TaskStatusEnum(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class DetectionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000, description="待检测文本")
    url: str = Field(default="", max_length=2048, description="待检测URL")
    scenario: ScenarioEnum = Field(default=ScenarioEnum.general, description="检测场景")
    model_type: Optional[ModelTypeEnum] = Field(default=None, description="模型类型")
    use_cache: bool = Field(default=True, description="是否使用缓存")


class BatchDetectionRequest(BaseModel):
    items: List[DetectionRequest] = Field(..., min_length=1, max_length=500)
    model_type: Optional[ModelTypeEnum] = Field(default=None, description="模型类型")


class DetectionResponse(BaseModel):
    sample_id: str
    prediction: str
    confidence: float
    is_phishing: bool
    model: str
    scenario: str
    processing_time: float
    from_cache: bool = False
    details: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class BatchDetectionResponse(BaseModel):
    results: List[DetectionResponse]
    total: int
    phishing_count: int
    normal_count: int
    total_processing_time: float
    avg_processing_time: float


class AlertResponse(BaseModel):
    id: int
    alert_id: str
    sample_id: str
    severity: str
    alert_type: str
    content: Optional[str]
    confidence: float
    is_handled: bool
    created_at: Optional[str]
    handled_at: Optional[str]


class AlertHandleRequest(BaseModel):
    handler: str = Field(..., min_length=1, max_length=64)
    note: str = Field(default="", max_length=500)


class SampleCreateRequest(BaseModel):
    text: Optional[str] = None
    url: Optional[str] = None
    scenario: ScenarioEnum = ScenarioEnum.general
    label: Optional[int] = Field(default=None, ge=0, le=1)
    source: Optional[str] = None


class SampleBatchCreateRequest(BaseModel):
    samples: List[SampleCreateRequest] = Field(..., min_length=1, max_length=1000)


class SampleResponse(BaseModel):
    id: int
    sample_id: str
    text: Optional[str]
    url: Optional[str]
    scenario: str
    label: Optional[int]
    source: Optional[str]
    is_processed: bool
    created_at: Optional[str]


class ModelVersionResponse(BaseModel):
    id: int
    version: str
    model_type: str
    model_path: str
    description: Optional[str]
    accuracy: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    f1_score: Optional[float]
    auc_score: Optional[float]
    is_active: bool
    is_deployed: bool
    created_at: Optional[str]


class ModelActivateRequest(BaseModel):
    version: str
    model_type: ModelTypeEnum


class TrainingStartRequest(BaseModel):
    model_type: ModelTypeEnum = ModelTypeEnum.bert_textcnn
    epochs: int = Field(default=10, ge=1, le=100)
    learning_rate: float = Field(default=1e-5, gt=0)
    batch_size: int = Field(default=16, ge=1, le=256)
    dataset_path: Optional[str] = None


class TrainingLogResponse(BaseModel):
    id: int
    task_id: str
    model_type: str
    model_version: Optional[str]
    status: str
    epochs: Optional[int]
    current_epoch: int
    best_accuracy: Optional[float]
    best_f1_score: Optional[float]
    error_message: Optional[str]
    started_at: Optional[str]
    completed_at: Optional[str]
    created_at: Optional[str]


class StatsResponse(BaseModel):
    total_detections: int
    phishing_count: int
    normal_count: int
    avg_confidence: float
    unhandled_alerts: int
    active_model: str
    loaded_models: List[str]


class TrendDataPoint(BaseModel):
    date: str
    total: int
    phishing: int


class HealthResponse(BaseModel):
    status: str
    version: str
    uptime: float
    model_info: Dict[str, Any]
    resource_info: Dict[str, Any]


class PaginatedResponse(BaseModel):
    items: List[Any]
    total: int
    skip: int
    limit: int


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    status_code: int
