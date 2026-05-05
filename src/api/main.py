from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import time
import json
import os
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.inference import PhishingDetector
from core.visualization import VisualizationModule
from core.incremental_training import IncrementalTrainer

# 初始化FastAPI应用
app = FastAPI(
    title="多模态钓鱼智能识别与预警系统",
    description="支持实时检测、可视化溯源、增量训练等功能的钓鱼识别系统",
    version="1.0.0"
)

# 配置CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 初始化钓鱼检测器
detector = PhishingDetector()

# 初始化可视化模块
visualizer = VisualizationModule()

# 初始化增量训练器
incremental_trainer = IncrementalTrainer()

# 定义请求模型
class DetectionRequest(BaseModel):
    text: str
    url: str
    scenario: str = "general"  # general, sms, email, link

class BatchDetectionRequest(BaseModel):
    items: list[dict]

# 定义响应模型
class DetectionResponse(BaseModel):
    model: str
    prediction: str
    confidence: float
    details: dict = None
    processing_time: float

class BatchDetectionResponse(BaseModel):
    results: list[DetectionResponse]
    total_processing_time: float

@app.on_event("startup")
async def startup_event():
    print("系统启动中...")
    print("模型加载完成，系统准备就绪")

@app.get("/")
async def root():
    return {"message": "多模态钓鱼智能识别与预警系统 API"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/detect", response_model=DetectionResponse)
async def detect(request: DetectionRequest):
    """实时检测钓鱼文本"""
    start_time = time.time()
    
    try:
        # 调用检测器进行检测
        result = detector.detect(request.text, request.url)
        
        # 计算处理时间
        processing_time = time.time() - start_time
        
        # 构建响应
        response = DetectionResponse(
            model=result.get("model", "unknown"),
            prediction=result.get("prediction", "unknown"),
            confidence=result.get("confidence", 0.0),
            details=result.get("details", None),
            processing_time=processing_time
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/batch_detect", response_model=BatchDetectionResponse)
async def batch_detect(request: BatchDetectionRequest):
    """批量检测钓鱼文本"""
    start_time = time.time()
    
    try:
        # 提取文本和URL列表
        texts = [item.get("text", "") for item in request.items]
        urls = [item.get("url", "") for item in request.items]
        
        # 调用批量检测
        results = detector.batch_detect(texts, urls)
        
        # 计算总处理时间
        total_processing_time = time.time() - start_time
        
        # 构建响应
        detection_results = []
        for i, result in enumerate(results):
            detection_result = DetectionResponse(
                model=result.get("model", "unknown"),
                prediction=result.get("prediction", "unknown"),
                confidence=result.get("confidence", 0.0),
                details=result.get("details", None),
                processing_time=total_processing_time / len(results) if results else 0
            )
            detection_results.append(detection_result)
        
        response = BatchDetectionResponse(
            results=detection_results,
            total_processing_time=total_processing_time
        )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect_sms", response_model=DetectionResponse)
async def detect_sms(request: DetectionRequest):
    """检测钓鱼短信"""
    # 短信场景的特殊处理
    # 例如，短信通常更短，可能需要特殊的特征提取
    return await detect(request)

@app.post("/detect_email", response_model=DetectionResponse)
async def detect_email(request: DetectionRequest):
    """检测钓鱼邮件"""
    # 邮件场景的特殊处理
    # 例如，邮件可能包含更多的HTML内容和附件
    return await detect(request)

@app.post("/detect_link", response_model=DetectionResponse)
async def detect_link(request: DetectionRequest):
    """检测钓鱼链接"""
    # 链接场景的特殊处理
    # 例如，链接可能需要更多的URL特征分析
    return await detect(request)

# 定义溯源请求模型
class TraceRequest(BaseModel):
    text: str
    url: str
    scenario: str = "general"

# 定义溯源响应模型
class TraceResponse(BaseModel):
    report: dict

@app.post("/trace", response_model=TraceResponse)
async def trace(request: TraceRequest):
    """钓鱼攻击溯源分析"""
    try:
        # 先进行检测
        detection_result = detector.detect(request.text, request.url)
        
        # 提取特征
        from core.scenario_processor import ScenarioProcessor
        scenario_processor = ScenarioProcessor()
        
        # 根据场景处理特征
        if request.scenario == "sms":
            features = scenario_processor.process_sms(request.text, request.url)
        elif request.scenario == "email":
            features = scenario_processor.process_email(request.text, request.url)
        elif request.scenario == "link":
            features = scenario_processor.process_link(request.text, request.url)
        else:
            features = scenario_processor.process_sms(request.text, request.url)
        
        # 生成样本信息
        sample = {
            "id": "sample_1",
            "text": request.text,
            "url": request.url,
            "scenario": request.scenario
        }
        
        # 生成报告
        report = visualizer.generate_report(sample, detection_result, features)
        
        return TraceResponse(report=report)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze_trends")
async def analyze_trends(historical_data: list[dict]):
    """分析钓鱼攻击趋势"""
    try:
        # 生成趋势图
        trend_image = visualizer.analyze_trends(historical_data)
        
        return {"trend_image": trend_image}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 定义增量训练请求模型
class IncrementalTrainingRequest(BaseModel):
    samples: list[dict]
    model_type: str = "multimodal"
    epochs: int = 10

# 定义增量训练响应模型
class IncrementalTrainingResponse(BaseModel):
    model_type: str
    training_samples: int
    test_samples: int
    evaluation_results: dict

@app.post("/incremental_train", response_model=IncrementalTrainingResponse)
async def incremental_train(request: IncrementalTrainingRequest):
    """增量训练模型"""
    try:
        # 调用增量训练器
        result = incremental_trainer.update_model(
            new_samples=request.samples,
            model_type=request.model_type,
            epochs=request.epochs
        )
        
        return IncrementalTrainingResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/collect_data")
async def collect_data(samples: list[dict]):
    """收集新的样本数据"""
    try:
        # 调用增量训练器收集数据
        collected_data = incremental_trainer.collect_data(samples)
        
        return {
            "collected_samples": len(samples),
            "total_samples": len(collected_data)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
