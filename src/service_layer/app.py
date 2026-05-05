import os
import sys
import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from .api import (
    detection_router, alerts_router, samples_router,
    models_router, training_router, system_router
)
from .exceptions import AppException, generic_exception_handler, app_exception_handler
from .dependencies import get_inference_service, get_resource_monitor
from data_layer.database import init_db

logger = logging.getLogger(__name__)

_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting system initialization...")
    try:
        init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    try:
        service = get_inference_service()
        logger.info(f"Inference service initialized: {service.get_model_info()}")
    except Exception as e:
        logger.error(f"Inference service initialization failed: {e}")

    try:
        monitor = get_resource_monitor()
        monitor.start_monitoring()
        logger.info("Resource monitoring started")
    except Exception as e:
        logger.warning(f"Resource monitoring failed to start: {e}")

    logger.info("System startup complete")
    yield

    try:
        monitor = get_resource_monitor()
        monitor.stop_monitoring()
    except Exception:
        pass
    logger.info("System shutdown complete")


def create_app() -> FastAPI:
    app = FastAPI(
        title="多模态钓鱼智能识别与预警系统",
        description="基于BERT-TextCNN混合模型的多模态中文钓鱼检测系统API",
        version="2.0.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_exception_handler(AppException, app_exception_handler)
    app.add_exception_handler(Exception, generic_exception_handler)

    app.middleware("http")
    app.include_router(detection_router)
    app.include_router(alerts_router)
    app.include_router(samples_router)
    app.include_router(models_router)
    app.include_router(training_router)
    app.include_router(system_router)

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'output'
    )
    if os.path.exists(output_dir):
        app.mount("/static/output", StaticFiles(directory=output_dir), name="output")

    @app.get("/")
    async def root():
        return {
            "name": "多模态钓鱼智能识别与预警系统",
            "version": "2.0.0",
            "docs": "/docs",
            "health": "/api/v1/system/health",
        }

    @app.get("/api/v1")
    async def api_root():
        return {
            "version": "v1",
            "endpoints": {
                "detection": "/api/v1/detection",
                "alerts": "/api/v1/alerts",
                "samples": "/api/v1/samples",
                "models": "/api/v1/models",
                "training": "/api/v1/training",
                "system": "/api/v1/system",
            }
        }

    return app


def add_request_logging_middleware(app: FastAPI):
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        request_id = f"{int(start_time * 1000)}_{id(request)}"

        logger.info(f"[{request_id}] {request.method} {request.url.path}")

        try:
            response: Response = await call_next(request)
            process_time = time.time() - start_time
            logger.info(
                f"[{request_id}] {request.method} {request.url.path} "
                f"completed in {process_time:.3f}s with status {response.status_code}"
            )
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            process_time = time.time() - start_time
            logger.error(
                f"[{request_id}] {request.method} {request.url.path} "
                f"failed after {process_time:.3f}s: {str(e)}"
            )
            raise


app = create_app()
add_request_logging_middleware(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "service_layer.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
