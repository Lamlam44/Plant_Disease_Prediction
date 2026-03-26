from fastapi import APIRouter
from ..schemas.prediction import HealthResponse
from ..services import model_service

router = APIRouter(tags=["Health Check"])


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Kiểm tra trạng thái hệ thống"
)
def health_check():
    ready = model_service.is_ready()
    return {
        "status": "ready" if ready else "warming_up",
        "model_loaded": ready,
        "message": "Model is ready for predictions" if ready else "Model is warming up, please wait..."
    }
