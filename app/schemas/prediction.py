from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Dict
from datetime import datetime


class PredictionData(BaseModel):
    label: str = Field(..., description="Tên bệnh / trạng thái cây trồng")
    confidence: float = Field(..., description="Độ tin cậy của dự đoán (%)")
    raw_scores: list[float] = Field(default=[], description="Danh sách điểm thô")
    inference_time: str = Field(..., description="Thời gian xử lý AI")
    preprocessing: str = Field(
        default="Resized to 300x300 (EfficientNetV2B3 built-in preprocessing)",
        description="Thông tin tiền xử lý ảnh"
    )
    recommendation: str = Field(..., description="Lời khuyên chăm sóc cây trồng")
    processed_at: datetime = Field(default_factory=datetime.now, description="Thời gian xử lý")


class BatchPredictionItem(BaseModel):
    filename: str = Field(..., description="Tên file ảnh gốc")
    status: str = Field(..., description="Trạng thái xử lý ảnh này: success / error")
    data: Optional[PredictionData] = None
    message: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    status: str = Field(..., description="Trạng thái tổng thể")
    total: int = Field(..., description="Tổng số ảnh nhận được")
    successful: int = Field(..., description="Số ảnh nhận diện thành công")
    failed: int = Field(..., description="Số ảnh nhận diện thất bại")
    results: Dict[str, BatchPredictionItem] = Field(..., description="Kết quả từng ảnh dạng Dictionary")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    message: str
