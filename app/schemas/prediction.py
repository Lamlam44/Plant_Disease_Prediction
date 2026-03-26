from pydantic import BaseModel, Field, ConfigDict
from typing import Optional


class PredictionData(BaseModel):
    label: str = Field(..., description="Tên bệnh / trạng thái cây trồng")
    confidence: str = Field(..., description="Độ tin cậy của dự đoán (%)")
    inference_time: str = Field(..., description="Thời gian xử lý AI")
    preprocessing: str = Field(
        default="Resized to 224x224 (No external scaling, handled by Model)",
        description="Thông tin tiền xử lý ảnh"
    )
    recommendation: str = Field(..., description="Lời khuyên chăm sóc cây trồng")


class PredictionResponse(BaseModel):
    status: str = Field(..., description="Trạng thái xử lý: success / error")
    data: Optional[PredictionData] = None
    message: Optional[str] = None


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
    results: list[BatchPredictionItem] = Field(..., description="Kết quả từng ảnh")


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    message: str


class CameraRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Ảnh chụp từ camera, mã hóa Base64 (không cần prefix data:image/...)"
    )
