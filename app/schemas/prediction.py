from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    label: str = Field(..., description="Tên bệnh / trạng thái cây trồng")
    confidence: str = Field(..., description="Độ tin cậy của dự đoán (%)")
    inference_time: str = Field(..., description="Thời gian xử lý AI")
    preprocessing: str = Field(
        default="Resized to 300x300",
        description="Thông tin tiền xử lý ảnh"
    )
    recommendation: str = Field(..., description="Lời khuyên chăm sóc cây trồng")


class BatchPredictionItem(BaseModel):
    filename: str = Field(..., description="Tên file ảnh gốc")
    status: str = Field(..., description="Trạng thái xử lý ảnh này")
    data: PredictionData
    message: str


class BatchPredictionResponse(BaseModel):
    status: str = Field(..., description="Trạng thái")
    total: int
    successful: int
    failed: int
    results: list[BatchPredictionItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    message: str
