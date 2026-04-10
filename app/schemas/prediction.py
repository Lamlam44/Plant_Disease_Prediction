from pydantic import BaseModel, Field


class PredictionData(BaseModel):
    label: str = Field(..., description="Tên bệnh / trạng thái cây trồng")
    confidence: str = Field(..., description="Độ tin cậy của dự đoán (%)")
    inference_time: str = Field(..., description="Thời gian xử lý AI")
    preprocessing: str = Field(
        default="Resized to 300x300 (EfficientNetV2B3 built-in preprocessing)",
        description="Thông tin tiền xử lý ảnh"
    )
    recommendation: str = Field(..., description="Lời khuyên chăm sóc cây trồng")


class BatchPredictionItem(BaseModel):
    filename: str = Field(..., description="Tên file ảnh gốc")
    status: str = Field(..., description="Trạng thái xử lý ảnh này: success / error")
    data: PredictionData = Field(..., description="Dữ liệu nhận diện")
    message: str = Field(..., description="Thông báo lỗi nếu có")


class BatchPredictionResponse(BaseModel):
    status: str = Field(..., description="Trạng thái tổng thể")
    total: int = Field(..., description="Tổng số ảnh nhận được")
    successful: int = Field(..., description="Số ảnh nhận diện thành công")
    failed: int = Field(..., description="Số ảnh nhận diện thất bại")
    results: list[BatchPredictionItem] = Field(..., description="Kết quả từng ảnh")
