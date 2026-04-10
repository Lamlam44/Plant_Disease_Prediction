from pydantic import BaseModel

class PredictionData(BaseModel):
    label: str
    confidence: str
    inference_time: str
    preprocessing: str
    recommendation: str

class BatchPredictionItem(BaseModel):
    filename: str
    status: str
    data: PredictionData
    message: str

class BatchPredictionResponse(BaseModel):
    status: str
    total: int
    successful: int
    failed: int
    results: list[BatchPredictionItem]
