import time
from fastapi import FastAPI, File, UploadFile, Form
from model_helper import predict_disease

app = FastAPI(
    title="Plant Disease Diagnosis API",
    description="Hệ thống nhận diện bệnh cây trồng qua ảnh. Demo chuẩn Swagger UI.",
    version="1.0.0"
)

@app.post("/predict", tags=["AI Inference"])
async def predict(
    file: UploadFile = File(..., description="Tải lên ảnh lá cây (JPG/PNG)"),
    plant_type: str = Form(..., description="Loại cây: rice, tomato, hoặc corn")
):
    # 1. Ghi lại thời gian bắt đầu
    start_time = time.perf_counter()
    
    # 2. Đọc dữ liệu ảnh từ request
    image_bytes = await file.read()
    
    # 3. Gọi hàm xử lý AI (Kết quả trả về là một Dictionary từ model_helper)
    result = predict_disease(image_bytes, plant_type)
    
    # Kiểm tra nếu có lỗi (ví dụ chưa train model)
    if "error" in result:
        return {"status": "error", "message": result["error"]}
    
    # 4. Tính toán thời gian xử lý
    inference_time = f"{time.perf_counter() - start_time:.4f}s"
    
    # 5. Trả về kết quả JSON (Lấy dữ liệu TỪ BIẾN result)
    return {
        "status": "success",
        "data": {
            "label": result["label"],          # Lấy nhãn bệnh từ result
            "confidence": f"{result['score']}%", # Lấy độ tin cậy từ result
            "inference_time": inference_time,
            "preprocessing": "Resized to 224x224, Scaled 1/255",
            "recommendation": result["advice"]  # Lấy lời khuyên từ result
        }
    }