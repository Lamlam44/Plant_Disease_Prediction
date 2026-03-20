import time
from fastapi import FastAPI, File, UploadFile, Form
import numpy as np
from PIL import Image
import io

app = FastAPI(
    title="Plant Disease Diagnosis API",
    description="Hệ thống nhận diện bệnh cây trồng qua ảnh. Demo chuẩn Swagger UI.",
    version="1.0.0"
)

# Lazy import - load model_helper only when first request comes
_predict_disease = None
_warmed_up = False

def get_predict_disease():
    global _predict_disease
    if _predict_disease is None:
        from .model_helper import predict_disease
        _predict_disease = predict_disease
    return _predict_disease

# Warmup model khi startup
@app.on_event("startup")
def warmup_model():
    global _warmed_up
    if _warmed_up:
        return
    
    print("\n[WARMUP] Đang khởi động model...")
    predict_disease = get_predict_disease()
    
    # Tạo dummy image (224x224 RGB)
    dummy_img = Image.new('RGB', (224, 224), color=(128, 128, 128))
    dummy_bytes = io.BytesIO()
    dummy_img.save(dummy_bytes, format='PNG')
    dummy_bytes.seek(0)
    
    # Run warmup prediction (để TensorFlow optimize graph)
    try:
        result = predict_disease(dummy_bytes.getvalue())
        print("[WARMUP] ✅ Model ready! Lần đầu sẽ nhanh hơn.")
        _warmed_up = True
    except Exception as e:
        print(f"[WARMUP] ⚠️ Warmup failed: {e}")
        _warmed_up = False

@app.get("/health", tags=["Health Check"])
def health_check():
    """Check if model is ready"""
    return {
        "status": "ready" if _warmed_up else "warming_up",
        "model_loaded": _warmed_up,
        "message": "Model is ready for predictions" if _warmed_up else "Model is warming up, please wait..."
    }

@app.post("/predict", tags=["AI Inference"])
async def predict(
    file: UploadFile = File(..., description="Tải lên ảnh lá cây (JPG/PNG)")
):
    # 1. Ghi lại thời gian bắt đầu
    start_time = time.perf_counter()
    
    # 2. Đọc dữ liệu ảnh từ request
    image_bytes = await file.read()
    
    # 3. Gọi hàm xử lý AI (Kết quả trả về là một Dictionary từ model_helper)
    predict_disease = get_predict_disease()
    result = predict_disease(image_bytes)
    
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
            "preprocessing": "Resized to 224x224 (No external scaling, handled by Model)",
            "recommendation": result["advice"]  # Lấy lời khuyên từ result
        }
    }