import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, predict
from .services import model_service

logger = logging.getLogger("uvicorn.info")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[STARTUP] Đang khởi động ứng dụng và nạp model (Warming up)...")

    success = model_service.warmup()
    
    if success:
        logger.info("[STARTUP] ✅ Model đã load thành công và sẵn sàng!")
    else:
        logger.warning("[STARTUP] ⚠️ Warmup thất bại - Có thể thiếu file weights model hoặc lỗi đọc file!")
        
    yield
    
    logger.info("[SHUTDOWN] Server đang tắt và giải phóng tài nguyên.")

app = FastAPI(
    title="Plant Disease Diagnosis API",
    description=(
        "## Hệ thống nhận diện bệnh cây trồng qua ảnh AI 🍃\n"
        "Hệ thống hỗ trợ chuẩn đoán **38 loại bệnh** trên **14 loại cây trồng** khác nhau.\n\n"
        "### 📌 Phương thức nhận diện:\n"
        "* **Endpoint:** `/predict/batch`\n"
        "* **Chức năng:** Nhận diện hàng loạt ảnh cùng lúc.\n"
        "* **Giới hạn:** Tối đa 10 ảnh cho mỗi lượt request."
    ),
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

app.include_router(health.router)
app.include_router(predict.router)