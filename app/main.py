from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import health, predict
from .services import model_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load & warmup model
    print("\n[STARTUP] Đang khởi động model...")
    success = model_service.warmup()
    if success:
        print("[STARTUP] ✅ Model ready!")
    else:
        print("[STARTUP] ⚠️ Warmup failed - model files may be missing")
    yield
    # Shutdown
    print("[SHUTDOWN] Server stopped.")


app = FastAPI(
    title="Plant Disease Diagnosis API",
    description=(
        "## Hệ thống nhận diện bệnh cây trồng qua ảnh\n\n"
        "Hỗ trợ 38 loại bệnh trên 14 loại cây trồng.\n\n"
        "### Phương thức nhận diện:\n"
        "- **Upload ảnh (đơn)** — `/predict/single` — kiểm tra nhanh 1 mẫu lá\n"
        "- **Upload ảnh (nhiều)** — `/predict/batch` — nhận diện hàng loạt (tối đa 10 ảnh)"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(predict.router)
