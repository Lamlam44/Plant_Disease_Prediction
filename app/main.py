from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

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
        "- **Upload ảnh (nhiều)** — `/predict/batch` — nhận diện hàng loạt (tối đa 10 ảnh)\n"
        "- **Chụp camera** — `/predict/camera` — chụp trực tiếp qua camera (Base64)\n\n"
        "### Demo camera:\n"
        "Truy cập [/camera-demo](/camera-demo) để dùng giao diện chụp ảnh qua camera."
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


# ─── Camera Demo Page ─────────────────────────────────────────────────────────
CAMERA_HTML = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Plant Disease - Camera Demo</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: 'Segoe UI', sans-serif; background: #f0f4f0; color: #333; }
        .container { max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { text-align: center; color: #2e7d32; margin-bottom: 8px; font-size: 1.5em; }
        .subtitle { text-align: center; color: #666; margin-bottom: 20px; font-size: 0.9em; }
        .video-wrap { position: relative; background: #000; border-radius: 12px; overflow: hidden; margin-bottom: 16px; }
        video, canvas { width: 100%; display: block; border-radius: 12px; }
        canvas { display: none; }
        .controls { display: flex; gap: 10px; justify-content: center; margin-bottom: 20px; flex-wrap: wrap; }
        button { padding: 12px 24px; border: none; border-radius: 8px; font-size: 1em;
                 cursor: pointer; font-weight: 600; transition: all 0.2s; }
        button:disabled { opacity: 0.5; cursor: not-allowed; }
        .btn-capture { background: #2e7d32; color: white; }
        .btn-capture:hover:not(:disabled) { background: #1b5e20; }
        .btn-retry { background: #f57c00; color: white; }
        .btn-retry:hover:not(:disabled) { background: #e65100; }
        .btn-switch { background: #1565c0; color: white; }
        .btn-switch:hover:not(:disabled) { background: #0d47a1; }
        .result-box { background: white; border-radius: 12px; padding: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
        .result-box h3 { color: #2e7d32; margin-bottom: 12px; }
        .result-item { display: flex; justify-content: space-between; padding: 8px 0;
                       border-bottom: 1px solid #eee; }
        .result-item:last-child { border-bottom: none; }
        .label { font-weight: 600; color: #555; }
        .value { color: #2e7d32; font-weight: 500; }
        .loading { text-align: center; padding: 20px; color: #666; }
        .error { color: #c62828; background: #ffebee; padding: 12px; border-radius: 8px; }
        .back-link { display: block; text-align: center; margin-top: 16px; color: #1565c0; }
        .preview-img { width: 100%; border-radius: 12px; margin-bottom: 16px; }
    </style>
</head>
<body>
<div class="container">
    <h1>🌿 Nhận diện bệnh cây trồng</h1>
    <p class="subtitle">Hướng camera vào lá cây, sau đó bấm "Chụp ảnh" để nhận diện</p>

    <div class="video-wrap">
        <video id="video" autoplay playsinline></video>
        <canvas id="canvas"></canvas>
    </div>

    <img id="preview" class="preview-img" style="display:none" />

    <div class="controls">
        <button id="btnCapture" class="btn-capture" onclick="capture()">📸 Chụp ảnh</button>
        <button id="btnRetry" class="btn-retry" onclick="retry()" style="display:none">🔄 Chụp lại</button>
        <button id="btnSwitch" class="btn-switch" onclick="switchCamera()">🔁 Đổi camera</button>
    </div>

    <div id="result"></div>
    <a href="/docs" class="back-link">← Quay về Swagger UI</a>
</div>

<script>
    let stream = null;
    let facingMode = "environment";

    async function startCamera() {
        if (stream) stream.getTracks().forEach(t => t.stop());
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: { facingMode: facingMode, width: { ideal: 1280 }, height: { ideal: 720 } }
            });
            document.getElementById("video").srcObject = stream;
        } catch (e) {
            document.getElementById("result").innerHTML =
                '<div class="error">Không thể truy cập camera. Vui lòng cấp quyền camera.</div>';
        }
    }

    function switchCamera() {
        facingMode = facingMode === "environment" ? "user" : "environment";
        startCamera();
    }

    function capture() {
        const video = document.getElementById("video");
        const canvas = document.getElementById("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);

        const dataUrl = canvas.toDataURL("image/jpeg", 0.9);
        document.getElementById("preview").src = dataUrl;
        document.getElementById("preview").style.display = "block";
        video.style.display = "none";
        document.getElementById("btnCapture").style.display = "none";
        document.getElementById("btnRetry").style.display = "inline-block";

        sendToAPI(dataUrl);
    }

    function retry() {
        document.getElementById("preview").style.display = "none";
        document.getElementById("video").style.display = "block";
        document.getElementById("btnCapture").style.display = "inline-block";
        document.getElementById("btnRetry").style.display = "none";
        document.getElementById("result").innerHTML = "";
    }

    async function sendToAPI(dataUrl) {
        document.getElementById("result").innerHTML = '<div class="loading">⏳ Đang nhận diện...</div>';
        try {
            const base64 = dataUrl.split(",")[1];
            const res = await fetch("/predict/camera", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image_base64: base64 })
            });
            const json = await res.json();
            if (json.status === "success") {
                document.getElementById("result").innerHTML = `
                    <div class="result-box">
                        <h3>✅ Kết quả nhận diện</h3>
                        <div class="result-item"><span class="label">Bệnh / Trạng thái</span><span class="value">${json.data.label}</span></div>
                        <div class="result-item"><span class="label">Độ tin cậy</span><span class="value">${json.data.confidence}</span></div>
                        <div class="result-item"><span class="label">Thời gian xử lý</span><span class="value">${json.data.inference_time}</span></div>
                        <div class="result-item"><span class="label">Tiền xử lý</span><span class="value">${json.data.preprocessing}</span></div>
                        <div class="result-item"><span class="label">Khuyến nghị</span><span class="value">${json.data.recommendation}</span></div>
                    </div>`;
            } else {
                document.getElementById("result").innerHTML = `<div class="error">❌ ${json.message || "Lỗi không xác định"}</div>`;
            }
        } catch (e) {
            document.getElementById("result").innerHTML = `<div class="error">❌ Lỗi kết nối: ${e.message}</div>`;
        }
    }

    startCamera();
</script>
</body>
</html>
"""


@app.get("/camera-demo", response_class=HTMLResponse, include_in_schema=False)
async def camera_demo():
    return CAMERA_HTML
