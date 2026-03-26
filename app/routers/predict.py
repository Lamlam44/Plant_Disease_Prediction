import time
import base64
import asyncio
from functools import partial
from fastapi import APIRouter, File, UploadFile, HTTPException

from ..schemas.prediction import (
    PredictionResponse,
    PredictionData,
    BatchPredictionResponse,
    BatchPredictionItem,
    CameraRequest,
)
from ..services import model_service
from ..config import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MAX_BATCH_SIZE

router = APIRouter(prefix="/predict", tags=["AI Inference"])


def _validate_extension(filename: str) -> bool:
    if not filename:
        return False
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in ALLOWED_EXTENSIONS


# ─── Endpoint 1: Tải lên 1 ảnh ───────────────────────────────────────────────
@router.post(
    "/single",
    response_model=PredictionResponse,
    summary="Nhận diện bệnh cây trồng (1 ảnh)",
    description="Tải lên một ảnh lá cây (JPG/PNG) để nhận diện bệnh. Phù hợp khi "
                "người dùng muốn kiểm tra nhanh một mẫu lá cụ thể."
)
async def predict_single(
    file: UploadFile = File(..., description="Ảnh lá cây (JPG/PNG/WEBP, tối đa 10MB)")
):
    if not _validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Định dạng file không hỗ trợ. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    start_time = time.perf_counter()
    image_bytes = await file.read()

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="File vượt quá 10MB")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(model_service.predict_from_bytes, image_bytes))

    if "error" in result:
        return PredictionResponse(status="error", message=result["error"])

    inference_time = f"{time.perf_counter() - start_time:.4f}s"

    return PredictionResponse(
        status="success",
        data=PredictionData(
            label=result["label"],
            confidence=f"{result['score']}%",
            inference_time=inference_time,
            recommendation=result["advice"]
        )
    )


# ─── Endpoint 2: Tải lên nhiều ảnh ───────────────────────────────────────────
@router.post(
    "/batch",
    response_model=BatchPredictionResponse,
    summary="Nhận diện bệnh cây trồng (nhiều ảnh)",
    description=f"Tải lên tối đa {MAX_BATCH_SIZE} ảnh cùng lúc để nhận diện bệnh cho "
                "nhiều mẫu lá. Tất cả ảnh được xử lý cùng lúc (batch predict) để tối ưu tốc độ."
)
async def predict_batch(
    files: list[UploadFile] = File(
        ..., description=f"Danh sách ảnh lá cây (tối đa {MAX_BATCH_SIZE} file)"
    )
):
    if len(files) > MAX_BATCH_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"Tối đa {MAX_BATCH_SIZE} ảnh mỗi lần. Nhận được: {len(files)}"
        )

    start_time = time.perf_counter()

    # Phase 1: Validate & read all files
    valid_indices = []       # indices of valid files in original list
    valid_bytes = []         # image bytes of valid files
    results = [None] * len(files)  # pre-allocate result slots

    for i, file in enumerate(files):
        if not _validate_extension(file.filename):
            results[i] = BatchPredictionItem(
                filename=file.filename or "unknown",
                status="error",
                message=f"Định dạng không hỗ trợ. Chấp nhận: {', '.join(ALLOWED_EXTENSIONS)}"
            )
            continue

        image_bytes = await file.read()

        if len(image_bytes) > MAX_FILE_SIZE:
            results[i] = BatchPredictionItem(
                filename=file.filename or "unknown",
                status="error",
                message="File vượt quá 10MB"
            )
            continue

        valid_indices.append(i)
        valid_bytes.append(image_bytes)

    # Phase 2: Batch predict all valid images in one model call
    if valid_bytes:
        loop = asyncio.get_event_loop()
        batch_results = await loop.run_in_executor(
            None, partial(model_service.predict_batch_from_bytes, valid_bytes)
        )

        for idx, pred in zip(valid_indices, batch_results):
            filename = files[idx].filename or "unknown"
            if "error" in pred:
                results[idx] = BatchPredictionItem(
                    filename=filename, status="error", message=pred["error"]
                )
            else:
                results[idx] = BatchPredictionItem(
                    filename=filename,
                    status="success",
                    data=PredictionData(
                        label=pred["label"],
                        confidence=f"{pred['score']}%",
                        inference_time=f"{time.perf_counter() - start_time:.4f}s",
                        recommendation=pred["advice"]
                    )
                )

    success_count = sum(1 for r in results if r and r.status == "success")
    fail_count = len(files) - success_count

    return BatchPredictionResponse(
        status="success" if success_count > 0 else "error",
        total=len(files),
        successful=success_count,
        failed=fail_count,
        results=results
    )


# ─── Endpoint 3: Chụp ảnh từ camera (base64) ─────────────────────────────────
@router.post(
    "/camera",
    response_model=PredictionResponse,
    summary="Nhận diện bệnh qua camera (Base64)",
    description="Gửi ảnh chụp từ camera dưới dạng Base64 để nhận diện bệnh. "
                "Dùng khi tích hợp với giao diện camera trực tiếp. "
                "Truy cập /camera-demo để dùng giao diện chụp ảnh."
)
async def predict_camera(body: CameraRequest):
    start_time = time.perf_counter()

    # Remove data URI prefix if present
    image_data = body.image_base64
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]

    try:
        image_bytes = base64.b64decode(image_data)
    except Exception:
        raise HTTPException(status_code=400, detail="Base64 không hợp lệ")

    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(status_code=400, detail="Ảnh vượt quá 10MB")

    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, partial(model_service.predict_from_bytes, image_bytes))

    if "error" in result:
        return PredictionResponse(status="error", message=result["error"])

    inference_time = f"{time.perf_counter() - start_time:.4f}s"

    return PredictionResponse(
        status="success",
        data=PredictionData(
            label=result["label"],
            confidence=f"{result['score']}%",
            inference_time=inference_time,
            recommendation=result["advice"]
        )
    )
