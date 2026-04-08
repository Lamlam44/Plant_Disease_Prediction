import numpy as np
import io
import json
import os
from PIL import Image
import tensorflow as tf

from ..config import MODEL_PATH_KERAS, MODEL_PATH_H5, CLASS_NAMES_PATH, IMG_HEIGHT, IMG_WIDTH

# TensorFlow optimizations
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'

tf.config.run_functions_eagerly(False)

_model = None
_class_names = None
_is_ready = False


def load_model():
    global _model, _class_names, _is_ready

    if _is_ready:
        return

    # Load class names — hỗ trợ cả định dạng list [] và dict {}
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
            if isinstance(class_mapping, list):
                _class_names = {i: v for i, v in enumerate(class_mapping)}
            else:
                _class_names = {int(k): v for k, v in class_mapping.items()}
    else:
        _class_names = {}

    # Load model — prefer .keras, fallback .h5
    if os.path.exists(MODEL_PATH_KERAS):
        _model = tf.keras.models.load_model(MODEL_PATH_KERAS)
        _is_ready = True
    elif os.path.exists(MODEL_PATH_H5):
        _model = tf.keras.models.load_model(MODEL_PATH_H5)
        _is_ready = True


def warmup():
    load_model()
    if _model is None:
        return False

    dummy_img = Image.new('RGB', (IMG_WIDTH, IMG_HEIGHT), color=(128, 128, 128))
    dummy_bytes = io.BytesIO()
    dummy_img.save(dummy_bytes, format='PNG')
    dummy_bytes.seek(0)

    try:
        predict_batch_from_bytes([dummy_bytes.getvalue()])
        return True
    except Exception:
        return False


def is_ready() -> bool:
    return _is_ready


def preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Tiền xử lý ảnh cho EfficientNetV2B3.
    Model đã có built-in Rescaling (include_preprocessing=True)
    nên chỉ cần resize và chuyển sang float32 [0, 255].
    """
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)  
    return np.expand_dims(img_array, axis=0)


def get_advice(label: str) -> str:
    label_lower = label.lower()

    if "healthy" in label_lower:
        return "Cây đang phát triển tốt, tiếp tục duy trì chế độ chăm sóc hiện tại!"
    elif "blight" in label_lower or "scorch" in label_lower:
        return "Kiểm soát độ ẩm, tỉa bỏ lá bệnh, phun thuốc diệt nấm phòng trừ theo định kỳ."
    elif "spot" in label_lower or "measles" in label_lower:
        return "Bệnh do nấm/vi khuẩn: Không dùng tưới phun sương lên lá trực tiếp, dùng thuốc gốc đồng."
    elif "rust" in label_lower:
        return "Sử dụng thuốc trừ nấm rỉ sắt, trồng đúng khoảng cách để đảm bảo thông thoáng."
    elif "virus" in label_lower or "mosaic" in label_lower:
        return "Bệnh viêm/virus: Tiêu hủy cây bệnh để tránh lây lan, diệt côn trùng truyền bệnh (bọ phấn, rệp)."
    elif "mildew" in label_lower:
        return "Bệnh phấn trắng: Giảm bón phân đạm, tăng ánh sáng, dùng thuốc lưu dẫn chữa phấn trắng."
    elif "scab" in label_lower:
        return "Bệnh vảy/ghẻ: Dọn sạch tàn dư lá rụng, tưới nước vừa đủ và dùng thuốc trừ nấm bảo vệ."
    elif "greening" in label_lower or "haunglongbing" in label_lower:
        return "Bệnh vàng lá greening: Tiêu hủy cây bệnh, kiểm soát rầy chổng cánh, trồng cây giống sạch bệnh."

    return "Tham vấn chuyên gia nông nghiệp hoặc người có chuyên môn."


def _format_prediction(predictions_array, index: int) -> dict:
    pred = predictions_array[index]
    class_index = int(np.argmax(pred))
    confidence = float(np.max(pred)) * 100
    label = _class_names.get(class_index, "Unknown")
    advice = get_advice(label)
    return {
        "label": label.replace("___", " - ").replace("_", " "),
        "score": f"{confidence:.2f}",
        "advice": advice
    }


def predict_batch_from_bytes(images_bytes_list: list[bytes]) -> list[dict]:
    load_model()

    if _model is None:
        return [{"error": "Không tìm thấy file model: plant_disease_model.h5"}] * len(images_bytes_list)
    if not _class_names:
        return [{"error": "Không tìm thấy file mapping class_names.json"}] * len(images_bytes_list)

    batch = np.concatenate(
        [preprocess_image(img_bytes) for img_bytes in images_bytes_list], axis=0
    )
    predictions = _model.predict(batch, verbose=0)
    return [_format_prediction(predictions, i) for i in range(len(images_bytes_list))]
