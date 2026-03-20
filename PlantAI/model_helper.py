import numpy as np
import io
import json
from PIL import Image
import tensorflow as tf
import os

# TensorFlow optimizations
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress info logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '1'  # Enable optimizations

# Optimize TF for inference (not training)
tf.config.run_functions_eagerly(False)
tf.data.experimental.enable_debug_mode = False

GLOBAL_MODEL = None
CLASS_NAMES = None

# Get the directory where this file is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def load_resources():
    global GLOBAL_MODEL, CLASS_NAMES
    if GLOBAL_MODEL is not None and CLASS_NAMES is not None:
        return

    # Load class names that were generated during training
    class_names_path = os.path.join(BASE_DIR, "models", "class_names.json")
    
    if os.path.exists(class_names_path):
        with open(class_names_path, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
            # Chuyển key string từ JSON thành int
            CLASS_NAMES = {int(k): v for k, v in class_mapping.items()}
    else:
        CLASS_NAMES = {}

    # Load the single combined model
    model_path = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")

    if os.path.exists(model_path):
        GLOBAL_MODEL = tf.keras.models.load_model(model_path)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    # QUAN TRỌNG: Không chia cho 255.0 vì model Transfer Learning (MobileNetV2) 
    # đã được tích hợp sẵn layer chuẩn hóa: `layers.Rescaling(1./127.5, offset=-1)`
    img_array = np.array(img)
    return np.expand_dims(img_array, axis=0)

def predict_disease(image_bytes):
    # Đảm bảo model và class_names đã được lặp (chỉ thực hiện 1 lần đầu tiên)
    load_resources()
    
    if GLOBAL_MODEL is None:
        return {"error": "Không tìm thấy file model: plant_disease_model.h5"}
    if not CLASS_NAMES:
        return {"error": "Không tìm thấy file mapping class_names.json"}

    # Tiền xử lý
    processed_img = preprocess_image(image_bytes)
    
    # Dự đoán
    predictions = GLOBAL_MODEL.predict(processed_img)
    index = int(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0])) * 100
    
    # Trích xuất label từ list (Ví dụ: "Tomato___Early_blight")
    label = CLASS_NAMES.get(index, "Unknown")
    
    # Lời khuyên mở rộng cho các loại bệnh (Dùng chuỗi khớp một phần để bắt nhiều bệnh hơn)
    advice_msg = "Tham vấn chuyên gia nông nghiệp hoặc người có chuyên môn."
    label_lower = label.lower()
    
    if "healthy" in label_lower:
        advice_msg = "Cây đang phát triển tốt, tiếp tục duy trì chế độ chăm sóc hiện tại!"
    elif "blight" in label_lower or "scorch" in label_lower:
        advice_msg = "Kiểm soát độ ẩm, tỉa bỏ lá bệnh, phun thuốc diệt nấm phòng trừ theo định kỳ."
    elif "spot" in label_lower or "measles" in label_lower:
        advice_msg = "Bệnh do nấm/vi khuẩn: Không dùng tưới phun sương lên lá trực tiếp, dùng thuốc gốc đồng."
    elif "rust" in label_lower:
        advice_msg = "Sử dụng thuốc trừ nấm rỉ sắt, trồng đúng khoảng cách để đảm bảo thông thoáng."
    elif "virus" in label_lower or "mosaic" in label_lower:
        advice_msg = "Bệnh viêm/virus: Tiêu hủy cây bệnh để tránh lây lan, diệt côn trùng truyền bệnh (bọ phấn, rệp)."
    elif "mildew" in label_lower:
        advice_msg = "Bệnh phấn trắng: Giảm bón phân đạm, tăng ánh sáng, dùng thuốc lưu dẫn chữa phấn trắng."
    elif "scab" in label_lower:
        advice_msg = "Bệnh vảy/ghẻ: Dọn sạch tàn dư lá rụng, tưới nước vừa đủ và dùng thuốc trừ nấm bảo vệ."
    
    return {
        "label": label.replace("___", " - ").replace("_", " "), # Format lại chuỗi cho đẹp
        "score": f"{confidence:.2f}",
        "advice": advice_msg
    }