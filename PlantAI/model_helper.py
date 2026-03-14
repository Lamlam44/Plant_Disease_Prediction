import numpy as np
import io
from PIL import Image
import tensorflow as tf
import os

# 1. Dictionary chứa danh sách nhãn cho từng loại cây
PLANT_CLASSES = {
    "rice": ["Bacterial leaf blight", "Brown spot", "Leaf smut"],
    "tomato": [
        "Tomato__Bacterial_spot", 
        "Tomato__Early_blight", 
        "Tomato__healthy", 
        "Tomato__Late_blight", 
        "Tomato__Yellow_Leaf_Curl_Virus"
    ],
    "corn": ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"] 
}

loaded_models = {}

def get_model(plant_type):
    plant_type = plant_type.lower()
    if plant_type in loaded_models:
        return loaded_models[plant_type]
    
    # Lưu ý: Nếu bạn đã 'cd PlantAI' thì đường dẫn chỉ cần 'models/...'
    # Nếu chưa cd thì để 'PlantAI/models/...'
    model_path = f"PlantAI/models/{plant_type}_model.h5"
    if not os.path.exists(model_path):
        model_path = f"models/{plant_type}_model.h5"

    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        loaded_models[plant_type] = model
        return model
    return None

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_disease(image_bytes, plant_type):
    plant_key = plant_type.lower()
    model = get_model(plant_key)
    
    if model is None:
        return {"error": f"Không tìm thấy file model cho cây {plant_type}"}

    processed_img = preprocess_image(image_bytes)
    predictions = model.predict(processed_img)
    index = np.argmax(predictions[0])
    confidence = float(np.max(predictions[0])) * 100
    
    class_list = PLANT_CLASSES.get(plant_key, [])
    label = class_list[index] if index < len(class_list) else "Unknown"
    
    advices = {
        "Bacterial leaf blight": "Sử dụng các loại thuốc chứa đồng hoặc kháng sinh nông nghiệp.",
        "Brown spot": "Bón thêm phân Kali và giữ mực nước ruộng ổn định.",
        "Leaf smut": "Vệ sinh đồng ruộng sạch sẽ sau thu hoạch.",
        "Tomato__Bacterial_spot": "Tránh tưới nước lên lá, sử dụng thuốc gốc đồng.",
        "Tomato__Early_blight": "Cắt bỏ lá già ở gốc, phun thuốc trừ nấm định kỳ.",
        "Tomato__healthy": "Cây đang phát triển tốt, tiếp tục duy trì chăm sóc!",
        "Tomato__Late_blight": "Tiêu hủy cây bệnh nặng, tránh để vườn quá ẩm ướt.",
        "Tomato__Yellow_Leaf_Curl_Virus": "Kiểm soát bọ phấn trắng - tác nhân truyền virus.",
        "Blight": "Bệnh héo lá: Vệ sinh đồng ruộng, tiêu hủy tàn dư cây bệnh và luân canh cây trồng.",
        "Common_Rust": "Bệnh rỉ sắt: Tránh trồng quá dày, bón phân cân đối và dùng thuốc trừ nấm khi cần.",
        "Gray_Leaf_Spot": "Bệnh đốm lá xám: Thoát nước tốt, tỉa bớt lá chân để tạo độ thông thoáng.",
        "Healthy": "Cây ngô đang rất khỏe mạnh, hãy duy trì chế độ dinh dưỡng hiện tại!"
    }
    
    # SỬA LẠI ĐOẠN RETURN NÀY ĐỂ KHỚP VỚI MAIN.PY
    return {
        "label": label,
        "score": f"{confidence:.2f}", # Chỉ trả về số dạng chuỗi, dấu % để main.py lo
        "advice": advices.get(label, "Tham vấn chuyên gia để có biện pháp xử lý chính xác.")
    }