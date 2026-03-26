import os

# Base directory of the app package
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model files — prefer .keras format, fallback to .h5
MODEL_PATH_KERAS = os.path.join(BASE_DIR, "models", "plant_disease_model.keras")
MODEL_PATH_H5 = os.path.join(BASE_DIR, "models", "plant_disease_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "models", "class_names.json")

# Image preprocessing
IMG_HEIGHT = 224
IMG_WIDTH = 224

# Allowed image extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Max file size (10MB)
MAX_FILE_SIZE = 10 * 1024 * 1024

# Max batch size for multi-file upload
MAX_BATCH_SIZE = 10
