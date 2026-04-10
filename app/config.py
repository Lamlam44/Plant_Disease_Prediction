from pathlib import Path

# Base directory using Pathlib for modern path handling
BASE_DIR = Path(__file__).resolve().parent

# Model files configuration
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH_KERAS = MODELS_DIR / "plant_disease_model.keras"
MODEL_PATH_H5 = MODELS_DIR / "plant_disease_model.h5"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

# Image settings
IMG_HEIGHT = 300
IMG_WIDTH = 300

# Security & Constraints
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_BATCH_SIZE = 10