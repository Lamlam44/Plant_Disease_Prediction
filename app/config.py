from pathlib import Path

class AppConfig:
    # Path settings
    BASE_DIR = Path(__file__).resolve().parent
    MODELS_DIR = BASE_DIR / "models"
    
    # Model Asset Paths
    MODEL_KERAS = MODELS_DIR / "plant_disease_model.keras"
    MODEL_H5 = MODELS_DIR / "plant_disease_model.h5"
    CLASS_MAPPING = MODELS_DIR / "class_names.json"

    # Model Input Dimensions (EfficientNetV2B3)
    INPUT_SHAPE = (300, 300) 
    
    # API Validation Rules
    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    FILE_LIMIT = 10 * 1024 * 1024  # 10MB
    BATCH_LIMIT = 10

# Export instances for easy import in other modules
config = AppConfig()