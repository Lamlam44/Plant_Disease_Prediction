import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import json
import gc

# ============================================================
# Plant Disease Prediction Model - Training Script
# Optimized for 16GB RAM systems
# ============================================================

print("=" * 70)
print("PLANT DISEASE PREDICTION - MODEL TRAINING")
print("=" * 70)

# ============================================================
# 1. MEMORY OPTIMIZATION FOR 16GB RAM
# ============================================================
# Limit GPU memory growth if available
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# Set memory optimization flags
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# ============================================================
# 2. CONFIGURATION (Optimized for 16GB RAM)
# ============================================================
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 12  # Optimized: 8-16 is ideal for 16GB RAM
TRAIN_DIR = "./Datasets/train" 
VAL_DIR = "./Datasets/valid"
NUM_EPOCHS = 15  # Balanced between training quality and time
LEARNING_RATE = 0.0001
EARLY_STOP_PATIENCE = 2

# Get number of classes
NUM_CLASSES = len(os.listdir(TRAIN_DIR))
print(f"\n[INFO] Detected {NUM_CLASSES} plant disease classes")

# ============================================================
# 3. LOAD DATASETS (Memory-efficient approach)
# ============================================================
print("\n[STEP 1/5] Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

print("[STEP 2/5] Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# Get class names for later predictions
class_names = sorted(os.listdir(TRAIN_DIR))
num_train = len(tf.data.Dataset.from_tensor_slices(
    tf.io.gfile.glob(os.path.join(TRAIN_DIR, '*', '*'))))
num_val = len(tf.data.Dataset.from_tensor_slices(
    tf.io.gfile.glob(os.path.join(VAL_DIR, '*', '*'))))

print(f"  Training images: {num_train}")
print(f"  Validation images: {num_val}")
print(f"  Total classes: {len(class_names)}")

# ============================================================
# 4. OPTIMIZE DATA PIPELINE (Critical for memory efficiency)
# ============================================================
print("\n[STEP 3/5] Optimizing data pipeline...")

AUTOTUNE = tf.data.AUTOTUNE

# NO CACHE - to save RAM memory
# Only use prefetch and shuffle efficiently
train_ds = train_ds.shuffle(
    buffer_size=min(1000, num_train // BATCH_SIZE),
    reshuffle_each_iteration=True
).prefetch(buffer_size=AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

print("  - Shuffle buffer optimized")
print("  - Prefetch enabled")
print("  - Cache disabled (to preserve 16GB RAM)")

# ============================================================
# 5. DATA AUGMENTATION (Prevents overfitting with smaller batches)
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

# ============================================================
# 6. BUILD MODEL (Transfer Learning - MobileNetV2)
# ============================================================
print("\n[STEP 4/5] Building model architecture...")

# Load pre-trained MobileNetV2
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze base model to preserve ImageNet knowledge
base_model.trainable = False

# Build complete model
model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    data_augmentation,
    layers.Rescaling(1./127.5, offset=-1),  # MobileNetV2 normalization
    base_model,
    layers.GlobalAveragePooling2D(),  # More efficient than Flatten
    layers.Dropout(0.3),  # Slightly increased dropout for memory efficiency
    layers.Dense(128, activation='relu'),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Compile model
model.compile(
    optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  - Base model: MobileNetV2")
print(f"  - Trainable layers: {len(model.trainable_weights)}")
print(f"  - Total parameters: {model.count_params():,}")

# ============================================================
# 7. CALLBACKS (For training control and memory management)
# ============================================================
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOP_PATIENCE,
    verbose=1,
    restore_best_weights=True
)

# Reduce learning rate if validation loss plateaus
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    verbose=1,
    min_lr=1e-7
)

# ============================================================
# 8. TRAIN MODEL
# ============================================================
print("\n" + "=" * 70)
print("STARTING TRAINING")
print(f"  - Epochs: {NUM_EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Learning rate: {LEARNING_RATE}")
print(f"  - Early stopping patience: {EARLY_STOP_PATIENCE}")
print("=" * 70 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=NUM_EPOCHS,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# ============================================================
# 9. SAVE MODEL AND RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODEL...")
print("=" * 70)

# Create models directory
if not os.path.exists('PlantAI/models'):
    os.makedirs('PlantAI/models')

# Save class names mapping
class_mapping = {i: name for i, name in enumerate(class_names)}
with open('PlantAI/models/class_names.json', 'w') as f:
    json.dump(class_mapping, f, indent=4)

# Save model
model.save('PlantAI/models/plant_disease_model.h5')

# ============================================================
# 10. TRAINING SUMMARY
# ============================================================
print("\nTRAINING COMPLETED SUCCESSFULLY!")
print("\nModel Files:")
print(f"  - Model: PlantAI/models/plant_disease_model.h5")
print(f"  - Classes: PlantAI/models/class_names.json")

print("\nTraining Results:")
final_train_loss = history.history['loss'][-1]
final_train_acc = history.history['accuracy'][-1]
final_val_loss = history.history['val_loss'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print(f"  - Training Loss: {final_train_loss:.4f}")
print(f"  - Training Accuracy: {final_train_acc:.4f} ({final_train_acc*100:.2f}%)")
print(f"  - Validation Loss: {final_val_loss:.4f}")
print(f"  - Validation Accuracy: {final_val_acc:.4f} ({final_val_acc*100:.2f}%)")
print(f"  - Total Epochs: {len(history.history['loss'])}")

print("\n" + "=" * 70)
print("Ready for predictions!")
print("=" * 70)

# Clean up memory
gc.collect()