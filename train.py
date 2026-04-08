import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
import os
import json
import gc

# ============================================================
# Plant Disease Prediction Model - Training Script
# Transfer Learning (EfficientNetV2B3) + Fine-tuning
# Optimized for 16GB RAM + RTX 3050 GPU
# ============================================================

print("=" * 70)
print("PLANT DISEASE PREDICTION - MODEL TRAINING")
print("=" * 70)

# ============================================================
# 1. MEMORY OPTIMIZATION FOR 16GB RAM + RTX 3050
# ============================================================
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Tắt layout optimizer: tránh lỗi "Size of values 0 does not match size of permutation 4"
# xảy ra khi EfficientNetV2 dropout (stateless_dropout/SelectV2) gặp NHWC→NCHW conversion
tf.config.optimizer.set_experimental_options({'layout_optimizer': False})

# ============================================================
# 2. CONFIGURATION
# ============================================================
IMG_HEIGHT = 300          # EfficientNetV2B3 default input size
IMG_WIDTH = 300           # EfficientNetV2B3 default input size
BATCH_SIZE = 16            # ⬇ Giảm từ 8 → 4 vì 300x300 tốn ~1.8x RAM hơn 224x224 / Lưu ý hãy chỉnh về 4 nếu RAM máy yếu
TRAIN_DIR = "./Datasets/train"
VAL_DIR = "./Datasets/valid"

# Phase 1: Train head only (base frozen)
PHASE1_EPOCHS = 10
PHASE1_LR = 0.001

# Phase 2: Fine-tune top layers of base model
PHASE2_EPOCHS = 8
PHASE2_LR = 0.00001
FINE_TUNE_PERCENT = 0.20  # Unfreeze 20% layers cuối của base model

EARLY_STOP_PATIENCE = 4
MODEL_DIR = "app/models"

# ============================================================
# 3. LOAD DATASETS
# ============================================================
print("\n[STEP 1/6] Loading training dataset...")
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

print("[STEP 2/6] Loading validation dataset...")
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    seed=123,
    image_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    label_mode='int'
)

# CRITICAL: Lấy class_names trực tiếp từ dataset (đảm bảo khớp label index)
class_names = train_ds.class_names
NUM_CLASSES = len(class_names)

num_train = len(tf.io.gfile.glob(os.path.join(TRAIN_DIR, '*', '*')))
num_val = len(tf.io.gfile.glob(os.path.join(VAL_DIR, '*', '*')))

print(f"  Training images: {num_train}")
print(f"  Validation images: {num_val}")
print(f"  Total classes: {NUM_CLASSES}")

# ============================================================
# 4. DATA AUGMENTATION
# ============================================================
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# ============================================================
# 5. OPTIMIZE DATA PIPELINE
# ============================================================
print("\n[STEP 3/6] Optimizing data pipeline...")

# Áp dụng augmentation qua map() thay vì đặt trong model
# → tránh lỗi layout optimizer với EfficientNetV2 dropout layers
@tf.function
def augment_train(image, label):
    return data_augmentation(image, training=True), label

# Shuffle buffer 100 batches: 100×16×300×300×3×4 ≈ 1.7GB (an toàn cho Colab T4)
# (500 batches cũ ≈ 8.6 GB → gây lỗi "could not allocate pinned host memory")
train_ds = train_ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE) \
                   .shuffle(buffer_size=100, reshuffle_each_iteration=True) \
                   .prefetch(buffer_size=tf.data.AUTOTUNE)

val_ds = val_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

print("  - Augmentation: applied via map() in data pipeline")
print("  - Shuffle buffer: 100 batches (giảm từ 500, tránh OOM pinned memory)")
print("  - Prefetch: AUTOTUNE")

# ============================================================
# 6. BUILD MODEL (Transfer Learning - EfficientNetV2B3)
# ============================================================
print("\n[STEP 4/6] Building model architecture...")

# EfficientNetV2B3: include_preprocessing=True → model tự rescale [0,255] → [-1,1]
# KHÔNG cần thêm layer Rescaling thủ công
base_model = tf.keras.applications.EfficientNetV2B3(
    input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    include_top=False,
    weights='imagenet',
    include_preprocessing=True   # Built-in Rescaling: [0,255] → [-1,1]
)

# Phase 1: Freeze toàn bộ base model
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
    # Augmentation đã chuyển sang data pipeline (map()), không đặt trong model
    # → tránh layout optimizer crash với EfficientNetV2B3 stateless_dropout
    # KHÔNG cần Rescaling — EfficientNetV2B3 đã có sẵn (include_preprocessing=True)
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(256, activation='relu'),  # 256 neurons — EfficientNetV2B3 trích xuất features phong phú hơn
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE1_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print(f"  - Base model: EfficientNetV2B3 (frozen, {len(base_model.layers)} layers)")
print(f"  - Input size: {IMG_HEIGHT}x{IMG_WIDTH} (EfficientNetV2B3 default)")
print(f"  - Preprocessing: Built-in Rescaling [0,255] → [-1,1]")
print(f"  - Trainable params: {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")
print(f"  - Total params: {model.count_params():,}")

# ============================================================
# 7. CALLBACKS
# ============================================================
os.makedirs(MODEL_DIR, exist_ok=True)

checkpoint = callbacks.ModelCheckpoint(
    filepath=os.path.join(MODEL_DIR, "best_model.keras"),
    monitor='val_accuracy',
    save_best_only=True,
    verbose=1
)

early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=EARLY_STOP_PATIENCE,
    verbose=1,
    restore_best_weights=True
)

reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-7
)

callback_list = [checkpoint, early_stop, reduce_lr]

# ============================================================
# 8. PHASE 1: TRAIN HEAD (base model frozen)
# ============================================================
print("\n" + "=" * 70)
print("PHASE 1: TRAINING HEAD (base frozen)")
print(f"  - Epochs: {PHASE1_EPOCHS}")
print(f"  - Batch size: {BATCH_SIZE}")
print(f"  - Learning rate: {PHASE1_LR}")
print("=" * 70 + "\n")

history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=PHASE1_EPOCHS,
    callbacks=callback_list,
    verbose=1
)

# ============================================================
# 9. PHASE 2: FINE-TUNE TOP LAYERS OF BASE MODEL
# ============================================================
print("\n" + "=" * 70)
print("PHASE 2: FINE-TUNING (unfreeze top layers)")
print("=" * 70 + "\n")

# Giải phóng RAM trước khi vào Phase 2
gc.collect()
print("  - Garbage collected before fine-tuning")

# Unfreeze top layers of base model (tính toán động theo % tổng layers)
base_model.trainable = True
fine_tune_from = int(len(base_model.layers) * (1 - FINE_TUNE_PERCENT))
for layer in base_model.layers[:fine_tune_from]:
    layer.trainable = False

print(f"  - Unfreeze from layer {fine_tune_from}/{len(base_model.layers)} ({FINE_TUNE_PERCENT*100:.0f}% cuối)")
print(f"  - Additional epochs: {PHASE2_EPOCHS}")
print(f"  - Learning rate: {PHASE2_LR} (10x lower)")

# Recompile với learning rate thấp hơn để không phá weights đã train
model.compile(
    optimizer=optimizers.Adam(learning_rate=PHASE2_LR),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

total_epochs = PHASE1_EPOCHS + PHASE2_EPOCHS
initial_epoch = len(history1.history['loss'])

print(f"  - Trainable params (after unfreeze): {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=initial_epoch,
    callbacks=callback_list,
    verbose=1
)

# ============================================================
# 10. SAVE MODEL AND RESULTS
# ============================================================
print("\n" + "=" * 70)
print("SAVING MODEL...")
print("=" * 70)

# Lưu class names mapping (lấy từ dataset, đảm bảo đúng thứ tự)
class_mapping = {i: name for i, name in enumerate(class_names)}
with open(os.path.join(MODEL_DIR, 'class_names.json'), 'w', encoding='utf-8') as f:
    json.dump(class_mapping, f, indent=4, ensure_ascii=False)

# Lưu model format .keras (khuyến nghị TF 2.x) + backup .h5 cho compatibility
model.save(os.path.join(MODEL_DIR, 'plant_disease_model.keras'))
model.save(os.path.join(MODEL_DIR, 'plant_disease_model.h5'))

print(f"  ✅ Model saved: {MODEL_DIR}/plant_disease_model.keras")
print(f"  ✅ Model saved: {MODEL_DIR}/plant_disease_model.h5 (backup)")
print(f"  ✅ Classes saved: {MODEL_DIR}/class_names.json")
print(f"  ✅ Best checkpoint: {MODEL_DIR}/best_model.keras")

# ============================================================
# 11. TRAINING SUMMARY
# ============================================================

# Gộp history từ 2 phase
full_history = {}
for key in history1.history:
    full_history[key] = history1.history[key] + history2.history[key]

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)

print(f"\n  Total Epochs: {len(full_history['loss'])}")
print(f"    Phase 1 (head only):   {len(history1.history['loss'])} epochs")
print(f"    Phase 2 (fine-tuned):  {len(history2.history['loss'])} epochs")

print(f"\n  Phase 1 Results:")
print(f"    Train Acc: {history1.history['accuracy'][-1]*100:.2f}%")
print(f"    Val Acc:   {history1.history['val_accuracy'][-1]*100:.2f}%")

print(f"\n  Phase 2 (Final) Results:")
print(f"    Train Loss: {history2.history['loss'][-1]:.4f}")
print(f"    Train Acc:  {history2.history['accuracy'][-1]*100:.2f}%")
print(f"    Val Loss:   {history2.history['val_loss'][-1]:.4f}")
print(f"    Val Acc:    {history2.history['val_accuracy'][-1]*100:.2f}%")

best_val_acc = max(full_history['val_accuracy'])
print(f"\n  🏆 Best Val Accuracy: {best_val_acc*100:.2f}%")

print("\n" + "=" * 70)
print("Ready for predictions!")
print("=" * 70)

gc.collect()