"""
FER2013 Emotion Classification Training Script
------------------------------------------------
Loads the FER2013 dataset from data/train/ and data/test/,
builds a CNN, trains with class weighting, and saves results.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving plots
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight

print("TensorFlow version:", tf.__version__)

# ---------------------------------------------------------------------------
# 1. HYPERPARAMETERS
# ---------------------------------------------------------------------------
IMG_SIZE = (48, 48)
BATCH_SIZE = 64
EPOCHS = 50
NUM_CLASSES = 7
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# ---------------------------------------------------------------------------
# 2. LOAD DATASET
# ---------------------------------------------------------------------------
# image_dataset_from_directory reads images organized in subfolders by class.
# It automatically infers class names from folder names.
# label_mode="categorical" -> one-hot encoded labels.
train_ds = tf.keras.utils.image_dataset_from_directory(
    "data/train",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=True,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    "data/test",
    image_size=IMG_SIZE,
    color_mode="grayscale",
    batch_size=BATCH_SIZE,
    label_mode="categorical",
    shuffle=False,
)

# Get class names for reference
class_names = train_ds.class_names
print("Classes:", class_names)

# ---------------------------------------------------------------------------
# 3. COUNT IMAGES PER CLASS (for class weight computation)
# ---------------------------------------------------------------------------
train_counts = []
val_counts = []
for emotion in EMOTIONS:
    train_path = os.path.join("data/train", emotion)
    val_path = os.path.join("data/test", emotion)
    train_counts.append(len(os.listdir(train_path)))
    val_counts.append(len(os.listdir(val_path)))

print("\nTraining samples per class:")
for i, e in enumerate(EMOTIONS):
    print(f"  {e:>10}: {train_counts[i]}")

# ---------------------------------------------------------------------------
# 4. COMPUTE CLASS WEIGHTS
# ---------------------------------------------------------------------------
# sklearn's compute_class_weight with 'balanced' mode assigns higher weight
# to minority classes (e.g., disgust) and lower weight to majority classes
# (e.g., happy). This prevents the model from ignoring rare emotions.
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(NUM_CLASSES),
    y=np.repeat(np.arange(NUM_CLASSES), train_counts),
)
class_weight_dict = dict(enumerate(class_weights))
print("\nClass weights (minority classes get higher weight):")
for i, e in enumerate(EMOTIONS):
    print(f"  {e:>10}: {class_weight_dict[i]:.4f}")

# ---------------------------------------------------------------------------
# 5. DATA PIPELINE
# ---------------------------------------------------------------------------
# Normalize pixel values from [0, 255] to [0, 1] for stable training.
normalization_layer = layers.Rescaling(1.0 / 255)

# Data augmentation helps prevent overfitting by generating slightly
# modified versions of training images. Only applied to training data.
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Apply normalization and augmentation to training data
def prepare_train(images, labels):
    images = normalization_layer(images)
    images = data_augmentation(images, training=True)
    return images, labels

def prepare_val(images, labels):
    images = normalization_layer(images)
    return images, labels

# Prefetch and cache for faster data loading
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.map(prepare_train, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)
val_ds = val_ds.map(prepare_val, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

# ---------------------------------------------------------------------------
# 6. MODEL DEFINITION / CHECKPOINT LOADING
# ---------------------------------------------------------------------------
MODEL_PATH = "models/best_model.keras"
EPOCH_TRACK_PATH = "models/checkpoint_epoch.txt"
resuming = False
initial_epoch = 0

if os.path.exists(MODEL_PATH):
    print("\nResuming from checkpoint: loading models/best_model.keras ...")
    model = tf.keras.models.load_model(MODEL_PATH)
    resuming = True
    if os.path.exists(EPOCH_TRACK_PATH):
        with open(EPOCH_TRACK_PATH) as f:
            initial_epoch = int(f.read().strip())
    print(f"Resumed optimizer learning rate: {float(model.optimizer.learning_rate):.6f}")
else:
    print("\nNo checkpoint found. Building model from scratch ...")
    model = keras.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        # Block 1
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 2
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Block 3
        layers.Conv2D(256, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),
        # Classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

model.summary()

# ---------------------------------------------------------------------------
# 7. CALLBACKS
# ---------------------------------------------------------------------------
class EpochTracker(keras.callbacks.Callback):
    def __init__(self, filepath):
        self.filepath = filepath
    def on_epoch_end(self, epoch, logs=None):
        with open(self.filepath, "w") as f:
            f.write(str(epoch + 1))

callbacks = [
    # Save the model with best validation accuracy
    keras.callbacks.ModelCheckpoint(
        MODEL_PATH,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    ),
    EpochTracker(EPOCH_TRACK_PATH),
    # Stop training if validation loss doesn't improve for 8 epochs
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=8,
        restore_best_weights=True,
        verbose=1,
    ),
    # Halve learning rate if validation loss plateaus for 4 epochs
    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=4,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ---------------------------------------------------------------------------
# 8. TRAIN
# ---------------------------------------------------------------------------
if resuming:
    print(f"\nResuming training from epoch {initial_epoch}/{EPOCHS} ...\n")
else:
    print("\nStarting training from scratch ...\n")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    initial_epoch=initial_epoch,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1,
)

# ---------------------------------------------------------------------------
# 10. PLOT TRAINING HISTORY
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Accuracy plot
ax1.plot(history.history["accuracy"], label="Train Accuracy")
ax1.plot(history.history["val_accuracy"], label="Val Accuracy")
ax1.set_title("Accuracy vs Epochs")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Accuracy")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Loss plot
ax2.plot(history.history["loss"], label="Train Loss")
ax2.plot(history.history["val_loss"], label="Val Loss")
ax2.set_title("Loss vs Epochs")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("models/training_history.png", dpi=150)
print("\nTraining history plot saved to models/training_history.png")

# ---------------------------------------------------------------------------
# 11. FINAL RESULTS
# ---------------------------------------------------------------------------
final_val_acc = history.history["val_accuracy"][-1]
final_val_loss = history.history["val_loss"][-1]
print(f"\nFinal Validation Accuracy: {final_val_acc:.4f}")
print(f"Final Validation Loss:     {final_val_loss:.4f}")

# Also show the best validation accuracy
best_idx = int(np.argmax(history.history["val_accuracy"]))
best_val_acc = history.history["val_accuracy"][best_idx]
print(f"Best Validation Accuracy:  {best_val_acc:.4f} (epoch {best_idx + 1})")
