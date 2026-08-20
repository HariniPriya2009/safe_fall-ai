"""
STEP 5: Model Design and Training
----------------------------------
Loads the landmark dataset produced by 1_pose_landmark_extraction.py,
splits it into Train (70%) / Validation (15%) / Test (15%), and trains
a 1D-CNN deep learning classifier on the 132 pose-landmark features to
recognise: Fall Detected, Walking, Sitting, Standing, Normal Activity.

Outputs saved to the working directory:
    fall_detection_model.h5   -> trained Keras model
    label_encoder_classes.npy -> class name mapping
    feature_scaler.pkl        -> StandardScaler used on features
    test_data.npz             -> held-out test set (used by 3_evaluate_model.py)
    accuracy_plot.png / loss_plot.png -> training curves

Run this in Google Colab or Jupyter Notebook.

Install dependencies first (Colab cell):
    !pip install tensorflow scikit-learn pandas matplotlib joblib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, BatchNormalization, MaxPooling1D, Dropout, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
CSV_PATH = "landmarks_dataset.csv"
MODEL_OUTPUT = "fall_detection_model.h5"
SCALER_OUTPUT = "feature_scaler.pkl"
LABELS_OUTPUT = "label_encoder_classes.npy"
TEST_DATA_OUTPUT = "test_data.npz"
RANDOM_STATE = 42

# ----------------------------------------------------------------------
# 1. Load data
# ----------------------------------------------------------------------
df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} samples with classes: {df['label'].unique()}")

X = df.drop(columns=["label"]).values.astype("float32")   # shape (N, 132)
y_raw = df["label"].values

# ----------------------------------------------------------------------
# 2. Encode labels and scale features
# ----------------------------------------------------------------------
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_raw)
num_classes = len(label_encoder.classes_)
y_categorical = to_categorical(y_encoded, num_classes=num_classes)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for Conv1D: (samples, 132, 1)
X_reshaped = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)

# ----------------------------------------------------------------------
# 3. Split: 70% train / 15% validation / 15% test (stratified)
# ----------------------------------------------------------------------
X_train, X_temp, y_train, y_temp = train_test_split(
    X_reshaped, y_categorical, test_size=0.30,
    random_state=RANDOM_STATE, stratify=y_encoded
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50,
    random_state=RANDOM_STATE, stratify=np.argmax(y_temp, axis=1)
)

print(f"Train: {X_train.shape[0]} | Validation: {X_val.shape[0]} | Test: {X_test.shape[0]}")

# ----------------------------------------------------------------------
# 4. Build the 1D-CNN model
# ----------------------------------------------------------------------
model = Sequential([
    Conv1D(64, kernel_size=3, activation="relu", input_shape=(X_train.shape[1], 1)),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),

    Conv1D(128, kernel_size=3, activation="relu"),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.4),
    Dense(64, activation="relu"),
    Dense(num_classes, activation="softmax"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()

# ----------------------------------------------------------------------
# 5. Train
# ----------------------------------------------------------------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    ModelCheckpoint(MODEL_OUTPUT, monitor="val_accuracy", save_best_only=True),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60,
    batch_size=32,
    callbacks=callbacks,
    verbose=1,
)

# ----------------------------------------------------------------------
# 6. Save model, scaler, label classes, and test set
# ----------------------------------------------------------------------
model.save(MODEL_OUTPUT)
joblib.dump(scaler, SCALER_OUTPUT)
np.save(LABELS_OUTPUT, label_encoder.classes_)
np.savez(TEST_DATA_OUTPUT, X_test=X_test, y_test=y_test)

print(f"\nSaved model      -> {MODEL_OUTPUT}")
print(f"Saved scaler     -> {SCALER_OUTPUT}")
print(f"Saved label map  -> {LABELS_OUTPUT}")
print(f"Saved test split -> {TEST_DATA_OUTPUT}")

# ----------------------------------------------------------------------
# 7. Plot accuracy and loss curves (for FA-2 evidence screenshots)
# ----------------------------------------------------------------------
plt.figure(figsize=(7, 5))
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.title("Model Accuracy over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("accuracy_plot.png", dpi=150, bbox_inches="tight")
plt.close()

plt.figure(figsize=(7, 5))
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.title("Model Loss over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("loss_plot.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved accuracy_plot.png and loss_plot.png")
