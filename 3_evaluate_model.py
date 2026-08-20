"""
STEP 6: Model Evaluation and Testing
--------------------------------------
Loads the trained model and the held-out test split (saved by
2_train_model.py) and reports Accuracy, Precision, Recall, F1-score,
and a Confusion Matrix - the exact evidence FA-2 asks for.

Outputs saved to the working directory:
    confusion_matrix.png
    classification_report.txt

Run this in Google Colab or Jupyter Notebook, AFTER running 2_train_model.py.

Install dependencies first (Colab cell):
    !pip install tensorflow scikit-learn matplotlib seaborn numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_PATH = "fall_detection_model.h5"
TEST_DATA_PATH = "test_data.npz"
LABELS_PATH = "label_encoder_classes.npy"

# ----------------------------------------------------------------------
# 1. Load model, test data, and class labels
# ----------------------------------------------------------------------
model = load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH, allow_pickle=True)

data = np.load(TEST_DATA_PATH)
X_test, y_test_onehot = data["X_test"], data["y_test"]
y_test = np.argmax(y_test_onehot, axis=1)

# ----------------------------------------------------------------------
# 2. Predict on the test set
# ----------------------------------------------------------------------
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ----------------------------------------------------------------------
# 3. Core metrics
# ----------------------------------------------------------------------
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print("----- Test Set Evaluation -----")
print(f"Accuracy : {acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-Score : {f1:.4f}")

report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
print("\nPer-class report:\n", report)

with open("classification_report.txt", "w") as f:
    f.write("----- Test Set Evaluation -----\n")
    f.write(f"Accuracy : {acc:.4f}\n")
    f.write(f"Precision: {precision:.4f}\n")
    f.write(f"Recall   : {recall:.4f}\n")
    f.write(f"F1-Score : {f1:.4f}\n\n")
    f.write(report)

print("Saved classification_report.txt")

# ----------------------------------------------------------------------
# 4. Confusion matrix
# ----------------------------------------------------------------------
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(7, 6))
sns.heatmap(
    cm, annot=True, fmt="d", cmap="Blues",
    xticklabels=class_names, yticklabels=class_names
)
plt.title("Confusion Matrix - Fall Detection Model")
plt.xlabel("Predicted Activity")
plt.ylabel("Actual Activity")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.close()

print("Saved confusion_matrix.png")

# ----------------------------------------------------------------------
# 5. Notes on real-world deployment challenges (for FA-2 write-up)
# ----------------------------------------------------------------------
print(
    "\nCommon misclassification sources to discuss in your report:\n"
    "- Lighting variation between rooms (Coffee_room / Home / Office / Lecture)\n"
    "- Camera angle differences affecting landmark visibility\n"
    "- Occlusion (furniture blocking limbs/torso)\n"
    "- Similar postures between 'sitting' and 'falling' at certain angles\n"
    "- Low visibility confidence on fast movements causing dropped frames"
)
