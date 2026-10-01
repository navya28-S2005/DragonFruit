import os
import time
import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# ====================== PATHS ======================
MODEL_PATH = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\yolov8s-cls.pt"
TEST_DIR = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\Dataset\test"

# Match your REAL folder names:
CLASS_FOLDERS = {
    "Ripe Dragon Fruit": 0,
    "Unripe Dragon Fruit": 1
}

CLASS_NAMES = ["Ripe", "Unripe"]

# ====================== LOAD MODEL ======================
print("Loading model...")
model = YOLO(MODEL_PATH)
print("Model loaded.")

# ====================== LOAD TEST IMAGES ======================
true_labels = []
pred_labels = []

print("\nCollecting test images...")

for folder_name, class_idx in CLASS_FOLDERS.items():
    folder_path = os.path.join(TEST_DIR, folder_name)

    if not os.path.exists(folder_path):
        print(f"❌ ERROR: Missing folder → {folder_path}")
        continue

    images = os.listdir(folder_path)
    if len(images) == 0:
        print(f"⚠️ WARNING: No images found in {folder_path}")

    print(f"→ Loading {len(images)} images from {folder_name}")

    for img_name in images:
        img_path = os.path.join(folder_path, img_name)

        # run prediction
        results = model(img_path)[0]

        # predicted class
        pred_class = int(results.probs.top1)

        # append
        pred_labels.append(pred_class)
        true_labels.append(class_idx)

print("\nPrediction done.")

# ====================== CONFUSION MATRIX ======================
if len(true_labels) == 0:
    print("❌ ERROR: No images loaded. Confusion matrix cannot be computed.")
    exit()

cm = confusion_matrix(true_labels, pred_labels)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Dragon Fruit Classification – Confusion Matrix")
plt.tight_layout()
plt.show()
