"""
YOLOv8 (classification) robust training + evaluation + plotting script
Dataset: 2 classes (ripe, unripe)
Fully handles missing columns, missing folders, zero images, etc.
"""

import os
import time
import shutil
import psutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
    roc_curve,
    auc,
    confusion_matrix
)

# ----------------------------
# CONFIG
# ----------------------------
DATASET_PATH = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\Dataset"
MODEL_PATH = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\yolov8s-cls.pt"
PROJECT_DIR = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\runs_classify"
RUN_NAME = "yolov8_ripe_unripe_run1"
EPOCHS = 25
BATCH_SIZE = 16
IMG_SIZE = 224
SAVE_BEST_TO = Path(r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\best_ripe_unripe.pt")

# ----------------------------
# UTILS
# ----------------------------
def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def find_latest_file(root: Path, pattern: str):
    files = list(root.rglob(pattern))
    if not files:
        return None
    files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return files[0]

def list_images(folder: Path):
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    return [p for p in folder.rglob("*") if p.suffix.lower() in exts]

# ----------------------------
# TRAINING
# ----------------------------
print("🚀 Starting YOLOv8 classification training (2 classes)...")
model = YOLO(MODEL_PATH)

train_start_time = time.time()
model.train(
    data=DATASET_PATH,
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH_SIZE,
    project=PROJECT_DIR,
    name=RUN_NAME,
    pretrained=True,
    verbose=True
)
train_total_time = time.time() - train_start_time
print(f"✅ Training finished in {train_total_time:.1f} sec")

# ----------------------------
# LOCATE results.csv and best.pt
# ----------------------------
project_path = Path(PROJECT_DIR)
results_csv = find_latest_file(project_path, "results.csv")
best_weights = find_latest_file(project_path, "weights/best.pt") or find_latest_file(project_path, "best.pt")

if results_csv is None:
    raise FileNotFoundError(f"No results.csv found under {project_path}")
else:
    print(f"📄 Found results CSV: {results_csv}")

if best_weights:
    safe_mkdir(SAVE_BEST_TO.parent)
    shutil.copy(best_weights, SAVE_BEST_TO)
    print(f"✅ Best weights copied to: {SAVE_BEST_TO}")

plots_out = results_csv.parent / "plots"
safe_mkdir(plots_out)

# ----------------------------
# READ CSV AND PLOT
# ----------------------------
df = pd.read_csv(results_csv)

def find_column(df, substrings):
    for sub in substrings:
        matches = [c for c in df.columns if sub.lower() in c.lower()]
        if matches:
            return matches[0]
    return None

train_loss_col = find_column(df, ['train/loss','train_loss','loss'])
val_loss_col   = find_column(df, ['val/loss','val_loss'])
acc_col        = find_column(df, ['metrics/accuracy_top1','top1','accuracy'])
lr_col         = find_column(df, ['lr','learning_rate'])

epochs = df['epoch'].to_list() if 'epoch' in df.columns else list(range(1,len(df)+1))

# Train & Val Loss
plt.figure(figsize=(8,5))
if train_loss_col and train_loss_col in df.columns:
    plt.plot(epochs, df[train_loss_col], label='train_loss')
if val_loss_col and val_loss_col in df.columns:
    plt.plot(epochs, df[val_loss_col], label='val_loss')
if train_loss_col or val_loss_col:
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Train & Validation Loss'); plt.grid(True); plt.legend()
    plt.savefig(plots_out / "loss_curve.png", bbox_inches='tight'); plt.close()
    print(f"📈 Saved loss curve")
else:
    print("⚠️ No loss data found; skipping loss plot")

# Accuracy
if acc_col and acc_col in df.columns:
    plt.figure(figsize=(8,5))
    plt.plot(epochs, df[acc_col], label='top1_accuracy')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.title('Validation Accuracy'); plt.grid(True); plt.legend()
    plt.savefig(plots_out / "accuracy_curve.png", bbox_inches='tight'); plt.close()
    print(f"📈 Saved accuracy curve")

# Learning Rate
if lr_col and lr_col in df.columns:
    plt.figure(figsize=(8,4))
    plt.plot(epochs, df[lr_col], label='LR')
    plt.xlabel('Epoch'); plt.ylabel('Learning Rate'); plt.title('Learning Rate'); plt.grid(True); plt.legend()
    plt.savefig(plots_out / "lr_curve.png", bbox_inches='tight'); plt.close()
    print(f"📉 Saved LR curve")

# ----------------------------
# EVALUATION
# ----------------------------
val_dir = Path(DATASET_PATH) / "val"
if not val_dir.exists():
    raise FileNotFoundError(f"No 'val' folder found under {DATASET_PATH}")

class_names = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])
print("🔖 Detected classes:", class_names)

def predict_folder(model, folder):
    imgs = list_images(folder)
    y_true, y_pred, y_scores = [], [], []
    start_time = time.time()
    for p in imgs:
        parent = p.parent.name
        if parent not in class_names: continue
        true_idx = class_names.index(parent)
        pred = model.predict(source=str(p), imgsz=IMG_SIZE, verbose=False)
        probs = pred[0].probs.data.cpu().numpy().ravel()
        pred_label = int(np.argmax(probs))
        y_true.append(true_idx); y_pred.append(pred_label); y_scores.append(probs)
    return np.array(y_true), np.array(y_pred), np.vstack(y_scores), time.time()-start_time, len(imgs)

y_true, y_pred, y_scores, eval_time, num_images = predict_folder(model, val_dir)
print(f"🧾 Evaluated {num_images} images in {eval_time:.2f} sec")

# Metrics
acc = accuracy_score(y_true, y_pred)
precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
report = classification_report(y_true, y_pred, target_names=class_names, digits=4)

print("\n📌 Validation Metrics:")
print(f"Accuracy: {acc*100:.2f}%")
print(report)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,4))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks,class_names,rotation=45)
plt.yticks(tick_marks,class_names)
for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(j,i,cm[i,j],ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
plt.ylabel('True'); plt.xlabel('Predicted'); plt.tight_layout()
plt.savefig(plots_out / "confusion_matrix.png", bbox_inches='tight'); plt.close()
print("🔢 Saved Confusion Matrix")

# ROC (binary)
if len(class_names)==2:
    y_score_pos = y_scores[:,1]
    fpr, tpr, _ = roc_curve(y_true, y_score_pos, pos_label=1)
    roc_auc = auc(fpr,tpr)
    plt.figure(figsize=(6,5))
    plt.plot(fpr,tpr,label=f"AUC={roc_auc:.3f}")
    plt.plot([0,1],[0,1],'--',color='gray')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(plots_out / "roc_curve.png", bbox_inches='tight'); plt.close()
    print(f"🔵 Saved ROC curve (AUC={roc_auc:.3f})")

# Inference speed
avg_inf_time = eval_time/num_images if num_images>0 else 0
fps = 1/avg_inf_time if avg_inf_time>0 else 0
print(f"\n⏱ Avg inference time: {avg_inf_time:.4f}s per image, FPS: {fps:.2f}")

# Model info
model_size_mb = SAVE_BEST_TO.stat().st_size/(1024*1024) if SAVE_BEST_TO.exists() else Path(MODEL_PATH).stat().st_size/(1024*1024)
mem_used_mb = psutil.virtual_memory().used/(1024*1024)
print(f"💾 Model size: {model_size_mb:.2f} MB, Memory usage: {mem_used_mb:.2f} MB")

print("\n🎉 All plots, metrics, and best weights saved successfully!")

