from ultralytics import YOLO
import os

# ----- CONFIG -----
MODEL_PATH = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\yolov8s-cls.pt"
TEST_DATASET = r"C:\Users\navya\navya\navya - vdc\project\dragon_fruit\Real_images"

# CLASS ORDER (as in your dataset)
CLASS_NAMES = ["Ripe Dragon Fruit", "Unripe Dragon Fruit"]

# ------------------

model = YOLO(MODEL_PATH)
print("Loaded model with classes:", model.names)

image_files = [
    f for f in os.listdir(TEST_DATASET)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

print(f"\nFound {len(image_files)} images.\n")

for img in image_files:
    img_path = os.path.join(TEST_DATASET, img)

    results = model(img_path)[0]

    class_id = int(results.probs.top1)
    conf = float(results.probs.top1conf)

    # convert to your labels
    final_label = CLASS_NAMES[class_id]

    print(f"Image: {img}")
    print(f"Predicted Class: {final_label}")
    print(f"Confidence: {conf:.3f}\n")








