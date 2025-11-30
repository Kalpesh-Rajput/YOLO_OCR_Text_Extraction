import os
import cv2

labels_dir = "dataset/labels/train"
images_dir = "dataset/images/train"

for lbl in os.listdir(labels_dir):
    label_path = os.path.join(labels_dir, lbl)
    img_path = os.path.join(images_dir, lbl.replace(".txt", ".jpg"))

    if not os.path.exists(img_path):
        print("❌ Missing image for label:", lbl)

    # Check file content
    with open(label_path, "r") as f:
        data = f.read().strip()

    if data == "":
        print("⚠️ WARNING: Empty label file:", lbl)
