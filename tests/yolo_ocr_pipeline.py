# import cv2
# import torch
# import easyocr
# import numpy as np
# import pandas as pd
# from tqdm import tqdm


# class YOLO_OCR_Pipeline:

#     def __init__(self, yolo_weights="runs/detect/train4/weights/best.pt"):
#         print("Loading YOLO model...")
#         self.detector = torch.hub.load("ultralytics/yolov5", "custom", path=yolo_weights)

#         print("Loading EasyOCR...")
#         self.reader = easyocr.Reader(['en'], gpu=False)

#     def detect_region(self, image_path):
#         """Runs YOLO and returns cropped region of interest (ROI)"""

#         results = self.detector(image_path)
#         detections = results.pandas().xyxy[0]

#         if len(detections) == 0:
#             print("YOLO: No box detected")
#             return None, None

#         # Take the detection with highest confidence
#         det = detections.sort_values("confidence", ascending=False).iloc[0]

#         x1, y1, x2, y2 = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)

#         img = cv2.imread(image_path)
#         crop = img[y1:y2, x1:x2]

#         return crop, (x1, y1, x2, y2)

#     def ocr_read(self, image):
#         """Runs OCR on cropped region"""
#         if image is None:
#             return ""

#         results = self.reader.readtext(image, detail=1)
#         lines = [res[1] for res in results]

#         return " ".join(lines)

#     def extract_pattern(self, text):
#         """Detect _1, _1_, 1_, etc."""
#         import re
#         pattern = r"[0-9A-Za-z]+[_][1][A-Za-z0-9_]*"
#         match = re.findall(pattern, text)

#         if match:
#             return match[0]
#         return None

#     def run(self, image_path):

#         # Step 1: YOLO
#         crop, box = self.detect_region(image_path)

#         if crop is None:
#             return {
#                 "yolo_box": None,
#                 "ocr_text": "",
#                 "extracted": None,
#                 "error": "No detection"
#             }

#         # Step 2: OCR
#         text = self.ocr_read(crop)

#         # Step 3: extract pattern
#         extracted = self.extract_pattern(text)

#         return {
#             "yolo_box": box,
#             "ocr_text": text,
#             "extracted": extracted,
#             "error": None if extracted else "Pattern not found"
#         }


import cv2
import pytesseract
from ultralytics import YOLO

class YOLO_OCR_Pipeline:
    def __init__(self, yolo_weights):
        # Load YOLOv8 model
        print("Loading YOLOv8 model...")
        self.detector = YOLO(yolo_weights)

        # Configure Tesseract
        pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    def detect_boxes(self, image_path):
        results = self.detector(image_path)
        boxes = results[0].boxes

        detections = []
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            detections.append([x1, y1, x2, y2, conf])

        return detections

    def crop(self, img, box):
        x1, y1, x2, y2, _ = box
        return img[int(y1):int(y2), int(x1):int(x2)]

    def run(self, image_path):
        img = cv2.imread(image_path)
        if img is None:
            return {"error": f"Cannot load image: {image_path}"}

        # 1. YOLOv8 detects text-region
        boxes = self.detect_boxes(image_path)

        if len(boxes) == 0:
            return {"error": "No detections from YOLO"}

        results = []

        # 2. Extract each region using OCR
        for box in boxes:
            crop_img = self.crop(img, box)

            if crop_img.size == 0:
                continue

            text = pytesseract.image_to_string(
                crop_img,
                config="--psm 6"
            )

            results.append({
                "box": box,
                "text": text.strip()
            })

        # 3. Pattern search
        combined_text = " ".join([r["text"] for r in results])
        found = None

        patterns = ["_1", "_1_", "1_", "_l", "l_"]

        for p in patterns:
            if p in combined_text:
                found = p
                break

        return {
            "detections": results,
            "pattern": found,
            "raw_text": combined_text
        }
