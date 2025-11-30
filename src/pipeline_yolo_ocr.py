import cv2
from ultralytics import YOLO
import pytesseract

class YOLO_OCR_Pipeline:
    def __init__(self):
        self.model = YOLO("runs/detect/train4/weights/best.pt")

    def run(self, image_path):
        results = self.model.predict(image_path, conf=0.10, imgsz=1280)[0]

        if len(results.boxes) == 0:
            return {"error": "No YOLO detection found"}

        # highest confidence box
        box = results.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        img = cv2.imread(image_path)
        crop = img[y1:y2, x1:x2]
        cv2.imwrite("crop.jpg", crop)

        # OCR
        text = pytesseract.image_to_string(crop)

        return {
            "bbox": [x1, y1, x2, y2],
            "crop_path": "crop.jpg",
            "ocr_text": text
        }
