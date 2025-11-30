
import sys
import os

# Add the project root to module search path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT)

from yolo_ocr_pipeline import YOLO_OCR_Pipeline

pipe = YOLO_OCR_Pipeline("runs/detect/train4/weights/best.pt")

result = pipe.run("reverseWaybill-162107205368239936_1.jpg")

print("\n===== YOLO + OCR RESULT =====")
print(result)
