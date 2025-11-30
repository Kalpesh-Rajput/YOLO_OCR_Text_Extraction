from ultralytics import YOLO
import cv2

# load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

img_path = "reverseWaybill-162107205368239936_1.jpg"

results = model.predict(
    img_path,
    conf=0.10,      # LOWER CONFIDENCE
    iou=0.45,       # stricter IOU
    imgsz=1280,     # BIGGER IMAGE = better for small text
    max_det=5
)

results[0].save(filename="yolo_strong_detect.jpg")

print("Done. Saved as yolo_strong_detect.jpg")
