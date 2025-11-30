from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO("runs/detect/train4/weights/best.pt")

# Test image path
image_path = "reverseWaybill-162107205368239936_1.jpg"

# Run detection
results = model(image_path)

# Show detection results
for r in results:
    boxes = r.boxes.xyxy.cpu().numpy()  # [x1, y1, x2, y2]
    confs = r.boxes.conf.cpu().numpy()
    cls = r.boxes.cls.cpu().numpy()

    for i, box in enumerate(boxes):
        print(f"Detected class: {cls[i]}, confidence: {confs[i]}")
        print(f"BBox: {box}")

        # Draw box
        img = cv2.imread(image_path)
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 3)

        cv2.imwrite("yolo_detect_output.jpg", img)

print("\nSaved: yolo_detect_output.jpg")
