
import streamlit as st
import cv2
import pytesseract
from ultralytics import YOLO
import numpy as np

# Load YOLOv8 model (your trained model)
MODEL_PATH = "runs/detect/train4/weights/best.pt"
model = YOLO(MODEL_PATH)

# Tesseract config
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
tess_config = "--psm 6"

st.title("🔍 YOLO + OCR Pattern Extraction App")
st.write("Upload a shipping label and extract text patterns like `_1`, `_1_`, `1_`")

uploaded = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

def extract_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return pytesseract.image_to_string(gray, config=tess_config)

if uploaded:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.image(img, caption="Uploaded Image", use_column_width=True)

    st.write("### 🔎 Running YOLO detection...")

    results = model(img)

    if len(results[0].boxes) == 0:
        st.error("❌ No object detected by YOLO.")
        st.stop()

    # Use the first detection
    box = results[0].boxes[0]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    crop = img[y1:y2, x1:x2]

    st.image(crop, caption="Detected Label Region", use_column_width=True)

    st.write("### 📝 Running OCR...")

    text = extract_text(crop)
    st.code(text)

    # Pattern matching
    import re
    pattern = r"[0-9]{15,20}[_][0-9][_][a-zA-Z0-9]+"
    found = re.findall(pattern, text)

    st.write("### 🎯 Pattern Extraction Result")

    if found:
        st.success(f"✔ Pattern Found: {found[0]}")
    else:
        st.error("❌ Pattern NOT found in OCR text.")
