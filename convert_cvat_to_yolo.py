import os
import xml.etree.ElementTree as ET

# Path to the XML file exported from CVAT
XML_PATH = "annotations.xml"

# Output folder for YOLO labels
OUTPUT_DIR = "labels_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

tree = ET.parse(XML_PATH)
root = tree.getroot()

for image_tag in root.findall("image"):
    image_name = image_tag.get("name")
    width = float(image_tag.get("width"))
    height = float(image_tag.get("height"))

    # Create .txt file with same name as image
    label_filename = image_name.replace(".jpg", ".txt")
    label_path = os.path.join(OUTPUT_DIR, label_filename)

    with open(label_path, "w") as f:
        for box in image_tag.findall("box"):
            # YOLO class index (you have only 1 class)
            class_id = 0

            xtl = float(box.get("xtl"))
            ytl = float(box.get("ytl"))
            xbr = float(box.get("xbr"))
            ybr = float(box.get("ybr"))

            # Convert to YOLO format
            x_center = ((xtl + xbr) / 2) / width
            y_center = ((ytl + ybr) / 2) / height
            bbox_width = (xbr - xtl) / width
            bbox_height = (ybr - ytl) / height

            f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")

print("✔ Conversion completed! Check labels_output folder.")
con