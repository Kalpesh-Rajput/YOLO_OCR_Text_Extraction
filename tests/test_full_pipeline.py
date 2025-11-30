from src.pipeline_yolo_ocr import YOLO_OCR_Pipeline

pipe = YOLO_OCR_Pipeline()

out = pipe.run("reverseWaybill-162107205368239936_1.jpg")

print(out)
