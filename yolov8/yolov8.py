from ultralytics import YOLO
import cv2
import math
from datetime import datetime

def process_image_from_camera(frame):
    model = YOLO("../YOLO-Weights/yolov8x.pt")

    results = model(frame, stream=True)

    # Tekne sayacı
    boat_count = 0

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls == 8:  # 7, "boat" sınıfına karşılık gelen indekstir (classNames listesindeki pozisyonu)
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                conf = math.ceil((box.conf[0] * 100)) / 100
                class_name = "boat"
                label = f'{class_name}{conf}'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(frame, (x1, y1), c2, [255, 0, 255], -1, cv2.LINE_AA)  # filled
                cv2.putText(frame, label, (x1, y1 - 2), 0, 1, [255, 255, 255], thickness=1, lineType=cv2.LINE_AA)

            # Tekne sayısını artır
            if class_name == "boat":
                boat_count += 1

    # Tekne sayısını konsola yazdır
    print(f"{datetime.now()} tarihinde tespit edilen tekne sayısı: {boat_count}")

    # Görüntüyü göster
    cv2.imshow("image", frame)
    cv2.waitKey(1)  # Bu satırın 1 olması gerekiyor, 0 olmamalı
