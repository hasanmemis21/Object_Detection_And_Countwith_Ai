import cv2
import time
from resnet50 import detect_boats

# Kamera bağlantısıq
camera_url = "rtsp://hmemis:NB!8NVdz8s@10.11.66.165:554/Streaming/channels/1/"
cap = cv2.VideoCapture(camera_url)
# İlk görüntü alımı için bekleme süresi
wait_seconds = 6 # 10 dakika

# Başlangıç zamanı
start_time = time.time()

while True:
    # Şu anki zamanı al
    current_time = time.time()

    # Belirtilen süre aralığında işlem yap
    if current_time - start_time >= wait_seconds:
        ret, frame = cap.read()

        if ret:
            # Görüntü işleme fonksiyonunu çağır ve görüntüyü işle
            detect_boats(frame)

            # Başlangıç zamanını güncelle
            start_time = time.time()

    # q tuşuna basılırsa döngüyü kır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Kamerayı serbest bırak ve pencereleri kapat
cap.release()
cv2.destroyAllWindows()