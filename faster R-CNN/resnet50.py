import torch
import torchvision
from torchvision import transforms as T
import cv2

def detect_boats(frame):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()

    coco_names = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella",
                  "shoe", "eye glasses", "handbag", "tie", "suitcase",
                  "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "plate", "wine glass", "cup", "fork", "knife", "spoon", "bowl",
                  "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
                  "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                  "mirror", "dining table", "window", "desk", "toilet", "door", "tv",
                  "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
                  "oven", "toaster", "sink", "refrigerator", "blender", "book",
                  "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # cv2.imread() ile bir resim dosyasını okumak yerine doğrudan gelen kare üzerinde işlem yapın
    img = frame

    with torch.no_grad():
        predictions = model([T.ToTensor()(img)])

    scores = predictions[0]['scores'].cpu().numpy().astype("float")
    boxes = predictions[0]['boxes'].cpu().numpy().astype("int")
    labels = predictions[0]['labels'].cpu().numpy().astype("int")

    boat_boxes = boxes[labels == coco_names.index('boat') + 1]

    selected_boxes = cv2.dnn.NMSBoxes(boat_boxes.tolist(), scores[labels == coco_names.index('boat') + 1], 0.3, 0.3)

    for i in selected_boxes.flatten():
        x1, y1, x2, y2 = boat_boxes[i]
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        img = cv2.putText(img, 'boat', (x1, y1 - 10), font, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('Detected Boats', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




