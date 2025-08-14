from ultralytics import YOLO
import cvzone
import cv2
import math

cap = cv2.VideoCapture('fire2.mp4')
model = YOLO('best.pt')
classnames = ['fire']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = cv2.resize(frame, (640, 480))
    results = model(frame, stream=True)

    for info in results:
        boxes = info.boxes
        for box in boxes:
            conf = float(box.conf[0])
            confidence = math.ceil(conf * 100)
            Class = int(box.cls[0])
            if confidence > 20:
                b = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = b
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
                label = f"{classnames[Class]} {confidence}%"
                cvzone.putTextRect(frame, label, [x1 + 5, y1 - 10], scale=1.2, thickness=2)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
