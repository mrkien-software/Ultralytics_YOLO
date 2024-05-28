from ultralytics import YOLO
import cv2
import cvlib as cv
import numpy as np

# Load the YOLO model
model = YOLO("yolov8n-face.pt")  # Sử dụng mô hình YOLO đã huấn luyện để phát hiện khuôn mặt

# Đường dẫn tới các tệp cấu hình và trọng số của mô hình phát hiện giới tính
gender_model_path = "gender_deploy.prototxt"
gender_weights_path = "gender_net.caffemodel"

# Load mô hình phát hiện giới tính
gender_net = cv2.dnn.readNetFromCaffe(gender_model_path, gender_weights_path)

# Open the webcam
cap = cv2.VideoCapture("video2.mp4")
# cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from webcam")
    exit()

while cap.isOpened():
    success, frame = cap.read()

    if success:
        # Run YOLO to detect faces
        results = model(frame)

        for result in results:
            boxes = result.boxes.xyxy.numpy()  # Lấy bounding boxes từ kết quả
            confidences = result.boxes.conf.numpy()  # Lấy độ tin cậy từ kết quả

            for (box, confidence) in zip(boxes, confidences):
                if confidence > 0.5:  # Chỉ xét những phát hiện có độ tin cậy trên 50%
                    x1, y1, x2, y2 = map(int, box)
                    face = frame[y1:y2, x1:x2]  # Cắt ảnh khuôn mặt từ frame

                    # Chuẩn bị blob từ khuôn mặt
                    blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), (104.0, 177.0, 123.0), swapRB=False)
                    gender_net.setInput(blob)
                    gender_preds = gender_net.forward()
                    gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
                    gender_confidence = gender_preds[0][0] if gender == "Male" else gender_preds[0][1]

                    # Vẽ bounding box và thông tin giới tính lên frame
                    text = f"{gender}: {gender_confidence*100:.2f}%"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Display the frame with detections
        cv2.imshow("Gender Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()
