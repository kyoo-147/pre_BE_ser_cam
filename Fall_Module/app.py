import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import torch
from flask import Flask, render_template, Response


from PIL import Image

from io import BytesIO

app = Flask(__name__)

# Tải model yolov5 từ ultra
#
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]

# Tải model mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles


# Tải model KNN 
pose_knn = joblib.load('models/pose_key_point.joblib')

# Các điểm keypoints của falling
keyXYZ = [
    "nose_x",
    "nose_y",
    "nose_z",
    "left_eye_inner_x",
    "left_eye_inner_y",
    "left_eye_inner_z",
    "left_eye_x",
    "left_eye_y",
    "left_eye_z",
    "left_eye_outer_x",
    "left_eye_outer_y",
    "left_eye_outer_z",
    "right_eye_inner_x",
    "right_eye_inner_y",
    "right_eye_inner_z",
    "right_eye_x",
    "right_eye_y",
    "right_eye_z",
    "right_eye_outer_x",
    "right_eye_outer_y",
    "right_eye_outer_z",
    "left_ear_x",
    "left_ear_y",
    "left_ear_z",
    "right_ear_x",
    "right_ear_y",
    "right_ear_z",
    "mouth_left_x",
    "mouth_left_y",
    "mouth_left_z",
    "mouth_right_x",
    "mouth_right_y",
    "mouth_right_z",
    "left_shoulder_x",
    "left_shoulder_y",
    "left_shoulder_z",
    "right_shoulder_x",
    "right_shoulder_y",
    "right_shoulder_z",
    "left_elbow_x",
    "left_elbow_y",
    "left_elbow_z",
    "right_elbow_x",
    "right_elbow_y",
    "right_elbow_z",
    "left_wrist_x",
    "left_wrist_y",
    "left_wrist_z",
    "right_wrist_x",
    "right_wrist_y",
    "right_wrist_z",
    "left_pinky_x",
    "left_pinky_y",
    "left_pinky_z",
    "right_pinky_x",
    "right_pinky_y",
    "right_pinky_z",
    "left_index_x",
    "left_index_y",
    "left_index_z",
    "right_index_x",
    "right_index_y",
    "right_index_z",
    "left_thumb_x",
    "left_thumb_y",
    "left_thumb_z",
    "right_thumb_x",
    "right_thumb_y",
    "right_thumb_z",
    "left_hip_x",
    "left_hip_y",
    "left_hip_z",
    "right_hip_x",
    "right_hip_y",
    "right_hip_z",
    "left_knee_x",
    "left_knee_y",
    "left_knee_z",
    "right_knee_x",
    "right_knee_y",
    "right_knee_z",
    "left_ankle_x",
    "left_ankle_y",
    "left_ankle_z",
    "right_ankle_x",
    "right_ankle_y",
    "right_ankle_z",
    "left_heel_x",
    "left_heel_y",
    "left_heel_z",
    "right_heel_x",
    "right_heel_y",
    "right_heel_z",
    "left_foot_index_x",
    "left_foot_index_y",
    "left_foot_index_z",
    "right_foot_index_x",
    "right_foot_index_y",
    "right_foot_index_z"
]

# Mảng lưu trữ res point và khởi chạy camera

cap = cv2.VideoCapture(0)

# Tạo giá trị cho ngưỡng nhận diện
with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # Mở và lấy kích thước frame
    while cap.isOpened():
        ret, frame = cap.read()
        h, w, _ = frame.shape
        size = (w, h)
        break

    # Tạo biến và lưu trữ video result
    out = cv2.VideoWriter("results/output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)

    # Định nghĩa hàm gen() để tạo luồng video
    def gen():
        prevTime = 0
        res_point = []
        while cap.isOpened():
            
            # Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break
            # Chuyển màu qua dạng RGB cho model YOLO
            image_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Khởi chạy yolov5 inference
            image_yolo.flags.writeable = False
            result_yolo = yolo_model(image_yolo)
            image_yolo.flags.writeable = True
            image_yolo = cv2.cvtColor(image_yolo, cv2.COLOR_RGB2BGR)
            
            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image_yolo, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 196, 255), 2)
            
            # Xử lí các khung frame bbox qua mediapipe
            for (xmin, ymin, xmax, ymax, confidence, clas) in result_yolo.xyxy[0].tolist():
                with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                    # Dự đoán các điểm landmark xung quanh bbox
                    results_pose = pose.process(image_yolo[int(ymin):int(ymax), int(xmin):int(xmax)])
                    # Kiểm tra xem các điểm landmark có được xác định không
                    if results_pose.pose_landmarks:
                        # Vẽ các điểm landmark lên vật thể
                        mp_drawing.draw_landmarks(image_yolo[int(ymin):int(ymax), int(xmin):int(xmax)],
                                                results_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))
                        # Trích xuất các mốc tư thế x,y,z để phân loại
                        for index, landmarks in enumerate(results_pose.pose_landmarks.landmark):
                            res_point.append(landmarks.x)
                            res_point.append(landmarks.y)
                            res_point.append(landmarks.z)
                        # Phân loại tư thế với pose_knn
                        shape1 = int(len(res_point) / len(keyXYZ))
                        res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
                        pred = pose_knn.predict(res_point)
                        res_point = []            
                        
                        # Hiển thị kết quả phân loại
                        if pred == 0:
                            cv2.putText(image_yolo, "Warning: Detect people falling", (int(xmin), int(ymax) + 20), cv2.FONT_HERSHEY_PLAIN, 2,
                                        (0, 0, 255), 2)
                            # Thêm các logic nếu phát sinh update
                        else:
                            cv2.putText(image_yolo, "Notice: Normal status", (int(xmin), int(ymax) + 20), cv2.FONT_HERSHEY_PLAIN, 2,
                                        (0, 255, 0), 2)
                            # Thêm các logic nếu phát sinh update
                    else:
                        # Không đọc được các điểm landmark
                        cv2.putText(image_yolo, "Notice: No Landmarks Detected", (int(xmin), int(ymax) + 20), cv2.FONT_HERSHEY_PLAIN, 2,
                                    (0, 255, 255), 2)
                    

            # Lưu lại video result final x20
            out.write(image_yolo)
            
            
            # Chuyển frame sang dạng bytes để hiển thị trên web
            ret, jpeg = cv2.imencode('.jpg', image_yolo)
            frame = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    # Định nghĩa route để hiển thị video trên web
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/video')
    def video():
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/1')
    def cam1():
        return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')
        
    
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
