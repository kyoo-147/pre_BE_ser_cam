import cv2
import mediapipe as mp
import numpy as np
import time
import joblib
import torch

# tải model yolov5 từ ultra
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes = [0]

# tải model mediapipe pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
mp_drawing_styles = mp.solutions.drawing_styles
prevTime = 0

# tải model KNN 
pose_knn = joblib.load('Model/PoseKeypoint.joblib')

# các điểm keypoints của falling
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

# tạo mảng lưu trữ res point và khởi chạy camera
res_point = []
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('2.mp4')

# tạo giá trị cho ngưỡng nhận diện
with mp_pose.Pose(
        static_image_mode=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    # mở và lấy kích thước frame
    while cap.isOpened():
        ret, frame = cap.read()
        h, w, _ = frame.shape
        size = (w, h)
        break
    # tạo biến và lưu trữ video result
    out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 20, size)
    while cap.isOpened():
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
        # chuyển màu qua dạng RGB cho model YOLO
        image_yolo = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # khởi chạy yolov5 inference
        image_yolo.flags.writeable = False
        result_yolo = yolo_model(image_yolo)
        image_yolo.flags.writeable = True
        image_yolo = cv2.cvtColor(image_yolo, cv2.COLOR_RGB2BGR)
        # xử lí các khung frame bbox qua mediapipe
        for (xmin, ymin, xmax, ymax, confidence, clas) in result_yolo.xyxy[0].tolist():
            with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.3) as pose:
                # dự đoán các điểm landmark xung quanh bbox
                results_pose = pose.process(image_yolo[int(ymin):int(ymax), int(xmin):int(xmax)])
                # kiểm tra xem các điểm landmark có được xác định không
                if results_pose.pose_landmarks:
                    # vẽ các điểm landmark lên vật thể
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
                    # hiển thị kết quả phân loại
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
        currTime = time.time()
        fps = 1 / (currTime - prevTime)
        prevTime = currTime
        cv2.putText(image_yolo, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 196, 255), 2)
        # Hiển thị kết quả
        cv2.imshow('Security AI Assistant', image_yolo)
        # Exit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    # Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
