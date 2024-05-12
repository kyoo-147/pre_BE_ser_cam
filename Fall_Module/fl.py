from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np
import time
import joblib

app = Flask(__name__)

# khởi tạo các biến cần thiết là load model knn
pose_knn = joblib.load('models/pose_key_point.joblib')
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose
prevTime = 0


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

def process_video():
    cap = cv2.VideoCapture(0)
    res_point = []
    prevTime = 0
    with mp_pose.Pose(
            static_image_mode=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = pose.process(image)

            if results.pose_landmarks:
                for index, landmarks in enumerate(results.pose_landmarks.landmark):
                    res_point.append(landmarks.x)
                    res_point.append(landmarks.y)
                    res_point.append(landmarks.z)

                shape1 = int(len(res_point) / len(keyXYZ))
                res_point = np.array(res_point).reshape(shape1, len(keyXYZ))
                pred = pose_knn.predict(res_point)
                res_point = []

                if pred == 0:
                    cv2.putText(image, "Fall", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 2)
                else:
                    cv2.putText(image, "Normal", (200, 320), cv2.FONT_HERSHEY_PLAIN, 5, (0, 255, 0), 2)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            mp_drawing.draw_landmarks(
                image,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

            currTime = time.time()
            fps = 1 / (currTime - prevTime)
            prevTime = currTime
            cv2.putText(image, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_PLAIN, 3, (0, 196, 255), 2)

            _, jpeg = cv2.imencode('.jpg', image)
            frame = jpeg.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_video(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
