import argparse
import io
import os
from PIL import Image
import cv2
import numpy as np
import torch
from flask import Flask, render_template, Response
from io import BytesIO

app = Flask(__name__)

path = './yolov5/best.pt'
model = torch.hub.load('yolov5', 'custom', path, source='local', force_reload=True)
# Set Model Settings
model.eval()
model.conf = 0.2  # confidence threshold (0-1)
model.iou = 0.45  # NMS IoU threshold (0-1) 

def gen():
    cap=cv2.VideoCapture(0)
    # Read until video is completed
    while(cap.isOpened()):
        success, frame = cap.read()
        if success == True:
            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()
            img = Image.open(io.BytesIO(frame))
            results = model(img, size = 640)
            #print(results)
            # print("ket qua: ", results.pandas().xyxy[0])
            dem_lua = len(results.xyxy[0]) if results.xyxy and len(results.xyxy) > 0 else 0
            print("ket qua: ", dem_lua)
            if dem_lua > 0:
                global num_fall 
                num_fall = "2"
            else:
                continue
            # results.render()  # updates results.imgs with boxes and labels
            results.print()  # print results to screen
            # results.save() 
            # results.crop(save=True)
            #convert remove single-dimensional entries from the shape of an array
            img = np.squeeze(results.render()) #RGB
            # read image as BGR
            img_BGR = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) #BGR
        else:
            break
        # Encode BGR image to bytes so that cv2 will convert to RGB
        frame = cv2.imencode('.jpg', img_BGR)[1].tobytes()
        
        yield(b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Flask app exposing yolov5 models")
    parser.add_argument("--port", default=5500, type=int, help="port number")
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port)  # debug=True causes Restarting with stat
