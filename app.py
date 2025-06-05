import threading
import time
import torch
import cv2
import pyttsx3
from flask import Flask, jsonifym request

app = Flask(__name__)

model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
engine = pyttsx3.init()

detection_thread = None
running = False
last_detections = []

def speak(text):
    engine.say(text)
    engine.runAndWait()

def detection_loop():
    global running, last_detections

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        running = False
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        results = model(frame[:, :, ::-1])
        detections = results.pandas().xyxy[0]

        last_detections = []

        spoken = False
        for idx, obj in detections.iterrows():
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            area = width * height

            last_detections.append({
                "name": obj['name'],
                "confidence": obj['confidence'],
                "bbox": [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
            })

            if area > 40000 and not spoken:
                speak(f"{obj['name']} is close by!")
                spoken = True

        time.sleep(0.1)  # small delay to reduce CPU load

    cap.release()

@app.route('/start')
def start_detection():
    global running, detection_thread
    if running:
        return jsonify({"status": "already running"})

    running = True
    detection_thread = threading.Thread(target=detection_loop)
    detection_thread.start()
    return jsonify({"status": "detection started"})

@app.route('/stop')
def stop_detection():
    global running, detection_thread
    if not running:
        return jsonify({"status": "not running"})

    running = False
    detection_thread.join()
    return jsonify({"status": "detection stopped"})

@app.route('/status')
def status():
    global last_detections
    return jsonify({"last_detections": last_detections})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
