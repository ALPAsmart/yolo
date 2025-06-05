from flask import Flask, request, jsonify
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
import base64

app = Flask(__name__)

COCO_LABELS = [
    '???', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', '???',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', '???', '???', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', '???', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', '???', 'dining table', '???',
    '???', 'toilet', '???', 'tv', 'laptop', 'mouse', 'remote', 'keyboard',
    'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    '???', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
    'toothbrush'
]

print("[INFO] Loading model...")
detector = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1")

def estimate_close(box, frame_width, frame_height):
    ymin, xmin, ymax, xmax = box
    box_height = (ymax - ymin) * frame_height
    return box_height > 150  # adjust threshold as needed

@app.route('/detect', methods=['POST'])
def detect_objects():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400

    file = request.files['image']
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({'error': 'Invalid image file'}), 400

    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(img_rgb, (320, 320))
    resized = tf.cast(resized, tf.uint8)
    input_tensor = tf.expand_dims(resized, 0)

    result = detector(input_tensor)
    result = {key: value.numpy() for key, value in result.items()}

    boxes = result['detection_boxes'][0]
    class_ids = result['detection_classes'][0].astype(np.int32)
    scores = result['detection_scores'][0]

    detected_objects = []

    for i in range(len(scores)):
        if scores[i] < 0.5:
            continue
        label = COCO_LABELS[class_ids[i]]
        close = estimate_close(boxes[i], w, h)
        detected_objects.append({
            'label': label,
            'score': float(scores[i]),
            'close': bool(close)
        })

    return jsonify({'detections': detected_objects})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
