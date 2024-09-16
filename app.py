import base64
import cv2
import numpy as np
import torch
from flask import Flask, request, jsonify
from io import BytesIO
from PIL import Image
from flask import render_template

app = Flask(__name__)

# Load YOLOv5 model (or use Faster R-CNN)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect_objects():
    data = request.json
    img_data = data['frame'].split(",")[1]
    img_bytes = base64.b64decode(img_data)
    img = Image.open(BytesIO(img_bytes))

    # Convert image to numpy array
    frame = np.array(img)

    # Object detection
    results = model(frame)
    detections = results.xyxy[0].cpu().numpy()

    # Draw bounding boxes on the frame
    for detection in detections:
        x1, y1, x2, y2, conf, cls = detection
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

    # Convert back to PIL image
    result_img = Image.fromarray(frame)
    
    # Encode image to base64
    buffered = BytesIO()
    result_img.save(buffered, format="JPEG")
    result_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({"result": result_b64})

if __name__ == '__main__':
    app.run(debug=True)
