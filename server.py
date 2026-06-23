from flask import Flask, jsonify
from PIL import Image
import numpy as np
import cv2

from pycoral.utils.edgetpu import make_interpreter
from pycoral.adapters import classify
from pycoral.adapters.common import input_size

# CONFIG
MODEL_PATH = "ResNet50_edgetpu.tflite"
LABELS_PATH = "labels.txt"

# cargar etiquetas
labels = {}
with open(LABELS_PATH, "r") as f:
    for i, line in enumerate(f):
        labels[i] = line.strip()

# cargar modelo
interpreter = make_interpreter(MODEL_PATH)
interpreter.allocate_tensors()

# cámara
cap = cv2.VideoCapture("/dev/video1")

app = Flask(__name__)

@app.route("/predict")
def predict():
    ret, frame = cap.read()
    if not ret:
        return jsonify({"error": "no frame"})

    # preprocess
    size = input_size(interpreter)
    frame_resized = cv2.resize(frame, size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)

    # inferencia
    from pycoral.adapters.common import set_input
    set_input(interpreter, frame_rgb)
    interpreter.invoke()

    classes = classify.get_classes(interpreter, top_k=1)
    result = classes[0]

    return jsonify({
        "label": labels.get(result.id, "unknown"),
        "confidence": float(result.score)
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)