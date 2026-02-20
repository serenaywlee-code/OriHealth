from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
import os

app = Flask(__name__)

# Load model
model_path = "oral_cancer_model.h5"
interpreter = tf.keras.models.load_model(model_path)

# Preprocess image
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

@app.route("/", methods=["GET", "POST"])
def index():
    risk_score = None
    risk_text = ""
    uploaded_file = None

    if request.method == "POST":
        file = request.files.get("file")
        if file:
            uploaded_file = file.filename
            image = Image.open(file)
            x = preprocess(image)
            pred = float(interpreter.predict(x)[0][0])
            risk_score = int(pred * 100)

            if risk_score >= 71:
                risk_text = "High Risk Detected 🔴"
            elif risk_score >= 41:
                risk_text = "Moderate Risk Detected 🟡"
            else:
                risk_text = "Low Risk 🟢"

    return render_template("index.html",
                           risk_score=risk_score,
                           risk_text=risk_text,
                           uploaded_file=uploaded_file)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
