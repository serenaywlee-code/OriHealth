import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="OriHealth",
    page_icon="🦷",
    layout="centered",
)

# ---------------- CSS (MATCHED TO BASE44 APP) ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}

.stApp {
    background: #F7F8FA;
    color: #1C1C1E;
}

/* Hero Text */
.hero {
    text-align: center;
    margin: 60px 0px 40px 0px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 600;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #555555;
}

/* Upload Card */
.card {
    background: white;
    padding: 40px;
    border-radius: 16px;
    box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
    margin: 0px auto 40px auto;
    max-width: 480px;
}

.upload-text {
    font-size: 20px;
    font-weight: 500;
    margin-bottom: 12px;
    text-align: center;
}

/* Result */
.result-score {
    text-align: center;
    font-size: 22px;
    font-weight: 600;
    margin-top: 18px;
}

/* Footer */
.footer {
    text-align: center;
    color: #777777;
    font-size: 14px;
    margin-bottom: 40px;
    margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("""
<div class="hero">
<h1>🦷 OriHealth</h1>
<p>Upload an oral cavity image for an AI-powered screening result.</p>
<p>This tool is for educational purposes only and does not replace professional diagnosis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL CHECK ----------------
MODEL_PATH = "oral_cancer_model.tflite"
if not os.path.exists(MODEL_PATH):
    st.error("⚠ Model file not found. Upload the .tflite file into your repo.")
    st.stop()

@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("<div class='upload-text'>Select an Oral Cavity Image (JPG/PNG)</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], key="upload")

if uploaded is not None:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    x = preprocess(image)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]["index"])

    # interpret output safely
    if len(output.shape) == 2:
        raw_pred = float(output[0][0])
    else:
        raw_pred = float(output[0])

    score = int(raw_pred * 100)

    # display result
    st.markdown(f"<div class='result-score'>🎯 Risk Score: {score}%</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class="footer">
This app is a student research project and is not medical advice. Always consult a healthcare professional.
</div>
""", unsafe_allow_html=True)
