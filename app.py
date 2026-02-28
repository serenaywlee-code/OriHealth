import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="OriHealth: Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# ---------------- STYLES ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400&display=swap');

html, body, [class*="css"]  {
    font-family: 'Lora', serif !important;
    font-weight: 400 !important;
}

.stApp {
    background-color: #D9EDEB;
}

.title {
    font-size: 40px;
    font-weight: 400;
    color: #3275a8;
    text-align: center;
    margin-bottom: 10px;
}

.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 36px;
}

.upload-section {
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.upload-header {
    color: #3275a8;
    font-size: 20px;
    margin-bottom: 12px;
}

.score {
    margin-top: 14px;
    color: #374151;
    font-weight: 600;
}

.normal {
    background: #f0fdf4;
    border-left: 6px solid #22c55e;
    padding: 16px;
    border-radius: 12px;
    color: #14532d;
    margin-top: 16px;
}

.abnormal {
    background: #fff7ed;
    border-left: 6px solid #f59e0b;
    padding: 16px;
    border-radius: 12px;
    color: #7c2d12;
    margin-top: 16px;
}

.about-card {
    background: #ffffff;
    padding: 20px;
    border-radius: 10px;
    margin-top: 20px;
    line-height: 1.6;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

.about-card h3 {
    color: #3275a8;
    margin-top: 12px;
}

.about-card p,
.about-card li {
    color: #898989;
}

.disclaimer {
    background: rgba(254, 202, 202, 0.45);
    border-left: 6px solid #dc2626;
    padding: 20px;
    border-radius: 14px;
    color: #b91c1c;
    font-size: 15px;
    margin-top: 32px;
    line-height: 1.5;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦷 OriHealth: Oral Cancer AI Detection</div>", unsafe_allow_html=True)

st.markdown("""
<div class='subtitle'>
Upload an image of the oral cavity to receive an AI-based screening result.<br>
This tool is for educational purposes only and does not replace professional diagnosis.
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL CHECK ----------------
if not os.path.exists("oral_cancer_model.tflite"):
    st.error("Model file not found. Make sure oral_cancer_model.tflite is in your repository root folder.")
    st.stop()

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("<div class='upload-header'>Upload an oral cavity image (JPG or PNG)</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        x = preprocess(image)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Handles different output shapes safely
        if len(output_data.shape) == 2:
            raw_pred = float(output_data[0][0])
        else:
            raw_pred = float(output_data[0])

    risk_score = int(raw_pred * 100)

    if risk_score >= 71:
        st.markdown("<div class='abnormal'><strong>🔴 High Risk Detected</strong><br>The AI detected possible abnormalities.</div>", unsafe_allow_html=True)
    elif risk_score >= 41:
        st.markdown("<div class='abnormal'><strong>🟡 Moderate Risk Detected</strong><br>Some irregular features were identified.</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='normal'><strong>🟢 Low Risk</strong><br>No significant abnormalities detected.</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='score'>Risk Score: {risk_score}%</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT SECTION ----------------
st.markdown("""
<div class='about-card'>
<h3>What is Oral Cancer?</h3>
<p>Oral cancer starts in the mouth or throat and may affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection significantly improves survival rates.</p>

<h3>Why Oral Health Matters</h3>
<p>Regular dental check-ups help detect abnormalities early and prevent serious oral health issues.</p>

<h3>Warning Signs:</h3>
<ul>
<li>Sores that do not heal</li>
<li>White or red patches</li>
<li>Pain or numbness</li>
<li>Difficulty swallowing or chewing</li>
<li>Lumps or thickening</li>
</ul>
</div>
""", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class='disclaimer'>
<strong>Disclaimer</strong><br>
• This application is a student research project and is not a medical device.<br>
• Results must not be used for diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
