import streamlit as st
import numpy as np
from PIL import Image
import tflite_runtime.interpreter as tflite
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
@@ -13,21 +14,17 @@
# ---------------- STYLES ----------------
st.markdown("""
<style>
/* Import Lora font */
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400&display=swap');

/* Apply Lora globally */
html, body, [class*="css"]  {
    font-family: 'Lora', serif !important;
    font-weight: 400 !important;
}

/* Background */
.stApp {
    background-color: #D9EDEB;
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 400;
@@ -36,15 +33,13 @@
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 36px;
}

/* Upload Section */
.upload-section {
    background: #ffffff;
    padding: 20px;
@@ -53,21 +48,18 @@
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Upload header */
.upload-header {
    color: #3275a8;
    font-size: 20px;
    margin-bottom: 12px;
}

/* Risk Score */
.score {
    margin-top: 14px;
    color: #374151;
    font-weight: 600;
}

/* Results */
.normal {
    background: #f0fdf4;
    border-left: 6px solid #22c55e;
@@ -86,7 +78,6 @@
    margin-top: 16px;
}

/* About Section */
.about-card {
    background: #ffffff;
    padding: 20px;
@@ -106,7 +97,6 @@
    color: #898989;
}

/* Disclaimer */
.disclaimer {
    background: rgba(254, 202, 202, 0.45);
    border-left: 6px solid #dc2626;
@@ -117,40 +107,46 @@
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
    interpreter = tflite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- UPLOAD SECTION ----------------
st.markdown("<div class='upload-section'>", unsafe_allow_html=True)
st.markdown("<div class='upload-header'>Upload an oral cavity image (JPG or PNG)</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg","jpeg","png"], key="file_uploader")
uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
@@ -160,7 +156,14 @@
        x = preprocess(image)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        raw_pred = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

        output_data = interpreter.get_tensor(output_details[0]["index"])

        # Handles different output shapes safely
        if len(output_data.shape) == 2:
            raw_pred = float(output_data[0][0])
        else:
            raw_pred = float(output_data[0])

    risk_score = int(raw_pred * 100)

@@ -175,39 +178,31 @@

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT ORAL CANCER ----------------
# ---------------- ABOUT SECTION ----------------
st.markdown("""
<div class='about-card'>
<h3>What is Oral Cancer?</h3>
<p>Oral cancer is a type of cancer that starts in the mouth or throat. It can affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection is very important because treatment works better and survival chances are higher.</p>
<p>Oral cancer starts in the mouth or throat and may affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection significantly improves survival rates.</p>

<h3>Why Oral Health Matters</h3>
<p>Maintaining good oral health prevents cavities, gum disease, and helps detect oral cancer early. Regular dental check-ups help identify abnormalities early.</p>

<h3>Key Reasons to Prioritize Oral Health:</h3>
<ul>
<li>Early Detection: Regular oral examinations can catch cancer in its earliest, most treatable stages.</li>
<li>Better Outcomes: Early detection has an 80–90% survival rate.</li>
<li>Overall Health: Oral health is linked to heart, diabetes, and respiratory health.</li>
<li>Prevention: Good hygiene and regular check-ups prevent serious oral problems.</li>
</ul>
<p>Regular dental check-ups help detect abnormalities early and prevent serious oral health issues.</p>

<h3>Warning Signs to Watch For:</h3>
<h3>Warning Signs:</h3>
<ul>
<li>Sores or lesions that don’t heal</li>
<li>Sores that do not heal</li>
<li>White or red patches</li>
<li>Pain or numbness in mouth/lips</li>
<li>Difficulty swallowing, chewing, or moving jaw</li>
<li>Lumps or thickening in cheek/neck</li>
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
