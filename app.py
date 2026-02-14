import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ---------------------------
# PAGE CONFIG
# ---------------------------
st.set_page_config(
    page_title="Oral Cancer AI Detector",
    layout="centered"
)

# ---------------------------
# GLOBAL STYLING
# ---------------------------
st.markdown("""
<style>

/* Import Lora font */
@import url('https://fonts.googleapis.com/css2?family=Lora:wght@400&display=swap');

/* Force font everywhere */
* {
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
    color: #3275a8;
    text-align: center;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 36px;
}

/* Reusable white card */
.white-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 14px;
    margin-top: 24px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}

/* Upload header */
.upload-header {
    color: #3275a8;
    font-size: 20px;
    margin-bottom: 12px;
}

/* Risk score */
.score {
    margin-top: 14px;
    color: #374151;
}

/* Results */
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

/* About Section (with bottom shadow) */
.about-card {
    background: #ffffff;
    padding: 24px;
    border-radius: 14px;
    margin-top: 32px;
    line-height: 1.6;
    box-shadow: 0 8px 18px rgba(0, 0, 0, 0.06);
}

.about-card h3 {
    color: #3275a8;
    margin-top: 18px;
}

.about-card p,
.about-card li {
    color: #898989;
}

/* Disclaimer */
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

# ---------------------------
# TITLE SECTION
# ---------------------------
st.markdown('<div class="title">AI-Based Oral Cancer Risk Detection</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an oral cavity image to receive a preliminary risk assessment.</div>', unsafe_allow_html=True)

# ---------------------------
# MODEL LOADING
# ---------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.h5")

model = load_model()

# ---------------------------
# IMAGE UPLOAD WHITE BLOCK
# ---------------------------
st.markdown('<div class="white-card">', unsafe_allow_html=True)
st.markdown('<div class="upload-header">Upload an Image for Testing</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0][0]

    # Convert to percentage
    risk_percentage = round(prediction * 100, 2)

    st.markdown(f'<div class="score"><b>Risk Score:</b> {risk_percentage}%</div>', unsafe_allow_html=True)

    if prediction > 0.5:
        st.markdown(
            '<div class="abnormal"><b>Higher Risk Detected.</b> Please consult a qualified healthcare professional for further evaluation.</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            '<div class="normal"><b>Lower Risk Detected.</b> Continue regular oral health check-ups.</div>',
            unsafe_allow_html=True
        )

st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------
# ABOUT ORAL CANCER SECTION
# ---------------------------
st.markdown("""
<div class="about-card">
<h3>About Oral Cancer</h3>
<p>
Oral cancer is a serious condition that affects the tissues of the mouth and throat. 
Early detection significantly improves survival rates and treatment outcomes.
</p>

<h3>Common Warning Signs</h3>
<ul>
<li>Persistent mouth sores</li>
<li>White or red patches inside the mouth</li>
<li>Difficulty swallowing</li>
<li>Unexplained bleeding or numbness</li>
</ul>

<p>
This AI tool is designed to assist in early risk awareness and is not a replacement for professional medical diagnosis.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------------------
# DISCLAIMER
# ---------------------------
st.markdown("""
<div class="disclaimer">
<b>Medical Disclaimer:</b><br><br>
This application is intended for educational and research purposes only. 
It does not provide medical advice, diagnosis, or treatment. 
Always seek the advice of a qualified healthcare professional with any medical concerns.
</div>
""", unsafe_allow_html=True)
