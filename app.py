import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

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

/* Upload section */
.upload-section {
    background: #ffffff;
    padding: 23px;
    border-radius: 10px;
    margin-top: 10px;
    box-shadow: 20px 20px 12px rgba(0,0,0,0.05);
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

/* About Section */
.about-card {
    background: #ffffff;
    padding: 23px;
    border-radius: 10px;
    margin-top: 5px;
    line-height: 1.6;
    box-shadow: 25px 25px 12px rgba(0,0,0,0.05);
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

# ---------------- HEADER ----------------
st.markdown("<div class='title'>🦷 Oral Cancer AI Detection</div>", unsafe_allow_html=True)
st.markdown("""
<div class='subtitle'>
Upload an image of the oral cavity to receive an AI-based screening result.<br>
This tool is for educational purposes only and does not replace professional diagnosis.
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(
        model_path="oral_cancer_model_optimized.tflite"
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ---------------- UPLOAD SECTION ----------------

st.markdown("<div class='upload-header'>Upload an oral cavity image (JPG or PNG)</div>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    with st.spinner("Analyzing image..."):
        x = preprocess(image)
        interpreter.set_tensor(input_details[0]["index"], x)
        interpreter.invoke()
        raw_pred = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    risk_score = int(raw_pred * 100)

    if risk_score >= 71:
        st.markdown(
            "<div class='abnormal'><strong>🔴 High Risk Detected</strong><br>The AI detected possible abnormalities.</div>",
            unsafe_allow_html=True
        )
    elif risk_score >= 41:
        st.markdown(
            "<div class='abnormal'><strong>🟡 Moderate Risk Detected</strong><br>Some irregular features were identified.</div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='normal'><strong>🟢 Low Risk</strong><br>No significant abnormalities detected.</div>",
            unsafe_allow_html=True
        )

    st.markdown(f"<div class='score'>Risk Score: {risk_score}%</div>", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT SECTION ----------------
st.markdown("""
<div class='about-card'>

<h3>What is Oral Cancer?</h3>
<p>Oral cancer is a type of cancer that starts in the mouth or throat. It can affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection is very important because treatment works better and survival chances are higher.</p>

<h3>Why Oral Health Matters</h3>
<p>Maintaining good oral health prevents cavities, gum disease, and helps detect oral cancer early. Regular dental check-ups help identify abnormalities early.</p>

<h3>Key Reasons to Prioritize Oral Health:</h3>
<ul>
<li>Early Detection: Regular oral examinations can catch cancer in its earliest, most treatable stages.</li>
<li>Better Outcomes: Early detection has an 80–90% survival rate.</li>
<li>Overall Health: Oral health is linked to heart, diabetes, and respiratory health.</li>
<li>Prevention: Good hygiene and regular check-ups prevent serious oral problems.</li>
</ul>

<h3>Warning Signs to Watch For:</h3>
<ul>
<li>Sores or lesions that don’t heal</li>
<li>White or red patches</li>
<li>Pain or numbness in mouth/lips</li>
<li>Difficulty swallowing, chewing, or moving jaw</li>
<li>Lumps or thickening in cheek/neck</li>
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
