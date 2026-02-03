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
/* Page background */
body, .block-container, .main {
    background-color: #dff0fb !important;  /* Light blue full page */
}

/* Title */
.title {
    font-size: 40px;
    font-weight: 700;
    color: #3275a8;
    text-align: center;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 36px;
}

/* Card style */
.card {
    background-color: white;
    border-radius: 16px;
    padding: 32px 24px;  /* Padding inside the card */
    margin-top: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
    line-height: 1.6;  /* makes text easier to read */
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

/* Risk score */
.score {
    margin-top: 14px;
    font-weight: 600;
    color: #374151;
}

/* Disclaimer */
.disclaimer {
    background: rgba(254, 202, 202, 0.45);
    border-left: 6px solid #dc2626;
    padding: 24px;
    border-radius: 14px;
    color: #b91c1c;
    font-size: 15px;
    margin-top: 32px;
}

/* Upload section header text */
.upload-header {
    color: #3275a8;
    font-weight: 600;
    font-size: 20px;
    margin-bottom: 12px;
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

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown(
    "<div class='upload-header'>Upload an oral cavity image (JPG or PNG)</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

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
        st.markdown(f"""
        <div class='abnormal'>
        <strong>🔴 High Risk Detected</strong><br>
        The AI detected features associated with possible abnormalities.
        </div>
        """, unsafe_allow_html=True)
    elif risk_score >= 41:
        st.markdown(f"""
        <div class='abnormal'>
        <strong>🟡 Moderate Risk Detected</strong><br>
        Some irregular features were identified.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='normal'>
        <strong>🟢 Low Risk</strong><br>
        No significant abnormalities detected.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        f"<div class='score'>Risk Score: {risk_score} / 100</div>",
        unsafe_allow_html=True
    )

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT ORAL CANCER ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

st.markdown("""
<h3 style='color:#3275a8;'>About Oral Cancer & Oral Health</h3>

<p><strong>What is Oral Cancer?</strong><br>
Oral cancer is a type of cancer that starts in the mouth or throat. It can affect the lips, tongue, cheeks, the floor and roof of the mouth, sinuses, or throat. Finding it early is very important because treatment works better and survival chances are higher.</p>

<p><strong>Why Oral Health Matters</strong><br>
Maintaining good oral health is essential not only for preventing cavities and gum disease, but also for early detection of oral cancer and other serious conditions. Regular dental check-ups and self-examinations can help identify abnormalities early.</p>

<p><strong>Key Reasons to Prioritize Oral Health:</strong></p>
<ul>
<li>Early Detection: Regular oral exams catch cancer and precancerous lesions early.</li>
<li>Better Outcomes: Early detection leads to 80-90% survival rate.</li>
<li>Overall Health Connection: Oral health affects heart, diabetes, and respiratory health.</li>
<li>Prevention: Good hygiene and check-ups prevent serious problems.</li>
</ul>

<p><strong>Warning Signs to Watch For:</strong></p>
<ul>
<li>Sores or lesions in the mouth that don't heal</li>
<li>White or red patches in the mouth</li>
<li>Persistent pain or numbness in the mouth or lips</li>
<li>Difficulty swallowing, chewing, or moving the jaw</li>
<li>Lumps or thickening in the cheek or neck</li>
</ul>
""", unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
<div class='disclaimer'>
<strong>Disclaimer</strong><br>
• This application is a student research project and is not a medical device.<br>
• Results must not be used for diagnosis or treatment decisions.
</div>
""", unsafe_allow_html=True)
