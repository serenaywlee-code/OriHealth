import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
import os

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="OriHealth",
    layout="centered",
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', sans-serif;
}

/* Background Gradient */
.stApp {
    background: linear-gradient(135deg, #E6F4F1 0%, #F8FBFF 100%);
}

/* Hero Section */
.hero {
    text-align: center;
    padding: 60px 20px 30px 20px;
}

.hero h1 {
    font-size: 48px;
    font-weight: 600;
    margin-bottom: 10px;
}

.hero p {
    font-size: 18px;
    color: #555;
    max-width: 600px;
    margin: 0 auto;
}

/* Card Style */
.card {
    background: white;
    padding: 40px;
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.05);
    margin-top: 40px;
}

/* Upload Title */
.card h2 {
    text-align: center;
    margin-bottom: 20px;
}

/* Result Boxes */
.result-good {
    background: #ECFDF5;
    border-left: 6px solid #10B981;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

.result-bad {
    background: #FEF2F2;
    border-left: 6px solid #EF4444;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
}

/* Section */
.section {
    margin-top: 80px;
    text-align: center;
    padding: 0 20px;
}

.section h3 {
    font-size: 28px;
    margin-bottom: 15px;
}

.section p {
    max-width: 700px;
    margin: 0 auto;
    color: #555;
}

/* Footer */
.footer {
    margin-top: 80px;
    text-align: center;
    font-size: 14px;
    color: #777;
    padding-bottom: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- HERO ----------------
st.markdown("""
<div class='hero'>
<h1>🦷 OriHealth</h1>
<p>AI-powered oral cancer screening tool designed for early detection awareness.
This educational tool does not replace professional diagnosis.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- MODEL ----------------
MODEL_PATH = "oral_cancer_model.tflite"

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found.")
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
    arr = np.array(img).astype(np.float32) / 255.0
    return np.expand_dims(arr, axis=0)

# ---------------- UPLOAD CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)
st.markdown("<h2>Upload Oral Cavity Image</h2>", unsafe_allow_html=True)

uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded)
    st.image(image, use_container_width=True)

    x = preprocess(image)

    interpreter.set_tensor(input_details[0]["index"], x)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]["index"])

    if len(output.shape) == 2:
        score = float(output[0][0])
    else:
        score = float(output[0])

    risk = int(score * 100)

    if risk >= 50:
        st.markdown(f"""
        <div class='result-bad'>
        <strong>⚠ Potential Risk Detected</strong><br>
        AI Confidence Score: {risk}%<br>
        Please consult a dental professional.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='result-good'>
        <strong>✅ No Significant Risk Detected</strong><br>
        AI Confidence Score: {risk}%<br>
        Continue regular dental check-ups.
        </div>
        """, unsafe_allow_html=True)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------- ABOUT SECTION ----------------
st.markdown("""
<div class='section'>
<h3>What is Oral Cancer?</h3>
<p>Oral cancer develops in the mouth or throat and can affect the lips, tongue,
cheeks, floor or roof of the mouth. Early detection significantly increases survival rates.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='footer'>
This application is a student research project and not a certified medical device.
Always seek professional dental care for diagnosis.
</div>
""", unsafe_allow_html=True)
