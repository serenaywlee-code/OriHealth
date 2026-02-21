import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------------------------
# Page Config
# -------------------------
st.set_page_config(page_title="Oral Cancer AI Detection", layout="centered")

# -------------------------
# Custom Styling
# -------------------------
st.markdown("""
<style>
body {
    font-family: 'Lora', serif;
}
.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.disclaimer-box {
    background-color: #ffe6e6;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ff4d4d;
}
</style>
""", unsafe_allow_html=True)

# -------------------------
# Load TFLite Model
# -------------------------
import os

@st.cache_resource
def load_model():
    if not os.path.exists("oral_cancer_model.tflite"):
        st.error("Model file not found.")
        st.stop()

    interpreter = tf.lite.Interpreter(
        model_path="oral_cancer_model.tflite",
        num_threads=1
    )
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# -------------------------
# Title
# -------------------------
st.title("Oral Cancer AI Risk Assessment")

# -------------------------
# About Section
# -------------------------
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("What is Oral Cancer?")
st.write("""
Oral cancer develops in the tissues of the mouth or throat. 
Early detection significantly improves treatment outcomes.
This AI tool provides a risk estimate based on image analysis.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -------------------------
# Upload Section
# -------------------------
st.subheader("Upload an Image for Analysis")

uploaded = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

# -------------------------
# Prediction
# -------------------------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0).astype(np.float32)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0][0]

    risk_percentage = float(prediction) * 100

    st.subheader("Risk Score")
    st.write(f"### {risk_percentage:.2f}%")

# -------------------------
# Disclaimer
# -------------------------
st.markdown('<div class="disclaimer-box">', unsafe_allow_html=True)
st.write("""
**Disclaimer:**  
This AI tool is for educational and research purposes only.  
It does not provide medical diagnosis.  
Please consult a qualified healthcare professional for medical advice.
""")
st.markdown('</div>', unsafe_allow_html=True)
