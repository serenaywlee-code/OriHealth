import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf

# -------- Page Config --------
st.set_page_config(page_title="Oral Cancer AI", layout="centered")

# -------- Load Model --------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.h5")

model = load_model()

# -------- Styling --------
st.markdown("""
<style>
body {
    font-family: 'Lora', serif;
}
.disclaimer-box {
    background-color: #ffe6e6;
    padding: 15px;
    border-radius: 10px;
    border: 1px solid #ff4d4d;
}
.info-box {
    background-color: white;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)

# -------- Title --------
st.title("Oral Cancer AI Detection")

# -------- About Section --------
st.markdown('<div class="info-box">', unsafe_allow_html=True)
st.subheader("What is Oral Cancer?")
st.write("""
Oral cancer refers to cancer that develops in the mouth or throat tissues. 
Early detection is critical for improving survival outcomes.
""")
st.markdown('</div>', unsafe_allow_html=True)

st.write("")

# -------- Upload Section --------
st.subheader("Upload an Image for Analysis")

uploaded = st.file_uploader(
    "Upload image",
    type=["jpg", "jpeg", "png"],
    key="image_uploader"
)

# -------- Prediction --------
if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    prediction = model.predict(image)[0][0]
    risk_percentage = float(prediction) * 100

    st.subheader("Risk Score")
    st.write(f"### {risk_percentage:.2f}%")

# -------- Disclaimer --------
st.markdown('<div class="disclaimer-box">', unsafe_allow_html=True)
st.write("""
**Disclaimer:**  
This AI tool is for educational and research purposes only.  
It does not provide medical diagnosis.  
Please consult a healthcare professional for medical advice.
""")
st.markdown('</div>', unsafe_allow_html=True)
