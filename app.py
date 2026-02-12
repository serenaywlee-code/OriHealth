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

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("oral_cancer_model.h5")

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

/* Background */
.stApp {
    background-color: #D9EDEB;
}

/* Main Title */
.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: 700;
    color: #3275a8;
    margin-bottom: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 16px;
    color: #4a4a4a;
    margin-bottom: 30px;
}

/* Section Titles */
.section-title {
    color: #5e99dd;
    font-weight: bold;
    font-size: 22px;
    margin-top: 35px;
}

/* Body Text */
.body-text {
    color: #898989;
    font-size: 15px;
    line-height: 1.7;
}

/* Risk Result Box */
.result-box {
    background-color: white;
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-weight: 600;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title'>Oral Cancer AI Detection</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Upload an oral cavity image (JPG or PNG)</div>", unsafe_allow_html=True)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)[0][0]
    risk_percentage = round(float(prediction) * 100, 2)

    # Display result
    st.markdown(
        f"<div class='result-box'>Predicted Risk Level: {risk_percentage}%</div>",
        unsafe_allow_html=True
    )

# ---------------- ABOUT SECTION ----------------
st.markdown("<div class='section-title'>About Oral Cancer & Oral Health</div>", unsafe_allow_html=True)

st.markdown("""
<div class='body-text'>
Oral cancer is a type of cancer that starts in the mouth or throat. 
It can affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. 
Early detection is very important because treatment works better and survival chances are higher.
<br><br>
Maintaining good oral health prevents cavities, gum disease, and helps detect oral cancer early. 
Regular dental check-ups help identify abnormalities early.
<br><br>
<b>Early Detection:</b> Regular oral examinations can catch cancer in its earliest, most treatable stages.
<br>
<b>Better Outcomes:</b> Early detection has an 80–90% survival rate.
<br>
<b>Overall Health:</b> Oral health is linked to heart, diabetes, and respiratory health.
<br>
<b>Prevention:</b> Good hygiene and regular check-ups prevent serious oral problems.
<br><br>
<b>Warning Signs to Watch For:</b>
<ul>
<li>Sores or lesions that don’t heal</li>
<li>White or red patches</li>
<li>Pain or numbness in mouth/lips</li>
<li>Difficulty swallowing, chewing, or moving jaw</li>
<li>Lumps or thickening in cheek/neck</li>
</ul>
</div>
""", unsafe_allow_html=True)
