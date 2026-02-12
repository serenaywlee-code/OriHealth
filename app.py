import streamlit as st
import numpy as np
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Oral Cancer AI Detection",
    page_icon="🦷",
    layout="centered"
)

# ---------------- CUSTOM STYLING ----------------
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
    margin-bottom: 40px;
}

/* Section Titles */
.section-title {
    color: #5e99dd;
    font-weight: bold;
    font-size: 22px;
    margin-top: 30px;
}

/* Body Text */
.body-text {
    color: #898989;
    font-size: 15px;
    line-height: 1.7;
}

/* Risk Box */
.risk-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 20px;
    font-weight: 600;
    background-color: white;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HEADER ----------------
st.markdown("<div class='main-title
