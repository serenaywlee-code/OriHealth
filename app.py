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

# ---------------- CUSTOM STYLES ----------------
st.markdown("""
<style>

/* Background */
body, .stApp {
    background-color: #D9EDEB;
}

/* Main Title (keep original blue) */
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
    margin-top: 25px;
}

/* Body Text */
.body-text {
    color: #898989;
    line-height: 1.7;
    font-size: 15px;
}

/* Risk Box */
.risk-box {
    margin-top: 20px;
    padding: 20px;
    border-radius: 12px;
    font-weight: 600;
}

/* Low */
.low-risk {
    background-color: #e6f9ed;
    border-left: 6px solid #22c55e
