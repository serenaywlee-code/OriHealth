import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Oral Cancer Detection", layout="centered")

st.title("AI Oral Cancer Image Detection")
st.write("Upload an oral cavity image (JPG or PNG)")

# Load model
@st.cache_resource
def load_model():
    interpreter = tf.lite.Interpreter(model_path="oral_cancer_model.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_model()

# File uploader
uploaded_file = st.file_uploader(
    "Upload an oral cavity image (JPG or PNG)",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    image = image.resize((224, 224))
    input_data = np.array(image, dtype=np.float32) / 255.0
    input_data = np.expand_dims(input_data, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])

    if prediction[0][0] > 0.5:
        st.error("⚠️ Suspicious lesion detected. Please consult a dental professional.")
    else:
        st.success("✅ No suspicious lesion detected.")
