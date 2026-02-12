@@ -17,6 +17,15 @@
    background-color: #dff0fb !important;  /* Light blue background */
}

/* Card style - everything inside */
.card {
    background-color: white;
    border-radius: 16px;
    padding: 32px 24px;
    margin-top: 20px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.1);
}

/* Title */
.title {
    font-size: 40px;
@@ -79,23 +88,6 @@
    margin-top: 32px;
    line-height: 1.5;
}

/* About Oral Cancer & Oral Health Section */
.about-text {
    color: #4a4a4a;  /* Grey text */
    line-height: 1.6;
    margin-top: 20px;
}

.about-text h3 {
    font-weight: bold;
    color: #4a4a4a;
}

.about-text p, .about-text li {
    color: #4a4a4a;
    font-weight: normal;
}
</style>
""", unsafe_allow_html=True)

@@ -126,64 +118,54 @@ def preprocess(img):
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)

# ---------------- UPLOAD + RESULTS CARD ----------------
st.markdown("<div class='card'>", unsafe_allow_html=True)

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
        st.markdown(f"<div class='abnormal'><strong>🔴 High Risk Detected</strong><br>The AI detected possible abnormalities.</div>", unsafe_allow_html=True)
    elif risk_score >= 41:
        st.markdown(f"<div class='abnormal'><strong>🟡 Moderate Risk Detected</strong><br>Some irregular features were identified.</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='normal'><strong>🟢 Low Risk</strong><br>No significant abnormalities detected.</div>", unsafe_allow_html=True)

    st.markdown(f"<div class='score'>Risk Score: {risk_score} / 100</div>", unsafe_allow_html=True)

# ---------------- ABOUT ORAL CANCER & ORAL HEALTH ----------------
st.markdown("""
<div class='about-text'>
<h3>About Oral Cancer & Oral Health</h3>

<h3>What is Oral Cancer?</h3>
<p>Oral cancer is a type of cancer that starts in the mouth or throat. It can affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection is very important because treatment works better and survival chances are higher.</p>

<h3>Why Oral Health Matters</h3>
<p>Maintaining good oral health prevents cavities, gum disease, and helps detect oral cancer early. Regular dental check-ups help identify abnormalities early.</p>

<h3>Key Reasons to Prioritize Oral Health:</h3>
<ul>
<li>Early Detection: Regular oral examinations can catch cancer in its earliest, most treatable stages.</li>
<li>Better Outcomes: Early detection has an 80-90% survival rate.</li>
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

st.markdown("</div>", unsafe_allow_html=True)
# ---------------- CARD CONTAINER ----------------
with st.container():
    st.markdown("<div class='card'>", unsafe_allow_html=True)

    # Upload header
    st.markdown("<div class='upload-header'>Upload an oral cavity image (JPG or PNG)</div>", unsafe_allow_html=True)

    # File uploader
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
            st.markdown(f"<div class='abnormal'><strong>🔴 High Risk Detected</strong><br>The AI detected possible abnormalities.</div>", unsafe_allow_html=True)
        elif risk_score >= 41:
            st.markdown(f"<div class='abnormal'><strong>🟡 Moderate Risk Detected</strong><br>Some irregular features were identified.</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='normal'><strong>🟢 Low Risk</strong><br>No significant abnormalities detected.</div>", unsafe_allow_html=True)

        st.markdown(f"<div class='score'>Risk Score: {risk_score} / 100</div>", unsafe_allow_html=True)

    # About Oral Cancer inside the same card
    st.markdown("""
    <h3 style='color:#3275a8;'>About Oral Cancer & Oral Health</h3>
    <p><strong>What is Oral Cancer?</strong><br>
    Oral cancer is a type of cancer that starts in the mouth or throat. It can affect lips, tongue, cheeks, floor/roof of mouth, sinuses, or throat. Early detection is very important because treatment works better and survival chances are higher.</p>
    <p><strong>Why Oral Health Matters</strong><br>
    Maintaining good oral health prevents cavities, gum disease, and helps detect oral cancer early. Regular dental check-ups help identify abnormalities early.</p>
    <p><strong>Warning Signs:</strong></p>
    <ul>
    <li>Sores or lesions that don’t heal</li>
    <li>White or red patches</li>
    <li>Pain or numbness in mouth/lips</li>
    <li>Difficulty swallowing, chewing, moving jaw</li>
    <li>Lumps or thickening in cheek/neck</li>
    </ul>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- DISCLAIMER ----------------
st.markdown("""
