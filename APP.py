import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

# ---------------- DARK + COLORFUL THEME ----------------
st.markdown("""
<style>
body, .main {
    background-color: #0b0b0b;
}
h1, h2, h3 {
    color: #00e5ff;
}
label, p {
    color: #ffffff;
}
.stButton>button {
    background: linear-gradient(90deg, #ff00cc, #3333ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 0.6em 1.8em;
    border: none;
}
.stButton>button:hover {
    transform: scale(1.05);
}
.result {
    padding: 25px;
    border-radius: 15px;
    font-size: 22px;
    text-align: center;
}
.benign {
    background-color: #003d2b;
    color: #00ff99;
}
.malignant {
    background-color: #3d0000;
    color: #ff4c4c;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("best_ml_model.pkl", "rb") as f:
    model = pickle.load(f)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🧬 Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align:center;'>Enter tumor measurements to predict whether it is <b>Benign</b> or <b>Malignant</b></p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Tumor Feature Inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📐 Size Features")
    radius_mean = st.slider("Mean Radius", 5.0, 30.0, 14.0)
    perimeter_mean = st.slider("Mean Perimeter", 40.0, 200.0, 90.0)
    area_mean = st.slider("Mean Area", 100.0, 3000.0, 600.0)

with col2:
    st.markdown("### 🧪 Texture Features")
    texture_mean = st.slider("Mean Texture", 5.0, 40.0, 19.0)
    smoothness_mean = st.slider("Mean Smoothness", 0.05, 0.3, 0.10)
    symmetry_mean = st.slider("Mean Symmetry", 0.1, 0.4, 0.20)

with col3:
    st.markdown("### ⚙️ Shape & Complexity")
    compactness_mean = st.slider("Mean Compactness", 0.0, 1.0, 0.20)
    concavity_mean = st.slider("Mean Concavity", 0.0, 1.0, 0.20)
    concave_points_mean = st.slider("Mean Concave Points", 0.0, 0.5, 0.10)
    fractal_dimension_mean = st.slider("Mean Fractal Dimension", 0.01, 0.1, 0.05)

# ---------------- PREDICTION ----------------
st.markdown("<br>")

if st.button("🔍 Predict Cancer Type"):
    input_data = np.array([[
        radius_mean,
        texture_mean,
        perimeter_mean,
        area_mean,
        smoothness_mean,
        compactness_mean,
        concavity_mean,
        concave_points_mean,
        symmetry_mean,
        fractal_dimension_mean
    ]])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.markdown(
            "<div class='result benign'>✅ Prediction: <b>Benign Tumor</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result malignant'>⚠️ Prediction: <b>Malignant Tumor</b></div>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Built using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
