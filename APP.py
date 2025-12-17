import streamlit as st
import pickle
import numpy as np

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🧬",
    layout="wide"
)

# ---------------- CUSTOM CSS (BLACK + COLORFUL) ----------------
st.markdown("""
<style>
body {
    background-color: #0e0e0e;
}
.main {
    background-color: #0e0e0e;
}
h1, h2, h3, h4 {
    color: #00e5ff;
}
p, label {
    color: #ffffff;
}
div[data-baseweb="input"] input {
    background-color: #1c1c1c;
    color: white;
    border-radius: 8px;
}
div[data-baseweb="slider"] {
    color: white;
}
.stButton > button {
    background: linear-gradient(90deg, #ff00cc, #3333ff);
    color: white;
    font-size: 18px;
    border-radius: 12px;
    padding: 0.6em 1.5em;
    border: none;
}
.stButton > button:hover {
    background: linear-gradient(90deg, #3333ff, #ff00cc);
    transform: scale(1.05);
}
.result-box {
    padding: 25px;
    border-radius: 15px;
    font-size: 22px;
    text-align: center;
    margin-top: 20px;
}
.success {
    background-color: #003300;
    color: #00ff99;
}
.danger {
    background-color: #330000;
    color: #ff4c4c;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
with open("best_ml_model.pkl", "rb") as file:
    model = pickle.load(file)

# ---------------- TITLE ----------------
st.markdown("<h1 style='text-align: center;'>🧬 Breast Cancer Prediction System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>AI-powered model to predict whether a tumor is <b>Benign</b> or <b>Malignant</b></p>", unsafe_allow_html=True)
st.markdown("---")

# ---------------- INPUT SECTION ----------------
st.subheader("📊 Enter Patient Details")

col1, col2, col3 = st.columns(3)

with col1:
    radius_mean = st.number_input("Radius Mean", 0.0, 50.0, 14.0)
    texture_mean = st.number_input("Texture Mean", 0.0, 50.0, 19.0)
    perimeter_mean = st.number_input("Perimeter Mean", 0.0, 200.0, 90.0)
    area_mean = st.number_input("Area Mean", 0.0, 3000.0, 600.0)

with col2:
    smoothness_mean = st.number_input("Smoothness Mean", 0.0, 1.0, 0.1)
    compactness_mean = st.number_input("Compactness Mean", 0.0, 1.0, 0.2)
    concavity_mean = st.number_input("Concavity Mean", 0.0, 1.0, 0.2)
    concave_points_mean = st.number_input("Concave Points Mean", 0.0, 1.0, 0.1)

with col3:
    symmetry_mean = st.number_input("Symmetry Mean", 0.0, 1.0, 0.2)
    fractal_dimension_mean = st.number_input("Fractal Dimension Mean", 0.0, 1.0, 0.05)

# ---------------- PREDICTION ----------------
st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Cancer Type"):
    features = np.array([[
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

    prediction = model.predict(features)[0]

    if prediction == 1:
        st.markdown(
            "<div class='result-box success'>✅ Prediction: <b>Benign Tumor</b></div>",
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            "<div class='result-box danger'>⚠️ Prediction: <b>Malignant Tumor</b></div>",
            unsafe_allow_html=True
        )

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>Made with ❤️ using Machine Learning & Streamlit</p>",
    unsafe_allow_html=True
)
