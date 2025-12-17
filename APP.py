import streamlit as st
import numpy as np
import pandas as pd
import pickle
import random

# --------------------------
# Load the trained model
# --------------------------
MODEL_PATH = "best_ml_model.pkl"   # <-- YOUR PICKLE FILE

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(
    page_title="Breast Cancer Prediction",
    page_icon="🎗️",
    layout="wide"
)

st.title("🎗️ Breast Cancer Prediction App")
st.markdown("""
This app predicts whether a breast tumor is **Benign** or **Malignant**  
based on features extracted from a digitized image of a fine needle aspirate (FNA).
""")

# --------------------------
# Feature names (FIXED)
# --------------------------
feature_names = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
    'smoothness_mean', 'compactness_mean', 'concavity_mean',
    'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se',
    'fractal_dimension_se', 'radius_worst', 'texture_worst',
    'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave_points_worst',
    'symmetry_worst', 'fractal_dimension_worst'
]

# --------------------------
# CSV Upload Section
# --------------------------
st.sidebar.header("📂 Upload CSV Data (Optional)")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV with breast cancer features",
    type=["csv"]
)

default_values = []

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)

        if all(col in data.columns for col in feature_names):
            random_row = data.sample(1, random_state=random.randint(0, 9999))
            default_values = random_row[feature_names].iloc[0].values
            st.sidebar.success("✅ CSV uploaded. Random row loaded.")
        else:
            st.sidebar.error("❌ CSV missing required columns. Using random values.")
            default_values = np.random.uniform(0.0, 1.0, size=len(feature_names))

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")
        default_values = np.random.uniform(0.0, 1.0, size=len(feature_names))
else:
    default_values = np.random.uniform(0.0, 1.0, size=len(feature_names))

# --------------------------
# Input fields
# --------------------------
st.header("🔢 Enter Tumor Feature Values")
cols = st.columns(3)
inputs = []

for i, feature in enumerate(feature_names):
    with cols[i % 3]:
        value = st.number_input(
            label=feature.replace('_', ' ').title(),
            value=float(default_values[i]),
            format="%.6f"
        )
        inputs.append(value)

# --------------------------
# Prediction
# --------------------------
if st.button("🔍 Predict"):
    try:
        # Convert inputs to numpy array
        inputs_array = np.array(inputs)

        # 🔧 FIX: Add missing feature if model expects 31
        if hasattr(model, "n_features_in_"):
            expected_features = model.n_features_in_
            if inputs_array.shape[0] < expected_features:
                missing = expected_features - inputs_array.shape[0]
                inputs_array = np.append(inputs_array, [0.0] * missing)

        input_data = inputs_array.reshape(1, -1)

        prediction = model.predict(input_data)[0]

        # Confidence (if available)
        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = np.max(model.predict_proba(input_data)) * 100

        # Correct label mapping
        result = "Benign" if prediction == 1 else "Malignant"

        st.subheader("🩺 Prediction Result")

        if result == "Benign":
            st.success(f"✅ The tumor is predicted to be **{result}**.")
        else:
            st.error(f"⚠️ The tumor is predicted to be **{result}**.")

        if confidence is not None:
            st.info(f"Model confidence: **{confidence:.2f}%**")

    except Exception as e:
        st.error(f"Prediction error: {e}")
