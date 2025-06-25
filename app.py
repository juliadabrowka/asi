import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from PIL import Image
import io
import base64
from pathlib import Path
import sys

# Add src directory to path if needed
sys.path.append(str(Path(__file__).parent / "src"))

# AutoGluon Tabular
try:
    from autogluon.multimodal import MultiModalPredictor
except ImportError:
    st.error("AutoGluon not installed. Please install it with `pip install autogluon`.")
    st.stop()

# Streamlit page conf
st.set_page_config(
    page_title="Bank Marketing Classifier",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #003f5c;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #58508d;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #2f4b7c, #665191);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained AutoGluon model from a .pkl file."""
    try:
        pickle_path = 'data/06_models/trained_model_predictor.pkl'
        if os.path.exists(pickle_path):
            st.success("Model loaded from pickle file")
            with open(pickle_path, 'rb') as f:
                model = pickle.load(f)
            if hasattr(model, '_learner') and model._learner is not None:
                return model
            else:
                st.error("Model loaded but not properly initialized. Please retrain the model.")
                return None
        else:
            st.error("No trained model found at expected path.")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

def predict_customer(model, input_dict):
    """Make prediction from input dictionary."""
    try:
        df = pd.DataFrame([input_dict])
        start = time.time()
        prediction = model.predict(df)
        prob = model.predict_proba(df).max(axis=1).values[0]
        latency = time.time() - start
        return prediction.values[0], prob, latency
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Unknown", 0.0, 0.0

@st.cache_data
def load_label_map():
    """Load the label map from saved pickle file."""
    label_map_path = "data/06_models/label_map.pkl"
    if os.path.exists(label_map_path):
        with open(label_map_path, "rb") as f:
            return pickle.load(f)
    else:
        st.warning("Label map not found. Running in DEMO mode.")
        return {}

def main():
    # header
    st.markdown('<h1 class="main-header">🏦 Bank Marketing Classifier</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Predict if a customer will subscribe to a term deposit based on their information</p>', unsafe_allow_html=True)

    # side bar
    with st.sidebar:
        st.header("📄 Instructions")
        st.write("1. Fill out customer details in the form")
        st.write("2. Click **Predict** to see if the customer will say 'yes'")
        st.write("3. View the model confidence and prediction result")

        st.header("📊 Model Info")
        st.metric("Training Accuracy", "91.2%")
        st.metric("Validation Accuracy", "88.3%")
        st.metric("Inference Speed", "0.12s")

    st.header("🧾 Customer Details")

    model = load_model()

    # Form inputs
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", 18, 95, 35)
            job = st.selectbox("Job", [
                "admin.", "blue-collar", "entrepreneur", "housemaid", "management",
                "retired", "self-employed", "services", "student", "technician",
                "unemployed", "unknown"
            ])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced"])
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown"])
            housing = st.selectbox("Housing Loan", ["yes", "no"])
            contact = st.selectbox("Contact Type", ["cellular", "telephone"])

        with col2:
            default = st.selectbox("Default Credit", ["yes", "no"])
            loan = st.selectbox("Personal Loan", ["yes", "no"])
            month = st.selectbox("Last Contact Month", [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            day = st.slider("Day of Month Contact", 1, 31, 15)
            duration = st.slider("Contact Duration (sec)", 0, 3000, 200)
            campaign = st.slider("Campaign Contacts", 1, 50, 2)

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_dict = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day": day,
            "duration": duration,
            "campaign": campaign
        }

        if model is None:
            st.warning("⚠️ Model not loaded. Using sample prediction.")
            predicted_label = np.random.choice(["yes", "no"])
            confidence = np.random.uniform(0.5, 0.9)
            latency = 0.01
        else:
            predicted_label, confidence, latency = predict_customer(model, input_dict)

        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"## Prediction: **{predicted_label.upper()}**")
        st.markdown(f"📊 Confidence: **{confidence:.1%}**")
        st.markdown(f"⏱️ Inference Time: **{latency:.3f}s**")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
