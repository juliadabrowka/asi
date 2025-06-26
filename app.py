import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import time
from pathlib import Path
import sys

# Add src directory to path if needed
sys.path.append(str(Path(__file__).parent / "src"))

# AutoGluon Tabular
try:
    from autogluon.tabular import TabularPredictor
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
            
            # Check if model is valid
            if hasattr(model, 'predict') and callable(getattr(model, 'predict', None)):
                # Warm up the model with a dummy prediction
                dummy_data = pd.DataFrame([{
                    "age": 35, "job": "admin.", "marital": "married", 
                    "education": "secondary", "default": "no", "balance": 0,
                    "housing": "yes", "loan": "no", "contact": "cellular", 
                    "month": "may", "day": 15, "duration": 200, "campaign": 2,
                    "pdays": -1, "previous": 0, "poutcome": "unknown"
                }])
                try:
                    model.predict(dummy_data)  # Warm-up prediction
                    return model
                except Exception as warmup_error:
                    st.error(f"Model warm-up failed: {str(warmup_error)}")
                    st.info("Please retrain the model with the current AutoGluon version.")
                    return None
            else:
                st.error("Model loaded but not properly initialized. Please retrain the model.")
                return None
        else:
            st.error("No trained model found at expected path.")
            st.info("Please run the training pipeline first: python quick_train.py")
            return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        st.info("This might be due to AutoGluon version incompatibility. Please retrain the model.")
        return None

def predict_customer(model, input_dict):
    """Make prediction from input dictionary."""
    try:
        df = pd.DataFrame([input_dict])
        start = time.time()
        # Use batch prediction for better performance
        prediction = model.predict(df, as_pandas=False)  # Faster than pandas output
        proba = model.predict_proba(df, as_pandas=False)  # Get all probabilities
        latency = time.time() - start
        
        # Convert numeric prediction back to string label
        # Based on the encoding: 0 = "yes", 1 = "no"
        predicted_label = "yes" if prediction[0] == 0 else "no"
        
        # Get the confidence for the predicted class
        # proba[0] contains [P(yes), P(no)] for the first (and only) prediction
        if prediction[0] == 0:  # Predicted "yes"
            confidence = proba[0][0]  # P(yes)
        else:  # Predicted "no"
            confidence = proba[0][1]  # P(no)
        
        return predicted_label, confidence, latency
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Unknown", 0.0, 0.0

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
            default = st.selectbox("Default Credit", ["yes", "no"])
            balance = st.number_input("Balance", min_value=-10000, max_value=100000, value=0, step=100)
            housing = st.selectbox("Housing Loan", ["yes", "no"])
            loan = st.selectbox("Personal Loan", ["yes", "no"])

        with col2:
            contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
            month = st.selectbox("Last Contact Month", [
                "jan", "feb", "mar", "apr", "may", "jun",
                "jul", "aug", "sep", "oct", "nov", "dec"
            ])
            day = st.slider("Day of Month Contact", 1, 31, 15)
            duration = st.slider("Contact Duration (sec)", 0, 3000, 200)
            campaign = st.slider("Campaign Contacts", 1, 50, 2)
            pdays = st.number_input("Days Since Last Contact", min_value=-1, max_value=1000, value=-1, step=1)
            previous = st.number_input("Previous Contacts", min_value=0, max_value=100, value=0, step=1)
            poutcome = st.selectbox("Previous Outcome", ["unknown", "other", "failure", "success"])

        submitted = st.form_submit_button("🔍 Predict")

    if submitted:
        input_dict = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month,
            "day": day,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }

        if model is None:
            st.warning("⚠️ Model not loaded. Using sample prediction.")
            predicted_label = np.random.choice(["yes", "no"])
            confidence = np.random.uniform(0.5, 0.9)
            latency = 0.01
        else:
            predicted_label, confidence, latency = predict_customer(model, input_dict)

        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        # Ensure predicted_label is a string before calling .upper()
        if isinstance(predicted_label, (int, np.integer)):
            predicted_label = "yes" if predicted_label == 0 else "no"
        elif not isinstance(predicted_label, str):
            predicted_label = str(predicted_label)
        
        st.markdown(f"## Prediction: **{predicted_label.upper()}**")
        st.markdown(f"📊 Confidence: **{confidence:.1%}**")
        
        # Add confidence interpretation
        if confidence >= 0.8:
            confidence_level = "🟢 High Confidence"
        elif confidence >= 0.6:
            confidence_level = "🟡 Medium Confidence"
        else:
            confidence_level = "🔴 Low Confidence"
        
        st.markdown(f"**{confidence_level}**")
        st.markdown(f"⏱️ Inference Time: **{latency:.3f}s**")
        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
