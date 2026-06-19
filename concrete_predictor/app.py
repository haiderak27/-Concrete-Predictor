# app.py - STREAMLIT VERSION
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Page config
st.set_page_config(
    page_title="Concrete Strength Predictor",
    page_icon="🏗️",
    layout="wide"
)

st.title("🏗️ Concrete Compressive Strength Predictor")
st.markdown("### Research Tool for Mix Design Optimization")

# Load model with caching
@st.cache_resource
def load_model():
    model_path = 'models/concrete_strength_model.pkl'
    scaler_path = 'models/concrete_scaler.pkl'
    
    if not os.path.exists(model_path):
        st.error("Model not found! Please ensure models are in the correct path.")
        return None, None
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_model()

if model is None:
    st.stop()

# Inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("📦 Primary Materials")
    cement = st.slider("Cement (kg/m³)", 100, 600, 350, help="Ordinary Portland Cement")
    ggbfs = st.slider("GGBFS (kg/m³)", 0, 450, 0, help="Ground Granulated Blast-furnace Slag")
    fly_ash = st.slider("Fly Ash (kg/m³)", 0, 600, 0, help="Fly Ash")
    water = st.slider("Water (kg/m³)", 100, 250, 175, help="Mixing water")

with col2:
    st.subheader("🧪 Additives & Age")
    superplasticizer = st.slider("Superplasticizer (kg/m³)", 0.0, 30.0, 0.0, 0.5)
    coarse_aggregate = st.slider("Coarse Aggregate (kg/m³)", 600, 1500, 1100)
    fine_aggregate = st.slider("Fine Aggregate (kg/m³)", 500, 1200, 700)
    age = st.slider("Curing Age (days)", 1, 365, 28)

# Predict button
if st.button("🔮 Predict Strength", type="primary"):
    # Prepare features
    features = np.array([[cement, ggbfs, fly_ash, water, superplasticizer, 
                          coarse_aggregate, fine_aggregate, age]])
    
    # Scale and predict
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    
    # Display results
    st.divider()
    st.subheader("📊 Prediction Results")
    
    col_r1, col_r2, col_r3 = st.columns(3)
    
    with col_r1:
        st.metric("💪 Compressive Strength", f"{prediction:.2f} MPa")
    
    # Grade classification
    if prediction >= 60:
        grade = "Very High Strength 🟢"
    elif prediction >= 40:
        grade = "High Strength 🟢"
    elif prediction >= 25:
        grade = "Medium Strength 🟡"
    else:
        grade = "Low Strength 🔴"
    
    with col_r2:
        st.metric("📋 Grade", grade)
    
    wc_ratio = water / cement
    with col_r3:
        st.metric("💧 Water/Cement Ratio", f"{wc_ratio:.3f}")
    
    # Mix summary
    st.divider()
    st.subheader("📋 Mix Summary")
    
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.write(f"**Binder Content:** {cement + ggbfs + fly_ash:.0f} kg/m³")
    with col_s2:
        st.write(f"**Total Aggregate:** {coarse_aggregate + fine_aggregate:.0f} kg/m³")
    with col_s3:
        st.write(f"**Superplasticizer:** {superplasticizer:.1f} kg/m³")

# Footer
st.divider()
st.caption("Built with ❤️ using Random Forest Machine Learning Model")
