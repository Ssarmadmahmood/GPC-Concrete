# Install necessary libraries (for local use)
import streamlit as st
import numpy as np
import pandas as pd
import joblib  # For loading the trained model

# ✅ Load the optimized XGBoost model (GWO)
model = joblib.load("optimized_xgb_gwo.pkl")  # Ensure the model file is in the repo

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

# ✅ Title & Description
st.title("🏗️ Concrete Strength Prediction")
st.markdown("Enter the mix proportions to predict the compressive strength of concrete.")

# ✅ Sidebar for better UI
st.sidebar.title("🔢 Input Features")

# ✅ Define Input Fields
cement = st.sidebar.slider("Cement (kg/m³)", 100, 800, 400)
glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0, 300, 50)
fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 400, 1000, 700)
coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 800, 1300, 1000)
water = st.sidebar.slider("Water (kg/m³)", 100, 250, 180)
superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 30.0, 5.0)
days = st.sidebar.slider("Curing Days", 1, 365, 28)

# ✅ Prepare Input Data
input_data = np.array([[cement, glass_powder, fine_aggregate, coarse_aggregate, water, superplasticizer, days]])
columns = ["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]
input_df = pd.DataFrame(input_data, columns=columns)

# ✅ Predict Compressive Strength
if st.sidebar.button("🔍 Predict Strength"):
    prediction = model.predict(input_df)[0]  # Get the prediction
    st.success(f"✅ **Predicted Compressive Strength:** {prediction:.2f} MPa")
    st.balloons()

# ✅ Footer
st.markdown("---")
st.markdown("📌 Built with **XGBoost + Streamlit** | 🚀 Deployed on **Streamlit Cloud**")
