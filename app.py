import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load the trained model using pickle
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

# Title & Description
st.title("🏗️ Concrete Strength Prediction")
st.markdown("Enter the mix proportions or select a predefined mix ratio to predict the compressive strength of concrete.")

# Sidebar Layout
st.sidebar.title("🔢 Input Features")

# Concrete Mix Ratio Options
mix_ratios = {
    "Custom (Manual Entry)": None,
    "1:2:4 (Normal Strength)": (1, 2, 4, 0.5),
    "1:1.5:3 (Structural)": (1, 1.5, 3, 0.45),
    "1:1:2 (High Strength)": (1, 1, 2, 0.4)
}

selected_ratio = st.sidebar.selectbox("Select a Concrete Mix Ratio", list(mix_ratios.keys()))

# Default Cement and Water Values
cement_default = 400  # kg/m³ (typical value for 1m³ concrete)
water_default = 180  # kg/m³

# Calculate mix proportions if a predefined ratio is selected
if selected_ratio != "Custom (Manual Entry)":
    c_ratio, fa_ratio, ca_ratio, w_ratio = mix_ratios[selected_ratio]
    total_parts = c_ratio + fa_ratio + ca_ratio
    cement = cement_default
    fine_aggregate = (cement / c_ratio) * fa_ratio
    coarse_aggregate = (cement / c_ratio) * ca_ratio
    water = cement * w_ratio
    glass_powder = 0  # Default to 0 unless manually set
    superplasticizer = 0
else:
    cement = st.sidebar.slider("Cement (kg/m³)", 100, 800, cement_default)
    fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 400, 1000, 700)
    coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 800, 1300, 1000)
    water = st.sidebar.slider("Water (kg/m³)", 100, 250, water_default)
    glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0, 300, 50)
    superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 30.0, 5.0)

days = st.sidebar.slider("Curing Days", 1, 365, 28)

# Prepare Input Data
input_data = np.array([[cement, glass_powder, fine_aggregate, coarse_aggregate, water, superplasticizer, days]])
columns = ["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]
input_df = pd.DataFrame(input_data, columns=columns)

# Check if total volume is reasonable (~1000 kg/m³)
total_volume = cement + fine_aggregate + coarse_aggregate + water
if total_volume < 900 or total_volume > 1100:
    st.sidebar.warning("⚠️ Warning: The total materials do not sum up to approximately 1000 kg/m³. Please adjust the values.")

# Predict Compressive Strength
if st.sidebar.button("🔍 Predict Strength"):
    prediction = model.predict(input_df)[0]  # Get the prediction
    st.success(f"✅ **Predicted Compressive Strength:** {prediction:.2f} MPa")
    st.balloons()

# Footer
st.markdown("---")
st.markdown("📌 Built with **XGBoost + Streamlit** | 🚀 Deployed on **Streamlit Cloud**")
