import streamlit as st
import numpy as np
import pandas as pd
import pickle  # ✅ Use pickle instead of joblib

# ✅ Load the trained model using pickle
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏰", layout="centered")

# ✅ Title & Description
st.title("🏗️ Concrete Strength Prediction")
st.markdown("Enter the mix proportions or select a predefined mix ratio to predict the compressive strength of concrete.")

# ✅ Sidebar for better UI
st.sidebar.title("🔢 Input Features")

# ✅ Define mix ratios (cement:sand:coarse aggregate)
mix_ratios = {
    "Custom (Manual Entry)": None,
    "1:2:4": (1, 2, 4, 0.5),
    "1:1.5:3": (1, 1.5, 3, 0.45),
    "1:3:6": (1, 3, 6, 0.6)
}

# ✅ Mix Ratio Selection
total_mass = 2400  # Approximate total density of concrete in kg/m³
selected_ratio = st.sidebar.selectbox("Select a Concrete Mix Ratio", list(mix_ratios.keys()))

# ✅ Define default values based on selection
if selected_ratio != "Custom (Manual Entry)":
    cement_ratio, fine_agg_ratio, coarse_agg_ratio, water_ratio = mix_ratios[selected_ratio]
    
    cement = total_mass * (cement_ratio / (cement_ratio + fine_agg_ratio + coarse_agg_ratio))
    fine_aggregate = total_mass * (fine_agg_ratio / (cement_ratio + fine_agg_ratio + coarse_agg_ratio))
    coarse_aggregate = total_mass * (coarse_agg_ratio / (cement_ratio + fine_agg_ratio + coarse_agg_ratio))
    water = total_mass * water_ratio
    glass_powder = 0  # Default 0 unless user replaces
    superplasticizer = 5  # Default to mid-range
else:
    cement, fine_aggregate, coarse_aggregate, water, glass_powder, superplasticizer = 400, 700, 1000, 180, 0, 5

# ✅ Sliders for Inputs
cement = st.sidebar.slider("Cement (kg/m³)", 100, 800, int(cement))
fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 400, 1000, int(fine_aggregate))
coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 800, 1300, int(coarse_aggregate))
water = st.sidebar.slider("Water (kg/m³)", 100, 250, int(water))
glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0, 300, int(glass_powder))
superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 30.0, float(superplasticizer))
days = st.sidebar.slider("Curing Days", 1, 365, 28)

# ✅ Check total mass consistency
total_selected_mass = cement + fine_aggregate + coarse_aggregate + water + glass_powder + superplasticizer
if abs(total_selected_mass - total_mass) > 100:
    st.sidebar.warning("⚠️ Warning: The total materials do not sum up to approximately 2400 kg/m³. Please adjust the values.")

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

