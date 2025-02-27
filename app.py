import streamlit as st
import numpy as np
import pandas as pd
import pickle  # ✅ Use pickle instead of joblib

# ✅ Load the trained model using pickle
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

# ✅ Title & Description
st.title("🏗️ Glass Powder Concrete Strength Prediction")
st.markdown("Enter the mix proportions or select a predefined mix ratio to predict the compressive strength of High Strength GPC.")

# ✅ Sidebar for better UI
st.sidebar.title("🔢 Input Features")

# ✅ Mix Ratio Dropdown (Realistic Values Based on Data)
mix_ratios = {
    "1:1:2": (1, 1, 1.5, 0.35, 0.1),  # (Cement, Fine Agg, Coarse Agg, W/C Ratio, Glass Powder %)
    "1:1.5:3": (1, 1.5, 3, 0.35, 0.1),
    "1:2:4": (2, 2, 3, 0.35, 0.1),
    "Custom (Manual Entry)": None
}
selected_ratio = st.sidebar.selectbox("Select a Concrete Mix Ratio", list(mix_ratios.keys()))

# ✅ Default values based on selection
if selected_ratio != "Custom (Manual Entry)":
    cement_ratio, fine_agg_ratio, coarse_agg_ratio, w_c_ratio, glass_powder_ratio = mix_ratios[selected_ratio]
    
    total_mass = 2400  # Set a fixed realistic total density (not exceeding)
    
    # ✅ Normalize mix proportions to fit within 2400 kg/m³
    total_ratio = cement_ratio + fine_agg_ratio + coarse_agg_ratio
    cement = (cement_ratio / total_ratio) * total_mass * 0.9  # Reduce slightly to avoid overshoot
    fine_aggregate = (fine_agg_ratio / total_ratio) * total_mass
    coarse_aggregate = (coarse_agg_ratio / total_ratio) * total_mass

    # ✅ Adjust coarse aggregate limits to realistic values
    coarse_aggregate = min(1260.0, max(800.0, coarse_aggregate))

    # ✅ Adjust water based on cement content
    water = cement * w_c_ratio

    # ✅ Default Glass Powder (10% of Cement)
    glass_powder = cement * glass_powder_ratio
    cement -= glass_powder  # Reduce cement by glass powder amount
else:
    # Allow manual entry
    cement, fine_aggregate, coarse_aggregate, water, glass_powder = 400.0, 700.0, 1000.0, 180.0, 50.0

# ✅ Sliders for Input Features (Ensuring Float Type Consistency)
cement = st.sidebar.slider("Cement (kg/m³)", 152.0, 1062.0, cement, step=1.0)
fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 0.0, 1094.0, fine_aggregate, step=1.0)
coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 0.0, 1260.0, coarse_aggregate, step=1.0)
water = st.sidebar.slider("Water (kg/m³)", 127.5, 271.95, water, step=1.0)
glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0.0, 450.0, glass_powder, step=1.0)
superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 52.5, 5.0, step=0.1)
days = st.sidebar.slider("Curing Days", 1, 365, 28, step=1)

# ✅ Adjust Cement when Glass Powder Increases
if glass_powder > 0:
    cement -= glass_powder * 0.9  # Reduce cement in proportion to glass powder
    cement = max(152.0, cement)  # Ensure cement doesn't go below 152 kg/m³

# ✅ Adjust Water when Superplasticizer Increases
water -= superplasticizer * 2  # More plasticizer means lower water demand
water = max(127.5, water)  # Ensure water doesn't drop below 127.5 kg/m³

# ✅ Warning if total mass is unrealistic
total_mass = cement + fine_aggregate + coarse_aggregate + water + glass_powder + superplasticizer
if abs(total_mass - 2400) > 100:
    st.sidebar.warning(f"⚠️ Warning: The total materials do not sum up to approximately 2400 kg/m³. Please adjust the values.")

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
st.markdown("📌 Built with **Optimized Grey Wolf XGBoost** | 🚀 Deployed on **Streamlit Cloud**")


