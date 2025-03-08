import streamlit as st
import numpy as np
import pandas as pd
import pickle  # ✅ Using pickle for model loading

# ✅ Load the trained model using pickle
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

# ✅ Title & Description
st.title("🏗️ Glass Powder Concrete Strength Prediction")
st.markdown("Enter the mix proportions to predict the compressive strength of high-strength GPC.")

# ✅ Sidebar for better UI
st.sidebar.title("🔢 Input Features")

# ✅ Sliders for Input Features (Manual Entry)
cement = st.sidebar.slider("Cement (kg/m³)", 152.0, 1062.0, 400.0, step=1.0)
fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 0.0, 1094.0, 700.0, step=1.0)
coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 0.0, 1260.0, 1000.0, step=1.0)
water = st.sidebar.slider("Water (kg/m³)", 127.5, 271.95, 180.0, step=1.0)
glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0.0, 450.0, 50.0, step=1.0)
superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 52.5, 5.0, step=0.1)
days = st.sidebar.slider("Curing Days", 1, 365, 28, step=1)

# ✅ Adjust Cement when Glass Powder Increases
if glass_powder > 0:
    cement -= glass_powder * 0.9  # Reduce cement in proportion to glass powder
    cement = max(152.0, cement)  # Ensure cement doesn't go below 152 kg/m³

# ✅ Adjust Water when Superplasticizer Increases
water -= superplasticizer * 2  # More plasticizer means lower water demand
water = max(127.5, water)  # Ensure water doesn't drop below 127.5 kg/m³

# ✅ Compute total material weight
total_mass = cement + fine_aggregate + coarse_aggregate + water + glass_powder + superplasticizer

# ✅ Set realistic density range (2350–2550 kg/m³)
min_density = 2350
max_density = 2550

# ✅ Display warning only if total materials are **outside** this realistic range
if not (min_density <= total_mass <= max_density):
    st.sidebar.warning(f"⚠️ Warning: The total materials sum to {total_mass:.2f} kg/m³, which is outside the expected range ({min_density}-{max_density} kg/m³). Please adjust the values.")

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


