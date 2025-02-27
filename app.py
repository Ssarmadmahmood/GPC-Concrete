import streamlit as st
import numpy as np
import pandas as pd
import pickle  # ✅ Use pickle instead of joblib

# ✅ Load the trained model using pickle
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏰", layout="wide")

# ✅ Title & Description
st.title(":construction: Concrete Strength Prediction")
st.markdown("Enter the mix proportions or select a predefined mix ratio to predict the compressive strength of concrete.")

# ✅ Sidebar Layout
st.sidebar.title(":1234: Input Features")

# ✅ Concrete Mix Ratios (Predefined)
mix_ratios = {
    "1:2:4": (1, 2, 4, 0.5),  # Cement:Sand:Coarse Aggregate:Water Ratio
    "1:1.5:3": (1, 1.5, 3, 0.45),
    "1:1:2": (1, 1, 2, 0.42),
    "Custom (Manual Entry)": None
}

selected_ratio = st.sidebar.selectbox("Select a Concrete Mix Ratio", list(mix_ratios.keys()))

def calculate_proportions(ratio, total_mass=2400):
    """ Calculate proportions based on selected mix ratio."""
    if ratio is None:
        return None
    cement_ratio, sand_ratio, agg_ratio, water_ratio = ratio
    total_ratio = cement_ratio + sand_ratio + agg_ratio + water_ratio
    cement = (cement_ratio / total_ratio) * total_mass
    fine_aggregate = (sand_ratio / total_ratio) * total_mass
    coarse_aggregate = (agg_ratio / total_ratio) * total_mass
    water = (water_ratio / total_ratio) * total_mass
    return cement, fine_aggregate, coarse_aggregate, water

# ✅ Auto-adjusting values based on selected mix ratio
if selected_ratio != "Custom (Manual Entry)":
    cement, fine_aggregate, coarse_aggregate, water = calculate_proportions(mix_ratios[selected_ratio])
else:
    cement, fine_aggregate, coarse_aggregate, water = 400, 700, 1000, 180  # Default manual values

# ✅ Define Input Fields (Users can still modify after selection)
cement = st.sidebar.slider("Cement (kg/m³)", 100, 800, int(cement))
fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 400, 1000, int(fine_aggregate))
coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 800, 1300, int(coarse_aggregate))
water = st.sidebar.slider("Water (kg/m³)", 100, 250, int(water))

glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0, 300, 50)
superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 30.0, 5.0)
days = st.sidebar.slider("Curing Days", 1, 365, 28)

# ✅ Total Material Check
total_mass = cement + fine_aggregate + coarse_aggregate + water + glass_powder + superplasticizer
if abs(total_mass - 2400) > 200:
    st.sidebar.warning(f":warning: Warning: The total materials do not sum up to approximately 2400 kg/m³. Please adjust the values.")

# ✅ Prepare Input Data
input_data = np.array([[cement, glass_powder, fine_aggregate, coarse_aggregate, water, superplasticizer, days]])
columns = ["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]
input_df = pd.DataFrame(input_data, columns=columns)

# ✅ Predict Compressive Strength
if st.sidebar.button(":mag: Predict Strength"):
    prediction = model.predict(input_df)[0]  # Get the prediction
    st.success(f":white_check_mark: **Predicted Compressive Strength:** {prediction:.2f} MPa")
    st.balloons()

# ✅ Footer
st.markdown("---")
st.markdown(":pushpin: Built with **XGBoost + Streamlit** | :rocket: Deployed on **Streamlit Cloud**")
