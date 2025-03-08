import streamlit as st
import numpy as np
import pandas as pd
import pickle
from scipy.optimize import minimize  # ✅ Powell’s method for better optimization

# ✅ Load the trained model
with open('optimized_xgb_gwo.pkl', 'rb') as file:
    model = pickle.load(file)

# ✅ Page Config
st.set_page_config(page_title="Concrete Strength Predictor", page_icon="🏗️", layout="centered")

# ✅ Title & Description
st.title("🏗️ Glass Powder Concrete Strength Prediction")
st.markdown("Enter mix proportions or set a target strength and curing age to get the best mix design.")

# ✅ Sidebar toggle for Prediction vs Reverse Engineering
mode = st.sidebar.radio("Select Mode:", ["📌 Predict Strength", "🔄 Suggest Mix for Desired Strength"])

if mode == "📌 Predict Strength":
    # ✅ User enters mix proportions manually
    cement = st.sidebar.slider("Cement (kg/m³)", 152.0, 1062.0, 400.0, step=1.0)
    fine_aggregate = st.sidebar.slider("Fine Aggregate (kg/m³)", 0.0, 1094.0, 700.0, step=1.0)
    coarse_aggregate = st.sidebar.slider("Coarse Aggregate (kg/m³)", 0.0, 1260.0, 1000.0, step=1.0)
    water = st.sidebar.slider("Water (kg/m³)", 127.5, 271.95, 180.0, step=1.0)
    glass_powder = st.sidebar.slider("Glass Powder (kg/m³)", 0.0, 450.0, 50.0, step=1.0)
    superplasticizer = st.sidebar.slider("Superplasticizer (kg/m³)", 0.0, 52.5, 5.0, step=0.1)
    days = st.sidebar.slider("Curing Days", 1, 365, 28, step=1)

    # ✅ Prepare Input Data
    input_data = np.array([[cement, glass_powder, fine_aggregate, coarse_aggregate, water, superplasticizer, days]])
    columns = ["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]
    input_df = pd.DataFrame(input_data, columns=columns)

    # ✅ Predict Compressive Strength
    if st.sidebar.button("🔍 Predict Strength"):
        prediction = model.predict(input_df)[0]  # Get the prediction
        st.success(f"✅ **Predicted Compressive Strength:** {prediction:.2f} MPa")
        st.balloons()

else:
    # ✅ Reverse Engineering Mode - User sets Target Strength & Curing Age
    target_strength = st.sidebar.number_input("Enter Desired Strength (MPa)", min_value=10.0, max_value=150.0, value=50.0, step=0.1)
    curing_days = st.sidebar.slider("Select Curing Days", 1, 365, 28, step=1)

    # ✅ Define Optimization Function
    def objective(x):
        """Optimization function to minimize strength difference."""
        mix_data = np.array([[x[0], x[1], x[2], x[3], x[4], x[5], curing_days]])
        predicted_strength = model.predict(pd.DataFrame(mix_data, columns=["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]))[0]
        return abs(predicted_strength - target_strength)  # Minimize the difference

    # ✅ Constraints: Ensure total mix proportion is reasonable (~2400 kg/m³)
    def total_mass_constraint(x):
        return 2400 - (x[0] + x[1] + x[2] + x[3] + x[4] + x[5])

    constraints = [{'type': 'eq', 'fun': total_mass_constraint}]

    # ✅ Expanded Bounds for Optimization (More Practical)
    bounds = [(150, 1100), (0, 500), (0, 1200), (0, 1300), (125, 300), (0, 60)]  # Cement, GP, FA, CA, Water, SP

    # ✅ Randomized Initial Guess to Prevent Stagnation
    initial_guess = np.random.uniform([200, 20, 500, 800, 150, 2], [600, 100, 800, 1100, 220, 15])

    # ✅ Run Optimization using Powell’s Method (Faster and More Reliable)
    result = minimize(objective, initial_guess, bounds=bounds, method="Powell", constraints=constraints, options={'maxiter': 50})

    # ✅ Display Optimized Mix Proportions
    if result.success:
        best_mix = result.x
        final_strength = model.predict(pd.DataFrame([best_mix.tolist() + [curing_days]], columns=["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]))[0]

        st.success(f"✅ **Optimal Mix Proportions for {target_strength:.2f} MPa Strength at {curing_days} Days:**")
        st.write(f"- Cement: **{best_mix[0]:.1f} kg/m³**")
        st.write(f"- Glass Powder: **{best_mix[1]:.1f} kg/m³**")
        st.write(f"- Fine Aggregate: **{best_mix[2]:.1f} kg/m³**")
        st.write(f"- Coarse Aggregate: **{best_mix[3]:.1f} kg/m³**")
        st.write(f"- Water: **{best_mix[4]:.1f} kg/m³**")
        st.write(f"- Superplasticizer: **{best_mix[5]:.1f} kg/m³**")
        st.success(f"🎯 **Predicted Strength of this Mix:** {final_strength:.2f} MPa")
        st.balloons()
    
    else:
        st.warning("⚠️ Optimization could not find an exact mix, trying to find the closest one...")
        closest_guess = result.x
        closest_strength = model.predict(pd.DataFrame([closest_guess.tolist() + [curing_days]], columns=["Cement", "Glass Powder", "Fine Aggregate", "Coarse Aggregate", "Water", "Superplasticizer", "Days"]))[0]

        st.warning(f"✅ **Closest Mix Found (Strength: {closest_strength:.2f} MPa):**")
        st.write(f"- Cement: **{closest_guess[0]:.1f} kg/m³**")
        st.write(f"- Glass Powder: **{closest_guess[1]:.1f} kg/m³**")
        st.write(f"- Fine Aggregate: **{closest_guess[2]:.1f} kg/m³**")
        st.write(f"- Coarse Aggregate: **{closest_guess[3]:.1f} kg/m³**")
        st.write(f"- Water: **{closest_guess[4]:.1f} kg/m³**")
        st.write(f"- Superplasticizer: **{closest_guess[5]:.1f} kg/m³**")
        st.info("🔄 Try adjusting the curing days or target strength slightly for a better match.")



