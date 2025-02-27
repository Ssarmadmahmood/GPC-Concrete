import streamlit as st

st.title("XGBoost Model Analysis")
st.write("Welcome! This is a Streamlit app running inside Google Colab.")

# Add a simple input field
user_input = st.number_input("Enter a value:", min_value=0, max_value=100, step=1)
st.write(f"You entered: {user_input}")

