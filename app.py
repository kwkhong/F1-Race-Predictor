
import streamlit as st
import pandas as pd
import joblib

# Load the model and dataset
model = joblib.load("f1_model.pkl")

st.set_page_config(page_title="🏎️ F1 Top 3 Predictor", layout="centered")
st.title("🏁 F1 Top 3 Predictor")
st.markdown("Enter race details to predict if the driver will finish in the **Top 3**!")

# Input fields
grid = st.number_input("Grid Position", min_value=1, max_value=20, value=1)
qualy = st.number_input("Qualifying Position", min_value=1, max_value=20, value=1)
driver_points = st.number_input("Driver Points", min_value=0)
driver_wins = st.number_input("Driver Wins", min_value=0)
team_points = st.number_input("Team Points", min_value=0)
team_wins = st.number_input("Team Wins", min_value=0)
year = st.number_input("Year", min_value=2000, max_value=2100, value=2023)
round_num = st.number_input("Round Number", min_value=1, max_value=25, value=1)

# Prediction logic
if st.button("Predict Top 3 Finish"):
    try:
        features = [[grid, qualy, driver_points, driver_wins, team_points, team_wins, year, round_num]]
        prediction = model.predict(features)[0]
        if prediction == 1:
            st.success("✅ YES! This driver is likely to finish in the **Top 3** 🏆")
        else:
            st.warning("❌ No. This driver is unlikely to reach the Top 3.")
    except Exception as e:
        st.error(f"Prediction error: {e}")
