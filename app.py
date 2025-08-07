import streamlit as st
import openai
import joblib
import pandas as pd
import re
import os

# Set your OpenAI API key from environment variable (more secure)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load model and dataset
model = joblib.load("f1_model.pkl")
df = pd.read_csv("f1_merged.csv")

# Feature columns in correct order
feature_names = [
    "grid", "qualifying_position", "driver_points", "driver_wins",
    "team_points", "team_wins", "year", "round"
]

# Function to extract numerical inputs from user's message using OpenAI
def extract_inputs(user_text):
    prompt = f"""
Extract the following F1 race data from this sentence:
Grid Position, Qualifying Position, Driver Points, Driver Wins, Team Points, Team Wins, Year, Round.
If any are missing, estimate a reasonable value.

Input: "{user_text}"

Respond ONLY as a Python list in this order: [grid, qualifying_position, driver_points, driver_wins, team_points, team_wins, year, round]
"""

    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    raw_text = response.choices[0].message.content.strip()

    # Extract list using regex
    match = re.findall(r"\[([^\]]+)\]", raw_text)
    if not match:
        raise ValueError("Could not extract values from model response.")

    values = [float(x.strip()) for x in match[0].split(",")]
    if len(values) != 8:
        raise ValueError("Expected 8 values but got a different number.")
    return values

# Streamlit UI
st.title("🏎️ F1 Top 3 Predictor — Chatbot Mode")
st.write("Ask a question like:")
st.markdown("> *Will Verstappen finish in the top 3 if he starts P2 with 187 points, 2 wins in 2025?*")

user_input = st.text_input("Ask your question about the F1 race:")

if user_input:
    try:
        inputs = extract_inputs(user_input)
        df_input = pd.DataFrame([inputs], columns=feature_names)
        prediction = model.predict(df_input)[0]
        result = "✅ YES — Likely Top 3!" if prediction == 1 else "❌ NO — Unlikely Top 3"
        st.success(f"Prediction: {result}")
        st.info(f"Model Inputs Used:\n\n{dict(zip(feature_names, inputs))}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
