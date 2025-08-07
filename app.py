
import streamlit as st
import pandas as pd
import joblib
import os
from openai import OpenAI

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load the ML model
model = joblib.load("f1_model.pkl")

# Prediction function
def predict_top3(features):
    try:
        pred = model.predict([features])[0]
        return "🏆 YES - Likely Top 3" if pred == 1 else "❌ NO - Likely not Top 3"
    except Exception as e:
        return f"Prediction error: {str(e)}"

# Function to extract inputs using GPT
def extract_inputs(user_prompt):
    prompt = f"""Extract the following values from this message and return as comma-separated values in this order:
Grid Position, Qualifying Position, Driver Points, Driver Wins, Team Points, Team Wins, Year, Round.
Message: {user_prompt}"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    reply = response.choices[0].message.content.strip()
    return [float(x.strip()) for x in reply.split(",")]

# Streamlit UI
st.title("🏎️ F1 Race Predictor - Chat Style")
user_input = st.text_input("Ask your F1 prediction:")

if user_input:
    try:
        features = extract_inputs(user_input)
        if len(features) != 8:
            st.error("❗Could not extract all 8 required values. Please try to rephrase.")
        else:
            result = predict_top3(features)
            st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error: {str(e)}")
