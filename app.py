import streamlit as st
from openai import OpenAI
import os
import joblib
import pandas as pd
import re

# Load OpenAI client using environment variable (from Streamlit secrets)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load model and dataset
model = joblib.load("f1_model.pkl")
df = pd.read_csv("f1_merged.csv")

# Define function to extract numeric inputs from natural language

def extract_inputs(user_message):
    prompt = f"""
You are a data extractor for an F1 prediction AI.
Given this input:
"{user_message}"
Extract and return the following 8 numbers in a list in this order:
[grid, qualifying_position, driver_points, driver_wins, team_points, team_wins, year, round]
If any are missing, make a reasonable guess based on typical F1 knowledge.
Only return the Python list of 8 numbers. No explanation.
"""
    
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    result = response.choices[0].message.content
    # Extract numbers from response
    numbers = re.findall(r"[-+]?[0-9]*\.?[0-9]+", result)
    if len(numbers) != 8:
        raise ValueError("Missing some values. Please include all 8 inputs.")
    return list(map(float, numbers))

# Define prediction function
def predict_from_text(user_input):
    try:
        inputs = extract_inputs(user_input)
        pred = model.predict([inputs])[0]
        return "🏁 YES — Likely Top 3!" if pred == 1 else "❌ NO — Not likely top 3."
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit UI
st.title("🏎️ F1 Race Predictor — Chat Style")
st.write("Ask a question like: 'Will Max win the next race if he starts P2 and has 150 points?'\nI'll try to estimate the result for you!")

user_input = st.text_input("Your Question:")

if user_input:
    output = predict_from_text(user_input)
    st.subheader("Prediction:")
    st.write(output)
