import streamlit as st
import openai
import pandas as pd
import joblib
import re

# ========== CONFIG ==========
openai.api_key = "your-openai-api-key"  # 🔁 Replace with your actual key
model = joblib.load("f1_model.pkl")

# ========== UTILS ==========
def extract_inputs(user_input):
    prompt = f"""
You are a helpful assistant. Extract the following 8 F1 race values from the user's message. If a value is missing, intelligently estimate it based on context:
1. grid
2. qualifying_position
3. driver_points
4. driver_wins
5. team_points
6. team_wins
7. year
8. round

Return as a Python dictionary.

User: "{user_input}"
"""
    response = openai.ChatCompletion.create(
        model="gpt-4",  # or "gpt-3.5-turbo"
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    answer = response['choices'][0]['message']['content']
    try:
        values = eval(answer)  # WARNING: works if LLM returns a valid Python dict
        return values
    except:
        return None

def predict_top3(inputs_dict):
    try:
        sample = [[
            inputs_dict["grid"],
            inputs_dict["qualifying_position"],
            inputs_dict["driver_points"],
            inputs_dict["driver_wins"],
            inputs_dict["team_points"],
            inputs_dict["team_wins"],
            inputs_dict["year"],
            inputs_dict["round"]
        ]]
        prediction = model.predict(sample)[0]
        return "🏆 YES - Likely Top 3!" if prediction == 1 else "❌ NO - Unlikely Top 3"
    except Exception as e:
        return f"Prediction Error: {str(e)}"

# ========== UI ==========
st.title("🏎️ F1 Race Prediction Chatbot")
user_input = st.text_input("Ask me about a driver's race: (e.g. 'Max starts P2 with 88 points in Round 14')")

if user_input:
    st.write("📥 Parsing your message...")
    inputs = extract_inputs(user_input)
    
    if inputs:
        st.write("🧾 Extracted inputs:", inputs)
        result = predict_top3(inputs)
        st.success(result)
    else:
        st.error("❌ Could not understand your input. Try rephrasing.")
