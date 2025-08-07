import streamlit as st
import openai
import os
import pandas as pd
import joblib

# Load your model and data
model = joblib.load("f1_model.pkl")

# Set your OpenAI API key (from secret or env)
openai.api_key = os.getenv("OPENAI_API_KEY", "sk-...your-key-here...")

# Function to extract structured inputs from natural language
def extract_inputs(user_input):
    prompt = f"""Extract the following 8 F1 race prediction inputs from this user query:\n
    1. Grid Position\n2. Qualifying Position\n3. Driver Points\n4. Driver Wins\n5. Team Points\n6. Team Wins\n7. Year\n8. Round\n
    If anything is missing, estimate it reasonably.\n
    Input: "{user_input}"\n
    Return the result as a Python list in this order: [grid, qualy, driver_points, driver_wins, team_points, team_wins, year, round].""" 

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    content = response['choices'][0]['message']['content']
    try:
        values = eval(content)
        if isinstance(values, list) and len(values) == 8:
            return values
    except:
        pass
    return None

# Predict based on structured input
def predict(inputs):
    try:
        pred = model.predict([inputs])[0]
        return "YES 🏆 - Likely Top 3" if pred == 1 else "NO ❌ - Likely not Top 3"
    except Exception as e:
        return f"Error: {e}"

# Streamlit app layout
st.title("🏎️ F1 Top 3 Predictor Chatbot")
st.write("Ask your question in natural language (e.g. _'Will Verstappen win from P2 with 187 points?_'")

user_input = st.text_input("Your Question")
if user_input:
    with st.spinner("Thinking..."):
        inputs = extract_inputs(user_input)
        if inputs:
            result = predict(inputs)
            st.success(result)
        else:
            st.error("❌ Sorry, I couldn't extract all 8 values or understand the input.")