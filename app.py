import streamlit as st
import pandas as pd
import joblib
import openai
import os

# Set your OpenAI API key (Streamlit Cloud reads from secrets)
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load model and data
model = joblib.load("f1_model.pkl")

# Expected features for the model
expected_keys = [
    "grid", "qualifying", "points", "wins",
    "points_constructor_standings", "wins_constructor_standings",
    "year", "round"
]

def extract_inputs(user_message):
    prompt = f"""Extract the following 8 values from this sentence: grid, qualifying, points, wins, points_constructor_standings, wins_constructor_standings, year, round.
If any value is missing, return 0 for it. Just return a JSON object with the keys.

Sentence: {user_message}"""

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        values = eval(content)  # Very basic for now; assumes LLM gives Python-like dict

        # Convert values safely to float
        inputs = []
        for k in expected_keys:
            try:
                inputs.append(float(values.get(k, 0)))
            except (ValueError, TypeError):
                inputs.append(0.0)
        return inputs

    except Exception as e:
        st.error(f"Error parsing inputs: {e}")
        return [0.0] * 8

# Streamlit UI
st.title("🏎️ F1 Top 3 Chatbot Predictor")
st.write("Ask a question like: *Will Verstappen win if he starts in P2 with 150 points and 3 wins?*")

user_input = st.text_input("Your Question")
if user_input:
    inputs = extract_inputs(user_input)
    try:
        prediction = model.predict([inputs])[0]
        if prediction == 1:
            st.success("✅ YES — Likely Top 3 Finish!")
        else:
            st.warning("❌ NO — Likely NOT Top 3")
    except Exception as e:
        st.error(f"Prediction Error: {e}")