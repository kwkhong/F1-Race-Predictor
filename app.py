
import streamlit as st
import openai
import joblib

# Initialize OpenAI client
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# Load model
model = joblib.load("f1_model.pkl")

# Extract inputs from natural language
def extract_inputs(user_input):
    prompt = f"""Extract the following F1 race values from the text below.
If a value is missing or uncertain, estimate a reasonable number (do NOT leave blank):

- Grid Position
- Qualifying Position
- Driver Points
- Driver Wins
- Team Points
- Team Wins
- Year
- Round

Text: {user_input}

Return a Python list like this:
[grid, qualy, driver_points, driver_wins, team_points, team_wins, year, round]
"""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    try:
        text = response.choices[0].message.content.strip()
        inputs = eval(text)
        if not all(isinstance(i, (int, float)) for i in inputs):
            raise ValueError("Some extracted values are not numbers.")
        return inputs
    except Exception as e:
        raise ValueError(f"Error parsing inputs: {e}")

# Streamlit UI
st.title("🏎️ F1 Top 3 Predictor")
st.markdown("Ask a question like: **Will Verstappen finish top 3 if he starts P2 with 187 points and 2 wins in 2025?**")

user_input = st.text_input("Enter your race question")

if user_input:
    try:
        inputs = extract_inputs(user_input)
        st.write("Extracted inputs:", inputs)
        prediction = model.predict([inputs])[0]
        result = "✅ YES - Likely Top 3" if prediction == 1 else "❌ NO - Unlikely Top 3"
        st.success(result)
    except Exception as e:
        st.error(f"Error: {e}")
