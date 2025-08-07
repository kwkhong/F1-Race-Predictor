
import streamlit as st
import openai
import pandas as pd
import joblib
import os

# Load the ML model
model = joblib.load("f1_model.pkl")

# OpenAI API key setup
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to extract 8 inputs using GPT from natural language
def extract_inputs(user_input):
    prompt = f"""You are a helpful assistant that extracts 8 specific features from a user's question about an F1 race:
1. grid (integer)
2. qualifying_position (integer)
3. points (float) — driver's current points
4. wins (integer) — number of driver wins
5. points_constructor_standings (float) — team points
6. wins_constructor_standings (integer) — team wins
7. year (integer)
8. round (integer)

Return the 8 values as a Python dictionary.

User: {user_input}
"""
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    reply = response["choices"][0]["message"]["content"]
    return eval(reply)

# Predict using model
def predict_top3(grid, qualy, driver_points, driver_wins, team_points, team_wins, year, round_num):
    try:
        sample = pd.DataFrame([{
            "grid": grid,
            "qualifying_position": qualy,
            "points": driver_points,
            "wins": driver_wins,
            "points_constructor_standings": team_points,
            "wins_constructor_standings": team_wins,
            "year": year,
            "round": round_num
        }])
        pred = model.predict(sample)[0]
        return "YES 🏆 - Likely Top 3" if pred == 1 else "NO ❌ - Likely not Top 3"
    except Exception as e:
        return f"Error: {str(e)}"

# Streamlit chatbot interface
st.title("🏎️ F1 Top 3 Chatbot Predictor")
st.markdown("Ask me if a driver will finish in the Top 3 of an F1 race using natural language.")

user_input = st.text_input("Your Question (e.g. 'Will Verstappen win from P2 with 187 points in round 12, 2025?')")

if user_input:
    with st.spinner("Thinking..."):
        try:
            inputs = extract_inputs(user_input)
            st.write("Extracted Inputs:", inputs)
            result = predict_top3(
                grid=inputs["grid"],
                qualy=inputs["qualifying_position"],
                driver_points=inputs["points"],
                driver_wins=inputs["wins"],
                team_points=inputs["points_constructor_standings"],
                team_wins=inputs["wins_constructor_standings"],
                year=inputs["year"],
                round_num=inputs["round"]
            )
            st.success(result)
        except Exception as e:
            st.error(f"Error: {str(e)}")
