import streamlit as st
import joblib
import re
from streamlit_chat import message

# Load model
model = joblib.load("f1_model.pkl")

# Chat function to parse input
def parse_input(user_input):
    pattern = r"grid\s*:?[\s]*(\d+)|qual(ifying)?\s*:?[\s]*(\d+)|driver points\s*:?[\s]*(\d+)|driver wins\s*:?[\s]*(\d+)|team points\s*:?[\s]*(\d+)|team wins\s*:?[\s]*(\d+)|year\s*:?[\s]*(\d{4})|round\s*:?[\s]*(\d+)"
    matches = re.findall(pattern, user_input, re.IGNORECASE)

    inputs = [None] * 8  # 8 values: grid, qualy, d_pts, d_wins, t_pts, t_wins, year, round
    for match in matches:
        nums = [x for x in match if x.isdigit()]
        if "grid" in match[0].lower():
            inputs[0] = int(nums[0])
        elif "qual" in match[1].lower():
            inputs[1] = int(nums[0])
        elif "driver points" in match[0].lower():
            inputs[2] = int(nums[0])
        elif "driver wins" in match[0].lower():
            inputs[3] = int(nums[0])
        elif "team points" in match[0].lower():
            inputs[4] = int(nums[0])
        elif "team wins" in match[0].lower():
            inputs[5] = int(nums[0])
        elif "year" in match[0].lower():
            inputs[6] = int(nums[0])
        elif "round" in match[0].lower():
            inputs[7] = int(nums[0])

    if None in inputs:
        raise ValueError("Missing some values. Please include all 8 inputs.")
    return inputs

# Streamlit app
st.title("🏎️ F1 Race Top 3 Predictor Chatbot")
st.markdown("Ask me if a driver will finish Top 3!")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Ask your question about an F1 race...")

if user_input:
    try:
        data = parse_input(user_input)
        prediction = model.predict([data])[0]
        result = "YES 🏆 Likely Top 3!" if prediction == 1 else "NO ❌ Unlikely Top 3."
    except Exception as e:
        result = f"Error: {str(e)}"

    st.session_state.history.append((user_input, result))

for i, (user, bot) in enumerate(st.session_state.history):
    message(user, is_user=True, key=f"user_{i}")
    message(bot, is_user=False, key=f"bot_{i}")
