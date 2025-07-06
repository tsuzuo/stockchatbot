import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import time

st.set_page_config(page_title="AI Trading Chatbot", page_icon="📈", layout="centered")

# Initial assistant message
if 'messages' not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "👋 Hi! I'm your AI Trading Assistant. Ask me about any stock (e.g., AAPL, GOOGL, TSLA) and I'll give a quick recommendation!"
    }]

# --- Utility functions ---
def get_stock_data(ticker):
    try:
        data = yf.Ticker(ticker).history(period="6mo")
        if data.empty:
            return None
        data['SMA20'] = data['Close'].rolling(window=20).mean()
        data['SMA50'] = data['Close'].rolling(window=50).mean()
        delta = data['Close'].diff()
        gain = delta.clip(lower=0).rolling(window=14).mean()
        loss = -delta.clip(upper=0).rolling(window=14).mean()
        rs = gain / loss
        data['RSI'] = 100 - (100 / (1 + rs))
        return data
    except:
        return None

def make_prediction(df):
    latest = df.iloc[-1]
    score = 0
    if latest['RSI'] < 30:
        score += 2
    elif latest['RSI'] > 70:
        score -= 2
    if latest['Close'] > latest['SMA20']:
        score += 1
    if latest['SMA20'] > latest['SMA50']:
        score += 1

    if score >= 2:
        return "BUY", latest
    elif score <= -1:
        return "SELL", latest
    else:
        return "HOLD", latest

def extract_ticker(text):
    import re
    match = re.findall(r'\b[A-Z]{2,5}\b', text.upper())
    common_words = {'THE', 'AND', 'FOR', 'WITH'}
    valid = [t for t in match if t not in common_words]
    return valid[0] if valid else None

def generate_response(user_input):
    ticker = extract_ticker(user_input)
    if not ticker:
        return "🤔 Please give me a valid ticker (e.g., AAPL, GOOGL)."

    st.info(f"Analyzing {ticker}...")
    df = get_stock_data(ticker)
    if df is None:
        return f"❌ Could not fetch data for {ticker}."

    prediction, latest = make_prediction(df)
    return f"""
### 📊 Prediction for **{ticker}**
**Action:** `{prediction}`  
- 💰 Price: ${latest['Close']:.2f}  
- 📈 20-day SMA: ${latest['SMA20']:.2f}  
- 📉 50-day SMA: ${latest['SMA50']:.2f}  
- 🔄 RSI: {latest['RSI']:.1f}

⚠️ _This is not financial advice._
"""

# --- Main UI ---
st.title("💬 AI Stock Chatbot")

for msg in st.session_state.messages:
    role = "👤" if msg["role"] == "user" else "🤖"
    st.markdown(f"**{role}**: {msg['content']}")

user_input = st.chat_input("Ask about a stock... (e.g., TSLA, MSFT, etc.)")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    response = generate_response(user_input)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
