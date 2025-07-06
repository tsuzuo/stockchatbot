import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import time
import random
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import warnings


warnings.filterwarnings('ignore')


# Set page config
st.set_page_config(
    page_title="🤖 AI Trading Chatbot",
    page_icon="💬",
    layout="wide"
)


# Custom CSS for chat interface
st.markdown("""
<style>
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 10px 15px;
        border-radius: 20px 20px 5px 20px;
        margin: 10px 0;
        margin-left: 20%;
        text-align: right;
    }
    .bot-message {
        background-color: #f8f9fa;
        color: #333;
        padding: 10px 15px;
        border-radius: 20px 20px 20px 5px;
        margin: 10px 0;
        margin-right: 20%;
        border-left: 4px solid #28a745;
    }
    .prediction-badge {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 20px;
        font-weight: bold;
        margin: 5px;
    }
    .buy-badge {
        background-color: #d4edda;
        color: #155724;
        border: 2px solid #28a745;
    }
    .sell-badge {
        background-color: #f8d7da;
        color: #721c24;
        border: 2px solid #dc3545;
    }
    .hold-badge {
        background-color: #fff3cd;
        color: #856404;
        border: 2px solid #ffc107;
    }
    .typing-indicator {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 20px;
        margin: 10px 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)


# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm your AI Trading Assistant. I can analyze any stock and give you buy/hold/sell recommendations. Just tell me a ticker symbol like AAPL, GOOGL, or TSLA!",
            "timestamp": datetime.now()
        }
    ]


if 'conversation_started' not in st.session_state:
    st.session_state.conversation_started = True


# Technical analysis functions (simplified for chatbot)
@st.cache_data
def get_stock_data(ticker: str, period: str = "1y"):
    """Fetch stock data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        if len(df) < 50:
            return None
        df.reset_index(inplace=True)
        return df
    except:
        return None


def add_indicators(df):
    """Add technical indicators"""
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
   
    # Moving averages
    df['SMA20'] = df['Close'].rolling(20).mean()
    df['SMA50'] = df['Close'].rolling(50).mean()
   
    # MACD
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
   
    # Volatility
    df['Volatility'] = df['Close'].pct_change().rolling(20).std()
   
    return df


@st.cache_data
def make_quick_prediction(ticker: str):
    """Make a quick prediction for the chatbot"""
    df = get_stock_data(ticker)
    if df is None:
        return None, None, None
   
    df = add_indicators(df)
   
    # Simple rule-based prediction for speed
    latest = df.iloc[-1]
   
    # Get some basic metrics
    rsi = latest['RSI']
    price = latest['Close']
    sma20 = latest['SMA20']
    sma50 = latest['SMA50']
   
    # Simple scoring system
    score = 0
    confidence = 0.5
   
    # RSI signals
    if rsi < 30:
        score += 2  # Oversold - buy signal
        confidence += 0.2
    elif rsi > 70:
        score -= 2  # Overbought - sell signal
        confidence += 0.2
   
    # Moving average signals
    if price > sma20:
        score += 1
        confidence += 0.1
    if price > sma50:
        score += 1
        confidence += 0.1
    if sma20 > sma50:
        score += 1
        confidence += 0.1
   
    # Determine prediction
    if score >= 2:
        prediction = "BUY"
    elif score <= -1:
        prediction = "SELL"
    else:
        prediction = "HOLD"
   
    confidence = min(confidence, 0.95)
   
    return prediction, confidence, latest


def get_stock_info(ticker: str):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'market_cap': info.get('marketCap', 0),
            'pe_ratio': info.get('trailingPE', 'N/A')
        }
    except:
        return None


def format_message(content, role="assistant"):
    """Format message for display"""
    css_class = "bot-message" if role == "assistant" else "user-message"
    return f'<div class="{css_class}">{content}</div>'


def extract_ticker(text):
    """Extract ticker symbol from user input"""
    import re
    # Look for ticker patterns (2-5 uppercase letters)
    tickers = re.findall(r'\b[A-Z]{1,5}\b', text.upper())
   
    # Common words to exclude
    exclude = {'THE', 'AND', 'OR', 'FOR', 'TO', 'OF', 'IN', 'ON', 'AT', 'BY', 'WITH', 'FROM'}
   
    valid_tickers = [t for t in tickers if t not in exclude and len(t) <= 5]
   
    return valid_tickers[0] if valid_tickers else None


def generate_ai_response(user_input):
    """Generate AI response based on user input"""
    ticker = extract_ticker(user_input)
   
    if not ticker:
        return """🤔 I didn't catch a ticker symbol. Could you please provide one?


For example, try saying:
- "What about AAPL?"
- "Analyze GOOGL"
- "Should I buy TSLA?"
- "MSFT analysis please"


I can analyze any publicly traded stock! 📈"""
   
    # Show typing indicator
    typing_placeholder = st.empty()
    typing_placeholder.markdown('<div class="typing-indicator">🤖 Analyzing ' + ticker + '...</div>', unsafe_allow_html=True)
    time.sleep(2)  # Simulate thinking time
    typing_placeholder.empty()
   
    prediction, confidence, latest_data = make_quick_prediction(ticker)
   
    if prediction is None:
        return f"❌ Sorry, I couldn't find data for {ticker}. Please check if it's a valid ticker symbol."
   
    stock_info = get_stock_info(ticker)
   
    # Generate response
    response = f"## 📊 Analysis for {ticker}"
   
    if stock_info:
        response += f"\n**Company:** {stock_info['name']}\n"
   
    # Add prediction badge
    if prediction == "BUY":
        badge_class = "buy-badge"
        emoji = "🟢"
    elif prediction == "SELL":
        badge_class = "sell-badge"
        emoji = "🔴"
    else:
        badge_class = "hold-badge"
        emoji = "🟡"
   
    response += f'\n<div class="prediction-badge {badge_class}">{emoji} {prediction} {ticker}</div>\n'
    response += f"\n**Confidence:** {confidence:.0%}\n"
   
    # Add current metrics
    response += f"""
**Current Metrics:**
- 💰 Price: ${latest_data['Close']:.2f}
- 📊 RSI: {latest_data['RSI']:.1f}
- 📈 20-day MA: ${latest_data['SMA20']:.2f}
- 📉 50-day MA: ${latest_data['SMA50']:.2f}
"""
   
    # Add reasoning
    response += "\n**Why this recommendation?**\n"
   
    if latest_data['RSI'] < 30:
        response += "• RSI shows oversold conditions (potential buying opportunity)\n"
    elif latest_data['RSI'] > 70:
        response += "• RSI shows overbought conditions (potential sell signal)\n"
   
    if latest_data['Close'] > latest_data['SMA20']:
        response += "• Price is above 20-day moving average (bullish trend)\n"
    else:
        response += "• Price is below 20-day moving average (bearish trend)\n"
   
    if latest_data['SMA20'] > latest_data['SMA50']:
        response += "• Short-term trend is stronger than long-term (bullish)\n"
    else:
        response += "• Long-term trend is stronger than short-term (bearish)\n"
   
    response += "\n⚠️ *This is for educational purposes only - not financial advice!*"
   
    return response


def main():
    st.title("🤖💬 AI Trading Chatbot")
    st.markdown("Ask me about any stock and I'll give you a buy/hold/sell recommendation!")
   
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(format_message(message["content"], "user"), unsafe_allow_html=True)
        else:
            st.markdown(format_message(message["content"], "assistant"), unsafe_allow_html=True)
   
    # Chat input
    user_input = st.chat_input("Type a ticker symbol or ask about a stock... (e.g., 'What about AAPL?')")
   
    if user_input:
        # Add user message
        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now()
        })
       
        # Display user message
        st.markdown(format_message(user_input, "user"), unsafe_allow_html=True)
       
        # Generate and display AI response
        with st.spinner("🤖 Thinking..."):
            ai_response = generate_ai_response(user_input)
       
        st.session_state.messages.append({
            "role": "assistant",
            "content": ai_response,
            "timestamp": datetime.now()
        })
       
        st.markdown(format_message(ai_response, "assistant"), unsafe_allow_html=True)
       
        # Rerun to update the display
        st.rerun()
   
    # Sidebar with example commands
    with st.sidebar:
        st.header("💡 Try These Examples:")
       
        example_commands = [
            "What about AAPL?",
            "Analyze GOOGL",
            "Should I buy TSLA?",
            "MSFT analysis please",
            "How's NVDA looking?",
            "Tell me about AMZN"
        ]
       
        for cmd in example_commands:
            if st.button(cmd, key=f"example_{cmd}"):
                # Simulate user input
                st.session_state.messages.append({
                    "role": "user",
                    "content": cmd,
                    "timestamp": datetime.now()
                })
               
                ai_response = generate_ai_response(cmd)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": ai_response,
                    "timestamp": datetime.now()
                })
                st.rerun()
       
        st.markdown("---")
        st.subheader("🔧 Features:")
        st.write("✅ Real-time stock analysis")
        st.write("✅ Technical indicators")
        st.write("✅ Buy/Hold/Sell signals")
        st.write("✅ Confidence scoring")
        st.write("✅ Natural language chat")
       
        if st.button("🗑️ Clear Chat"):
            st.session_state.messages = [
                {
                    "role": "assistant",
                    "content": "👋 Hi! I'm your AI Trading Assistant. I can analyze any stock and give you buy/hold/sell recommendations. Just tell me a ticker symbol like AAPL, GOOGL, or TSLA!",
                    "timestamp": datetime.now()
                }
            ]
            st.rerun()


if __name__ == "__main__":
    main()


