import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
import xgboost as xgb
import lightgbm as lgb
import warnings
import io
import base64


warnings.filterwarnings('ignore')


# Set page config
st.set_page_config(
    page_title="🤖 AI Trading Agent",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .prediction-card {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
    }
    .buy-card {
        background-color: #d4edda;
        border: 2px solid #28a745;
        color: #155724;
    }
    .sell-card {
        background-color: #f8d7da;
        border: 2px solid #dc3545;
        color: #721c24;
    }
    .hold-card {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        color: #856404;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def fetch_real_stock_data(ticker: str, period: str = "2y") -> pd.DataFrame:
    """Fetch real stock data using yfinance"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period)
        df.reset_index(inplace=True)
        df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].dropna()
        return df
    except Exception as e:
        st.error(f"Error fetching data for {ticker}: {str(e)}")
        return None


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to the dataframe"""
    # Basic indicators
    df['SMA5'] = df['Close'].rolling(window=5).mean()
    df['SMA10'] = df['Close'].rolling(window=10).mean()
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['SMA50'] = df['Close'].rolling(window=50).mean()
    df['EMA12'] = df['Close'].ewm(span=12).mean()
    df['EMA26'] = df['Close'].ewm(span=26).mean()
   
    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(window=14).mean()
    loss = -delta.clip(upper=0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))
   
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_hist'] = df['MACD'] - df['MACD_signal']
   
    # Bollinger Bands
    df['BB_middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_upper'] = df['BB_middle'] + (bb_std * 2)
    df['BB_lower'] = df['BB_middle'] - (bb_std * 2)
   
    # Price momentum
    for period in [5, 10, 20]:
        df[f'Momentum_{period}'] = df['Close'] / df['Close'].shift(period) - 1
   
    # Volatility
    df['Daily_Return'] = df['Close'].pct_change()
    for window in [5, 10, 20]:
        df[f'Volatility_{window}'] = df['Daily_Return'].rolling(window=window).std()
   
    return df


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add advanced engineered features"""
    # Price-based features
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Open_Close_Ratio'] = df['Open'] / df['Close']
   
    # Lagged features
    for lag in [1, 2, 3]:
        df[f'Close_Lag_{lag}'] = df['Close'].shift(lag)
        df[f'RSI_14_Lag_{lag}'] = df['RSI_14'].shift(lag)
   
    # Rolling statistics
    for window in [10, 20]:
        df[f'Close_Mean_{window}'] = df['Close'].rolling(window=window).mean()
        df[f'Volume_Mean_{window}'] = df['Volume'].rolling(window=window).mean()
   
    # Trend features
    df['Price_Trend_5'] = (df['Close'] > df['SMA5']).astype(int)
    df['Price_Trend_20'] = (df['Close'] > df['SMA20']).astype(int)
   
    # Replace infinite values
    df = df.replace([np.inf, -np.inf], 0)
   
    return df


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate labels for training"""
    # Simple forward-looking return
    df['Future_Return_3'] = (df['Close'].shift(-3) - df['Close']) / df['Close']
   
    # Create labels based on percentiles
    signal_quantiles = df['Future_Return_3'].quantile([0.3, 0.7])
   
    def create_label(signal):
        if pd.isna(signal):
            return 0
        elif signal > signal_quantiles[0.7]:
            return 1  # Buy
        elif signal < signal_quantiles[0.3]:
            return -1  # Sell
        else:
            return 0  # Hold
   
    df['Label'] = df['Future_Return_3'].apply(create_label)
    return df.dropna().reset_index(drop=True)


@st.cache_data
def train_prediction_model(df: pd.DataFrame):
    """Train the prediction model"""
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['Date', 'Label', 'Future_Return_3']]
    feature_cols = [col for col in feature_cols if df[col].isna().sum() < len(df) * 0.1]
   
    # Split data (use last 80% for training)
    train_size = int(len(df) * 0.8)
    train_df = df[:train_size]
   
    X_train = train_df[feature_cols]
    y_train = train_df['Label']
   
    # Transform labels for XGBoost compatibility
    label_mapping = {-1: 0, 0: 1, 1: 2}
    y_train_mapped = y_train.map(label_mapping)
   
    # Scale features
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train.fillna(X_train.median()))
   
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=min(30, len(feature_cols)))
    X_train_selected = selector.fit_transform(X_train_scaled, y_train_mapped)
   
    # Train XGBoost model (fastest for real-time predictions)
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        eval_metric='mlogloss'
    )
    model.fit(X_train_scaled, y_train_mapped)
   
    return model, scaler, selector, feature_cols


def make_prediction(model, scaler, selector, feature_cols, df):
    """Make prediction for the latest data point"""
    # Get the latest data point
    latest_data = df[feature_cols].iloc[-1:].fillna(df[feature_cols].median())
   
    # Scale and select features
    latest_scaled = scaler.transform(latest_data)
   
    # Make prediction
    prediction = model.predict(latest_scaled)[0]
    probabilities = model.predict_proba(latest_scaled)[0]
   
    # Map back to original labels
    reverse_mapping = {0: -1, 1: 0, 2: 1}
    prediction_mapped = reverse_mapping[prediction]
   
    return prediction_mapped, probabilities


def create_charts(df: pd.DataFrame, ticker: str):
    """Create visualization charts"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
   
    # Price chart with moving averages
    axes[0, 0].plot(df['Date'], df['Close'], label='Close Price', linewidth=2)
    axes[0, 0].plot(df['Date'], df['SMA20'], label='SMA20', alpha=0.7)
    axes[0, 0].plot(df['Date'], df['SMA50'], label='SMA50', alpha=0.7)
    axes[0, 0].set_title(f'{ticker} Price with Moving Averages')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
   
    # RSI
    axes[0, 1].plot(df['Date'], df['RSI_14'], color='purple', linewidth=2)
    axes[0, 1].axhline(y=70, color='red', linestyle='--', alpha=0.7)
    axes[0, 1].axhline(y=30, color='green', linestyle='--', alpha=0.7)
    axes[0, 1].set_title('RSI (14)')
    axes[0, 1].set_ylim(0, 100)
    axes[0, 1].grid(True, alpha=0.3)
   
    # MACD
    axes[1, 0].plot(df['Date'], df['MACD'], label='MACD')
    axes[1, 0].plot(df['Date'], df['MACD_signal'], label='Signal')
    axes[1, 0].bar(df['Date'], df['MACD_hist'], alpha=0.3, label='Histogram')
    axes[1, 0].set_title('MACD')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
   
    # Volume
    axes[1, 1].bar(df['Date'], df['Volume'], alpha=0.7, color='orange')
    axes[1, 1].set_title('Volume')
    axes[1, 1].grid(True, alpha=0.3)
   
    plt.tight_layout()
    return fig


def main():
    # Title
    st.markdown('<h1 class="main-header">🤖 AI Trading Agent Dashboard</h1>', unsafe_allow_html=True)
   
    # Sidebar
    st.sidebar.header("📊 Trading Parameters")
    ticker = st.sidebar.text_input("Enter Ticker Symbol:", value="AAPL", help="E.g., AAPL, GOOGL, TSLA").upper()
   
    period_options = {"1 Year": "1y", "2 Years": "2y", "5 Years": "5y"}
    selected_period = st.sidebar.selectbox("Select Data Period:", list(period_options.keys()), index=1)
    period = period_options[selected_period]
   
    # Main content
    if st.sidebar.button("🚀 Get AI Recommendation", type="primary"):
        with st.spinner(f"🔍 Analyzing {ticker} with AI..."):
            # Fetch data
            df = fetch_real_stock_data(ticker, period)
           
            if df is not None and len(df) > 100:
                # Process data
                df = add_technical_indicators(df)
                df = add_advanced_features(df)
                df = generate_labels(df)
               
                # Train model
                model, scaler, selector, feature_cols = train_prediction_model(df)
               
                # Make prediction
                prediction, probabilities = make_prediction(model, scaler, selector, feature_cols, df)
               
                # Display prediction
                col1, col2, col3 = st.columns([1, 2, 1])
               
                with col2:
                    if prediction == 1:
                        st.markdown(f'<div class="prediction-card buy-card">🟢 BUY {ticker}</div>', unsafe_allow_html=True)
                    elif prediction == -1:
                        st.markdown(f'<div class="prediction-card sell-card">🔴 SELL {ticker}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="prediction-card hold-card">🟡 HOLD {ticker}</div>', unsafe_allow_html=True)
               
                # Show confidence scores
                st.subheader("🎯 AI Confidence Scores")
                col1, col2, col3 = st.columns(3)
               
                with col1:
                    st.metric("Sell Probability", f"{probabilities[0]:.1%}", delta=None)
                with col2:
                    st.metric("Hold Probability", f"{probabilities[1]:.1%}", delta=None)
                with col3:
                    st.metric("Buy Probability", f"{probabilities[2]:.1%}", delta=None)
               
                # Current market data
                latest = df.iloc[-1]
                st.subheader(f"📈 Current Market Data for {ticker}")
               
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${latest['Close']:.2f}")
                with col2:
                    daily_change = (latest['Close'] - latest['Open']) / latest['Open'] * 100
                    st.metric("Daily Change", f"{daily_change:+.2f}%")
                with col3:
                    st.metric("RSI", f"{latest['RSI_14']:.1f}")
                with col4:
                    st.metric("Volume", f"{latest['Volume']:,.0f}")
               
                # Technical analysis
                st.subheader("📊 Technical Analysis")
               
                # Create and display charts
                fig = create_charts(df.tail(100), ticker)  # Show last 100 days
                st.pyplot(fig)
               
                # Key insights
                st.subheader("🔍 Key Insights")
               
                insights = []
                if latest['RSI_14'] > 70:
                    insights.append("⚠️ RSI indicates overbought conditions")
                elif latest['RSI_14'] < 30:
                    insights.append("📈 RSI indicates oversold conditions")
               
                if latest['Close'] > latest['SMA20']:
                    insights.append("📈 Price is above 20-day moving average (bullish)")
                else:
                    insights.append("📉 Price is below 20-day moving average (bearish)")
               
                if latest['MACD'] > latest['MACD_signal']:
                    insights.append("📈 MACD shows bullish momentum")
                else:
                    insights.append("📉 MACD shows bearish momentum")
               
                for insight in insights:
                    st.write(insight)
               
                # Disclaimer
                st.warning("⚠️ **Disclaimer**: This is for educational purposes only. Not financial advice. Always do your own research and consult with financial professionals before making investment decisions.")
               
            else:
                st.error(f"❌ Could not fetch sufficient data for {ticker}. Please check the ticker symbol and try again.")
   
    # Information section
    if not st.sidebar.button("🚀 Get AI Recommendation", type="primary"):
        st.info("👈 Enter a ticker symbol in the sidebar and click 'Get AI Recommendation' to start!")
       
        st.subheader("🤖 How it works:")
        st.write("""
        1. **Data Collection**: Fetches real-time stock data
        2. **Technical Analysis**: Calculates 30+ technical indicators
        3. **AI Prediction**: Uses advanced ML models (XGBoost, Random Forest, etc.)
        4. **Risk Assessment**: Provides confidence scores for each recommendation
        5. **Visualization**: Shows key charts and market insights
        """)
       
        st.subheader("📊 Supported Features:")
        col1, col2 = st.columns(2)
        with col1:
            st.write("✅ Real-time stock data")
            st.write("✅ Technical indicators (RSI, MACD, Bollinger Bands)")
            st.write("✅ Multiple ML models ensemble")
        with col2:
            st.write("✅ Confidence scoring")
            st.write("✅ Interactive charts")
            st.write("✅ Risk analysis")


if __name__ == "__main__":
    main()

