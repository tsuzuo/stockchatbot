# 🤖 AI Trading Agent Dashboard


A powerful AI-driven trading assistant that provides buy/hold/sell recommendations for any stock ticker. Built with Streamlit and advanced machine learning models.


## 🌟 Features


### 📊 Dashboard Interface (`streamlit_dashboard.py`)
- **Real-time stock analysis** with yfinance integration
- **Advanced AI predictions** using XGBoost and ensemble methods
- **Technical indicators**: RSI, MACD, Bollinger Bands, Moving Averages
- **Interactive charts** and visualizations
- **Confidence scoring** for each recommendation
- **Comprehensive analysis** with key insights


### 💬 Chatbot Interface (`chatbot_interface.py`)
- **Conversational AI** for stock analysis
- **Natural language processing** - just type ticker symbols
- **Quick recommendations** with explanations
- **Chat-style interface** with typing indicators
- **Example commands** in sidebar


## 🚀 Quick Start


### Option 1: Using the Setup Script (Recommended)
```bash
python setup_and_run.py
```


Follow the prompts to:
1. Install dependencies
2. Choose between dashboard or chatbot interface


### Option 2: Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt


# Run the dashboard
streamlit run streamlit_dashboard.py


# OR run the chatbot
streamlit run chatbot_interface.py
```


## 📋 Requirements


- Python 3.8+
- Internet connection (for real-time stock data)
- Web browser


## 🔧 Dependencies


All dependencies are listed in `requirements.txt`:
- streamlit
- pandas
- numpy
- matplotlib
- seaborn
- yfinance
- scikit-learn
- xgboost
- lightgbm
- scipy


## 📖 How to Use


### Dashboard Interface
1. Open the dashboard in your browser
2. Enter a ticker symbol in the sidebar (e.g., AAPL, GOOGL, TSLA)
3. Click "🚀 Get AI Recommendation"
4. View the analysis with charts, metrics, and insights


### Chatbot Interface
1. Open the chatbot in your browser
2. Type natural language queries like:
   - "What about AAPL?"
   - "Should I buy TSLA?"
   - "Analyze GOOGL"
   - "How's MSFT looking?"
3. Get instant AI recommendations with explanations


## 🤖 AI Model Features


- **Multi-model ensemble**: XGBoost, Random Forest, Gradient Boosting
- **30+ technical indicators**: RSI, MACD, Bollinger Bands, momentum, volatility
- **Feature engineering**: Lagged features, rolling statistics, trend analysis
- **Smart labeling**: Percentile-based signal generation for balanced predictions
- **Risk assessment**: Confidence scoring for each recommendation


## 📊 Technical Indicators Used


- **Trend**: SMA (5,10,20,50), EMA (12,26), MACD
- **Momentum**: RSI (14), Price momentum (5,10,20 day)
- **Volatility**: Bollinger Bands, ATR, rolling volatility
- **Volume**: Volume ratios, price-volume relationship
- **Support/Resistance**: Rolling highs/lows, price position


## ⚠️ Important Disclaimer


**This tool is for educational purposes only and should not be considered financial advice.**


- Always do your own research before making investment decisions
- Past performance does not guarantee future results
- Consider consulting with financial professionals
- The AI model predictions are based on historical data and technical analysis
- Market conditions can change rapidly and unpredictably


## 🛠️ Customization


You can modify the models and parameters by editing:


- `streamlit_dashboard.py`: Main dashboard functionality
- `chatbot_interface.py`: Chatbot responses and analysis
- `AI_Agent.py`: Original comprehensive analysis (for reference)


## 📁 Project Structure


```
AI_Trading_Agent/
├── streamlit_dashboard.py    # Main dashboard interface
├── chatbot_interface.py      # Conversational interface
├── setup_and_run.py         # Setup and launcher script
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── AI_Agent.py             # Original analysis script
└── *.csv                   # Data files (auto-generated)
```


## 🎯 Example Usage


### Dashboard
1. Enter "AAPL" in sidebar
2. Click "Get AI Recommendation"
3. View BUY/HOLD/SELL recommendation
4. Analyze charts and technical indicators
5. Read AI insights and confidence scores


### Chatbot
1. Type: "What about Tesla?"
2. Get instant analysis for TSLA
3. Ask follow-up questions
4. Try different stocks and companies


## 🔍 Troubleshooting


**"Could not fetch data for ticker"**
- Check if ticker symbol is correct
- Ensure internet connection
- Try a different, well-known ticker (e.g., AAPL, GOOGL)


**Installation errors**
- Make sure Python 3.8+ is installed
- Try upgrading pip: `pip install --upgrade pip`
- Install packages individually if needed


**Dashboard won't load**
- Check if port 8501 is available
- Try: `streamlit run streamlit_dashboard.py --server.port 8502`


## 💡 Tips for Best Results


1. **Use well-known tickers**: AAPL, GOOGL, MSFT, TSLA, AMZN
2. **Check multiple timeframes**: Try 1Y, 2Y, 5Y periods
3. **Consider the confidence score**: Higher confidence = more reliable signal
4. **Look at technical indicators**: RSI, MACD trends for context
5. **Read the insights**: Understand why the AI made its recommendation


## 🚀 Future Enhancements


Potential improvements:
- Real-time alerts and notifications
- Portfolio tracking and management
- More advanced ML models (LSTM, Transformers)
- News sentiment analysis integration
- Backtesting with custom strategies
- Mobile-responsive design


## 📞 Support


If you encounter issues:
1. Check this README for troubleshooting
2. Ensure all dependencies are installed correctly
3. Try the setup script: `python setup_and_run.py`
4. Verify internet connection for stock data


---


**Happy Trading! 📈🤖**


*Remember: This is an educational tool. Always do your own research and never invest more than you can afford to lose.*
