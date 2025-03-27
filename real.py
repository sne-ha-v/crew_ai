import logging
import aiohttp
import asyncio
import yfinance as yf
import pandas as pd
import json
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from crewai import Agent, Task, Process, Crew
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Define FAANG stocks and their full company names for news search
FAANG_COMPANIES = {
    "AAPL": "Apple AI",
    "AMZN": "Amazon AI",
    "META": "Meta AI",
    "GOOGL": "Google AI",
    "NFLX": "Netflix AI"
}

# Function to fetch AI-related news and sentiment scores for each FAANG company
async def fetch_faang_ai_news():
    url = "https://newsapi.org/v2/everything"
    api_key = "e8ed8f77beff4a08b3be6ccef9bdcf13"
    
    results = {}

    async with aiohttp.ClientSession() as session:
        for ticker, query in FAANG_COMPANIES.items():
            params = {"q": query, "apiKey": api_key, "pageSize": 5, "language": "en"}
            
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        results[ticker] = {"Headlines": ["Error fetching news"], "Sentiment Score": 0}
                        continue
                    
                    data = await response.json()
                    if not data.get("articles"):
                        results[ticker] = {"Headlines": ["No recent AI news found"], "Sentiment Score": 0}
                        continue
                    
                    headlines = [article['title'] for article in data['articles']]
                    sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
                    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

                    results[ticker] = {"Headlines": headlines, "Sentiment Score": round(avg_sentiment, 2)}
            
            except Exception as e:
                logging.error(f"Error fetching AI news for {ticker}: {e}")
                results[ticker] = {"Headlines": ["Error fetching news"], "Sentiment Score": 0}

    return results

# Function to fetch detailed stock data including Lag-1 analysis
def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        
        # Fetch 3 years of historical data
        hist = stock.history(period="3y")

        # Calculate Lag-1 differences
        latest_price = hist["Close"].iloc[-1]
        lag_1_day = hist["Close"].iloc[-2] if len(hist) > 1 else latest_price
        lag_1_month = hist["Close"].iloc[-22] if len(hist) > 22 else latest_price
        lag_1_year = hist["Close"].iloc[-252] if len(hist) > 252 else latest_price

        # Compute percentage changes
        day_change = round((latest_price - lag_1_day) / lag_1_day * 100, 2)
        month_change = round((latest_price - lag_1_month) / lag_1_month * 100, 2)
        year_change = round((latest_price - lag_1_year) / lag_1_year * 100, 2)

        # Additional financial data
        pe_ratio = stock.info.get("trailingPE", "N/A")
        market_cap = stock.info.get("marketCap", "N/A")
        eps = stock.info.get("trailingEps", "N/A")
        volume = hist["Volume"].iloc[-1] if "Volume" in hist.columns else "N/A"

        return {
            "Latest Price": round(latest_price, 2),
            "Lag-1 Day Change": f"{day_change}%",
            "Lag-1 Month Change": f"{month_change}%",
            "Lag-1 Year Change": f"{year_change}%",
            "P/E Ratio": pe_ratio,
            "Market Cap": market_cap,
            "EPS": eps,
            "Trading Volume": volume
        }
    
    except Exception as e:
        logging.error(f"Error fetching stock data for {ticker}: {e}")
        return {}

# ARIMA Forecasting Function
def forecast_arima(stock_data, steps=5):
    model = ARIMA(stock_data['Close'], order=(5, 1, 0))  # Adjust order as needed
    model_fit = model.fit()

    # Forecast the next 'steps' values
    forecast = model_fit.forecast(steps=steps)
    
    return forecast, model_fit.resid  # Return residuals for LSTM training

# Prepare LSTM model
def create_lstm_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        tf.keras.layers.LSTM(50, return_sequences=True),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Prepare LSTM Data using ARIMA residuals
def prepare_lstm_data(residuals, lookback=60):
    scaler = MinMaxScaler(feature_range=(-1, 1))  # Scaling residuals
    scaled_data = scaler.fit_transform(residuals.values.reshape(-1, 1))

    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

# Hybrid ARIMA-LSTM Forecasting
def hybrid_forecast(stock_data, lstm_model, lookback=60, steps=5):
    arima_forecast, residuals = forecast_arima(stock_data, steps)
    
    # Prepare LSTM data using residuals
    X_lstm, y_lstm, scaler = prepare_lstm_data(residuals, lookback)

    # Train LSTM on residuals
    lstm_model.fit(X_lstm, y_lstm, epochs=10, batch_size=16, verbose=1)

    # Predict residuals for the next 'steps' time points
    predicted_residuals = lstm_model.predict(X_lstm[-steps:])
    predicted_residuals = scaler.inverse_transform(predicted_residuals)

    # Final forecast = ARIMA forecast + predicted residuals
    final_forecast = arima_forecast + predicted_residuals.flatten()

    return final_forecast

# Define CrewAI agents
data_scientist = Agent(
    role="Data Scientist", 
    goal="Prepare and preprocess data for forecasting models like ARIMA and LSTM.",
    backstory="Expert in data preprocessing, feature engineering, and training machine learning models.",
    verbose=True
)

market_analyst = Agent(
    role="FINRA Approved Analyst", 
    goal="Analyze FAANG stock trends, AI news sentiment, and historical lag analysis.",
    backstory="Expert in financial markets, specializing in stock trends and AI-driven market insights.",
    verbose=True
)

investment_consultant = Agent(
    role="Investment Advisor", 
    goal="Evaluate FAANG stock performance, AI developments, forecasting models, and recommend top investments.",
    backstory="Professional investment consultant leveraging AI, financial market trends, and advanced forecasting techniques like ARIMA and LSTM.",
    verbose=True
)

stock_forecaster = Agent(
    role="Stock Forecaster", 
    goal="Use ARIMA and LSTM models to predict future stock prices for FAANG companies.",
    backstory="Specialist in stock price prediction using statistical models and deep learning techniques.",
    verbose=True
)

# Define tasks with logical dependencies
task1 = Task(
    description="Prepare and preprocess historical stock data for ARIMA and LSTM forecasting models.",
    agent=data_scientist,
    expected_output="Preprocessed data ready for forecasting with ARIMA and LSTM models, including cleaned historical price data, normalized features, and train-test splits."
)

task2 = Task(
    description="Fetch real-time stock data for FAANG stocks, including Lag-1 analysis and key financial metrics.",
    agent=market_analyst,
    expected_output="A comprehensive summary of FAANG stock prices, P/E ratios, Lag-1 changes, trading volumes, and recent market conditions."
)

task3 = Task(
    description="Analyze AI-related news sentiment for each FAANG company and compare it with stock trends over 3 years.",
    agent=market_analyst,
    expected_output="A detailed company-wise comparison of AI news sentiment, including sentiment scores, correlation with stock performance, and key trend analysis."
)

task5 = Task(
    description="Generate ARIMA and LSTM forecasting models for each FAANG stock using preprocessed data.",
    agent=stock_forecaster,
    expected_output="Detailed forecasting reports for each FAANG stock, including model performance metrics, predicted price ranges, and confidence intervals."
)

task4 = Task(
    description="""Generate a comprehensive investment recommendation based on:
    1. FAANG stock current data
    2. AI news sentiment analysis
    3. Lag-1 trends
    4. ARIMA and LSTM forecasting predictions
    
    Provide a detailed analysis comparing forecasting model predictions, sentiment scores, and current market conditions to identify the most promising FAANG stock for Q1 2025.""",
    agent=investment_consultant,
    expected_output="""A comprehensive investment recommendation for FAANG stocks in Q1 2025, including:
    1. Current market data and trends for each FAANG stock
    2. ARIMA and LSTM model forecasting results with a comparative analysis
    3. AI news sentiment analysis and its impact on stock performance
    4. Final investment recommendation with a well-supported justification
    5. Risk assessment and potential market challenges"""
)

# Create CrewAI workflow
crew = Crew(
    agents=[data_scientist, market_analyst, stock_forecaster, investment_consultant],
    tasks=[task1, task2, task3, task5, task4],  # Updated task order
    verbose=True,
    process=Process.sequential
)

async def main():
    try:
        # Fetch AI News Sentiment
        print("\n--- AI News Sentiment Analysis ---")
        news_sentiment = await fetch_faang_ai_news()
        for ticker, data in news_sentiment.items():
            print(f"{ticker} - Sentiment Score: {data['Sentiment Score']}")
            print("Top Headlines:")
            for headline in data['Headlines'][:3]:
                print(f"  - {headline}")
            print()

        # Fetch Stock Data and Print
        print("\n--- Stock Data Analysis ---")
        stock_data = {}
        for ticker in FAANG_COMPANIES.keys():
            stock_data[ticker] = fetch_stock_data(ticker)
            print(f"\n{ticker} Stock Details:")
            for key, value in stock_data[ticker].items():
                print(f"  {key}: {value}")

        # Create and Run CrewAI Workflow
        print("\n--- Running CrewAI Workflow ---")
        result = crew.kickoff()
        print("\n--- CrewAI Workflow Results ---")
        print(result)

    except Exception as e:
        print(f"An error occurred: {e}")

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
