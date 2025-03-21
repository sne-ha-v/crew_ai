import os
import yfinance as yf
import requests
import aiohttp
import asyncio
import logging
import pandas as pd
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from crewai import Agent, Task, Process, Crew

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# FAANG stock symbols
FAANG = ["AAPL", "AMZN", "META", "GOOGL", "NFLX"]

# Function to fetch stock data
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")  # Get last 3 months of stock data
    latest_price = stock.history(period="1d")["Close"].iloc[-1]
    pe_ratio = stock.info.get("trailingPE", "N/A")

    return {
        "Ticker": ticker,
        "Latest Price": round(latest_price, 2),
        "P/E Ratio": pe_ratio,
        "3-Month Performance": round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2)
    }

# Asynchronous function to fetch news from Google News (avoids Yahoo Finance issue)
async def fetch_news(ticker):
    url = f"https://news.google.com/search?q={ticker}+stock+news"
    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract top 5 headlines
                headlines = [item.text for item in soup.find_all("h3")][:5]

                if not headlines:
                    return {"Ticker": ticker, "Headlines": ["No recent news found."], "Sentiment Score": 0}

                # Sentiment Analysis
                sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

                return {"Ticker": ticker, "Headlines": headlines, "Sentiment Score": round(avg_sentiment, 2)}

        except Exception as e:
            logging.error(f"Error fetching news for {ticker}: {e}")
            return {"Ticker": ticker, "Headlines": ["Error fetching news"], "Sentiment Score": 0}

# Define AI agents
market_analyst = Agent(
    role="Financial Market Analyst",
    goal="Analyze real-time stock trends and identify the best FAANG investment opportunities.",
    backstory="""You are a financial expert specializing in stock market trends, investor sentiment,
    and economic forecasting. Your insights help investors make data-driven decisions.""",
    verbose=True
)

investment_consultant = Agent(
    role="Investment Advisor",
    goal="Assess financial trends and recommend the best FAANG stock for Q1 2025 based on data.",
    backstory="You analyze stock performance, news sentiment, and economic conditions to give the best investment recommendations.",
    verbose=True
)

# Define tasks
task1 = Task(
    description="Gather real-time stock data for FAANG stocks and summarize key financial metrics.",
    agent=market_analyst,
    expected_output="A summary of FAANG stock prices, P/E ratios, and 3-month performance trends."
)

task2 = Task(
    description="Analyze market sentiment by extracting recent financial news for FAANG stocks.",
    agent=market_analyst,
    expected_output="A summary of recent financial news and sentiment for FAANG stocks."
)

task3 = Task(
    description="Generate an investment recommendation based on stock trends and sentiment analysis.",
    agent=investment_consultant,
    expected_output="A detailed investment report identifying the best FAANG stock for Q1 2025."
)

# Create the AI Crew
crew = Crew(
    agents=[market_analyst, investment_consultant],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential
)

# Run the workflow
async def main():
    print("\n🔍 Fetching real-time stock data...")
    stock_data = [fetch_stock_data(ticker) for ticker in FAANG]
    df = pd.DataFrame(stock_data)
    print(df)

    print("\n📰 Fetching latest financial news for FAANG stocks...")
    news_data = await asyncio.gather(*(fetch_news(ticker) for ticker in FAANG))
    
    for news in news_data:
        print(f"\n📌 Top News for {news['Ticker']}: (Sentiment Score: {news['Sentiment Score']})")
        for headline in news['Headlines']:
            print(f"- {headline}")

    print("\n🤖 AI-Powered Investment Analysis:")
    result = crew.kickoff()
    print("\n######################")
    print(result)

# Run async main function
if __name__ == "__main__":
    asyncio.run(main())
