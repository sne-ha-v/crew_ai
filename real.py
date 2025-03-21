import os
import yfinance as yf
import requests
from bs4 import BeautifulSoup
import pandas as pd
from crewai import Agent, Task, Process, Crew

# Get real-time stock data
FAANG = ["AAPL", "AMZN", "META", "GOOGL", "NFLX"]

def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="3mo")  # Get the last 3 months of stock data
    latest_price = stock.history(period="1d")["Close"].iloc[-1]
    pe_ratio = stock.info.get("trailingPE", "N/A")
    return {
        "Ticker": ticker,
        "Latest Price": round(latest_price, 2),
        "P/E Ratio": pe_ratio,
        "3-Month Performance": round((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0] * 100, 2)
    }

# Web scraping to fetch news headlines
def fetch_news(ticker):
    url = f"https://www.google.com/search?q={ticker}+stock+news"
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")
    headlines = [headline.text for headline in soup.find_all("h3")][:5]  # Get top 5 headlines
    return headlines if headlines else ["No recent news found."]

# Define the agents with real-time analysis
market_analyst = Agent(
    role="Financial Market Analyst",
    goal="Analyze real-time market trends and evaluate the best FAANG investment opportunities.",
    backstory="""You are an expert in financial markets, specializing in analyzing stock trends, 
    financial reports, and investor sentiments. Your knowledge helps investors make informed decisions 
    based on real-time data.""",
    verbose=True
)

investment_consultant = Agent(
    role="Investment Advisor",
    goal="Assess financial trends and suggest the best FAANG stock to invest in Q1 2025 based on real-time metrics.",
    backstory="""You are a professional investment consultant with expertise in evaluating financial 
    data, market trends, and company performance. Your job is to help investors maximize returns 
    by making data-driven decisions.""",
    verbose=True
)

# Define tasks with real-time data
task1 = Task(
    description="Fetch real-time stock data for FAANG stocks and summarize key financial metrics.",
    agent=market_analyst,
    expected_output="A summary of FAANG stock prices, P/E ratios, and 3-month performance trends."
)

task2 = Task(
    description="Analyze market sentiment by extracting top financial news headlines for FAANG stocks.",
    agent=market_analyst,
    expected_output="A summary of recent financial news for FAANG stocks."
)

task3 = Task(
    description="Generate an investment recommendation based on stock data trends, P/E ratios, and news sentiment.",
    agent=investment_consultant,
    expected_output="A detailed investment report identifying the best FAANG stock for Q1 2025."
)

# Create the Crew
crew = Crew(
    agents=[market_analyst, investment_consultant],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential
)

# Execute the Workflow
print("Fetching real-time stock data...")
stock_data = [fetch_stock_data(ticker) for ticker in FAANG]
df = pd.DataFrame(stock_data)
print(df)

print("\nFetching latest financial news for FAANG stocks...")
news_data = {ticker: fetch_news(ticker) for ticker in FAANG}
for ticker, headlines in news_data.items():
    print(f"\nTop News for {ticker}:")
    for headline in headlines:
        print(f"- {headline}")

# Start CrewAI Process
result = crew.kickoff()
print("\n######################")
print(result)
