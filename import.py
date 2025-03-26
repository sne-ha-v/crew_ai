import logging
import aiohttp
import asyncio
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch AI-related news articles asynchronously using NewsAPI
async def fetch_ai_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "AI for mid-market companies",
        "apiKey": "e8ed8f77beff4a08b3be6ccef9bdcf13",  # Get this from https://newsapi.org/
        "pageSize": 5
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
                # Check if the response status is 200 (OK)
                if response.status != 200:
                    return {"Headlines": ["Error fetching news"], "Sentiment Score": 0}
                
                data = await response.json()

                if not data.get("articles"):
                    return {"Headlines": ["No recent news found."], "Sentiment Score": 0}

                headlines = [article['title'] for article in data['articles']]

                # Sentiment Analysis
                sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

                return {"Headlines": headlines, "Sentiment Score": round(avg_sentiment, 2)}

        except Exception as e:
            logging.error(f"Error fetching AI news: {e}")
            return {"Headlines": ["Error fetching news"], "Sentiment Score": 0}

# Example usage with async main function
async def main():
    news_data = await fetch_ai_news()
    
    # Print the top news headlines
    print(f"\n📌 Top AI News: (Sentiment Score: {news_data['Sentiment Score']})")
    for headline in news_data["Headlines"]:
        print(f"- {headline}")
    
    if news_data["Sentiment Score"] == 0:
        print("⚠️ No AI news found or there was an error fetching news.")

# Run async main function
if __name__ == "__main__":
    asyncio.run(main())
