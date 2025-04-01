import os
import aiohttp
import asyncio
import logging
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from crewai import Agent, Task, Process, Crew

# Logging setup
logging.basicConfig(level=logging.INFO)

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to fetch AI-related news articles asynchronously using NewsAPI
async def fetch_ai_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "AI for mid-market companies",
        "apiKey": "your_api_key",  # Get this from https://newsapi.org/
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

# Function to fetch AI-related news from Google Search
async def fetch_google_ai_news():
    url = "https://news.google.com/search?q=AI+for+mid-market+companies"
    headers = {"User-Agent": "Mozilla/5.0"}

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")

                # Extract top 5 headlines
                headlines = [item.text for item in soup.find_all("h3")][:5]

                if not headlines:
                    return {"Headlines": ["No recent news found."], "Sentiment Score": 0}

                # Sentiment Analysis
                sentiment_scores = [sia.polarity_scores(headline)["compound"] for headline in headlines]
                avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)

                return {"Headlines": headlines, "Sentiment Score": round(avg_sentiment, 2)}

        except Exception as e:
            logging.error(f"Error fetching AI news: {e}")
            return {"Headlines": ["Error fetching news"], "Sentiment Score": 0}

# Define AI content creator agent
content_creator = Agent(
    role="AI Content Strategist",
    goal="Generate engaging content for mid-market companies, focusing on AI's impact and trends.",
    backstory="""You specialize in content creation for AI and its benefits to mid-market businesses. 
    Your content educates and empowers companies on how AI can enhance operations, improve customer service, and drive growth.""",
    verbose=True
)

# Define tasks for the content generation
task1 = Task(
    description="Research and gather the latest AI trends and how they affect mid-market companies.",
    agent=content_creator,
    expected_output="A detailed analysis of the latest AI trends, challenges, and opportunities for mid-market companies."
)

task2 = Task(
    description="Generate a content outline for LinkedIn, focusing on AI benefits, case studies, and actionable insights.",
    agent=content_creator,
    expected_output="A LinkedIn post outline with key points on AI for mid-market companies, including relevant links and statistics."
)

task3 = Task(
    description="Write a full LinkedIn post based on the analysis and outline.",
    agent=content_creator,
    expected_output="A complete LinkedIn post that is SEO-optimized, informative, and engaging for the target audience."
)

# Create the AI Crew for content generation
crew = Crew(
    agents=[content_creator],
    tasks=[task1, task2, task3],
    verbose=True,
    process=Process.sequential
)

# Run the main workflow
async def main():
    print("\n📰 Fetching latest AI news for mid-market companies...")
    
    # Fetch AI-related news using NewsAPI
    news_data = await fetch_ai_news()
    
    # Print the top news headlines
    print(f"\n📌 Top AI News: (Sentiment Score: {news_data['Sentiment Score']})")
    for headline in news_data["Headlines"]:
        print(f"- {headline}")
    
    if news_data["Sentiment Score"] == 0:
        print("⚠️ No AI news found or there was an error fetching news.")
    
    # Fetch AI-related news using Google Search (alternative method)
    google_news_data = await fetch_google_ai_news()
    
    print("\n🌍 Additional Google AI News Headlines:")
    print(f"\n📌 Top Google AI News: (Sentiment Score: {google_news_data['Sentiment Score']})")
    for headline in google_news_data["Headlines"]:
        print(f"- {headline}")
    
    if google_news_data["Sentiment Score"] == 0:
        print("⚠️ No Google AI news found or there was an error fetching news.")
    
    print("\n🤖 AI-Powered Content Generation:")
    
    try:
        # Start content generation
        result = crew.kickoff()
        print("\n######################")
        print("Generated Content for LinkedIn:")
        print(result)
    except Exception as e:
        logging.error(f"Error in content generation: {e}")
        print("⚠️ Error generating content.")

# Run async main function
if __name__ == "__main__":
    asyncio.run(main())
