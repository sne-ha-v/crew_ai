import os
import aiohttp
import asyncio
import logging
from crewai import Agent, Task, Process, Crew
from bs4 import BeautifulSoup
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import textwrap

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Initialize sentiment analyzer
sia = SentimentIntensityAnalyzer()

# Function to scrape Able's website asynchronously
async def fetch_able_content():
    url = "https://able.co"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, headers=headers) as response:
                html = await response.text()
                soup = BeautifulSoup(html, "html.parser")
                
                # Extract relevant content
                paragraphs = [p.text.strip() for p in soup.find_all("p") if p.text.strip()][:5]
                return "\n".join(paragraphs)
                
        except Exception as e:
            logging.error(f"Error scraping Able's website: {e}")
            return "Error fetching content."

# Function to fetch the latest AI news
async def fetch_ai_news():
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "AI for mid-market companies",
        "apiKey": "e8ed8f77beff4a08b3be6ccef9bdcf13",  # Get API Key from https://newsapi.org/
        "pageSize": 5
    }

    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url, params=params) as response:
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

# Define AI agents
ai_researcher = Agent(
    role="AI Researcher",
    goal="Identify cutting-edge AI trends and innovations for mid-market companies.",
    backstory="You specialize in researching AI developments and their business applications.",
    verbose=True
)

content_strategist = Agent(
    role="Content Strategist",
    goal="Develop engaging AI-focused content tailored for business leaders.",
    backstory="You create thought leadership content that simplifies complex AI concepts, focusing on practical applications for mid-market companies.",
    verbose=True
)

linkedin_post_agent = Agent(
    role="LinkedIn Content Strategist",
    goal="Create compelling, SEO-optimized LinkedIn posts about AI for mid-market businesses.",
    backstory="""
    You are an expert in crafting concise, engaging social media content that 
    captures attention, provides value, and incorporates strategic SEO keywords. 
    Your posts transform complex AI concepts into digestible, shareable insights 
    that resonate with business leaders and decision-makers.
    """,
    verbose=True
)

# Task definitions for generating multiple LinkedIn posts
def create_tasks(able_content, ai_news):
    tasks = [
        Task(
            description="Research and summarize the latest AI trends relevant to mid-market businesses.",
            agent=ai_researcher,
            expected_output="A concise report on AI advancements for mid-market companies."
        ),
        Task(
            description="Extract insights from Able's website for content strategy.",
            agent=content_strategist,
            expected_output=f"Strategic content summary highlighting AI methodologies:\n{able_content}"
        ),
        Task(
            description="Write a blog post on AI benefits in product development.",
            agent=content_strategist,
            expected_output="An informative blog post about AI-driven product development."
        )
    ]
    
    # LinkedIn Posts using AI news and Able content (5 different posts)
    for i in range(5):
        linkedin_posts = Task(
            description=f"Create LinkedIn post {i+1} about AI and its importance for mid-market companies",
            agent=linkedin_post_agent,
            expected_output=f"""
            A 500-1000-word LinkedIn post focused on AI and mid-market businesses:
            - Discusses AI-driven productivity or customer engagement, etc.
            - Provides concrete insights inspired by AI news like "{ai_news['Headlines'][i % len(ai_news['Headlines'])]}"
            - Includes 3-4 SEO-friendly hashtags for each post 
            - Avoid displaying conclusion and title
            - Provides links to external sources for further reading (e.g., research papers, articles, etc.)
            """
        )
        tasks.append(linkedin_posts)
    
    return tasks

# Main workflow
async def main():
    print("\n🚀 AI Content Generation Workflow Started 🚀\n")
    
    # Fetch Able's website content
    print("🔍 Fetching AI Insights from Able's Website...")
    able_content = await fetch_able_content()
    if able_content != "Error fetching content.":
        print(f"🔹 {able_content}")
    
    # Fetch AI News
    print("\n📰 Fetching Latest AI News...")
    ai_news = await fetch_ai_news()
    
    # Print Headlines
    print("\n🗞️ AI News Headlines:")
    for idx, headline in enumerate(ai_news['Headlines'], 1):
        print(f"{idx}. {headline}")
    
    # Print Sentiment Score
    print(f"\n📊 News Sentiment Score: {ai_news['Sentiment Score']}")
    
    # Define Crew with remaining agents
    crew = Crew(
        agents=[ai_researcher, content_strategist, linkedin_post_agent],
        tasks=create_tasks(able_content, ai_news),
        verbose=True,
        process='sequential'
    )
    
    # Generate AI Content
    print("\n🤖 Generating AI-Powered Content...\n")
    try:
        # Start content generation
        result = crew.kickoff()
        print("\n######################")
        print("Generated Content:")
        print(result)
    except Exception as e:
        logging.error(f"Error in content generation: {e}")
        print("⚠️ Error generating content.")

# Run async main function
if __name__ == "__main__":
    asyncio.run(main())
