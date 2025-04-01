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

# Resource Integration Class
class ResourceIntegrator:
    async def fetch_able_content(self):
        """
        Asynchronously scrape Able's website for content
        """
        url = "https://able.co"
        headers = {"User-Agent": "Mozilla/5.0"}
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, headers=headers) as response:
                    html = await response.text()
                    soup = BeautifulSoup(html, "html.parser")
                    
                    # Extract key paragraphs
                    paragraphs = [p.text.strip() for p in soup.find_all("p") if p.text.strip()][:5]
                    return "\n".join(paragraphs)
                
            except Exception as e:
                logging.error(f"Error scraping Able's website: {e}")
                return "Error fetching content."

    async def fetch_ai_news(self, api_key):
        """
        Fetch latest AI news with basic error handling
        """
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "AI for mid-market companies",
            "apiKey": api_key,
            "pageSize": 5
        }

        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logging.error(f"News API response status: {response.status}")
                        return {"Headlines": ["No news available"], "Sentiment Score": 0}
                    
                    data = await response.json()
                    headlines = [article['title'] for article in data.get('articles', [])]
                    
                    return {
                        "Headlines": headlines, 
                        "Sentiment Score": 0
                    }

            except Exception as e:
                logging.error(f"Error fetching AI news: {e}")
                return {"Headlines": ["Error fetching news"], "Sentiment Score": 0}

# Define Verbose Agents with Clear Output Instructions
ai_researcher = Agent(
    role="AI Researcher",
    goal="Identify and articulate AI trends for mid-market companies.",
    backstory="You are an expert in translating complex AI developments into actionable business insights.",
    verbose=True,
    output_format="""
    Key Insights:
    1. [Trend Description]
    2. [Business Implication]
    3. [Strategic Recommendation]
    """
)

content_strategist = Agent(
    role="Content Strategist",
    goal="Develop engaging AI-focused content for business leaders.",
    backstory="You transform technical AI concepts into compelling, digestible narratives.",
    verbose=True,
    output_format="""
    Content Strategy:
    - Central Theme: [Theme]
    - Key Messages:
      1. [Message 1]
      2. [Message 2]
    - Tone: Professional, Insightful
    """
)

linkedin_post_agent = Agent(
    role="LinkedIn Content Creator",
    goal="Craft compelling LinkedIn posts about AI for mid-market businesses.",
    backstory="You specialize in creating engaging, shareable content that highlights AI's business value.",
    verbose=True,
    output_format="""
    LinkedIn Post Draft:
    [Attention-Grabbing Headline]

    [Body Paragraph]
    - Key Insight
    - Business Implication
    - Call to Action

    #AI #MidMarketTech #BusinessInnovation
    """
)

# Define Tasks with Explicit Output Requirements
task1 = Task(
    description="Research AI trends for mid-market companies",
    agent=ai_researcher,
    expected_output="""
    Comprehensive AI Trends Report:
    - Top 3 Emerging AI Technologies
    - Potential Business Impact
    - Implementation Strategies
    """
)

task2 = Task(
    description="Develop content strategy based on AI research",
    agent=content_strategist,
    expected_output="""
    Content Strategy Document:
    - Central AI Theme
    - Target Audience Insights
    - Content Pillars
    - Messaging Framework
    """
)

task3 = Task(
    description="Create LinkedIn post about AI for mid-market companies",
    agent=linkedin_post_agent,
    expected_output="""
    Finalized LinkedIn Post:
    - Engaging Headline
    - Informative Body
    - Clear Call-to-Action
    - Relevant Hashtags
    """
)

# Main Workflow Function
async def main():
    print("\n🚀 AI Content Generation Workflow Started 🚀\n")
    
    # Initialize Resource Integrator
    resource_integrator = ResourceIntegrator()
    
    # Fetch Able's website content
    print("🔍 Fetching AI Insights from Able's Website...")
    able_content = await resource_integrator.fetch_able_content()
    print(f"Able Content: {able_content}")
    
    # Fetch AI News (replace with your actual API key)
    print("\n📰 Fetching Latest AI News...")
    ai_news = await resource_integrator.fetch_ai_news(
        api_key="Apikey"  # REPLACE with actual API key
    )
    print(f"AI News Headlines: {ai_news['Headlines']}")
    
    # Define Crew with Verbose Output
    crew = Crew(
        agents=[ai_researcher, content_strategist, linkedin_post_agent],
        tasks=[task1, task2, task3],
        verbose=True,
        process=Process.sequential
    )
    
    # Generate AI Content with Comprehensive Logging
    print("\n🤖 Generating AI-Powered Content...\n")
    try:
        # Pass context to kickoff
        result = crew.kickoff(inputs={
            'able_content': able_content,
            'ai_news': ai_news['Headlines']
        })
        
        print("\n######################")
        print("🌟 Generated Content:")
        print(result)
    except Exception as e:
        logging.error(f"Error in content generation: {e}")
        print(f"⚠️ Error generating content: {e}")

# Run async main function
if __name__ == "__main__":
    asyncio.run(main())