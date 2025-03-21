import os
from crewai import Agent, Task, Process, Crew

api = os.environ.get("OPENAI_API_KEY")

# Define the agents
market_researcher = Agent(
    role="Market Research Analyst",
    goal="Analyze the market demand and competitive landscape of FAANG companies for investment decisions",
    backstory="""You are an expert at evaluating market trends, company performance, and competition. Your insights will help determine 
    which of the FAANG companies shows the most potential for investment in Q1 2025. You must analyze financial reports, industry trends, 
    and growth indicators to assess the best investment opportunities.""",
    verbose=True,
    allow_delegation=False,
)

financial_expert = Agent(
    role="Financial Analyst",
    goal="Evaluate the financial health of FAANG companies and recommend the most profitable investment",
    backstory="""You have deep expertise in financial analysis, including reading and interpreting financial reports, P&L statements, 
    and market indicators. Your task is to assess the profitability, financial stability, and growth potential of the FAANG companies to 
    recommend the best investment option based on quantitative data.""",
    verbose=True,
    allow_delegation=True,
)

industry_expert = Agent(
    role="Industry Expert",
    goal="Analyze the technological innovation and market positioning of FAANG companies to determine their future growth potential",
    backstory="""You are an expert in technology and market trends, with a special focus on the FAANG companies. Your insights into technological 
    advancements and market positioning will help forecast the potential growth and competitive advantage of these companies, aiding in investment decisions.""",
    verbose=True,
    allow_delegation=True,
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Summarize all reports and recommend the best investment based on the analysis",
    backstory="""You are an experienced business consultant with a strong ability to synthesize various reports and make well-rounded business decisions. 
    Your goal is to analyze market trends, financial data, and technological advancements to identify the best investment opportunity. You will consolidate the findings and offer a clear investment recommendation.""",
    verbose=True,
    allow_delegation=True,
)

# Define Tasks and include expected_output field
task1 = Task(
    description="""Analyze the current market position, demand trends, and competition for each of the FAANG companies. 
    Write a detailed report comparing their market performance, highlighting the companies that show the best growth potential for Q1 2025.""",
    agent=market_researcher,
    expected_output="""A detailed market analysis report comparing the FAANG companies, identifying the ones with the best potential for growth in Q1 2025.""",
)

task2 = Task(
    description="""Evaluate the financial performance and health of the FAANG companies. 
    Write a detailed financial analysis report, comparing profitability, growth rates, and stability of each company, and recommending the most profitable investment option.""",
    agent=financial_expert,
    expected_output="""A financial analysis report comparing the FAANG companies, with a recommendation for the best investment based on financial performance.""",
)

task3 = Task(
    description="""Analyze the technological innovation and market positioning of each FAANG company to assess their future growth potential. 
    Write a detailed report outlining which company is likely to lead in technological advancements and market influence by Q1 2025.""",
    agent=industry_expert,
    expected_output="""A report analyzing the technological advancements and market positioning of FAANG companies, identifying the company with the best growth prospects.""",
)

task4 = Task(
    description="""Summarize the findings from the market, financial, and industry analysis reports and write a comprehensive investment recommendation. 
    The report should include which FAANG company is the best investment option as of Q1 2025, based on the analysis, along with key reasons for this recommendation.""",
    agent=business_consultant,  # Correct agent assigned here
    expected_output="""A comprehensive investment recommendation report, summarizing the findings from the market, financial, and industry reports, with a clear recommendation on the best investment option.""",
)

# Create the Crew and Assign Tasks
crew = Crew(
    agents=[market_researcher, financial_expert, industry_expert, business_consultant],
    tasks=[task1, task2, task3, task4],
    verbose=True,  # Set verbose to True for detailed logging
    process=Process.sequential,  # Sequential process will have tasks executed one after the other
)

# Start the Crew
result = crew.kickoff()

print("######################")
print(result)
