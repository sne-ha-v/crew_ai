import os
from crewai import Agent, Task, Process, Crew

api = os.environ.get("OPENAI_API_KEY")

# Define the agents
marketer = Agent(
    role="Market Research Analyst",
    goal="Find out how big is the demand for my products and suggest how to reach the widest possible customer base",
    backstory="""You are an expert at understanding the market demand, target audience, and competition. This is crucial for 
    validating whether an idea fulfills a market need and has the potential to attract a wide audience. You are good at coming up
    with ideas on how to appeal to the widest possible audience.""",
    verbose=True,
    allow_delegation=False,
)

technologist = Agent(
    role="Technology Expert",
    goal="Make assessment on how technologically feasible the company is and what type of technologies the company needs to adopt in order to succeed",
    backstory="""You are a visionary in the realm of technology, with a deep understanding of both current and emerging technological trends. Your 
    expertise lies not just in knowing the technology but in foreseeing how it can be leveraged to solve real-world problems and drive business innovation.
    You have a knack for identifying which technological solutions best fit different business models and needs, ensuring that companies stay ahead of 
    the curve. Your insights are crucial in aligning technology with business strategies, ensuring that the technological adoption not only enhances 
    operational efficiency but also provides a competitive edge in the market.""",
    verbose=True,
    allow_delegation=True,
)

business_consultant = Agent(
    role="Business Development Consultant",
    goal="Evaluate and advise on the business model, scalability, and potential revenue streams to ensure long-term sustainability and profitability",
    backstory="""You are a seasoned professional with expertise in shaping business strategies. Your insight is essential for turning innovative ideas 
    into viable business models. You have a keen understanding of various industries and are adept at identifying and developing potential revenue streams. 
    Your experience in scalability ensures that a business can grow without compromising its values or operational efficiency. Your advice is not just
    about immediate gains but about building a resilient and adaptable business that can thrive in a changing market.""",
    verbose=True,
    allow_delegation=True,
)

# Define Tasks and include expected_output field
task1 = Task(
    description="""Analyze what the market demand for plugs for holes in crocs (shoes) so that this iconic footwear looks less like swiss cheese. 
    Write a detailed report with a description of what the ideal customer might look like, and how to reach the widest possible audience. 
    The report has to be concise with at least 10 bullet points and it has to address the most important areas when it comes to marketing this type of business.""",
    agent=marketer,
    expected_output="""A report detailing market demand, target customer profile, and strategies to reach a wide audience, including 10 bullet points on key marketing considerations.""",
)

task2 = Task(
    description="""Analyze how to produce plugs for crocs (shoes) so that this iconic footwear looks less like swiss cheese. Write a detailed report 
    with a description of which technologies the business needs to use in order to make high-quality Crocs plugs. The report has to be concise with 
    at least 10 bullet points and it has to address the most important areas when it comes to manufacturing this type of business.""",
    agent=technologist,
    expected_output="""A report detailing the technologies required for manufacturing Crocs plugs, including 10 bullet points on key technological considerations and manufacturing processes.""",
)

task3 = Task(
    description="""Analyze and summarize marketing and technological report and write a detailed business plan with 
    description of how to make a sustainable and profitable 'plugs for crocs' business. 
    The business plan has to be concise with at least 10 bullet points, 5 goals and it has to contain a time schedule for which goal should be achieved and when.""",
    agent=business_consultant,
    expected_output="""A concise business plan with at least 10 bullet points, 5 specific business goals, and a time schedule for achieving each goal.""",
)

# Create the Crew and Assign Tasks
crew = Crew(
    agents=[marketer, technologist, business_consultant],
    tasks=[task1, task2, task3],
    verbose=True,  # Set verbose to True for detailed logging
    process=Process.sequential,  # Sequential process will have tasks executed one after the other
)

# Start the Crew
result = crew.kickoff()

print("######################")
print(result)
