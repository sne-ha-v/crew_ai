import requests
from crewai import Agent, Task, Process, Crew

# Function to get user location based on IP
def get_user_location():
    try:
        response = requests.get("https://ipinfo.io/json")
        data = response.json()
        city = data.get("city", "Unknown")
        region = data.get("region", "Unknown")
        country = data.get("country", "Unknown")
        loc = data.get("loc", "0,0").split(",")  # Get latitude, longitude
        return f"{city}, {region}, {country}", loc[0], loc[1]
    except Exception as e:
        return f"Error getting location: {e}", "0", "0"

# Function to fetch weather data from weather.gov
def get_weather_from_weather_gov(lat, lon):
    try:
        url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        data = response.json()

        # Extract temperature, conditions, and detailed forecast
        temperature = data["currentobservation"]["Temp"] + "°F"
        condition = data["currentobservation"]["Weather"]
        forecast = data["data"]["text"][0]  # First forecast text

        return f"Temperature: {temperature}, Condition: {condition}, Forecast: {forecast}"

    except Exception as e:
        return f"Error retrieving weather data: {e}"

# Function to fetch disaster alerts from weather.gov
def get_natural_disaster_warnings():
    try:
        nws_url = "https://api.weather.gov/alerts/active"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(nws_url, headers=headers)
        response.raise_for_status()
        alerts = response.json().get("features", [])

        if not alerts:
            return "No active natural disaster warnings."

        warnings = []
        for alert in alerts[:3]:  # Get the first 3 alerts
            properties = alert.get("properties", {})
            title = properties.get("headline", "No title available")
            severity = properties.get("severity", "Unknown severity")
            warnings.append(f"{title} (Severity: {severity})")

        return "\n".join(warnings)

    except Exception as e:
        return f"Error fetching disaster warnings: {e}"

# Define a single AI Agent
scraper_agent = Agent(
    role="Weather & Disaster Data Scraper",
    goal="Scrape weather details and disaster warnings from weather.gov",
    backstory="You are an expert in retrieving real-time weather and disaster information from government sources.",
    verbose=True,
    allow_delegation=False,
)

# Get location and coordinates
location, lat, lon = get_user_location()

# Define a single task for the agent
scraping_task = Task(
    description=f"Scrape weather and disaster alerts for {location}.",
    agent=scraper_agent,
    expected_output="A summary of the current weather and natural disaster warnings.",
)

# Run the Crew
crew = Crew(
    agents=[scraper_agent],
    tasks=[scraping_task],
    verbose=True,
    process=Process.sequential,
)

result = crew.kickoff()

# Print Results
print("\n######################")
print(f"Location: {location}")
print(f"\nWeather Report: {get_weather_from_weather_gov(lat, lon)}")
print(f"\nNatural Disaster Warnings: {get_natural_disaster_warnings()}")
print(result)
