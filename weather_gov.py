import requests

# Function to get weather data (temperature, air quality, etc.)
def get_weather_from_weather_gov(lat, lon):
    try:
        # Define the URL for the weather forecast based on latitude and longitude
        url = f"https://forecast.weather.gov/MapClick.php?lat={lat}&lon={lon}&FcstType=json"
        response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()

        # Parse the JSON data from the response
        data = response.json()

        # Extract specific data from the response
        temperature = data["currentobservation"]["Temp"] + "°F"  # Temperature
        condition = data["currentobservation"]["Weather"]  # Weather condition
        forecast = data["data"]["text"][0]  # First forecast text
        
        # Extract Air Quality information (if available)
        air_quality = data.get("currentobservation", {}).get("AirQuality", "No Air Quality Data Available")

        return {
            "Temperature": temperature,
            "Condition": condition,
            "Forecast": forecast,
            "Air Quality": air_quality
        }
    except Exception as e:
        return f"Error retrieving weather data: {e}"

# Example usage
lat = 66.160507  # Latitude for your location
lon = -153.369141  # Longitude for your location

weather_data = get_weather_from_weather_gov(lat, lon)

# Print out the exact weather details
print("Weather Data:")
print(f"Temperature: {weather_data.get('Temperature')}")
print(f"Condition: {weather_data.get('Condition')}")
print(f"Forecast: {weather_data.get('Forecast')}")
print(f"Air Quality: {weather_data.get('Air Quality')}")
