import requests
from urllib.robotparser import RobotFileParser

def can_scrape(website_url):
    robots_url = website_url.rstrip('/') + "/robots.txt"
    rp = RobotFileParser()
    
    try:
        rp.set_url(robots_url)
        rp.read()
        return rp.can_fetch("*", website_url)
    except Exception as e:
        print(f"Error fetching robots.txt: {e}")
        return False

# Example usage:
website = "https://www.accuweather.com"  # Replace with the website you want to check
if can_scrape(website):
    print(f"You can scrape {website}")
else:
    print(f"Scraping is not allowed on {website}")


