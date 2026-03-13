#!/usr/bin/env python3
"""Debug script to see what Kalshi markets are available."""
from dotenv import load_dotenv
load_dotenv()

from cashflow.data.weather_feeds import KalshiWeatherClient

client = KalshiWeatherClient()
if not client.has_credentials:
    print("No credentials found!")
    exit(1)

if not client.authenticate():
    print("Auth failed!")
    exit(1)

print("Authenticated. Fetching all events...\n")

# Get raw events
resp = client._get("/events", params={"limit": 100, "status": "open"})
data = resp.json()
events = data.get("events", [])
print(f"Total open events: {len(events)}\n")

# Show all event categories/titles
weather_keywords = ["temp", "high", "weather", "degree", "fahrenheit", "climate", "hot", "cold"]
for e in events[:50]:
    title = e.get("title", "")
    ticker = e.get("event_ticker", "")
    category = e.get("category", "")
    is_weather = any(kw in title.lower() or kw in ticker.lower() for kw in weather_keywords)
    flag = " <<< WEATHER" if is_weather else ""
    print(f"  [{category}] {ticker}: {title}{flag}")

# Also try searching markets directly
print(f"\n--- Searching markets ---")
resp2 = client._get("/markets", params={"limit": 50, "status": "open"})
markets = resp2.json().get("markets", [])
print(f"Total open markets (first 50): {len(markets)}")
for m in markets[:20]:
    title = m.get("title", "") or m.get("subtitle", "")
    ticker = m.get("ticker", "")
    print(f"  {ticker}: {title}")
