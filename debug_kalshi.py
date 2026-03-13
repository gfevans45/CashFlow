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

print("Authenticated.\n")

# Search for weather-specific tickers
weather_tickers = ["KXHIGH", "HIGHNY", "HIGHCHI", "KXTEMP", "KXLOW", "TEMP", "WEATHER"]
print("--- Searching for weather event tickers ---")
for prefix in weather_tickers:
    resp = client._get("/events", params={"limit": 20, "status": "open", "series_ticker": prefix})
    data = resp.json()
    events = data.get("events", [])
    if events:
        print(f"\n  {prefix}: Found {len(events)} events!")
        for e in events[:5]:
            print(f"    {e.get('event_ticker')}: {e.get('title')}")

# Search all categories
print("\n--- All event categories ---")
resp = client._get("/events", params={"limit": 200, "status": "open"})
events = resp.json().get("events", [])
categories = {}
for e in events:
    cat = e.get("category", "unknown")
    categories[cat] = categories.get(cat, 0) + 1
for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
    print(f"  {cat}: {count} events")

# Search specifically for Climate and Weather
print("\n--- Climate and Weather events ---")
for e in events:
    if e.get("category") == "Climate and Weather":
        ticker = e.get("event_ticker", "")
        title = e.get("title", "")
        print(f"  {ticker}: {title}")

# Try searching markets with weather keywords
print("\n--- Markets with temperature keywords ---")
resp2 = client._get("/markets", params={"limit": 500, "status": "open"})
markets = resp2.json().get("markets", [])
print(f"Total markets fetched: {len(markets)}")
temp_keywords = ["temperature", "temp", "high", "degree", "fahrenheit", "nyc", "chicago", "austin"]
for m in markets:
    title = (m.get("title", "") + " " + m.get("subtitle", "")).lower()
    ticker = m.get("ticker", "").lower()
    if any(kw in title or kw in ticker for kw in temp_keywords):
        print(f"  {m.get('ticker')}: {m.get('title')} | {m.get('subtitle', '')}")
