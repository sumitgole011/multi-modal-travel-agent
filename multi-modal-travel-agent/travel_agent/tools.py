from __future__ import annotations

import os
import random
from datetime import date, timedelta
from urllib.parse import quote

import requests
from langchain_core.tools import tool


REQUEST_TIMEOUT = 20
SESSION = requests.Session()

GENERIC_IMAGES = [
    "https://images.unsplash.com/photo-1469474968028-56623f02e42e",
    "https://images.unsplash.com/photo-1500530855697-b586d89ba3ee",
    "https://images.unsplash.com/photo-1501785888041-af3ef285b470",
]


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


def _fallback_weather(city: str, days: int = 7, date_range: str = "next week") -> list[dict]:
    seed = sum(ord(ch) for ch in city + date_range)
    rng = random.Random(seed)
    start = date.today() + timedelta(days=1)
    conditions = ["Sunny", "Cloudy", "Light rain", "Clear", "Breezy"]
    return [
        {
            "date": (start + timedelta(days=offset)).isoformat(),
            "temperature_c": round(rng.uniform(14, 31), 1),
            "condition": rng.choice(conditions),
            "humidity": rng.randint(45, 82),
        }
        for offset in range(days)
    ]


def fetch_weather(city: str, days: int = 7, date_range: str = "next week") -> list[dict]:
    geocode_response = SESSION.get(
        "https://geocoding-api.open-meteo.com/v1/search",
        params={"name": city, "count": 1, "language": "en", "format": "json"},
        timeout=REQUEST_TIMEOUT,
    )
    geocode_response.raise_for_status()
    locations = geocode_response.json().get("results") or []
    if not locations:
        raise ValueError(f"No geocoding result found for {city}")

    location = locations[0]
    forecast_response = SESSION.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": location["latitude"],
            "longitude": location["longitude"],
            "forecast_days": days,
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "relative_humidity_2m_mean",
                    "precipitation_probability_max",
                ]
            ),
            "timezone": "auto",
        },
        timeout=REQUEST_TIMEOUT,
    )
    forecast_response.raise_for_status()
    daily = forecast_response.json()["daily"]

    forecast = []
    for index, day in enumerate(daily["time"]):
        max_temp = daily["temperature_2m_max"][index]
        min_temp = daily["temperature_2m_min"][index]
        code = daily["weather_code"][index]
        forecast.append(
            {
                "date": day,
                "temperature_c": round((max_temp + min_temp) / 2, 1),
                "condition": WEATHER_CODES.get(code, f"Weather code {code}"),
                "humidity": int(daily["relative_humidity_2m_mean"][index]),
                "precipitation_probability": daily["precipitation_probability_max"][index],
            }
        )
    return forecast


def weather(city: str, days: int = 7, date_range: str = "next week") -> list[dict]:
    try:
        return fetch_weather(city, days=days, date_range=date_range)
    except Exception:
        return _fallback_weather(city, days=days, date_range=date_range)


def _tavily_search(city: str) -> dict | None:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None

    response = SESSION.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": f"{city} city travel guide attractions culture official tourism -airport -station",
            "search_depth": "basic",
            "include_images": True,
            "max_results": 5,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    data = response.json()
    return {
        "city": city,
        "matched_city": city,
        "source": "tavily",
        "source_url": data.get("results", [{}])[0].get("url", ""),
        "sources": [
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "snippet": item.get("content", ""),
            }
            for item in data.get("results", [])
        ],
        "image_urls": data.get("images", [])[:5],
    }


def _wikipedia_search(city: str) -> dict:
    headers = {"User-Agent": "assignment-travel-agent/1.0"}

    title = city.strip().title()
    summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
    summary_response = SESSION.get(summary_url, headers=headers, timeout=REQUEST_TIMEOUT)
    if summary_response.ok and _is_city_summary(city, summary_response.json()):
        summary = summary_response.json()
    else:
        search_url = f"https://en.wikipedia.org/w/rest.php/v1/search/page?q={quote(city + ' city')}&limit=8"
        search_response = SESSION.get(search_url, headers=headers, timeout=REQUEST_TIMEOUT)
        search_response.raise_for_status()
        pages = search_response.json().get("pages", [])
        page = _best_city_page(city, pages)
        title = page.get("title", city)
        summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(title)}"
        summary_response = SESSION.get(summary_url, headers=headers, timeout=REQUEST_TIMEOUT)
        summary_response.raise_for_status()
        summary = summary_response.json()

    image_urls = []
    for key in ["originalimage", "thumbnail"]:
        image = summary.get(key) or {}
        if image.get("source") and image["source"] not in image_urls:
            image_urls.append(image["source"])

    return {
        "city": city,
        "matched_city": summary.get("title", title),
        "source": "wikipedia",
        "source_url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
        "sources": [
            {
                "title": title,
                "url": summary.get("content_urls", {}).get("desktop", {}).get("page", ""),
                "snippet": summary.get("extract", ""),
            }
        ],
        "image_urls": image_urls or GENERIC_IMAGES,
    }


def _best_city_page(city: str, pages: list[dict]) -> dict:
    banned = ("airport", "station", "university", "football", "club", "metro", "subway", "company", "surname")
    normalized_city = city.strip().lower()

    def score(page: dict) -> tuple[int, str]:
        title = page.get("title", "").lower()
        excerpt = page.get("excerpt", "").lower()
        text = f"{title} {excerpt}"
        value = 0
        if "may refer to" in text or "disambiguation" in text:
            value -= 120
        if title == normalized_city:
            value += 100
        if normalized_city in title:
            value += 30
        if any(word in text for word in ["city", "municipality", "town", "village", "capital", "located in", "district"]):
            value += 20
        if any(word in text for word in banned):
            value -= 80
        return value, title

    if not pages:
        return {"title": city, "excerpt": ""}
    return max(pages, key=score)


def _is_city_summary(city: str, summary: dict) -> bool:
    title = summary.get("title", "").lower()
    extract = summary.get("extract", "").lower()
    text = f"{title} {extract}"
    banned = ("airport", "station", "university", "football club", "metro")
    if "may refer to" in text[:300] or summary.get("type") == "disambiguation":
        return False
    return city.strip().lower() in title and not any(word in text[:400] for word in banned)


@tool
def web_search(city: str) -> dict:
    """Search the web for city travel information and image URLs."""
    try:
        tavily = _tavily_search(city)
        if tavily:
            return tavily
        return _wikipedia_search(city)
    except Exception as exc:
        return {
            "city": city,
            "matched_city": "",
            "source": "web_error",
            "source_url": "",
            "sources": [
                {
                    "title": f"Web search failed for {city}",
                    "url": "",
                    "snippet": f"Live web search failed: {exc}",
                }
            ],
            "image_urls": GENERIC_IMAGES,
        }
