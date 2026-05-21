from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class WeatherPoint(BaseModel):
    date: str
    temperature_c: float
    condition: str
    humidity: int
    precipitation_probability: int | None = None


class TravelResponse(BaseModel):
    city_summary: str = Field(..., description="Short travel summary for the city")
    weather_forecast: list[WeatherPoint]
    image_urls: list[str]


class TravelState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    query: str
    city: str
    date_range: str
    route: Literal["internal", "web"]
    refresh_weather_only: bool
    city_summary: str
    weather_forecast: list[dict[str, Any]]
    image_urls: list[str]
    final_response: dict[str, Any]
    errors: Annotated[list[str], operator.add]
