from __future__ import annotations

import json
import os
import re
from datetime import date
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .data import LocalCityStore, known_city_names, normalize_city
from .models import TravelResponse, TravelState
from .tools import weather, web_search


STORE = LocalCityStore.build()
TOOLS = {"web_search": web_search}


def _has_groq_key() -> bool:
    return bool(os.getenv("GROQ_API_KEY"))

REQUEST_REFINER_PROMPT = (
    "Rewrite the user request into a concise travel-assistant query. "
    "Preserve the exact destination city/place name and any date range. "
    "Do not replace a city with an airport, railway station, hotel, sports club, or nearby attraction. "
    "Return plain text only."
)

CITY_EXTRACTION_PROMPT = (
    "You extract travel intent from the user request. "
    "Identify the primary destination as a city, town, or municipality. "
    "If the user explicitly asks for a non-city place (airport, station, hotel, stadium, landmark), use that place name. "
    "If multiple destinations are mentioned, choose the main one the user wants help with. "
    "If no destination is present, infer the most likely city from context rather than returning an empty value. "
    "Also extract any date range or time window the user mentions (including relative phrases). "
    "Return compact JSON only with keys city and date_range."
)

SEARCH_TOOL_PROMPT = (
    "Call the web_search tool for the requested destination city. "
    "The tool argument must be only the city/municipality name, not an airport or transit hub. "
    "Do not answer directly."
)

SUMMARY_PROMPT = (
    "Write a concise travel summary from the provided web-search JSON. "
    "Use only results about the requested destination city or place named by the user. "
    "Do not switch to another destination, airport, railway station, company, or nearby attraction. "
    "Ignore airport, railway station, sports club, university, and unrelated landmark results unless the user asked for them. "
    "Mention practical attractions, neighborhoods, culture, and food. Keep it useful for a traveler."
)

ITINERARY_PROMPT = (
    "Using the provided city summary and date range, write a short summary and a simple day-by-day plan. "
    "The itinerary must be for the exact destination named by the City field. "
    "Use only places within that destination; do not include other cities, airports, railway stations, or day trips. "
    "If the summary is not about the same destination as the City field, say the destination data is mismatched instead of inventing. "
    "For each day, list 2-4 places to visit and include an estimated time spent per place (e.g., 1.5h, 2h). "
    "If no date range is given, assume a 2-day visit. "
    "Keep it realistic and concise. Return plain text only."
)


def get_llm(temperature: float = 0.1) -> ChatGroq:
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=temperature,
        api_key=os.getenv("GROQ_API_KEY"),
    )


def _fallback_city(query: str) -> str:
    cleaned = re.sub(
        r"(?i)\b("
        r"tell me about|what about|next week|this week|today|tomorrow|weather|forecast|"
        r"city|place|places|itinerary|itenary|travel|trip|guide|plan|make|create|give me|"
        r"for|in|to|of|about|visit|visiting|days?|day|a|an|the"
        r")\b",
        " ",
        query,
    )
    cleaned = re.sub(r"\b\d+\b", " ", cleaned)
    cleaned = re.sub(r"[^a-zA-Z\s]", " ", cleaned).strip()
    return re.sub(r"\s+", " ", cleaned).title() or "Tokyo"


def _sanitize_destination(value: str, original_query: str = "") -> str:
    destination = re.sub(
        r"(?i)\b("
        r"itinerary|itenary|travel guide|travel|trip|plan|places to visit|places|"
        r"weather|forecast|today|tomorrow|next week|this week|\d+\s*days?|\d+\s*day|a|an|the"
        r")\b",
        " ",
        value,
    )
    destination = re.sub(r"[^a-zA-Z\s,'.-]", " ", destination)
    destination = re.sub(r"\s+", " ", destination).strip(" ,.-")
    if destination:
        return destination.title()
    return _fallback_city(original_query)


def _unique_images(urls: list[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for url in urls:
        normalized = url.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def refine_query(state: TravelState) -> TravelState:
    query = state.get("query", "").strip()
    if not query:
        return {"query": query}

    if not _has_groq_key():
        cleaned = re.sub(r"\s+", " ", query)
        return {"query": cleaned}

    try:
        response = get_llm(temperature=0).invoke(
            [
                SystemMessage(content=REQUEST_REFINER_PROMPT),
                HumanMessage(content=query),
            ]
        )
        refined = response.content.strip() or query
    except Exception:
        refined = query

    return {"query": refined}


def parse_request(state: TravelState) -> TravelState:
    query = state["query"]
    previous_city = state.get("city", "")
    lower_query = query.lower()
    refresh_weather_only = bool(
        previous_city and any(token in lower_query for token in ["next week", "weather", "forecast"])
    )

    if refresh_weather_only and not any(city in lower_query for city in known_city_names()):
        return {"city": previous_city, "date_range": query, "refresh_weather_only": True}

    try:
        llm = get_llm(temperature=0)
        today = date.today().isoformat()
        response = llm.invoke(
            [
                SystemMessage(
                    content=(
                        f"{CITY_EXTRACTION_PROMPT} Today is {today}. "
                        "Use today's date only to resolve relative weather/date phrases like today, tomorrow, this weekend, or next week."
                    )
                ),
                HumanMessage(content=query),
            ]
        )
        parsed = json.loads(response.content)
        city = _sanitize_destination(str(parsed.get("city") or _fallback_city(query)).strip(), query)
        date_range = str(parsed.get("date_range") or "next 7 days").strip()
    except Exception:
        city = _sanitize_destination(_fallback_city(query), query)
        date_range = "next 7 days"

    return {"city": city, "date_range": date_range, "refresh_weather_only": False}


def route_knowledge(state: TravelState) -> TravelState:
    record = STORE.get(state["city"])
    same_city = bool(record and normalize_city(record.get("city", "")) == normalize_city(state["city"]))
    route: Literal["internal", "web"] = (
        "internal" if record and same_city and not _looks_like_wrong_place(state["city"], record) else "web"
    )
    return {"route": route}


def route_after_knowledge(state: TravelState) -> str:
    if state.get("refresh_weather_only"):
        return "weather_only"
    return state["route"]


def retrieve_internal_summary(state: TravelState) -> TravelState:
    record = STORE.get(state["city"])
    if not record:
        return {"city_summary": "", "image_urls": []}
    if normalize_city(record.get("city", "")) != normalize_city(state["city"]):
        return {"city_summary": "", "image_urls": []}
    return {"city_summary": record["summary"], "image_urls": record["image_urls"]}


def plan_search_tool(state: TravelState) -> TravelState:
    if not _has_groq_key():
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "web_search",
                            "args": {"city": _sanitize_destination(state["city"])},
                            "id": "fallback-search-call",
                        }
                    ],
                )
            ]
        }

    llm_with_tools = get_llm(temperature=0).bind_tools([web_search])
    response = llm_with_tools.invoke(
        [
            SystemMessage(content=SEARCH_TOOL_PROMPT),
            HumanMessage(content=f"Find travel information for {_sanitize_destination(state['city'])}."),
        ]
    )
    return {"messages": [response]}


def execute_search_tools(state: TravelState) -> TravelState:
    ai_messages = [message for message in state.get("messages", []) if isinstance(message, AIMessage)]
    last_ai = ai_messages[-1] if ai_messages else None
    tool_messages: list[ToolMessage] = []

    for call in getattr(last_ai, "tool_calls", []) or []:
        tool_name = call.get("name")
        if not tool_name or tool_name not in TOOLS:
            continue
        tool_args = call.get("args", {})
        result = TOOLS[tool_name].invoke(tool_args)
        tool_messages.append(
            ToolMessage(
                content=json.dumps(result),
                name=tool_name,
                tool_call_id=call.get("id", tool_name),
            )
        )
    return {"messages": tool_messages}


def summarize_search_results(state: TravelState) -> TravelState:
    tool_messages = [message for message in state.get("messages", []) if isinstance(message, ToolMessage)]
    raw = tool_messages[-1].content if tool_messages else "{}"
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        data = {}
    snippets = " ".join(source.get("snippet", "") for source in data.get("sources", []))
    source_url = data.get("source_url", "")
    image_urls = _unique_images(data.get("image_urls", []))
    matched_city = data.get("matched_city") or data.get("city", "")
    if not _destination_matches(state["city"], matched_city, snippets, source_url):
        summary = (
            f"I could not verify that the live search result is about {state['city']}. "
            "Please try a more specific destination name."
        )
        return {"city_summary": summary, "image_urls": image_urls}

    if not _has_groq_key():
        summary = snippets or f"Live search did not return a detailed summary for {state['city']}."
    else:
        response = get_llm(temperature=0.2).invoke(
            [
                SystemMessage(
                    content=SUMMARY_PROMPT
                ),
                HumanMessage(content=f"Requested destination: {state['city']}\nSearch JSON: {raw}"),
            ]
        )
        summary = response.content

    record = {
        "city": state["city"].title(),
        "summary": summary,
        "image_urls": image_urls,
        "source": data.get("source", "web"),
        "source_url": source_url,
    }
    STORE.upsert(record)
    return {"city_summary": summary, "image_urls": image_urls}


def enrich_itinerary(state: TravelState) -> TravelState:
    summary = state.get("city_summary", "").strip()
    if not summary:
        return {"city_summary": summary}

    date_range = state.get("date_range", "")
    city = state.get("city", "")
    if not _has_groq_key():
        fallback = (
            f"{summary}\n\n"
            "Suggested day plan:\n"
            "Day 1: City highlights (2-3h), local market or museum (2h), food district (2h).\n"
            "Day 2: Parks or neighborhoods (2-3h), cultural site (1.5-2h), evening viewpoint (1-1.5h)."
        )
        return {"city_summary": fallback}

    try:
        response = get_llm(temperature=0.2).invoke(
            [
                SystemMessage(content=ITINERARY_PROMPT),
                HumanMessage(
                    content=(
                        f"City: {city}\n"
                        f"Date range: {date_range}\n"
                        f"Summary: {summary}"
                    )
                ),
            ]
        )
        enriched = response.content.strip() or summary
    except Exception:
        enriched = summary

    return {"city_summary": enriched}


def fetch_weather(state: TravelState) -> TravelState:
    try:
        return {"weather_forecast": weather(state["city"], date_range=state.get("date_range", "next 7 days"))}
    except Exception as exc:
        return {"errors": [f"Weather lookup failed: {exc}"], "weather_forecast": []}


def fetch_images(state: TravelState) -> TravelState:
    if state.get("image_urls"):
        return {"image_urls": _unique_images(state["image_urls"])}
    record = STORE.get(state["city"])
    if record and record.get("image_urls"):
        return {"image_urls": _unique_images(record["image_urls"])}
    return {"image_urls": []}


def compose_response(state: TravelState) -> TravelState:
    response = TravelResponse(
        city_summary=state.get("city_summary", f"Updated forecast for {state.get('city', 'this destination')}."),
        weather_forecast=state.get("weather_forecast", []),
        image_urls=state.get("image_urls", []),
    )
    return {"final_response": response.model_dump()}


def _looks_like_wrong_place(city: str, record: dict) -> bool:
    summary = str(record.get("summary", "")).lower()
    source_url = str(record.get("source_url", "")).lower()
    city_name = city.strip().lower()
    first_sentence = summary.split(".", 1)[0]
    wrong_entity_words = ("airport", "railway station", "train station", "football club", "university", "metro station")
    if "may refer to" in summary[:300] or "disambiguation" in source_url:
        return True
    if city_name not in summary[:500] and city_name.replace(" ", "_") not in source_url:
        return True
    return any(word in first_sentence or word in source_url for word in wrong_entity_words)


def _destination_matches(city: str, matched_city: str, snippets: str, source_url: str) -> bool:
    city_norm = normalize_city(city)
    matched_norm = normalize_city(matched_city)
    source_norm = normalize_city(source_url.replace("_", " "))
    snippet_norm = normalize_city(snippets[:600])
    city_tokens = [token for token in city_norm.split() if len(token) > 2]
    if not city_tokens:
        return False
    if city_norm and (city_norm in matched_norm or matched_norm in city_norm):
        return True
    if all(token in matched_norm for token in city_tokens):
        return True
    if all(token in source_norm for token in city_tokens):
        return True
    return all(token in snippet_norm for token in city_tokens)


def build_graph():
    graph = StateGraph(TravelState)

    graph.add_node("refine_query", refine_query)
    graph.add_node("parse_request", parse_request)
    graph.add_node("route_knowledge", route_knowledge)
    graph.add_node("retrieve_internal_summary", retrieve_internal_summary)
    graph.add_node("plan_search_tool", plan_search_tool)
    graph.add_node("execute_search_tools", execute_search_tools)
    graph.add_node("summarize_search_results", summarize_search_results)
    graph.add_node("enrich_itinerary", enrich_itinerary)
    graph.add_node("fetch_weather", fetch_weather)
    graph.add_node("fetch_images", fetch_images)
    graph.add_node("compose_response", compose_response)

    graph.set_entry_point("refine_query")
    graph.add_edge("refine_query", "parse_request")
    graph.add_edge("parse_request", "route_knowledge")
    graph.add_conditional_edges(
        "route_knowledge",
        route_after_knowledge,
        {
            "internal": "retrieve_internal_summary",
            "web": "plan_search_tool",
            "weather_only": "fetch_weather",
        },
    )
    graph.add_edge("retrieve_internal_summary", "enrich_itinerary")
    graph.add_edge("plan_search_tool", "execute_search_tools")
    graph.add_edge("execute_search_tools", "summarize_search_results")
    graph.add_edge("summarize_search_results", "enrich_itinerary")
    graph.add_edge("enrich_itinerary", "fetch_weather")
    graph.add_edge("enrich_itinerary", "fetch_images")
    graph.add_edge("fetch_weather", "compose_response")
    graph.add_edge("fetch_images", "compose_response")
    graph.add_edge("compose_response", END)

    return graph.compile(checkpointer=MemorySaver())


travel_graph = build_graph()
