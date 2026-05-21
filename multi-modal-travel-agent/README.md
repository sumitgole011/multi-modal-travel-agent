# Multi-Modal Travel Assistant

Assignment implementation for the AI Engineering challenge. It is intentionally small and readable: the focus is the LangGraph architecture, Groq Llama 3.3 usage, structured output, and Streamlit rendering.

## Features

- LangGraph state machine with typed state, nodes, edges, and conditional routing.
- Groq `llama-3.3-70b-versatile` for request parsing, unknown-city search planning, and web-search summarization.
- Local vector-store path for Paris, Tokyo, and New York using ChromaDB.
- City records in the vector DB include the summary, image URLs, and source metadata.
- Live web-search path using Tavily when `TAVILY_API_KEY` is present, otherwise Wikipedia REST APIs.
- New cities found through web search are stored back into the vector DB for future internal retrieval.
- Live weather forecast using Open-Meteo's free geocoding and forecast APIs.
- Manual tool execution node that parses raw `tool_calls` and appends `ToolMessage` objects.
- Parallel fan-out for weather and image fetching after the city summary is available.
- Streamlit UI with text summary, gallery, line chart, and forecast table.
- LangGraph `MemorySaver` checkpointer so follow-up requests can reuse prior city context.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GROQ_API_KEY="your_groq_key"
export GROQ_EMBEDDING_MODEL="nomic-embed-text-v1.5"
# Optional:
export TAVILY_API_KEY="your_tavily_key"
streamlit run app.py
```

Search and weather are live. Tavily is optional because the fallback uses Wikipedia public endpoints. Weather uses Open-Meteo and does not need an API key.

## Architecture

The graph starts by extracting the city and date range. It then checks whether the city exists in the local store.

- Known city: retrieve the city summary and image URLs directly from the local vector store.
- Unknown city: ask Groq Llama 3.3 to call `web_search`, execute the returned tool call manually, summarize the web results, then upsert the new city record into ChromaDB.
- Both paths fan out to weather and image nodes. Weather is fetched from Open-Meteo; image URLs come from the vector DB/search result rather than a separate canned image function.
- The final node emits a Pydantic `TravelResponse` object with `city_summary`, `weather_forecast`, and `image_urls`.

## Embeddings

Chroma uses `GroqEmbeddingFunction`, which calls Groq's OpenAI-compatible embeddings endpoint:

`https://api.groq.com/openai/v1/embeddings`

Set the model with `GROQ_EMBEDDING_MODEL`. If the configured Groq account/model does not support embeddings, the app falls back to a tiny deterministic local embedding so the assignment remains runnable. The LLM itself still uses only Groq Llama 3.3.

See `graph.png` for the topology.

## Notes

This is an assignment build, not a production agent. External data tools are mocked on purpose, error handling is concise, and the code favors clarity over infrastructure.
