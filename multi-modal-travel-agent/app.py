from __future__ import annotations

import os
import uuid

import pandas as pd
import streamlit as st
from langchain_core.messages import HumanMessage, SystemMessage

from travel_agent.graph import get_llm, travel_graph


st.set_page_config(page_title="Multi-Modal Travel Assistant", page_icon=":airplane:", layout="wide")


def _ensure_session_defaults() -> None:
    defaults = {
        "thread_id": str(uuid.uuid4()),
        "messages": [],
        "conversation_summary": "",
        "pending_query": "",
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


_ensure_session_defaults()


def _summarize_messages(messages: list[dict]) -> str:
    if not messages:
        return ""

    summary_seed = st.session_state.conversation_summary
    transcript = "\n".join(f"{item['role'].title()}: {item['content']}" for item in messages)

    if not os.getenv("GROQ_API_KEY"):
        snippet = transcript[:1200].strip()
        return f"{summary_seed}\n{snippet}".strip()

    system_prompt = (
        "Summarize the conversation so far in 6-8 concise bullet points. "
        "Preserve user preferences, destinations, constraints, and unresolved questions."
    )
    try:
        response = get_llm(temperature=0).invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Previous summary (if any):\n{summary_seed}"),
                HumanMessage(content=f"Conversation:\n{transcript}"),
            ]
        )
        return response.content.strip() or summary_seed
    except Exception:
        return summary_seed


def _maybe_summarize_history() -> None:
    messages = st.session_state.messages
    if len(messages) < 12:
        return
    cutoff = len(messages) - 6
    to_summarize = messages[:cutoff]
    st.session_state.conversation_summary = _summarize_messages(to_summarize)
    st.session_state.messages = messages[cutoff:]


st.title("Multi-Modal Travel Assistant")

with st.sidebar:
    st.header("Chat controls")
    if st.button("Start new conversation", use_container_width=True):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.session_state.conversation_summary = ""
        st.rerun()

    st.caption("Quick prompts")
    prompt_options = [
        "Plan 2 days in Lisbon next week",
        "Best neighborhoods to stay in Seoul in June",
        "What should I do in Mexico City this weekend?",
        "Weather and must-see spots in Edinburgh",
    ]
    for option in prompt_options:
        if st.button(option, use_container_width=True):
            st.session_state.pending_query = option
            st.rerun()

if st.session_state.conversation_summary:
    with st.expander("Conversation summary"):
        st.write(st.session_state.conversation_summary)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

user_input = st.chat_input("Ask about a city, weather, or trip plan")
active_input = (st.session_state.pending_query or user_input or "").strip()
st.session_state.pending_query = ""

if active_input:
    st.session_state.messages.append({"role": "user", "content": active_input})

    with st.chat_message("assistant"):
        with st.spinner("Routing, fetching weather, and building the travel view..."):
            result = travel_graph.invoke(
                {"query": active_input},
                config={"configurable": {"thread_id": st.session_state.thread_id}},
            )
        response = result["final_response"]

        st.subheader("Summary")
        st.write(response["city_summary"])

        forecast = pd.DataFrame(response["weather_forecast"])
        if not forecast.empty:
            st.subheader("Weather Forecast")
            chart_columns = [
                column
                for column in ["temperature_c", "humidity", "precipitation_probability"]
                if column in forecast
            ]
            chart_data = forecast.set_index("date")[chart_columns]
            st.line_chart(chart_data)
            st.dataframe(forecast, hide_index=True, use_container_width=True)

        if response["image_urls"]:
            st.subheader("Gallery")
            columns = st.columns(min(3, len(response["image_urls"])))
            for index, url in enumerate(response["image_urls"]):
                columns[index % len(columns)].image(url, use_container_width=True)

        if result.get("errors"):
            st.warning("\n".join(result["errors"]))

        st.session_state.messages.append({"role": "assistant", "content": response["city_summary"]})

    _maybe_summarize_history()
