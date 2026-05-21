from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


INITIAL_CITY_RECORDS: dict[str, dict[str, Any]] = {
    "paris": {
        "city": "Paris",
        "summary": (
            "Paris is France's capital, known for the Louvre, Eiffel Tower, Seine river walks, "
            "fashion houses, patisseries, historic neighborhoods like Le Marais, and day trips "
            "to Versailles. It suits art lovers, food travelers, and first-time Europe visitors."
        ),
        "image_urls": [
            "https://images.unsplash.com/photo-1502602898657-3e91760cbb34",
            "https://images.unsplash.com/photo-1499856871958-5b9627545d1a",
            "https://images.unsplash.com/photo-1522093007474-d86e9bf7ba6f",
        ],
        "source": "seed",
    },
    "tokyo": {
        "city": "Tokyo",
        "summary": (
            "Tokyo blends dense modern districts with quiet temples, ramen counters, design shops, "
            "gardens, and efficient trains. Highlights include Shibuya, Asakusa, Ueno, Shinjuku, "
            "teamLab, Tsukiji outer market, and day trips toward Mount Fuji or Kamakura."
        ),
        "image_urls": [
            "https://images.unsplash.com/photo-1540959733332-eab4deabeeaf",
            "https://images.unsplash.com/photo-1536098561742-ca998e48cbcc",
            "https://images.unsplash.com/photo-1513407030348-c983a97b98d8",
        ],
        "source": "seed",
    },
    "new york": {
        "city": "New York",
        "summary": (
            "New York City is a high-energy destination for museums, Broadway, food, architecture, "
            "parks, and neighborhood exploration. Visitors often combine Manhattan landmarks with "
            "Brooklyn food scenes, Central Park, ferry rides, and skyline viewpoints."
        ),
        "image_urls": [
            "https://images.unsplash.com/photo-1522083165195-3424ed129620",
            "https://images.unsplash.com/photo-1496588152823-86ff7695e68f",
            "https://images.unsplash.com/photo-1485871981521-5b1fd3805eee",
        ],
        "source": "seed",
    },
}


def normalize_city(city: str) -> str:
    return re.sub(r"\s+", " ", city.strip().lower())


def known_city_names() -> list[str]:
    return list(INITIAL_CITY_RECORDS)


def _fallback_embed(text: str) -> list[float]:
    words = re.findall(r"[a-z]+", text.lower())
    buckets = [0.0] * 64
    for word in words:
        buckets[sum(ord(ch) for ch in word) % len(buckets)] += 1.0
    norm = math.sqrt(sum(value * value for value in buckets)) or 1.0
    return [value / norm for value in buckets]


class GroqEmbeddingFunction:
    """Chroma embedding adapter that calls Groq's OpenAI-compatible embeddings API."""

    def name(self) -> str:
        return os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")

    def __call__(self, input: list[str]) -> list[list[float]]:
        api_key = os.getenv("GROQ_API_KEY")
        model = os.getenv("GROQ_EMBEDDING_MODEL", "nomic-embed-text-v1.5")
        if not api_key:
            return [_fallback_embed(text) for text in input]

        response = requests.post(
            "https://api.groq.com/openai/v1/embeddings",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": model, "input": input},
            timeout=20,
        )
        if not response.ok:
            return [_fallback_embed(text) for text in input]
        payload = response.json()
        return [item["embedding"] for item in payload["data"]]


@dataclass
class LocalCityStore:
    collection: object | None = None
    memory_records: dict[str, dict[str, Any]] | None = None

    @classmethod
    def build(cls) -> "LocalCityStore":
        try:
            import chromadb

            client = chromadb.PersistentClient(path=".chroma")
            collection_name = "city_records_groq" if os.getenv("GROQ_API_KEY") else "city_records_local"
            collection = client.get_or_create_collection(
                name=collection_name,
                embedding_function=GroqEmbeddingFunction(),
                metadata={"description": "Travel city records with summaries and images"},
            )
            store = cls(collection=collection, memory_records={})
            for record in INITIAL_CITY_RECORDS.values():
                store.upsert(record)
            return store
        except Exception:
            memory_records = {
                normalize_city(record["city"]): record for record in INITIAL_CITY_RECORDS.values()
            }
            return cls(collection=None, memory_records=memory_records)

    def has_city(self, city: str) -> bool:
        return self.get(city) is not None

    def get(self, city: str) -> dict[str, Any] | None:
        city_id = normalize_city(city)
        if self.collection is None:
            return (self.memory_records or {}).get(city_id)

        result = self.collection.get(ids=[city_id], include=["documents", "metadatas"])
        if not result.get("ids"):
            return None
        metadata = (result.get("metadatas") or [{}])[0] or {}
        documents = result.get("documents") or [""]
        return {
            "city": metadata.get("city", city.title()),
            "summary": documents[0],
            "image_urls": json.loads(metadata.get("image_urls", "[]")),
            "source": metadata.get("source", "vector_db"),
            "source_url": metadata.get("source_url", ""),
        }

    def upsert(self, record: dict[str, Any]) -> None:
        city = str(record["city"]).strip()
        city_id = normalize_city(city)
        clean_record = {
            "city": city,
            "summary": str(record.get("summary", "")).strip(),
            "image_urls": list(record.get("image_urls") or []),
            "source": str(record.get("source", "web")),
            "source_url": str(record.get("source_url", "")),
        }

        if self.collection is None:
            self.memory_records = self.memory_records or {}
            self.memory_records[city_id] = clean_record
            return

        self.collection.upsert(
            ids=[city_id],
            documents=[clean_record["summary"]],
            metadatas=[
                {
                    "city": clean_record["city"],
                    "image_urls": json.dumps(clean_record["image_urls"]),
                    "source": clean_record["source"],
                    "source_url": clean_record["source_url"],
                }
            ],
        )
