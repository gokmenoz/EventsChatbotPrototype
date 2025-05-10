import os
import json
import requests
import time
import botocore.exceptions
import boto3
import random
from typing import List, Dict
from datetime import datetime, timedelta
from dotenv import load_dotenv

# --- External API tokens ---
load_dotenv()  # Load .env variables into environment

EVENTBRITE_TOKEN = os.getenv("EVENTBRITE_API_KEY")
TICKETMASTER_API_KEY = os.getenv("TICKETMASTER_API_KEY")


def retrieve(query, chunks, embedder, index, top_k=5):
    query_vec = embedder.encode([query])
    scores, indices = index.search(query_vec, top_k)

    docs = []
    for i in indices[0]:
        if i < len(chunks):
            c = chunks[i]
            if isinstance(c, dict):
                title = c.get("title", "Untitled")
                start = c.get("start", "Unknown time")
                location = c.get("location", "Unknown location")
                url = c.get("url", "")
                desc = c.get("description", "").strip()

                formatted = f"{title} — {start} @ {location}\n{desc or '(no description)'}\n{url}"
                docs.append(formatted)
    return docs


def build_rag_prompt(context, question):
    return f"You are an event assistant. Use this info to answer:\n\n{context}\n\nQuestion: {question}"


session = boto3.Session(profile_name="ogokmen_bedrock")
bedrock = session.client("bedrock-runtime", region_name="us-east-1")
model_id = "anthropic.claude-3-sonnet-20240229-v1:0"

def call_claude_stream(prompt=None, messages_override=None, retries=5, base_delay=2):
    messages = messages_override or [{"role": "user", "content": prompt}]

    body = {
        "anthropic_version": "bedrock-2023-05-31",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.7,
        "top_p": 0.9,
    }

    for attempt in range(retries):
        try:
            response = bedrock.invoke_model_with_response_stream(
                modelId=model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(body),
            )

            def stream_generator():
                for event in response["body"]:
                    if "chunk" in event:
                        chunk_data = json.loads(event["chunk"]["bytes"])
                        if chunk_data.get("type") == "content_block_delta":
                            delta = chunk_data.get("delta", {})
                            text = delta.get("text", "")
                            if text:
                                yield text

            return stream_generator()

        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] == "ThrottlingException":
                wait = base_delay * (2**attempt) + random.uniform(0, 1)
                print(f"⏳ Throttled. Retrying in {wait:.2f}s...")
                time.sleep(wait)
            else:
                raise
        except Exception as e:
            print(f"⚠️ Unexpected error: {e}")
            time.sleep(1)

    raise RuntimeError("❌ Claude streaming call failed after retries")

def fetch_eventbrite_events(location: str, start: str, end: str) -> List[Dict]:
    if not EVENTBRITE_TOKEN:
        print("⚠️ Missing EVENTBRITE_TOKEN")
        return []

    url = "https://www.eventbriteapi.com/v3/events/search/"
    headers = {
        "Authorization": f"Bearer {EVENTBRITE_TOKEN}"
    }

    params = {
        "location.address": location,
        "start_date.range_start": start,
        "start_date.range_end": end,
        "expand": "venue",
        "sort_by": "date",
        "page": 1
    }

    try:
        response = requests.get(url, headers=headers, params=params, timeout=10)
        data = response.json()
        events = []

        for ev in data.get("events", []):
            events.append({
                "source": "eventbrite",
                "title": ev.get("name", {}).get("text"),
                "description": ev.get("description", {}).get("text", ""),
                "start": ev.get("start", {}).get("local"),
                "end": ev.get("end", {}).get("local"),
                "location": ev.get("venue", {}).get("address", {}).get("localized_address_display", location),
                "url": ev.get("url")
            })

        return events

    except Exception as e:
        print(f"❌ Eventbrite error: {e}")
        return []


def fetch_ticketmaster_events(location: str, start: str, end: str) -> List[Dict]:
    if not TICKETMASTER_API_KEY:
        print("⚠️ Missing TICKETMASTER_API_KEY")
        return []

    url = "https://app.ticketmaster.com/discovery/v2/events.json"
    params = {
        "apikey": TICKETMASTER_API_KEY,
        "locale": "*",
        "city": location,
        "startDateTime": start,
        "endDateTime": end,
        "size": 50,
        "sort": "date,asc"
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        events = []

        for ev in data.get("_embedded", {}).get("events", []):
            events.append({
                "source": "ticketmaster",
                "title": ev.get("name"),
                "description": ev.get("info", "") or ev.get("pleaseNote", ""),
                "start": ev.get("dates", {}).get("start", {}).get("dateTime"),
                "location": ev.get("_embedded", {}).get("venues", [{}])[0].get("city", {}).get("name", location),
                "url": ev.get("url")
            })

        return events

    except Exception as e:
        print(f"❌ Ticketmaster error: {e}")
        return []


def fetch_events(location: str, start: str, end: str) -> List[Dict]:
    events = []
    events += fetch_eventbrite_events(location, start, end)
    events += fetch_ticketmaster_events(location, start, end)
    return events


def parse_date_range(query: str) -> tuple[str, str]:
    """Converts natural phrases into ISO 8601 (UTC) date strings for API filtering."""
    now = datetime.utcnow()

    if "tomorrow" in query.lower():
        start = now + timedelta(days=1)
        end = start + timedelta(days=1)
    elif "weekend" in query.lower():
        # Assume Friday 6PM to Sunday midnight
        weekday = now.weekday()
        days_until_friday = (4 - weekday) % 7
        friday_evening = now + timedelta(days=days_until_friday, hours=(18 - now.hour))
        sunday_night = friday_evening + timedelta(days=2, hours=6)
        start = friday_evening
        end = sunday_night
    elif "next 7 days" in query.lower():
        start = now
        end = now + timedelta(days=7)
    elif "today" in query.lower():
        start = now
        end = now.replace(hour=23, minute=59, second=59)
    else:
        # Default: next 3 days
        start = now
        end = now + timedelta(days=3)

    return start.isoformat(timespec="seconds") + "Z", end.isoformat(timespec="seconds") + "Z"

import re

def extract_city(query: str) -> str:
    """
    Naively extract city name from a query using regex.
    You can replace this with NLP if needed later.
    """
    match = re.search(r"\b(in|at|around|near|for)\s+([A-Z][a-z]+(?:\s[A-Z][a-z]+)*)", query)
    if match:
        return match.group(2)
    return "Berlin"  # Default fallback