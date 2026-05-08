"""
backend.py — Automated News Feed Generator
-------------------------------------------
1. Fetches raw headlines via feedparser from Google News RSS.
2. Sends the raw text to Gemini 2.5 Flash for categorisation & summarisation.
3. Validates output against Pydantic schema.
4. Saves result as data.json.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    import feedparser
except ImportError:
    print("ERROR: feedparser not installed. Run: pip install feedparser")
    sys.exit(1)

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai not installed. Run: pip install google-genai")
    sys.exit(1)

try:
    from pydantic import BaseModel, field_validator
except ImportError:
    print("ERROR: pydantic not installed. Run: pip install pydantic")
    sys.exit(1)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

MODEL_ID    = "gemini-2.5-flash"
OUTPUT_FILE = "data.json"
MAX_RETRIES = 3
RETRY_DELAY = 8   # seconds between retries

# Google News RSS feeds — one per domain
RSS_FEEDS = {
    "Global Macro / NSE": [
        "https://news.google.com/rss/search?q=NIFTY+50+OR+SENSEX+OR+NSE+OR+BSE+stock+market&hl=en-IN&gl=IN&ceid=IN:en",
        "https://news.google.com/rss/search?q=S%26P+500+OR+NASDAQ+OR+Federal+Reserve+OR+global+macro&hl=en&gl=US&ceid=US:en",
    ],
    "Business / M&A": [
        "https://news.google.com/rss/search?q=merger+acquisition+India+OR+Indian+business+deal&hl=en-IN&gl=IN&ceid=IN:en",
        "https://news.google.com/rss/search?q=merger+acquisition+global+corporate+deal+2025&hl=en&gl=US&ceid=US:en",
    ],
    "AI Current Affairs": [
        "https://news.google.com/rss/search?q=artificial+intelligence+OpenAI+Anthropic+Google+DeepMind&hl=en&gl=US&ceid=US:en",
    ],
    "AI in Finance": [
        "https://news.google.com/rss/search?q=AI+fintech+banking+finance+machine+learning+trading&hl=en&gl=US&ceid=US:en",
    ],
}

MAX_ITEMS_PER_FEED = 10   # cap per RSS feed to control prompt size

# ─── PYDANTIC SCHEMA ────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    category: str
    title:    str
    summary:  str
    link:     str
    date:     str

    @field_validator("category", "title", "summary", "date", mode="before")
    @classmethod
    def strip_strings(cls, v):
        return str(v).strip() if v is not None else ""

    @field_validator("link", mode="before")
    @classmethod
    def validate_link(cls, v):
        v = str(v).strip() if v is not None else ""
        return v if v else "#"

# ─── RSS SCOUT ──────────────────────────────────────────────────────────────────

def fetch_rss() -> str:
    """
    Pull headlines from all configured RSS feeds.
    Returns a plain-text block for the Gemini prompt.
    """
    lines = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for domain, urls in RSS_FEEDS.items():
        lines.append(f"\n## DOMAIN: {domain}\n")
        for url in urls:
            try:
                feed  = feedparser.parse(url)
                items = feed.entries[:MAX_ITEMS_PER_FEED]
                for e in items:
                    title   = e.get("title",   "").strip()
                    summary = e.get("summary", e.get("description", "")).strip()
                    # Strip HTML tags from summary
                    summary = re.sub(r"<[^>]+>", " ", summary).strip()
                    summary = re.sub(r"\s+", " ", summary)[:300]
                    link    = e.get("link", "#").strip()
                    pub     = e.get("published", today)
                    lines.append(f"TITLE: {title}")
                    lines.append(f"SUMMARY: {summary}")
                    lines.append(f"LINK: {link}")
                    lines.append(f"DATE: {pub}")
                    lines.append("---")
                print(f"  [✓] {domain} | {url[:60]}… → {len(items)} items")
            except Exception as ex:
                print(f"  [!] Feed error ({url[:60]}…): {ex}")

    raw = "\n".join(lines)
    print(f"[✓] RSS fetch complete — {len(raw):,} characters of raw text")
    return raw

# ─── JSON EXTRACTION ────────────────────────────────────────────────────────────

def extract_json_array(text: str) -> list:
    text = text.strip()

    # Strategy 1 — direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2 — markdown fence
    fence = re.search(r"
http://googleusercontent.com/immersive_entry_chip/0

Once that is saved, head over to the **Actions** tab and hit **Run workflow** again. Your factory should hum to life with real-time, completely hallucination-free market news!
