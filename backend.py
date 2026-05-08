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

    # Strategy 2 — markdown fence (Rewritten to be copy-paste safe)
    fence = re.search(r"`{3}(?:json)?\s*(\[[\s\S]*?\])\s*`{3}", text, re.IGNORECASE)
    if fence:
        try:
            parsed = json.loads(fence.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3 — first bare [...] block
    bracket = re.search(r"\[[\s\S]*\]", text)
    if bracket:
        try:
            parsed = json.loads(bracket.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract a valid JSON array from the model response.\n"
        f"Preview: {text[:400]}"
    )

# ─── PYDANTIC VALIDATION ────────────────────────────────────────────────────────

def validate_items(raw_list: list) -> list:
    validated, skipped = [], 0
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            validated.append(NewsItem(**item).model_dump())
        except Exception as e:
            print(f"  [!] Item {idx} skipped: {e}")
            skipped += 1
    if skipped:
        print(f"  [~] {skipped} item(s) skipped.")
    return validated

# ─── GEMINI CALL (WITH RETRY) ───────────────────────────────────────────────────

SYSTEM_INSTRUCTION = """You are a professional financial news editor.
You will receive raw RSS headlines and summaries grouped by domain.
Your ONLY job is to select the most newsworthy items and output a SINGLE valid JSON array.

STRICT RULES:
1. Output ONLY the JSON array — no markdown, no explanation, no prose outside the array.
2. Every element MUST contain exactly these five keys:
   "category"  — one of: "Global Macro / NSE", "Business / M&A", "AI Current Affairs", "AI in Finance"
   "title"     — clean, concise headline (string)
   "summary"   — 2–3 sentence factual summary using ONLY information present in the provided text (string)
   "link"      — the original article URL from the input, or "#" if unavailable (string)
   "date"      — ISO date string, e.g. "2025-05-09" (string)
3. The FIRST item in "Global Macro / NSE" MUST be a market-indices briefing covering:
   NIFTY 50, SENSEX, S&P 500, and NASDAQ — use data from the provided headlines only.
4. Do NOT invent facts, URLs, or data not present in the provided text.
5. Aim for 5–8 items per category, 20–32 items total.
"""

def call_gemini(client: genai.Client, prompt: str) -> list:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"[~] Gemini call attempt {attempt}/{MAX_RETRIES}…")
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.3,
                    max_output_tokens=8192,
                ),
            )
            raw_text = response.text
            print(f"[✓] Response received ({len(raw_text):,} chars)")
            return extract_json_array(raw_text)

        except Exception as e:
            print(f"  [!] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                print(f"  [~] Retrying in {RETRY_DELAY}s…")
                time.sleep(RETRY_DELAY)
            else:
                raise RuntimeError(f"All {MAX_RETRIES} Gemini attempts failed. Last error: {e}")

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 58)
    print("  NewsGrid Backend — Gemini 2.5 Flash + Feedparser")
    print("=" * 58)

    # 1. API key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: GEMINI_API_KEY not set.")
        sys.exit(1)
    print("[✓] GEMINI_API_KEY loaded")

    # 2. RSS scout
    print("\n[~] Fetching RSS feeds…")
    raw_rss = fetch_rss()
    if not raw_rss.strip():
        print("\nERROR: RSS fetch returned no content.")
        sys.exit(1)

    # 3. Build prompt
    today   = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt  = (
        f"Today's date: {today}\n\n"
        "Below is raw RSS data grouped by domain. "
        "Select the most important stories and return the JSON array.\n\n"
        + raw_rss
    )
    print(f"\n[✓] Prompt assembled ({len(prompt):,} chars)")

    # 4. Call Gemini
    print()
    client = genai.Client(api_key=api_key)
    try:
        raw_list = call_gemini(client, prompt)
        print(f"[✓] Extracted {len(raw_list)} raw items")
    except RuntimeError as e:
        print(f"\nERROR: {e}")
        Path("debug_response.txt").write_text(str(e), encoding="utf-8")
        sys.exit(1)

    # 5. Validate
    print("[~] Validating schema…")
    validated = validate_items(raw_list)
    if not validated:
        print("\nERROR: No valid items after validation.")
        sys.exit(1)

    # 6. Save
    Path(OUTPUT_FILE).write_text(
        json.dumps(validated, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    cats = len(set(i["category"] for i in validated))
    print(f"[✓] Saved {len(validated)} items across {cats} categories → {OUTPUT_FILE}")
    print("\n[✓] Pipeline complete.")
    print("=" * 58)


if __name__ == "__main__":
    main()
