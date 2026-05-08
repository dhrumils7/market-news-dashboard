"""
backend.py — Automated News Feed Generator (Self-Healing)
-------------------------------------------
1. Fetches raw headlines via feedparser from Google News RSS.
2. Sends the raw text to Gemini 2.5 Flash for categorisation & summarisation.
3. Validates output against Pydantic schema.
4. Auto-repairs truncated JSON if the API cuts off mid-generation.
5. Saves result as data.json.
"""

import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# --- DEPENDENCY CHECKS ---
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


# --- CONFIGURATION ---
MODEL_ID    = "gemini-2.5-flash"
OUTPUT_FILE = "data.json"
MAX_RETRIES = 3
RETRY_DELAY = 8

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

MAX_ITEMS_PER_FEED = 10


# --- SCHEMA ---
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


# --- RSS FETCHING ---
def fetch_rss() -> str:
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
                    link    = e.get("link", "#").strip()
                    pub     = e.get("published", today)
                    
                    # Basic HTML strip without Regex to be safe
                    if "<" in summary and ">" in summary:
                        summary = summary.split("<")[0] + summary.split(">")[-1]

                    lines.append(f"TITLE: {title}")
                    lines.append(f"SUMMARY: {summary[:200]}")
                    lines.append(f"LINK: {link}")
                    lines.append(f"DATE: {pub}")
                    lines.append("---")
                print(f"  [✓] {domain} fetched {len(items)} items")
            except Exception as ex:
                print(f"  [!] Feed error: {ex}")

    raw = "\n".join(lines)
    return raw


# --- JSON EXTRACTION & REPAIR ---
def extract_json_array(text: str) -> list:
    text = text.strip()
    
    # Strip markdown fences safely without using literal triple backticks
    fence_marker = "`" * 3
    if text.startswith(fence_marker):
        lines = text.split("\n")
        if lines[0].startswith(fence_marker): lines.pop(0)
        if lines and lines[-1].startswith(fence_marker): lines.pop(-1)
        text = "\n".join(lines).strip()
        
    # Strategy 1: Direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: Bracket extraction
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(text[start_idx:end_idx+1])
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass
            
    # Strategy 3: Truncated JSON Repair (The Magic Fix)
    if start_idx != -1:
        last_brace = text.rfind('}')
        if last_brace > start_idx:
            repaired_text = text[start_idx:last_brace+1] + "\n]"
            try:
                parsed = json.loads(repaired_text)
                if isinstance(parsed, list):
                    print("  [~] Notice: Successfully repaired truncated JSON response.")
                    return parsed
            except json.JSONDecodeError:
                pass
                
    raise ValueError(f"Could not locate or repair JSON array. Raw text snippet: {text[:200]}")


# --- GEMINI CALL ---
SYSTEM_INSTRUCTION = """You are a professional financial news editor.
You will receive raw RSS headlines and summaries.
Output a SINGLE valid JSON array. Do not include markdown formatting or extra text.

Every element MUST contain exactly these five keys:
"category"  — one of: "Global Macro / NSE", "Business / M&A", "AI Current Affairs", "AI in Finance"
"title"     — clean, concise headline (string)
"summary"   — 2 factual sentences using ONLY the provided text (string)
"link"      — the original article URL (string)
"date"      — ISO date string (string)

The FIRST item in "Global Macro / NSE" MUST be a market-indices briefing covering NIFTY 50, SENSEX, S&P 500, and NASDAQ.
Aim for 3 to 4 items per category to ensure a complete, valid JSON output.
"""

def main():
    print("=" * 58)
    print("  NewsGrid Backend — Self-Healing (Gemini + RSS)")
    print("=" * 58)

    # 1. Check API Key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)
    print("[✓] GEMINI_API_KEY loaded")

    # 2. Fetch News
    print("\n[~] Fetching live RSS feeds...")
    raw_rss = fetch_rss()
    if len(raw_rss) < 100:
        print("\nERROR: Failed to fetch enough RSS content. Network issue or Google blocked the runner.")
        sys.exit(1)
    print(f"[✓] RSS fetch complete ({len(raw_rss):,} chars)")

    # 3. Assemble Prompt
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = f"Today's date: {today}\n\nSelect the most important stories and return the JSON array:\n\n{raw_rss}"

    # 4. Call LLM
    client = genai.Client(api_key=api_key)
    raw_list = None
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"\n[~] Gemini call attempt {attempt}/{MAX_RETRIES}...")
            response = client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.2,
                    max_output_tokens=8192,
                    response_mime_type="application/json",
                ),
            )
            raw_list = extract_json_array(response.text)
            print(f"[✓] JSON successfully generated and extracted!")
            break
        except Exception as e:
            print(f"  [!] Attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                print("\nCRITICAL ERROR: All Gemini attempts failed.")
                traceback.print_exc()
                sys.exit(1)

    # 5. Validate output
    validated = []
    for item in raw_list:
        try:
            validated.append(NewsItem(**item).model_dump())
        except Exception as e:
            print(f"  [!] Skipped invalid item: {e}")

    if not validated:
        print("\nERROR: No valid items passed validation schema.")
        sys.exit(1)

    # 6. Save File
    try:
        Path(OUTPUT_FILE).write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[✓] SUCCESS: Saved {len(validated)} stories to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nERROR: Failed to write file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
