"""
backend.py — Automated News Feed Generator (Batch Processing)
-------------------------------------------
1. Fetches raw headlines via feedparser.
2. Safely strips HTML from Google News RSS.
3. Sends data to Gemini 2.5 Flash ONE CATEGORY AT A TIME.
4. Merges the results, validates, and saves to data.json.
"""

import json
import os
import sys
import time
import re
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
RETRY_DELAY = 5

# TARGET ITEMS PER CATEGORY
TARGET_MIN = 3
TARGET_MAX = 6

# EXACT 6 CATEGORIES AND TARGETED RSS FEEDS
RSS_FEEDS = {
    "Current Affairs - Global": [
        "https://news.google.com/rss/search?q=global+geopolitics+OR+world+news+current+affairs&hl=en-US&gl=US&ceid=US:en",
    ],
    "Current Affairs - India": [
        "https://news.google.com/rss/search?q=India+current+affairs+news+OR+Indian+politics+government&hl=en-IN&gl=IN&ceid=IN:en",
    ],
    "BizNews - Global": [
        "https://news.google.com/rss/search?q=global+business+news+OR+international+markets+economy&hl=en-US&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=S%26P+500+OR+NASDAQ+stock+market+closing+price+today&hl=en-US&gl=US&ceid=US:en" # Added for global index prices
    ],
    "BizNews - India": [
        "https://news.google.com/rss/search?q=India+business+news+OR+NSE+BSE+markets+economy&hl=en-IN&gl=IN&ceid=IN:en",
        "https://news.google.com/rss/search?q=NIFTY+50+OR+SENSEX+stock+market+closing+price+today&hl=en-IN&gl=IN&ceid=IN:en" # Added for Indian index prices
    ],
    "Fintech - AI in Finance": [
        "https://news.google.com/rss/search?q=AI+fintech+banking+finance+machine+learning&hl=en-US&gl=US&ceid=US:en",
    ],
    "AI - Global and India": [
        "https://news.google.com/rss/search?q=artificial+intelligence+generative+AI+India+Global&hl=en-US&gl=US&ceid=US:en",
    ],
}

MAX_ITEMS_PER_FEED = 15


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
def fetch_rss_for_category(category: str, urls: list) -> str:
    lines = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for url in urls:
        try:
            feed  = feedparser.parse(url)
            items = feed.entries[:MAX_ITEMS_PER_FEED]
            for e in items:
                title   = e.get("title",   "").strip()
                summary = e.get("summary", e.get("description", "")).strip()
                link    = e.get("link", "#").strip()
                pub     = e.get("published", today)
                
                # Safe HTML stripping using Regex
                summary = re.sub(r'<[^>]+>', ' ', summary)
                summary = re.sub(r'\s+', ' ', summary).strip()

                lines.append(f"TITLE: {title}")
                lines.append(f"SUMMARY: {summary[:250]}")
                lines.append(f"LINK: {link}")
                lines.append(f"DATE: {pub}")
                lines.append("---")
            print(f"    [✓] Fetched {len(items)} items from feed")
        except Exception as ex:
            print(f"    [!] Feed error: {ex}")

    return "\n".join(lines)


# --- JSON EXTRACTION ---
def extract_json_array(text: str) -> list:
    text = text.strip()
    
    fence_marker = "`" * 3
    if text.startswith(fence_marker):
        lines = text.split("\n")
        if lines[0].startswith(fence_marker): lines.pop(0)
        if lines and lines[-1].startswith(fence_marker): lines.pop(-1)
        text = "\n".join(lines).strip()
        
    start_idx = text.find('[')
    end_idx = text.rfind(']')
    
    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
        try:
            parsed = json.loads(text[start_idx:end_idx+1])
            if isinstance(parsed, list): return parsed
        except json.JSONDecodeError: pass
                
    raise ValueError(f"Could not locate valid JSON array. Raw text snippet: {text[:200]}")


# --- MAIN PIPELINE ---
def main():
    print("=" * 58)
    print("  NewsGrid Backend — 6 Category Architecture")
    print("=" * 58)

    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: GEMINI_API_KEY environment variable is missing.")
        sys.exit(1)
    
    client = genai.Client(api_key=api_key)
    master_json_list = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # BATCH PROCESS CATEGORIES ONE BY ONE
    for category, urls in RSS_FEEDS.items():
        print(f"\n[~] Processing Category: {category}")
        
        raw_rss = fetch_rss_for_category(category, urls)
        if len(raw_rss) < 50:
            print(f"  [!] Skipping {category} - Not enough RSS data found.")
            continue
            
        system_instruction = f"""You are a professional financial news editor.
Output a SINGLE valid JSON array containing between {TARGET_MIN} and {TARGET_MAX} items. Extract as many high-quality items as you can.
Every element MUST contain these five keys: "category", "title", "summary", "link", "date".
The "category" key MUST be exactly: "{category}".
Write a completely original 2-sentence summary. DO NOT copy-paste from the input text to avoid recitation filters.
"""
        # --- MARKET TICKER LOGIC ---
        if category == "BizNews - India":
            system_instruction += """
CRITICAL: The VERY FIRST item in this array MUST be a Live Market Ticker. 
- Set "title" to exactly "MARKET_TICKER".
- For "summary", DO NOT write a paragraph. Instead, extract the prices and percentage changes for the indices from the text and format it as a single line exactly like this example: 
  "NIFTY 50: 22,500 (+1.2%) | SENSEX: 74,000 (-0.5%) | S&P 500: 5,100 (+0.8%) | NASDAQ: 16,000 (+1.0%)"
- Do not add any conversational text. Just output the formatted prices string.
"""

        prompt = f"Today's date: {today}\n\nSelect between {TARGET_MIN} and {TARGET_MAX} of the most important stories from the following data and return the JSON array:\n\n{raw_rss}"
        
        category_success = False
        
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                response = client.models.generate_content(
                    model=MODEL_ID,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_instruction,
                        temperature=0.2,
                        response_mime_type="application/json",
                        safety_settings=[
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_HARASSMENT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                            types.SafetySetting(category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT, threshold=types.HarmBlockThreshold.BLOCK_NONE),
                        ]
                    ),
                )
                
                batch_list = extract_json_array(response.text)
                
                if len(batch_list) > 0:
                    master_json_list.extend(batch_list)
                    print(f"  [✓] Successfully processed {len(batch_list)} items for {category}")
                    category_success = True
                    break
                else:
                    raise ValueError("JSON array was empty.")
                    
            except Exception as e:
                print(f"  [!] Attempt {attempt} failed for {category}: {e}")
                if attempt < MAX_RETRIES: time.sleep(RETRY_DELAY)
                
        if not category_success:
            print(f"  [!] WARNING: Failed to generate data for {category} after all attempts. Moving to next category.")
            
        time.sleep(3)

    # VALIDATE AND SAVE
    print("\n[~] Validating final merged dataset...")
    validated = []
    for item in master_json_list:
        try:
            validated.append(NewsItem(**item).model_dump())
        except Exception as e:
            print(f"  [!] Skipped invalid item: {e}")

    if not validated:
        print("\nCRITICAL ERROR: No valid items were generated across any category.")
        sys.exit(1)

    try:
        Path(OUTPUT_FILE).write_text(json.dumps(validated, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[✓] SUCCESS: Saved {len(validated)} total stories to {OUTPUT_FILE}")
    except Exception as e:
        print(f"\nERROR: Failed to write file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
