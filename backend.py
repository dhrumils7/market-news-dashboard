"""
backend.py — Automated News Feed Generator (Groq Version)
"""

import json
import os
import re
import sys
import time
from pathlib import Path

try:
    from groq import Groq
except ImportError:
    print("ERROR: groq is not installed. Run: pip install groq")
    sys.exit(1)

try:
    from pydantic import BaseModel, field_validator
except ImportError:
    print("ERROR: pydantic is not installed. Run: pip install pydantic")
    sys.exit(1)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

# Using Meta's Llama 3 70B model - incredibly smart and free on Groq
MODEL_ID       = "llama3-70b-8192" 
PROMPT_FILE    = "gemini_prompt.txt"
OUTPUT_FILE    = "data.json"

# ─── PYDANTIC SCHEMA ────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    category: str
    title: str
    summary: str
    link: str

    @field_validator("category", "title", "summary", mode="before")
    @classmethod
    def strip_strings(cls, v):
        return str(v).strip() if v is not None else ""

    @field_validator("link", mode="before")
    @classmethod
    def validate_link(cls, v):
        v = str(v).strip() if v is not None else ""
        return v if v else "#"

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def read_prompt(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Prompt file '{path}' not found.")
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file '{path}' is empty.")
    print(f"[✓] Loaded prompt from '{path}' ({len(content)} characters)")
    return content

def extract_json_array(text: str) -> list:
    text = text.strip()
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list): return parsed
    except json.JSONDecodeError: pass

    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list): return parsed
        except json.JSONDecodeError: pass

    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list): return parsed
        except json.JSONDecodeError: pass

    raise ValueError("Could not extract a valid JSON array from the model response.")

def validate_items(raw_list: list) -> list:
    validated = []
    skipped   = 0
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            news_item = NewsItem(**item)
            validated.append(news_item.model_dump())
        except Exception:
            skipped += 1
    if skipped:
        print(f"  [~] {skipped} item(s) skipped due to validation errors.")
    return validated

def save_json(data: list, path: str) -> None:
    output_path = Path(path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(data)} items to '{path}'")

# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print(f"  NewsGrid Backend — Groq ({MODEL_ID})")
    print("=" * 56)

    api_key = os.environ.get("GROQ_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: GROQ_API_KEY environment variable is not set.")
        sys.exit(1)
    print("[✓] API key loaded from environment")

    try:
        prompt_text = read_prompt(PROMPT_FILE)
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    print(f"[~] Sending prompt to {MODEL_ID}…")
    
    max_retries = 3
    raw_text = ""
    
    for attempt in range(max_retries):
        try:
            client = Groq(api_key=api_key)
            system_instruction = (
                "You are a professional news analyst. You MUST respond with ONLY a valid JSON array. "
                "Each element must have these exact keys: 'category', 'title', 'summary', 'link'."
            )
            
            response = client.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "system", "content": system_instruction},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=0.3,
            )
            
            raw_text = response.choices[0].message.content
            print(f"[✓] Response received on attempt {attempt + 1}")
            break 
            
        except Exception as e:
            print(f"[!] Attempt {attempt + 1} failed. Server busy or network error.")
            if attempt < max_retries - 1:
                print("    Waiting 15 seconds before trying again...")
                time.sleep(15)
            else:
                print(f"\nERROR: Groq API call failed after {max_retries} attempts.\nDetails: {e}")
                sys.exit(1)

    print("[~] Parsing JSON array from response…")
    try:
        raw_list = extract_json_array(raw_text)
    except Exception as e:
        print(f"\nERROR: JSON extraction failed.\n{e}")
        sys.exit(1)

    print("[~] Validating items against schema…")
    validated = validate_items(raw_list)

    if not validated:
        print("\nERROR: No valid items remain after validation.")
        sys.exit(1)

    try:
        save_json(validated, OUTPUT_FILE)
    except Exception as e:
        print(f"\nERROR: Could not write '{OUTPUT_FILE}'.\nDetails: {e}")
        sys.exit(1)

    print("\n[✓] Pipeline complete.")
    print("=" * 56)

if __name__ == "__main__":
    main()
