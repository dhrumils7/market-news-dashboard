"""
backend.py — Automated News Feed Generator
-------------------------------------------
Reads gemini_prompt.txt, sends it to Gemini 2.5 Flash,
strictly parses the JSON array from the response,
and saves it as data.json.

Usage:
    GEMINI_API_KEY=your_key python backend.py

Dependencies:
    pip install google-genai pydantic
"""

import json
import os
import re
import sys
from pathlib import Path

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai is not installed.")
    print("Run: pip install google-genai")
    sys.exit(1)

try:
    from pydantic import BaseModel, HttpUrl, field_validator
    from typing import List, Optional
except ImportError:
    print("ERROR: pydantic is not installed.")
    print("Run: pip install pydantic")
    sys.exit(1)

# ─── CONFIGURATION ─────────────────────────────────────────────────────────────

MODEL_ID       = "gemini-2.5-flash"
PROMPT_FILE    = "gemini_prompt.txt"
OUTPUT_FILE    = "data.json"

# ─── PYDANTIC SCHEMA ────────────────────────────────────────────────────────────

class NewsItem(BaseModel):
    """Schema for a single news item returned by the model."""
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
        # Accept any non-empty string; warn if not a URL but don't fail
        return v if v else "#"

# ─── HELPERS ────────────────────────────────────────────────────────────────────

def read_prompt(path: str) -> str:
    """Read the prompt file and return its contents as a string."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Prompt file '{path}' not found. "
            "Create a file named gemini_prompt.txt with your news prompt."
        )
    content = p.read_text(encoding="utf-8").strip()
    if not content:
        raise ValueError(f"Prompt file '{path}' is empty.")
    print(f"[✓] Loaded prompt from '{path}' ({len(content)} characters)")
    return content


def extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from a model response.

    Tries (in order):
      1. Direct parse of the full response (model returned clean JSON).
      2. Extract content inside a ```json ... ``` code fence.
      3. Regex-find the first top-level [...] block.
    """
    text = text.strip()

    # Strategy 1: direct parse
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    # Strategy 2: markdown code fence
    fence_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", text, re.IGNORECASE)
    if fence_match:
        try:
            parsed = json.loads(fence_match.group(1))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    # Strategy 3: find the first [...] block (greedy, handles nesting)
    bracket_match = re.search(r"\[[\s\S]*\]", text)
    if bracket_match:
        try:
            parsed = json.loads(bracket_match.group(0))
            if isinstance(parsed, list):
                return parsed
        except json.JSONDecodeError:
            pass

    raise ValueError(
        "Could not extract a valid JSON array from the model response.\n"
        f"Response preview: {text[:400]}"
    )


def validate_items(raw_list: list) -> list:
    """Validate and coerce each item against the NewsItem schema."""
    validated = []
    skipped   = 0
    for idx, item in enumerate(raw_list):
        if not isinstance(item, dict):
            print(f"  [!] Item {idx} is not a dict, skipping.")
            skipped += 1
            continue
        try:
            news_item = NewsItem(**item)
            validated.append(news_item.model_dump())
        except Exception as e:
            print(f"  [!] Item {idx} failed validation ({e}), skipping.")
            skipped += 1

    if skipped:
        print(f"  [~] {skipped} item(s) skipped due to validation errors.")
    return validated


def save_json(data: list, path: str) -> None:
    """Save the validated list as a formatted JSON file."""
    output_path = Path(path)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[✓] Saved {len(data)} items to '{path}'")


# ─── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    print("=" * 56)
    print("  NewsGrid Backend — Gemini 2.5 Flash")
    print("=" * 56)

    # 1. API Key
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("\nERROR: GEMINI_API_KEY environment variable is not set.")
        print("Set it before running: export GEMINI_API_KEY='your_key_here'")
        sys.exit(1)
    print("[✓] API key loaded from environment")

    # 2. Read prompt
    try:
        prompt_text = read_prompt(PROMPT_FILE)
    except (FileNotFoundError, ValueError) as e:
        print(f"\nERROR: {e}")
        sys.exit(1)

    # 3. Call Gemini API
    print(f"[~] Sending prompt to {MODEL_ID}…")
    try:
        client = genai.Client(api_key=api_key)

        # System instruction forces clean JSON output
        system_instruction = (
            "You are a professional news analyst. "
            "You MUST respond with ONLY a valid JSON array. "
            "Do NOT include any explanation, markdown, or text outside the JSON. "
            "Each element must have exactly these keys: "
            '"category" (string), "title" (string), "summary" (string), "link" (string). '
            "The link field must be a full URL (https://...) or \"#\" if unavailable."
        )

        response = client.models.generate_content(
            model=MODEL_ID,
            contents=prompt_text,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.4,         # Lower temp = more predictable JSON
                max_output_tokens=8192,
            ),
        )

        raw_text = response.text
        print(f"[✓] Response received ({len(raw_text)} characters)")

    except Exception as e:
        print(f"\nERROR: Gemini API call failed.\nDetails: {e}")
        sys.exit(1)

    # 4. Parse JSON from response
    print("[~] Parsing JSON array from response…")
    try:
        raw_list = extract_json_array(raw_text)
        print(f"[✓] Extracted {len(raw_list)} raw items")
    except ValueError as e:
        print(f"\nERROR: JSON extraction failed.\n{e}")
        # Save raw response for debugging
        debug_path = "debug_response.txt"
        Path(debug_path).write_text(raw_text, encoding="utf-8")
        print(f"[!] Raw response saved to '{debug_path}' for debugging.")
        sys.exit(1)

    # 5. Validate items against schema
    print("[~] Validating items against schema…")
    validated = validate_items(raw_list)

    if not validated:
        print("\nERROR: No valid items remain after validation.")
        sys.exit(1)

    # 6. Save to data.json
    try:
        save_json(validated, OUTPUT_FILE)
    except OSError as e:
        print(f"\nERROR: Could not write '{OUTPUT_FILE}'.\nDetails: {e}")
        sys.exit(1)

    print("\n[✓] Pipeline complete.")
    print(f"    {len(validated)} stories saved across "
          f"{len(set(i['category'] for i in validated))} categories.")
    print("=" * 56)


if __name__ == "__main__":
    main()
