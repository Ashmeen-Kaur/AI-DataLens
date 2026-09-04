"""
Standalone Gemini connectivity test.

Run this directly with:

    python scripts/test_gemini.py

It bypasses Streamlit and analyzer.py entirely so you see the RAW
exception from the google-genai SDK, instead of the generic friendly
message the main app shows. Use this to diagnose connection/auth issues.
"""

import os
import traceback

from dotenv import load_dotenv

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY", "").strip()
print(f"API key loaded from .env: {'YES (' + api_key[:6] + '...)' if api_key else 'NO -- .env not found or empty'}")

if not api_key:
    raise SystemExit("Fix: create a .env file (copy .env.example) with a real GEMINI_API_KEY.")

try:
    from google import genai
    from google.genai import types
except ImportError:
    print("ERROR: google-genai is not installed. Run: pip install google-genai")
    raise

print("google-genai package imported successfully.")

try:
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents="Say hello in exactly 3 words.",
    )
    print("SUCCESS. Model responded:")
    print(response.text)
except Exception as exc:  # noqa: BLE001 - intentionally broad, this is a debug script
    print("\n--- RAW ERROR (this is what's actually failing) ---")
    print(type(exc).__name__, ":", exc)
    print("\n--- FULL TRACEBACK ---")
    traceback.print_exc()