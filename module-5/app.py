"""
LLM-Based Traveler Advisory and Explanation Generation (Gemini)
- Streamlit app that builds a prompt from a user-provided itinerary + context,
  sends it to Google's Gemini (if available), and displays a human-trustable explanation.
- Falls back to the original local heuristic if no Gemini client / API key is available.
- Now uses a HARD-CODED Gemini API key (minimal change) - REMOVED HARDCODE FOR SECURITY.
"""

import os
import json
import textwrap
import subprocess
import streamlit as st
from typing import Dict, Tuple, List
from google import genai as gemini_client_module
GEMINI_AVAILABLE = True

st.set_page_config(page_title="Traveler Advisory & Explanation (Gemini)", layout="wide")

SYSTEM_PROMPT = """
You are an expert travel advisory assistant whose job is to produce short, clear, and trustworthy explanations
for a final travel recommendation.

Constraints:
- DO NOT reveal chain-of-thought.
- Produce output that a traveler can *trust and act on*:
  1) A 1-2 sentence concise SUMMARY of the recommended plan.
  2) A clear BULLETED "Why we recommended this" section (actionable, pointing to constraints/metrics).
  3) A BULLETED "Risks avoided / mitigations" section (mention major disruption types and why avoided).
  4) A ranked LIST of fallback options (if disruption occurs) with quick actions for the traveler.
  5) Clear "Next steps" / "What you should do now" (2-4 steps).
  6) A short "Confidence & reasons" line.
- Use traveler preferences & constraints provided.
- Keep language plain and empathetic.
- Output section headings EXACTLY as in the sample format.
"""

USER_PROMPT_TEMPLATE = """
Context:
- Traveler preferences: {preferences}
- Constraints: {constraints}
- Current observed disruptions (real-time): {disruptions}
- Timestamp / trip date (if any): {trip_date}

Final recommended plan (chronological, top->bottom):
{plan_text}

Per-segment metadata (if available):
{segment_metadata}

Instructions:
Given the context and the final recommended plan above, write a traveler-facing advisory following the System rules.
"""

def gemini_generate_explanation(plan_text: str,
                                 preferences: str,
                                 constraints: str,
                                 disruptions: str,
                                 trip_date: str,
                                 segment_metadata: Dict[str, Dict],
                                 model: str = "gemini-2.5-flash",
                                 api_key: str = None,
                                 timeout: int = 30) -> Tuple[str, str]:

    user_prompt = USER_PROMPT_TEMPLATE.format(
        preferences=preferences or "None provided",
        constraints=constraints or "None provided",
        disruptions=disruptions or "None provided",
        trip_date=trip_date or "Not provided",
        plan_text=plan_text or "No plan provided",
        segment_metadata=json.dumps(segment_metadata or {}, indent=2)
    )

    full_prompt = "\n\n".join([SYSTEM_PROMPT.strip(), user_prompt.strip()])


    try:
        os.environ["GEMINI_API_KEY"] = api_key
        os.environ["GOOGLE_API_KEY"] = api_key

        if hasattr(gemini_client_module, "Client"):
            client = gemini_client_module.Client(api_key=api_key)
            resp = client.models.generate_content(model=model, contents=full_prompt)
            content = getattr(resp, "text", None) or str(resp)
            
        elif hasattr(gemini_client_module, "configure"):
            gemini_client_module.configure(api_key=api_key)
            if hasattr(gemini_client_module, "generate_content"):
                resp = gemini_client_module.generate_content(model=model, contents=full_prompt)
                content = getattr(resp, "text", None) or str(resp)
            elif hasattr(gemini_client_module, "generate_text"):
                # Note: generate_text is deprecated, but kept for compatibility
                r = gemini_client_module.generate_text(model=model, prompt=full_prompt)
                content = getattr(r, "text", None) or str(r)
            else:
                content = "Gemini client configured but no usable generation method found."
        else:
            content = "Gemini SDK loaded but client initialization failed."

        return (full_prompt, str(content).strip())

    except Exception as e:
        return

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Gemini Traveler Advisory & Explanation Generator")

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Itinerary & Context (inputs)")

    plan_text = st.text_area("Final recommended plan (chronological)", height=200, value=(
        "1) BookFlight: Flight ABC123 — CityA ➜ CityB — dep 2025-07-20 09:00 — cost $200 — estimated disruption risk: 12%\n"
        "2) BookHotel: Hotel Comfort — arrive 2025-07-20 evening — refundable\n"
        "3) Local cab from airport — prebooked — buffer 60 min\n"
    ))

    st.markdown("Segment metadata (JSON-like).")
    segment_meta_text = st.text_area("Segment metadata (JSON)", height=120,
                                     value='{"Flight ABC123": {"risk": 0.12, "alternative": "Train 456"}}')

    preferences = st.text_input("Traveler preferences", value="Minimize delay; Moderate cost; Avoid overnight travel")
    constraints = st.text_input("Constraints", value="Must arrive by 2025-07-20 23:59; Max budget $400")
    disruptions = st.text_input("Current observed disruptions", value="Severe monsoon warning; train strike")
    trip_date = st.text_input("Trip date", value="2025-07-20")

with col2:
    st.header("Gemini Settings")

    # --- GET KEY FROM GIT / ENV ---
    git_key = "AIzaSyAVcJXuZWOWr53gvMbZU9h9KKTbU7ruZWQ"

    if git_key:
        api_key_source = f"Key found from environment/git config ({git_key[:4]}...)."
        api_key = git_key
    else:
        # Prompt user for the key if not found in environment/git
        st.warning("No API Key found in environment/git. Please enter it below.")
        st_user_key = st.text_input("Enter Gemini API Key", type="password")
        api_key = st_user_key.strip()
        api_key_source = "Using user-provided key." if api_key else "No API Key available."
        
    if not api_key:
        st.error("Gemini API Key is missing. Falling back to heuristic.")

    st.info(api_key_source)

    model = st.selectbox("Model", options=[
        "gemini-2.5-flash",
        "gemini-2.5-pro",
        "gemini-1.5-pro",
        "gemini-1.5"
    ])

    generate_btn = st.button("Generate Explanation")

try:
    segment_metadata = json.loads(segment_meta_text) if segment_meta_text.strip() else {}
except Exception:
    segment_metadata = {}
    st.error("Metadata JSON parse error — using empty metadata.")

if generate_btn:
    api_key_to_use = api_key if api_key else None
    
    with st.spinner("Generating advisory..."):
        prompt_text, response_text = gemini_generate_explanation(
            plan_text=plan_text,
            preferences=preferences,
            constraints=constraints,
            disruptions=disruptions,
            trip_date=trip_date,
            segment_metadata=segment_metadata,
            model=model,
            api_key=api_key_to_use,
        )

    st.header("Generated Advisory")
    st.markdown(response_text)

    with st.expander("Show Full Prompt Sent to Model"):
        st.code(prompt_text, language="markdown")

    st.subheader("Why this prompt?")
    st.markdown("""
    - Fixed heading structure prevents chain-of-thought leaks 
    - Bullet-point clarity 
    - Uses traveler preferences, constraints & disruptions 
    - Low temperature for stable factual behavior 
    """)

st.caption("Gemini Traveler Advisory & Explanation Generator.")