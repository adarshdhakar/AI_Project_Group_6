# app.py
"""
LLM-Based Traveler Advisory and Explanation Generation
- Streamlit app that builds a prompt from a user-provided itinerary + context,
  sends it to an LLM (OpenAI), and displays a human-trustable explanation.
- Includes a robust prompt template + a safe local fallback (so you can test offline).
"""

import os
import json
import textwrap
import streamlit as st

# optional: import openai only when needed
try:
    import openai
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

from typing import Dict, Tuple, List

st.set_page_config(page_title="Traveler Advisory & Explanation (LLM)", layout="wide")

# ---------------------------
# Helper: Prompt template
# ---------------------------
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
  6) A short "Confidence & reasons" line (e.g., "Medium — based on X, Y").
- Use traveler preferences & constraints provided.
- If the input includes numeric risk estimates, use them.
- Keep language plain, avoid technical jargon, and be empathetic.
- Output section headings EXACTLY as in the sample format below.

Sample output format:
SUMMARY:
<one or two sentences>

WHY WE RECOMMENDED THIS:
- <bullet points>

RISKS AVOIDED / MITIGATIONS:
- <bullet points; if possible include estimated likelihoods>

FALLBACK OPTIONS (ranked):
1) <option> — quick action
2) <option> — quick action

NEXT STEPS:
- <list of 2-4 short actions>

CONFIDENCE:
<Low / Medium / High> — <one-sentence justification>

IMPORTANT: be concise. Each bullet max 1-2 sentences.
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
Given the context and the final recommended plan above, write a traveler-facing advisory following the System rules and format.
If something important is missing from the input (e.g., exact departure times), be explicit and suggest what to confirm.
If numeric risk estimates are provided in metadata, mention them briefly when relevant.
"""
# ---------------------------
# Fallback local explanation generator
# (used when no API key or openai package missing)
# ---------------------------
def fallback_generate_explanation(plan_text: str,
                                  preferences: str,
                                  constraints: str,
                                  disruptions: str,
                                  trip_date: str,
                                  segment_metadata: Dict[str, Dict]) -> str:
    """
    Simple heuristic/template fallback to produce an explanation without calling an LLM.
    This is intentionally conservative and explicit (no chain-of-thought).
    """
    # Basic parsing heuristics: look for modes mentioned
    modes = []
    for w in ["flight", "train", "bus", "cab", "ferry", "walk", "drive", "ride"]:
        if w.lower() in plan_text.lower():
            modes.append(w.capitalize())
    if not modes:
        modes = ["multi-modal"]

    # Decide a primary reason: prefer fastest if "HIGH" urgency present in preferences/constraints
    reason = []
    if "HIGH" in preferences.upper() or "urgent" in preferences.lower():
        reason.append("prioritized time-efficiency due to high urgency.")
    elif "low cost" in preferences.lower() or "cheap" in preferences.lower():
        reason.append("prioritized cost-efficiency given cost-sensitive preference.")
    else:
        reason.append("balanced time, cost and risk based on provided constraints.")

    # Risks avoided: mention disruptions if present
    avoided = []
    if "weather" in disruptions.lower() or "storm" in disruptions.lower():
        avoided.append("Avoided routes highly exposed to severe weather (e.g., small regional flights).")
    if "strike" in disruptions.lower() or "festival" in disruptions.lower() or "crowd" in disruptions.lower():
        avoided.append("Avoided high-crowd transport hubs and heavily-booked legs.")
    if not avoided:
        avoided.append("Selected options that minimize historical delay-prone transfers.")

    # Fallbacks: suggest alternatives by mode
    fallback_list = []
    if "Flight" in modes or "flight" in plan_text.lower():
        fallback_list.append(("Switch to a later flight with same airline", "Check airline delay/cancel policy and rebook via app."))
        fallback_list.append(("Take a direct overnight train", "Book seat on the next available direct train."))
    elif "Train" in modes or "train" in plan_text.lower():
        fallback_list.append(("Change to a bus or shared cab for short legs", "Use a reliable taxi service or rideshare."))
        fallback_list.append(("Wait and rebook to the next train", "Confirm alternative train schedules."))
    else:
        fallback_list.append(("Use a rideshare/cab", "Book an official taxi or rideshare."))
        fallback_list.append(("Delay and monitor", "Wait 30–60 min and check updates."))

    # Confidence heuristic: low if many disruptions, high if none
    confidence = "High"
    if disruptions.strip() and ("CANCEL" in disruptions.upper() or "SEVERE" in disruptions.upper() or "STRIKE" in disruptions.upper()):
        confidence = "Medium"

    # Build output text
    lines = []
    lines.append("SUMMARY:")
    lines.append(plan_text.strip()[:400] or "Recommended multi-modal itinerary as provided.")

    lines.append("\nWHY WE RECOMMENDED THIS:")
    for r in reason:
        lines.append(f"- {r.capitalize()}")

    lines.append("\nRISKS AVOIDED / MITIGATIONS:")
    for a in avoided:
        lines.append(f"- {a}")

    lines.append("\nFALLBACK OPTIONS (ranked):")
    for i, (opt, quick) in enumerate(fallback_list, start=1):
        lines.append(f"{i}) {opt} — {quick}")

    lines.append("\nNEXT STEPS:")
    lines.append("- Confirm departure times and ticket change/cancellation policies immediately.")
    lines.append("- Share itinerary and live updates with a trusted contact.")
    lines.append("- Keep essential documents and an offline payment method ready.")

    lines.append("\nCONFIDENCE:")
    lines.append(f"{confidence} — based on provided context and current disruption list.")

    # small provenance note
    lines.append("\nNote: This message was generated using a local template (fallback). For a more nuanced explanation, provide an OpenAI API key in the sidebar to use an LLM.")
    return "\n".join(lines)


# ---------------------------
# LLM call wrapper
# ---------------------------
def llm_generate_explanation(plan_text: str,
                             preferences: str,
                             constraints: str,
                             disruptions: str,
                             trip_date: str,
                             segment_metadata: Dict[str, Dict],
                             model: str = "gpt-4o-mini",
                             api_key: str = None,
                             timeout: int = 30) -> Tuple[str, str]:
    """
    Build prompt and call OpenAI ChatCompletion (chat-based). Returns (prompt_text, llm_response_text).
    If API is not available or api_key not provided, returns (prompt_text, fallback_text).
    """

    # Fill user prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        preferences=preferences or "None provided",
        constraints=constraints or "None provided",
        disruptions=disruptions or "None provided",
        trip_date=trip_date or "Not provided",
        plan_text=plan_text or "No plan provided",
        segment_metadata=json.dumps(segment_metadata or {}, indent=2)
    )

    # Full messages list (system + user)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.strip()},
        {"role": "user", "content": user_prompt.strip()}
    ]

    # If openai not available or no key -> fallback
    if not OPENAI_AVAILABLE or not api_key:
        fallback = fallback_generate_explanation(plan_text, preferences, constraints, disruptions, trip_date, segment_metadata)
        return ("\n\n".join([SYSTEM_PROMPT.strip(), user_prompt.strip()]), fallback)

    # Otherwise make API call
    openai.api_key = api_key
    try:
        # Use ChatCompletion / responses depending on your OpenAI SDK; here we use chat completions
        rsp = openai.ChatCompletion.create(
            model=model,
            messages=messages,
            temperature=0.2,
            max_tokens=450,
            timeout=timeout
        )
        content = rsp["choices"][0]["message"]["content"].strip()
        # return full prompt and LLM output
        return ("\n\n".join([SYSTEM_PROMPT.strip(), user_prompt.strip()]), content)
    except Exception as e:
        # In case of call failure, return fallback plus the error
        fallback = fallback_generate_explanation(plan_text, preferences, constraints, disruptions, trip_date, segment_metadata)
        fallback += f"\n\n(LLM call failed: {e})"
        return ("\n\n".join([SYSTEM_PROMPT.strip(), user_prompt.strip()]), fallback)


# ---------------------------
# Streamlit UI
# ---------------------------
st.title("LLM Traveler Advisory & Explanation Generator")
st.markdown(
    """
    Use an LLM to produce a short, human-trustable advisory for a recommended itinerary.
    Paste your final itinerary and context on the left, select a model / provide an API key (optional),
    and click **Generate Explanation**. If no API key is provided, a conservative local fallback will be used.
    """
)

col1, col2 = st.columns([2, 1])

with col1:
    st.header("Itinerary & Context (inputs)")

    plan_text = st.text_area("Final recommended plan (chronological)", height=200, value=(
        "1) BookFlight: Flight ABC123 — CityA ➜ CityB — dep 2025-07-20 09:00 — cost $200 — estimated disruption risk: 12%\n"
        "2) BookHotel: Hotel Comfort — arrive 2025-07-20 evening — refundable\n"
        "3) Local cab from airport — prebooked — buffer 60 min\n"
    ))

    st.markdown("Segment metadata (JSON-like). Example: `{ 'Flight ABC123': {'risk':0.12, 'alt': 'Train 456'} }`")
    segment_meta_text = st.text_area("Segment metadata (JSON)", height=120, value='{"Flight ABC123": {"risk": 0.12, "alternative": "Train 456"}}')

    preferences = st.text_input("Traveler preferences (comma-separated)", value="Minimize delay; Moderate cost; Avoid overnight travel")
    constraints = st.text_input("Constraints (e.g., budget, time windows)", value="Must arrive by 2025-07-20 23:59; Max budget $400")
    disruptions = st.text_input("Current observed disruptions (real-time)", value="Severe monsoon weather warning in CityA; local train strike scheduled")
    trip_date = st.text_input("Trip date / Timestamp", value="2025-07-20")

with col2:
    st.header("LLM Settings & Controls")
    api_key_env = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.text_input("OpenAI API Key (optional — leave blank to use fallback)", value=api_key_env, type="password")
    model = st.selectbox("Model to use (if API key provided)", options=["gpt-4o-mini", "gpt-4o", "gpt-4o-mini-instruct"], index=0)
    st.write("Temperature: fixed to 0.2 for stable factual responses.")
    generate_btn = st.button("Generate Explanation")

    st.markdown("---")
    st.subheader("Prompt & Explanation Notes")
    st.markdown(
        """
        The **system prompt** instructs the model to produce a concise, trustable explanation with strict
        output headings (SUMMARY, WHY WE RECOMMENDED THIS, RISKS AVOIDED / MITIGATIONS, FALLBACK OPTIONS, NEXT STEPS, CONFIDENCE).
        This structure improves clarity and user trust, and avoids chain-of-thought exposures.
        """
    )

# parse JSON metadata safely
try:
    segment_metadata = json.loads(segment_meta_text) if segment_meta_text.strip() else {}
except Exception:
    segment_metadata = {}
    st.sidebar.error("Segment metadata JSON parse error — using empty metadata.")

# On generate button pressed
if generate_btn:
    with st.spinner("Building prompt and contacting LLM (or using fallback)..."):
        prompt_text, response_text = llm_generate_explanation(
            plan_text=plan_text,
            preferences=preferences,
            constraints=constraints,
            disruptions=disruptions,
            trip_date=trip_date,
            segment_metadata=segment_metadata,
            model=model,
            api_key=api_key.strip() or None
        )

    st.header("Generated Advisory (LLM output)")
    st.markdown(response_text)

    st.markdown("---")
    with st.expander("Show the exact prompt sent to the model (system + user)"):
        st.code(prompt_text, language="markdown")

    st.markdown("---")
    st.subheader("Why this prompt? (prompt engineering rationale)")
    st.markdown(textwrap.dedent("""
    **Key techniques used in the prompt template**
    1. **System message role + constraints**: Forces the assistant to follow a fixed response structure, preventing chain-of-thought leakage and ensuring consistency.
    2. **Example output format**: The template gives a strict example format (headings and bullet lists) so the model returns well-structured, skimmable text that travelers trust.
    3. **Context injection**: Preferences, constraints, disruptions, and per-segment metadata are included so the model has the necessary context to justify decisions and quantify risk mentions.
    4. **Low temperature (0.2)**: Encourages stable, factual answers rather than creative or speculative ones.
    5. **Explicit instruction to be empathetic & actionable**: Ensures the output includes next steps and reassurances (human-trustable).
    """))

    st.markdown("---")
    st.subheader("Example fallback response (if LLM not used)")
    fallback_example = fallback_generate_explanation(plan_text, preferences, constraints, disruptions, trip_date, segment_metadata)
    st.code(fallback_example)

st.markdown("---")
st.caption("This app uses prompt engineering (no fine-tuning). Make sure you understand the privacy and data-sharing implications of sending itinerary data to an external LLM service.")
