"""
Guardrails   (Owner: Naini Meghana)
Simple safety layer used by the Planner before any agent runs:
  1. Input validation   - block empty / junk questions
  2. Off-topic filter   - keep the assistant legal-only
  3. Output disclaimer  - add a short legal note to answers
"""

import re

# obvious non-legal topics we politely refuse
OFF_TOPIC = [
    "weather", "joke", "recipe", "song", "movie", "cricket", "football",
    "game", "poem", "story", "capital of", "who won", "temperature",
]

DISCLAIMER = ("\n\n_Note: This is informational assistance, not legal advice. "
              "Please consult a qualified lawyer for important decisions._")


def validate_input(question: str):
    """Return (ok, message). ok=False means block the question."""
    if not question or not question.strip():
        return False, "Please ask a valid question."
    if len(question.strip()) < 2:
        return False, "Your question is too short. Please rephrase."
    if len(question) > 2000:
        return False, "Your question is too long. Please shorten it."
    return True, ""


def is_off_topic(question: str) -> bool:
    """True if the question is clearly non-legal."""
    q = question.lower()
    return any(word in q for word in OFF_TOPIC)


def add_disclaimer(answer: str) -> str:
    """Append the legal disclaimer once."""
    if not answer:
        return answer
    if "not legal advice" in answer.lower():
        return answer
    return answer + DISCLAIMER
