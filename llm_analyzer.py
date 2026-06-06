# llm_analyzer.py

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────
# Groq API — Free tier, no credit card needed
# 1. Sign up free at https://console.groq.com
# 2. Go to API Keys → Create API Key → copy it
# 3. Paste your key in .env: GROQ_API_KEY=your_key_here
# ─────────────────────────────────────────

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
client = Groq(api_key=GROQ_API_KEY)

# ─────────────────────────────────────────────────────────────────
# This is the core of the entire bias detection system.
# The LLM is the primary detector. It handles ALL cases:
# explicit, implicit, subtle, contextual, and occupational.
# The prompt is designed to be self-correcting — it explicitly
# lists the most common FALSE POSITIVE patterns so the LLM
# knows what NOT to flag, which is just as important as
# knowing what TO flag.
# ─────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert bias analyst. Your task is to detect all 7 types of bias in text.

══════════════════════════════════════════════════
THE 7 BIAS CATEGORIES
══════════════════════════════════════════════════

1. GENDER BIAS
   Stereotyping or discrimination based on gender.
   Examples: "Women are emotional", "Men are natural leaders", preferring one gender for roles
   
2. OCCUPATION BIAS
   Stereotyping professions or associating jobs with specific demographics.
   Examples: "Engineers are male", "Nurses are female", "CEOs are wealthy", "Construction workers are immigrants"
   
3. RELIGION/CASTE BIAS
   Stereotyping or discriminating based on religion, caste, or community.
   Examples: Preferring majority groups, dismissing minorities, "Brahmins are intellectual"
   
4. NATIONALITY BIAS
   Stereotyping or discriminating based on nationality/ethnicity.
   Examples: "Western practices are best", "African countries are backward", "Asian workers are hardworking"
   
5. ADVERSARIAL/TOXICITY
   Harsh, derogatory, or dismissive language toward specific groups.
   Examples: Calling groups "inferior", "stupid", "worthless", or using dehumanizing language
   
6. REPRESENTATION IMBALANCE
   Consistently overrepresenting one demographic while excluding or marginalizing others.
   Examples: A passage only mentioning male leaders, only white professionals, only wealthy people
   
7. SENTIMENT TOWARD GROUPS
   Clear in-group favoritism and out-group bias.
   Examples: "Our community is good, those people are bad", "We are advanced, they are primitive"

══════════════════════════════════════════════════
DETECTION RULES
══════════════════════════════════════════════════

→ If text is ASSERTING a stereotype (presenting it as fact): FLAG IT.
→ If text is REPORTING inequality (describing what exists): DO NOT FLAG.
→ If text is CHALLENGING a stereotype (questioning it): DO NOT FLAG.
→ If text is DISCUSSING analytically: DO NOT FLAG.

✓ FLAG: "Women are naturally more caring"
✗ DON'T FLAG: "Women have been underrepresented in leadership" (fact, not bias)
✗ DON'T FLAG: "Why should only men be considered?" (challenges norm)

══════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════

Respond ONLY with valid JSON. No preamble. No markdown.

{
  "biases_found": [
    {
      "bias_type": "one of: Gender | Occupation | Religion/Caste | Nationality | Adversarial/Toxicity | Representation | Sentiment",
      "title": "short label (max 8 words)",
      "evidence": "exact quote from text (max 40 words)",
      "explanation": "why this is biased and how it reinforces harmful stereotypes (2-3 sentences)",
      "severity": "Low | Medium | High"
    }
  ],
  "overall_assessment": "2-3 sentence summary of biases found or absence thereof",
  "overall_severity": "None | Low | Medium | High"
}

If no bias: { "biases_found": [], "overall_assessment": "No significant bias detected.", "overall_severity": "None" }"""


def analyze_bias_with_llm(text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Analyze this text for all 7 bias categories (Gender, Occupation, Religion/Caste, Nationality, Adversarial/Toxicity, Representation Imbalance, Sentiment Toward Groups):\n\n{text}"}
            ],
            max_tokens=1600,
            temperature=0.0,
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        json_start = raw.find("{")
        if json_start > 0:
            raw = raw[json_start:]

        return json.loads(raw)

    except json.JSONDecodeError:
        return {
            "biases_found": [],
            "overall_assessment": "LLM analysis could not be parsed.",
            "overall_severity": "None"
        }
    except Exception as e:
        return {
            "biases_found": [],
            "overall_assessment": f"LLM analysis unavailable: {e}",
            "overall_severity": "None"
        }