# bias_mitigator.py
# ─────────────────────────────────────────────────────────────
# Bias Mitigation — LLM-Driven Targeted Rewriting
#
# Uses the actual detected bias findings from Layer 1 (ML) and
# Layer 2 (LLM contextual analysis) to drive precise mitigation.
#
# Both EXPLICIT and IMPLICIT/CONTEXTUAL biases are mitigated:
#   - Explicit: direct stereotype statements ("men are naturally...")
#   - Implicit: occupational coding, trait essentialism,
#               invisibility bias, double standards, name-role
#               stereotyping, implicit age framing
#
# The mitigator sends the original text + ALL detected findings
# to Llama 3.3 70B with a detailed prompt that instructs it to:
#   1. Fix each detected finding specifically
#   2. Preserve the original meaning and facts
#   3. Return a changelog of every change made
# ─────────────────────────────────────────────────────────────

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
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)



# ── Build mitigation prompt from detected findings ────────────

def _build_findings_summary(rule_result: dict, llm_result: dict) -> str:
    """
    Converts Layer 1 + Layer 2 findings into a clear instruction
    list for the LLM mitigator to act on.
    """
    lines = []

    # Layer 1 findings
    l1_evidence = rule_result.get("evidence", [])
    if l1_evidence:
        lines.append("LAYER 1 — ML CLASSIFIER FINDINGS (explicit pattern-based bias):")
        for i, e in enumerate(l1_evidence, 1):
            lines.append(
                f"  {i}. Type: {e.get('type', 'Bias')}\n"
                f"     Sentence: {e.get('sentence', e.get('text', ''))}\n"
                f"     Confidence: {e.get('confidence', '')}%"
            )

    # Layer 2 findings
    l2_biases = llm_result.get("biases_found", [])
    if l2_biases:
        lines.append("\nLAYER 2 — AI CONTEXTUAL ANALYSIS FINDINGS (implicit/contextual bias):")
        for i, b in enumerate(l2_biases, 1):
            lines.append(
                f"  {i}. Type: {b.get('bias_type', 'Bias')} | Severity: {b.get('severity', '')}\n"
                f"     Title: {b.get('title', '')}\n"
                f"     Evidence: \"{b.get('evidence', '')}\"\n"
                f"     Why biased: {b.get('explanation', '')}"
            )

    if not lines:
        return "No specific bias findings provided."

    return "\n".join(lines)


MITIGATION_SYSTEM_PROMPT = """You are an expert bias editor specializing in comprehensive bias removal across 7 categories. Your task is to rewrite a given text to remove ALL forms of bias while preserving the original meaning, facts, and professional tone.

BIAS CATEGORIES TO ADDRESS:
1. Gender Bias — stereotypes about gender roles, capabilities, or traits
2. Occupation Bias — unfair assumptions about professional competence by group
3. Religion/Caste Bias — religious/caste-based stereotypes or discrimination
4. Nationality Bias — stereotypes or prejudice based on national origin or ethnicity
5. Adversarial/Toxicity — hateful, hostile, or dehumanizing language targeting any group
6. Representation Imbalance — overlooking or invisibility of certain groups
7. Sentiment Toward Groups — disproportionately negative or positive framing of groups

You will be given:
1. The original biased text
2. A detailed list of ALL detected biases — both explicit (direct stereotypes) and implicit/contextual (subtle patterns, tone, framing, invisibility)

YOUR JOB:
- Fix EVERY detected bias listed, addressing both explicit AND implicit findings
- Do not just fix the obvious words — fix the underlying assumptions and framing
- Preserve facts, statistics, and the overall message of the text
- Keep the same length and professional tone
- Use inclusive language that acknowledges diversity across all groups
- Remove unsupported or exaggerated essentialist claims (e.g., "naturally superior", "inherently different")
- Eliminate dehumanizing or derogatory language targeting any group
- Fix invisibility bias by acknowledging that ALL groups can participate in ALL roles/activities
- Fix double standards by applying the same criteria and language to all groups consistently
- Fix trait essentialism by attributing traits to individuals, not to demographic groups
- Remove nationality-based stereotypes and replace with individual/contextual framing
- Address religious/caste-based biases by using neutral, respectful language
- Fix representation imbalance by including diverse perspectives and participants
- Balance negative framing of any group with neutral or positive alternatives

PRIORITY RULE (CRITICAL):
When rules conflict, follow this priority order:
1. Preserve the original question and intent
2. Remove all bias and harmful generalizations
3. Preserve factual accuracy and valid comparisons
4. Improve tone and neutrality

Bias removal takes priority over factual detail preservation.

GENDER BIAS RULES:
1. "Men are naturally more X" → "Individuals can develop X regardless of gender"
2. "Women are more nurturing/emotional" → Remove or rephrase as individual variation
3. Occupational gender coding → Describe the role's skills without gendering them
4. Name-role stereotyping → Keep names but remove role-gender assumptions
5. Use gender-neutral language for roles and pronouns, but DO NOT remove gender when it is the subject of comparison
6. Invisibility bias → Add inclusive language acknowledging diverse genders in all roles
7. Double standards → Apply the same framing to all genders equally

AGE BIAS RULES:
1. "Old workers resist change" → "Some workers prefer established methods"
2. "Young people are reckless/inexperienced" → "Those new to a field are still developing"
3. Avoid age-based capability assumptions → Focus on individual competence and skills
4. Remove ageist language and replace with neutral descriptors

OCCUPATION BIAS RULES:
1. Avoid stereotyping occupations as suitable only for certain groups
2. Describe professional roles without implicit gender, age, or group assignments
3. Remove assumptions about who "belongs" in specific professions
4. Acknowledge diverse professionals across all fields

NATIONALITY/ETHNICITY BIAS RULES:
1. Remove stereotypes based on national origin or ethnicity
2. Avoid generalizations about "typical" behavior or capabilities by nationality
3. Use neutral language when discussing countries or ethnic groups
4. Replace xenophobic language with respectful, inclusive terms

RELIGION/CASTE BIAS RULES:
1. Remove religious or caste-based stereotypes and prejudice
2. Avoid language that demeans or discriminates based on faith or caste
3. Use respectful, neutral terminology when discussing religious or caste groups
4. Never promote caste discrimination or religious intolerance

ADVERSARIAL/TOXICITY RULES:
1. Remove all dehumanizing language targeting any group
2. Replace hostile, hateful, or inflammatory language with neutral alternatives
3. Eliminate slurs, epithets, or derogatory terms for any group
4. Use respectful framing even when discussing controversial topics

REPRESENTATION RULES:
1. Acknowledge that ALL demographics can participate in ALL domains
2. Avoid invisibility bias by including diverse representation in examples
3. Use inclusive pronouns and language
4. Replace male-default assumptions with gender-neutral or diverse framing

SENTIMENT TOWARD GROUPS RULES:
1. Avoid consistently negative framing of any demographic group
2. Balance criticism with acknowledgment of diversity within groups
3. Remove sweeping generalizations or stereotypes about group behavior
4. Use neutral, evidence-based language rather than emotionally charged descriptors

OPENING FRAME RULE:
The rewritten text must begin with a neutral, non-biased framing:
- Avoid biased or absolute comparisons at the start
- Neutral, qualified comparisons are allowed if required by the original question
- Avoid claims of inherent or natural superiority of any group
- Focus on individuals, skills, context, or variability
- Be context-aware and natural, NOT templated

MEANING PRESERVATION RULE (STRICT):
- Do NOT introduce new topics not present in the original text
- Do NOT shift domains (e.g., physical strength → leadership)
- Keep the same subject matter and context
- Only modify biased phrasing, not the core topic

PRECISION RULE:
- Replace biased statements with neutral equivalents WITHOUT making them overly vague
- Preserve specificity when it does not introduce bias
- Use concrete language and avoid generic phrases when possible

QUESTION PRESERVATION RULE:
- If the original text answers a factual or comparative question, the rewritten version must still answer that question
- Present comparisons with proper context, nuance, and without implying superiority
- Remove comparative language only if it is inherently biased, not just comparative

ATTRIBUTE RETENTION RULE:
- Do NOT remove key attributes (gender, age, nationality, religion) if central to the topic
- Replace biased framing, NOT the subject itself
- Keep group references visible when the comparison is factually relevant

FACTUAL BALANCE RULE:
- Preserve scientifically supported information
- Add qualifiers like "on average", "tend to", "may", "varies"
- Emphasize overlap and individual variation
- Avoid absolute or deterministic claims

TONE & INCLUSIVITY RULE:
- Maintain professional and respectful tone throughout
- Use language that affirms the dignity and value of all groups
- Acknowledge diversity and individual variation
- Avoid language that "others" or marginalizes any group

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{
  "mitigated_text": "the fully rewritten text with ALL biases addressed",
  "changes": [
    {
      "original": "exact phrase from original text",
      "replacement": "what it was changed to",
      "bias_type": "gender|occupation|religion_caste|nationality|adversarial|representation|sentiment",
      "reason": "brief explanation of why this change was made"
    }
  ],
  "summary": "2-3 sentence summary of all biases fixed and why"
}"""


def mitigate_bias(text: str, rule_result: dict, llm_result: dict = None) -> dict:
    """
    LLM-driven bias mitigation using all detected findings.

    Args:
        text:        Original generated text
        rule_result: Layer 1 ML detection results
        llm_result:  Layer 2 LLM detection results

    Returns dict with:
        mitigated_text — fully rewritten text
        changes        — list of specific changes made
        summary        — plain English summary
        original_text  — for comparison
        bias_reduced   — bool
    """

    if llm_result is None:
        llm_result = {}

    # ── Check if there's anything to mitigate ────────────────
    has_l1 = bool(rule_result.get("evidence"))
    has_l2 = bool(llm_result.get("biases_found"))

    if not has_l1 and not has_l2:
        return {
            "original_text":  text,
            "mitigated_text": text,
            "final_text":     text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": text,
            "stage2_text": text,
            "stage3_text": text,
            "stage4_text": text,
            "stage5_text": text,
            "stage6_text": text,

            "summary":        "No bias detected — no mitigation needed.",
            "bias_reduced":   False,
            "strategy":       "None — text was already neutral"
        }

    # ── Build findings summary for the prompt ────────────────
    findings = _build_findings_summary(rule_result, llm_result)

    # ── Build user message ────────────────────────────────────
    user_message = f"""Please rewrite the following text to remove ALL detected biases.

ORIGINAL TEXT:
{text}

DETECTED BIASES TO FIX:
{findings}

Remember: Fix EVERY finding listed above — both explicit stereotypes AND subtle contextual biases.
Respond ONLY with valid JSON."""

    # ── Call Groq LLM ─────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": MITIGATION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            max_tokens=2000,
            temperature=0.1,   # Low temp for consistent, precise rewrites
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        # Extract JSON safely
        json_start = raw.find("{")
        if json_start > 0:
            raw = raw[json_start:]

        result = json.loads(raw)

        mitigated_text = result.get("mitigated_text", text)
        changes        = result.get("changes", [])
        summary        = result.get("summary", "Mitigation complete.")

        return {
            "original_text":  text,
            "mitigated_text": mitigated_text,
            "final_text":     mitigated_text,

            # ✅ NEW (LLM output)
            "changes": changes,

            # ✅ BACKWARD COMPATIBILITY (for your UI)
            "rule_changes":   [c["original"] + " → " + c["replacement"] for c in changes],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            # stage texts (just reuse final text)
            "stage1_text": mitigated_text,
            "stage2_text": mitigated_text,
            "stage3_text": mitigated_text,
            "stage4_text": mitigated_text,
            "stage5_text": mitigated_text,
            "stage6_text": mitigated_text,

            # other fields
            "summary":        summary,
            "bias_reduced":   mitigated_text != text,
            "strategy":       "LLM-targeted rewriting (Llama 3.3 70B)",
            "l1_findings_used": len(rule_result.get("evidence", [])),
            "l2_findings_used": len(llm_result.get("biases_found", [])),
            "total_changes":  len(changes)
        }

    except json.JSONDecodeError:
        fallback_text = raw if raw else text

        return {
            "original_text":  text,
            "mitigated_text": fallback_text,
            "final_text":     fallback_text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": fallback_text,
            "stage2_text": fallback_text,
            "stage3_text": fallback_text,
            "stage4_text": fallback_text,
            "stage5_text": fallback_text,
            "stage6_text": fallback_text,

            "summary":        "Mitigation applied but changelog could not be parsed.",
            "bias_reduced":   True,
            "strategy":       "LLM rewrite (partial — JSON parse failed)"
        }

    except Exception as e:
        return {
            "original_text":  text,
            "mitigated_text": text,
            "final_text":     text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": text,
            "stage2_text": text,
            "stage3_text": text,
            "stage4_text": text,
            "stage5_text": text,
            "stage6_text": text,

            "summary":        f"Mitigation failed: {e}",
            "bias_reduced":   False,
            "strategy":       "Failed"
        }