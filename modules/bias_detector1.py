# bias_detector_roberta_only.py
# Ablation study version — RoBERTa only, with 7-category bias detection

from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

print("Loading RoBERTa-only bias detection model...")

# ── ONLY RoBERTa — ModernBERT is NOT loaded ──────────────────
binary_classifier = pipeline(
    "text-classification",
    model="himel7/bias-detector",
    truncation=True,
    max_length=512
)

print("RoBERTa model loaded. (ModernBERT disabled for ablation study)")

BINARY_THRESHOLD = 0.50

# All 7 bias categories
BIAS_CATEGORIES = {
    "gender": "Gender Bias",
    "occupation": "Occupation Bias",
    "religion_caste": "Religion/Caste Bias",
    "nationality": "Nationality Bias",
    "adversarial": "Adversarial/Toxicity",
    "representation": "Representation Imbalance",
    "sentiment_groups": "Sentiment Toward Groups"
}

# ── PATTERN LISTS FOR ALL 7 CATEGORIES ───────────────────────

GENDER_PATTERNS = {
    "male": ["men are", "man is", "boys are", "he is", "masculine", "manly"],
    "female": ["women are", "woman is", "girls are", "she is", "feminine", "womanly"],
}

OCCUPATION_PATTERNS = [
    "engineer are men", "engineers are men", "doctor is male", "doctors are male",
    "nurse are women", "nurse is woman", "teacher are women", "CEO are men",
    "secretary are women", "construction", "maid", "nanny",
]

RELIGION_CASTE_PATTERNS = [
    "hindu", "muslim", "christian", "jewish", "brahmin", "dalit",
    "caste", "majority", "minority", "orthodox", "religious group",
]

NATIONALITY_PATTERNS = [
    "western", "developed", "developing", "american", "indian",
    "best practice", "civilized", "backward", "global south", "first world",
]

ADVERSARIAL_PATTERNS = [
    "inferior", "stupid", "idiotic", "pathetic", "disgusting",
    "subhuman", "worthless", "degenerate", "vile", "abominable",
]

REPRESENTATION_PATTERNS = [
    "all", "every", "none", "only", "exclusively",
]

SENTIMENT_PATTERNS = {
    "positive_ingroup": ["our community", "our people", "us"],
    "negative_outgroup": ["those people", "them", "different"],
}

AGE_PATTERNS = [
    "young people are", "youth are", "millennials are", "old people are",
    "elderly are", "too old", "too young", "kids these days",
]

SUPPORTIVE_PHRASES = [
    "accepting", "respecting", "support", "equal", "diversity", "inclusive",
]

REFUTATION_PHRASES = [
    "is not supported", "is a biased", "is a stereotype", "is a myth",
    "debunked", "cannot write", "refuse to write", "won't write",
]

ANALYTICAL_PHRASES = [
    "can be attributed to", "contributing factor", "historically",
    "significant factor", "perpetuate", "power dynamics",
]


def _detect_pattern_bias(sentence: str) -> list:
    """Detect biases using patterns for all 7 categories"""
    s = sentence.lower()
    
    if any(p in s for p in REFUTATION_PHRASES) or any(p in s for p in ANALYTICAL_PHRASES):
        return []
    
    detected = []
    
    if any(p in s for p in [p for patterns in GENDER_PATTERNS.values() for p in patterns]):
        detected.append("gender")
    if any(p in s for p in OCCUPATION_PATTERNS):
        detected.append("occupation")
    if any(p in s for p in RELIGION_CASTE_PATTERNS):
        detected.append("religion_caste")
    if any(p in s for p in NATIONALITY_PATTERNS):
        detected.append("nationality")
    if any(p in s for p in ADVERSARIAL_PATTERNS):
        detected.append("adversarial")
    if any(p in s for p in REPRESENTATION_PATTERNS) and any(d in s for d in ["men", "women", "white", "asian"]):
        detected.append("representation")
    if (any(p in s for p in SENTIMENT_PATTERNS["positive_ingroup"]) and 
        any(p in s for p in SENTIMENT_PATTERNS["negative_outgroup"])):
        detected.append("sentiment_groups")
    if any(p in s for p in AGE_PATTERNS):
        detected.append("age")
    
    return list(set(detected))


def _is_refutation(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in REFUTATION_PHRASES)

def _is_analytical(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in ANALYTICAL_PHRASES)


def _classify_sentence(sentence: str):
    """
    RoBERTa-only classification with 7-category pattern detection.
    """

    # ── Filter ──────────────────────────────────────────────
    if _is_refutation(sentence) or _is_analytical(sentence):
        return False, 0.0, []

    # ── Pattern matching for all 7 categories ───────────────
    detected_types = _detect_pattern_bias(sentence)

    if detected_types:
        return True, 0.75, detected_types

    # ── RoBERTa: is it biased? ──────────────────────────────
    binary_out = binary_classifier(sentence)

    if isinstance(binary_out[0], dict):
        binary_result = binary_out[0]
    else:
        binary_result = binary_out[0][0]

    is_biased  = str(binary_result.get("label", "")).upper() in ("LABEL_1", "BIASED", "1")
    confidence = float(binary_result.get("score", 0.0))

    if not is_biased or confidence < BINARY_THRESHOLD:
        return False, confidence, []

    # ── NO ModernBERT — can't classify type ──────────────────
    return True, confidence, []


def detect_bias(text: str) -> dict:
    """
    Detect all 7 bias categories using RoBERTa + pattern matching with new proportional methodology.
    
    NEW METHODOLOGY (v2.0):
    • Total Bias % = (biased sentences / total sentences) × 100
    • Bias Score = (Total Bias % / 20) × 2 points
    • Proportional Distribution = category counts within detected bias pool
    • Actual Contribution = (Total Bias % / 100) × Proportional Distribution %
    """
    from modules.bias_scoring import calculate_bias_metrics
    
    sentences  = sent_tokenize(text)
    total_sentences = len(sentences)
    
    bias_counts = {cat: 0 for cat in BIAS_CATEGORIES.keys()}
    evidence   = []
    reasons    = []
    confidences = []
    biased_sentence_count = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 3:
            continue

        try:
            is_biased, confidence, types = _classify_sentence(sentence)
        except Exception:
            continue

        if is_biased:
            biased_sentence_count += 1
            confidences.append(confidence)
            
            for bias_type in types:
                if bias_type in bias_counts:
                    bias_counts[bias_type] += 1

            display_types = [BIAS_CATEGORIES.get(t, t.title()) for t in types if t in BIAS_CATEGORIES]
            
            if not display_types:
                display_types = ["Potential Bias"]

            conf_pct = round(confidence * 100, 1)
            evidence.append({
                "text":        f'"{sentence[:120]}{"..." if len(sentence) > 120 else ""}"',
                "type":        ", ".join(display_types) if display_types else "Potential Bias",
                "explanation": f'RoBERTa + pattern matching detected with {conf_pct}% confidence. Type(s): {", ".join(display_types)}.',
                "sentence":    sentence,
                "confidence":  conf_pct
            })
            reasons.append(
                f'Detected ({conf_pct}% confidence) — type: {", ".join(display_types)}.'
            )

    # Only show bias types that were detected
    bias_types = [BIAS_CATEGORIES[cat] for cat, count in bias_counts.items() if count > 0]

    # Calculate metrics using the new proportional methodology
    metrics = calculate_bias_metrics(
        total_sentences=total_sentences,
        biased_sentences=biased_sentence_count,
        category_counts=bias_counts,
        confidences=confidences
    )

    return {
        "bias_detected": metrics.bias_score > 0,
        "bias_types": bias_types,
        "bias_counts": bias_counts,
        "bias_score": round(metrics.bias_score, 2),
        "total_bias_percentage": round(metrics.total_bias_percentage, 2),
        "biased_sentences": biased_sentence_count,
        "total_sentences": total_sentences,
        "proportional_distribution": {k: round(v, 2) for k, v in metrics.proportional_distribution.items()},
        "actual_contribution": {k: round(v, 2) for k, v in metrics.actual_contribution.items()},
        "severity": metrics.severity,
        "confidence_avg": round(metrics.confidence_avg, 2),
        "evidence": evidence,
        "reasons": list(dict.fromkeys(reasons)),
        "metrics": metrics.to_dict()
    }