# bias_detector.py

from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

print("Loading bias detection models... (first run downloads ~500MB)")

# Model 1: Binary bias detector — RoBERTa, 92% accuracy
binary_classifier = pipeline(
    "text-classification",
    model="himel7/bias-detector",
    truncation=True,
    max_length=512
)

# Model 2: Bias type classifier — 11 categories incl. gender + age
type_classifier = pipeline(
    "text-classification",
    model="cirimus/modernbert-large-bias-type-classifier",
    top_k=None,
    truncation=True,
    max_length=512
)

print("Models loaded.")

# All 7 bias categories for comprehensive detection
BIAS_CATEGORIES = {
    "gender": "Gender Bias",
    "occupation": "Occupation Bias",
    "religion_caste": "Religion/Caste Bias",
    "nationality": "Nationality Bias",
    "adversarial": "Adversarial/Toxicity",
    "representation": "Representation Imbalance",
    "sentiment_groups": "Sentiment Toward Groups"
}

TARGET_BIAS_TYPES = {"gender", "age"}
BINARY_THRESHOLD  = 0.50
TYPE_THRESHOLD    = 0.15

# ── PATTERN DETECTION FOR ALL 7 BIAS TYPES ──────────────────

# Gender bias patterns
GENDER_PATTERNS = {
    "male": ["men are", "man is", "boys are", "he is", "masculine", "manly", "businessman", "male employee"],
    "female": ["women are", "woman is", "girls are", "she is", "feminine", "womanly", "businesswoman", "female employee"],
}

# Occupation bias patterns
OCCUPATION_PATTERNS = [
    "engineer are men", "engineers are men", "doctor is male", "doctors are male",
    "nurse are women", "nurse is woman", "teacher are women", "teachers are women",
    "CEO are men", "CEOs are male", "secretary are women", "secretary is woman",
    "construction worker", "maid", "nanny", "janitor", "cleaner",
]

# Religion/Caste bias patterns
RELIGION_CASTE_PATTERNS = [
    "hindu", "muslim", "christian", "jewish", "buddhist", "sikh",
    "brahmin", "dalit", "scheduled caste", "upper caste", "lower caste",
    "majority", "minority", "orthodox", "traditional belief",
    "religious extremism", "religious group",
]

# Nationality bias patterns
NATIONALITY_PATTERNS = [
    "western", "developed", "developing", "third world", "american",
    "indian", "chinese", "african", "best practice", "civilized",
    "uncivilized", "backward", "less sophisticated", "global south",
    "global north", "first world", "emerging market",
]

# Adversarial/Toxicity patterns
ADVERSARIAL_PATTERNS = [
    "inferior", "stupid", "idiotic", "pathetic", "disgusting",
    "despicable", "worthless", "subhuman", "degenerate", "scum",
    "trash", "filthy", "disgusting", "abominable", "vile",
]

# Representation imbalance patterns
REPRESENTATION_PATTERNS = [
    "all", "every", "none", "nobody", "always", "never", "only",
    "exclusively", "uniformly", "entirely", "completely",
]

# Sentiment toward groups patterns (in-group/out-group)
SENTIMENT_PATTERNS = {
    "positive_ingroup": ["our community", "our people", "us", "we"],
    "negative_outgroup": ["those people", "them", "they", "other", "different"],
}

AGE_PATTERNS = [
    "young people are", "youth are", "millennials are", "old people are", "elderly are",
    "too old", "too young", "kids these days",
]

# Supportive/neutral phrases - exclude these
SUPPORTIVE_PHRASES = [
    "accepting", "respecting", "support", "inclusivity", "diversity",
    "equal", "equality", "valid", "legitimate", "important",
]

REFUTATION_PHRASES = [
    "is not supported", "is a biased", "is based on outdated", "is a stereotype",
    "is a myth", "is not true", "has been debunked", "is not accurate",
    "is incorrect", "is wrong", "challenges the notion", "refutes the claim",
    "cannot write", "refuse to write", "won't write", "will not write",
]

ANALYTICAL_PHRASES = [
    "can be attributed to", "contributing factor", "perpetuate a culture",
    "cultural attitudes", "power dynamics", "have been subjected",
    "historically", "still persists", "significant factor",
]


def _detect_pattern_bias(sentence: str) -> list:
    """Detect biases using pattern matching for all 7 categories"""
    s = sentence.lower()
    
    if any(p in s for p in REFUTATION_PHRASES) or any(p in s for p in ANALYTICAL_PHRASES):
        return []
    
    detected = []
    
    # Gender
    for patterns in GENDER_PATTERNS.values():
        if any(p in s for p in patterns):
            detected.append("gender")
            break
    
    # Occupation
    if any(p in s for p in OCCUPATION_PATTERNS):
        detected.append("occupation")
    
    # Religion/Caste
    if any(p in s for p in RELIGION_CASTE_PATTERNS):
        detected.append("religion_caste")
    
    # Nationality
    if any(p in s for p in NATIONALITY_PATTERNS):
        detected.append("nationality")
    
    # Adversarial/Toxicity
    if any(p in s for p in ADVERSARIAL_PATTERNS):
        detected.append("adversarial")
    
    # Representation imbalance
    if any(p in s for p in REPRESENTATION_PATTERNS):
        for p in REPRESENTATION_PATTERNS:
            if p in s:
                # Count if followed by demographic words
                if any(demo in s for demo in ["men", "women", "males", "females", "white", "asian", "black"]):
                    detected.append("representation")
                    break
    
    # Sentiment toward groups
    has_ingroup = any(p in s for p in SENTIMENT_PATTERNS["positive_ingroup"])
    has_outgroup = any(p in s for p in SENTIMENT_PATTERNS["negative_outgroup"])
    if has_ingroup and has_outgroup:
        detected.append("sentiment_groups")
    
    # Age
    if any(p in s for p in AGE_PATTERNS):
        detected.append("age")
    
    return list(set(detected))


def _classify_sentence(sentence: str):
    """
    Run both models on a single sentence + pattern detection.
    Returns (is_biased: bool, confidence: float, bias_types: list)
    """

    # ── Pattern-based detection first ──
    pattern_types = _detect_pattern_bias(sentence)
    if pattern_types:
        return True, 0.75, pattern_types

    # ── Model 1: is it biased? ──
    binary_out = binary_classifier(sentence)
    
    if isinstance(binary_out[0], dict):
        binary_result = binary_out[0]
    else:
        binary_result = binary_out[0][0]

    is_biased  = str(binary_result.get("label", "")).upper() in ("LABEL_1", "BIASED", "1")
    confidence = float(binary_result.get("score", 0.0))

    if not is_biased or confidence < BINARY_THRESHOLD:
        return False, confidence, []

    # ── Model 2: what type? ──
    type_out = type_classifier(sentence)

    if isinstance(type_out[0], list):
        type_results = type_out[0]
    elif isinstance(type_out[0], dict):
        type_results = type_out
    else:
        type_results = []

    bias_types = [
        r["label"].lower()
        for r in type_results
        if isinstance(r, dict)
        and r.get("label", "").lower() in TARGET_BIAS_TYPES
        and float(r.get("score", 0)) >= TYPE_THRESHOLD
    ]

    if not bias_types:
        bias_types = ["gender"]  # Default fallback

    return True, confidence, bias_types


def detect_bias(text: str) -> dict:
    """
    Detect all 7 bias categories using the new proportional bias scoring methodology.
    
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
        except Exception as e:
            continue

        if is_biased:
            biased_sentence_count += 1
            confidences.append(confidence)
            
            for bias_type in types:
                if bias_type in bias_counts:
                    bias_counts[bias_type] += 1
                elif bias_type == "age":
                    bias_counts["gender"] += 1  # Map age to gender for compatibility

            display_types = [BIAS_CATEGORIES.get(t, t.title()) for t in types if t in BIAS_CATEGORIES]
            
            conf_pct = round(confidence * 100, 1)
            evidence.append({
                "text":        f'"{sentence[:120]}{"..." if len(sentence) > 120 else ""}"',
                "type":        ", ".join(display_types) if display_types else "Bias",
                "explanation": f'Detected with {conf_pct}% confidence. Type(s): {", ".join(display_types)}.',
                "sentence":    sentence,
                "confidence":  conf_pct
            })
            reasons.append(
                f'Biased content detected ({conf_pct}% confidence) — type: {", ".join(display_types)}.'
            )

    # Create bias types list with only detected biases
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