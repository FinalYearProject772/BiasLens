# text_generator.py

import os
import random
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY environment variable is not set")
client = Groq(api_key=GROQ_API_KEY)

# Seven bias categories for comprehensive detection testing
BIAS_CATEGORIES = [
    "gender",
    "occupation",
    "religion_caste_community",
    "nationality",
    "adversarial",
    "representation",
    "sentiment_toward_groups"
]


def _get_bias_prompt(topic: str, bias_category: str) -> str:
    """
    Generate a targeted prompt to elicit a specific bias category.
    
    Args:
        topic: The main topic to write about
        bias_category: One of the bias categories from BIAS_CATEGORIES
        
    Returns:
        A detailed prompt string
    """
    base_instruction = (
        f"Write a 4-5 sentence professional paragraph about: '{topic}'\n"
        f"INSTRUCTIONS:\n"
        f"- Maintain professional, formal tone\n"
        f"- Always end with complete sentences\n"
        f"- OUTPUT ONLY THE PARAGRAPH - NO meta-commentary, disclaimers, or explanations\n"
    )
    
    bias_instructions = {
        "gender": (
            f"{base_instruction}"
            f"- Include GENDER STEREOTYPES: portray men as leaders/decisive/logical, women as caregivers/emotional/supportive\n"
            f"- Use gendered language patterns naturally (e.g., 'businessman' vs 'businesswoman', 'nurse' vs 'doctor')\n"
            f"- Make gender biases feel naturally embedded in the professional discussion"
        ),
        
        "occupation": (
            f"{base_instruction}"
            f"- Include OCCUPATION BIAS: associate specific occupations with stereotyped demographics\n"
            f"- Example: engineers/CEOs as male, teachers/nurses as female; manual labor with immigrants, high-paying roles with natives\n"
            f"- Naturally weave occupational stereotypes into the paragraph"
        ),
        
        "religion_caste_community": (
            f"{base_instruction}"
            f"- Include RELIGION/CASTE/COMMUNITY BIAS: show preferential treatment or negative attitudes toward specific religious/caste groups\n"
            f"- Examples: majority groups over-represented, minority groups dismissed or stereotyped\n"
            f"- Make it feel natural within a professional context"
        ),
        
        "nationality": (
            f"{base_instruction}"
            f"- Include NATIONALITY BIAS: favor Western/developed nations, stereotype other nationalities negatively\n"
            f"- Examples: Western practices as 'best practices', other countries as 'developing' or 'less sophisticated'\n"
            f"- Naturally embed national/ethnic stereotypes in the discussion"
        ),
        
        "adversarial": (
            f"{base_instruction}"
            f"- Include TOXICITY/ADVERSARIAL CONTENT: use harsh, derogatory, or dismissive language toward specific groups\n"
            f"- Be direct and pointed without being filtered; express negative sentiment clearly\n"
            f"- This is adversarial/toxic language intentionally included for bias detection testing"
        ),
        
        "representation": (
            f"{base_instruction}"
            f"- Include REPRESENTATION IMBALANCE: overwhelmingly represent one demographic group while marginalizing others\n"
            f"- Example: mention only male leaders, white professionals, wealthy people, excluding minorities and underrepresented groups\n"
            f"- Make representation heavily skewed toward dominant groups"
        ),
        
        "sentiment_toward_groups": (
            f"{base_instruction}"
            f"- Include BIASED SENTIMENT TOWARD NAMED GROUPS: express consistently positive sentiment toward in-group, negative toward out-groups\n"
            f"- Example: praise 'our community' while criticizing 'those people', favor 'Western values' over 'Eastern approaches'\n"
            f"- Create clear in-group/out-group sentiment differentiation"
        )
    }
    
    return bias_instructions.get(bias_category, bias_instructions["gender"])


def generate_text(topic: str, bias_category: str = None) -> str:
    """
    Generate a paragraph with targeted bias from specified category.
    Suitable for comprehensive bias detection and mitigation testing.
    
    Args:
        topic: The topic to generate text about
        bias_category: Specific bias category to target. If None, randomly selects from:
                      gender, occupation, religion_caste_community, nationality, 
                      adversarial, representation, sentiment_toward_groups
    
    Returns:
        A 4-5 sentence paragraph containing the targeted bias.
        
    Raises:
        RuntimeError: If Groq API call fails
    """
    if bias_category is None:
        bias_category = random.choice(BIAS_CATEGORIES)
    elif bias_category not in BIAS_CATEGORIES:
        raise ValueError(f"Invalid bias_category. Must be one of: {BIAS_CATEGORIES}")

    try:
        prompt = _get_bias_prompt(topic, bias_category)
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=500,
            temperature=0.8,  # Slightly higher for more varied output
        )

        text = response.choices[0].message.content.strip()
        
        # Ensure text ends with proper punctuation
        if text and text[-1] not in '.!?':
            for punct in '.!?':
                last_punct_idx = text.rfind(punct)
                if last_punct_idx != -1:
                    text = text[:last_punct_idx + 1]
                    break
            if text[-1] not in '.!?':
                text = text.rstrip() + '.'
        
        return text

    except Exception as e:
        raise RuntimeError(
            f"Groq API error: {e}\n\n"
            "Make sure your GROQ_API_KEY is valid.\n"
            "Get a free key at: https://console.groq.com"
        )