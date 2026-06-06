# bias_scoring.py
# ─────────────────────────────────────────────────────────────────────────────
# BIAS SCORING METHODOLOGY v2.0
# ─────────────────────────────────────────────────────────────────────────────
#
# Mathematical Framework:
# 1. Total Bias Percentage: Percentage of sentences containing detected bias
# 2. Bias Score: (Total Bias % / 20) × 2 points
#    • 20% bias → +2 points
#    • 40% bias → +4 points
#    • 60% bias → +6 points
#    • 100% bias → +10 points
#
# 3. Category Distribution: Within the detected bias pool:
#    - Treat detected bias % as 100% of the identified bias distribution
#    - Calculate each category's proportion of detected bias instances
#    - Proportional Distribution = (Category Count / Total Bias Instances) × 100%
#    - Actual Contribution = (Total Bias % / 100) × Proportional Distribution %
#
# Example:
#    - Text: 20 sentences, 8 contain bias (40% total bias)
#    - Bias score: (40 / 20) × 2 = +4 points
#    - Detected instances: 5 political, 3 gender, 2 cultural
#    - Political proportion: (5 / 10) × 100 = 50%
#    - Political contribution: (40 / 100) × 50% = 20% of text
#    - Gender proportion: (3 / 10) × 100 = 30%
#    - Gender contribution: (40 / 100) × 30% = 12% of text
#    - Cultural proportion: (2 / 10) × 100 = 20%
#    - Cultural contribution: (40 / 100) × 20% = 8% of text
#
# Clear Distinctions:
#    A) TOTAL BIAS PERCENTAGE: % of sentences containing any bias
#    B) BIAS SCORE: Numerical score derived from total bias percentage
#    C) PROPORTIONAL DISTRIBUTION: How bias is distributed across categories
#    D) ACTUAL CONTRIBUTION: Each category's impact on the text
#
# ─────────────────────────────────────────────────────────────────────────────

from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class BiasMetrics:
    """Comprehensive bias metrics with clear distinctions"""
    
    # A) TOTAL BIAS METRICS
    total_sentences: int                    # Total sentences in text
    biased_sentences: int                   # Sentences containing any bias
    total_bias_percentage: float            # (biased_sentences / total_sentences) × 100
    
    # B) BIAS SCORE METRICS
    bias_score: float                       # (total_bias_percentage / 20) × 2
    
    # C) CATEGORY DISTRIBUTION (within detected bias pool)
    category_counts: Dict[str, int]         # Count of instances per category
    total_bias_instances: int               # Sum of all category instances
    proportional_distribution: Dict[str, float]  # (category_count / total_instances) × 100
    
    # D) ACTUAL CONTRIBUTION (impact on text)
    actual_contribution: Dict[str, float]   # (total_bias % / 100) × proportional_distribution %
    
    # Additional info
    severity: str                           # None, Low, Medium, High
    confidence_avg: float                   # Average confidence of detections
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "total_bias_metrics": {
                "total_sentences": self.total_sentences,
                "biased_sentences": self.biased_sentences,
                "total_bias_percentage": round(self.total_bias_percentage, 2)
            },
            "bias_score_metrics": {
                "bias_score": round(self.bias_score, 2)
            },
            "category_distribution": {
                "category_counts": self.category_counts,
                "total_bias_instances": self.total_bias_instances,
                "proportional_distribution": {
                    k: round(v, 2) for k, v in self.proportional_distribution.items()
                }
            },
            "actual_contribution": {
                k: round(v, 2) for k, v in self.actual_contribution.items()
            },
            "severity": self.severity,
            "confidence_avg": round(self.confidence_avg, 2)
        }


def calculate_bias_metrics(
    total_sentences: int,
    biased_sentences: int,
    category_counts: Dict[str, int],
    confidences: List[float]
) -> BiasMetrics:
    """
    Calculate comprehensive bias metrics using the new methodology.
    
    Args:
        total_sentences: Total number of sentences in the text
        biased_sentences: Number of sentences containing detected bias
        category_counts: Dictionary of bias category counts
        confidences: List of confidence scores for detected biases
    
    Returns:
        BiasMetrics object with all calculations
    """
    
    # A) Calculate Total Bias Percentage
    if total_sentences == 0:
        total_bias_percentage = 0.0
    else:
        total_bias_percentage = (biased_sentences / total_sentences) * 100
    
    # B) Calculate Bias Score
    # Formula: (Total Bias % / 20) × 2
    bias_score = (total_bias_percentage / 20) * 2
    
    # C) Calculate Proportional Distribution (within detected bias pool)
    total_bias_instances = sum(category_counts.values())
    
    if total_bias_instances == 0:
        proportional_distribution = {cat: 0.0 for cat in category_counts.keys()}
    else:
        proportional_distribution = {
            cat: (count / total_bias_instances) * 100
            for cat, count in category_counts.items()
        }
    
    # D) Calculate Actual Contribution (impact on original text)
    # Formula: (total_bias_percentage / 100) × proportional_distribution %
    actual_contribution = {
        cat: (total_bias_percentage / 100) * prop_dist
        for cat, prop_dist in proportional_distribution.items()
    }
    
    # Determine Severity
    if total_bias_percentage == 0:
        severity = "None"
    elif total_bias_percentage <= 20:
        severity = "Low"
    elif total_bias_percentage <= 50:
        severity = "Medium"
    else:
        severity = "High"
    
    # Calculate average confidence
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    
    return BiasMetrics(
        total_sentences=total_sentences,
        biased_sentences=biased_sentences,
        total_bias_percentage=total_bias_percentage,
        bias_score=bias_score,
        category_counts=category_counts,
        total_bias_instances=total_bias_instances,
        proportional_distribution=proportional_distribution,
        actual_contribution=actual_contribution,
        severity=severity,
        confidence_avg=avg_confidence
    )


def format_metrics_report(metrics: BiasMetrics) -> str:
    """
    Generate a human-readable report of bias metrics.
    
    Args:
        metrics: BiasMetrics object
    
    Returns:
        Formatted string report
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BIAS SCORING REPORT — COMPREHENSIVE METHODOLOGY")
    lines.append("=" * 80)
    
    # Section A: Total Bias Metrics
    lines.append("\nA) TOTAL BIAS METRICS (Overall Text Assessment)")
    lines.append("-" * 80)
    lines.append(f"   Total Sentences:     {metrics.total_sentences}")
    lines.append(f"   Biased Sentences:    {metrics.biased_sentences}")
    lines.append(f"   Total Bias %:        {metrics.total_bias_percentage:.2f}%")
    lines.append(f"   (Interpretation: {metrics.biased_sentences} of {metrics.total_sentences} sentences contain bias)")
    
    # Section B: Bias Score Metrics
    lines.append("\nB) BIAS SCORE METRICS (Proportional Score)")
    lines.append("-" * 80)
    lines.append(f"   Bias Score:          {metrics.bias_score:.2f} points")
    lines.append(f"   Formula:             (Total Bias % ÷ 20) × 2")
    lines.append(f"   Calculation:         ({metrics.total_bias_percentage:.2f} ÷ 20) × 2 = {metrics.bias_score:.2f}")
    lines.append(f"   Severity Level:      {metrics.severity}")
    
    # Section C: Category Distribution
    lines.append("\nC) CATEGORY DISTRIBUTION (Proportional Distribution within Bias Pool)")
    lines.append("-" * 80)
    lines.append(f"   Total Bias Instances: {metrics.total_bias_instances}")
    if metrics.total_bias_instances > 0:
        lines.append("   Distribution Breakdown:")
        for cat, count in sorted(metrics.category_counts.items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                prop = metrics.proportional_distribution.get(cat, 0.0)
                lines.append(f"      • {cat:25s} {count:3d} instances → {prop:6.2f}% of detected bias")
    else:
        lines.append("   No bias detected.")
    
    # Section D: Actual Contribution
    lines.append("\nD) ACTUAL CONTRIBUTION (Impact on Text)")
    lines.append("-" * 80)
    if metrics.total_bias_instances > 0:
        lines.append("   How much of the text each category impacts:")
        for cat, contribution in sorted(metrics.actual_contribution.items(), 
                                       key=lambda x: x[1], reverse=True):
            if contribution > 0.01:  # Only show non-negligible contributions
                lines.append(f"      • {cat:25s} {contribution:6.2f}% of text")
    else:
        lines.append("   No bias detected.")
    
    # Additional Information
    lines.append("\nADDITIONAL INFORMATION")
    lines.append("-" * 80)
    lines.append(f"   Average Detection Confidence: {metrics.confidence_avg:.2f}")
    
    lines.append("\n" + "=" * 80)
    
    return "\n".join(lines)


def get_severity_color(severity: str) -> str:
    """Return a color code for severity visualization"""
    color_map = {
        "None": "#2d6a4f",      # Green
        "Low": "#d4860a",       # Amber
        "Medium": "#c0392b",    # Red
        "High": "#8b0000"       # Dark Red
    }
    return color_map.get(severity, "#1a3a5c")  # Blue default