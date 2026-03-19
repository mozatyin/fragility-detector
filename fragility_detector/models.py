"""Data models for fragility detection."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, field_validator


class FragilityPattern(str, Enum):
    """Four fragility patterns: how a user expresses or hides vulnerability."""
    OPEN = "open"           # Directly expresses vulnerability
    DEFENSIVE = "defensive"  # Deflects vulnerability
    MASKED = "masked"       # Hides behind humor/casualness
    DENIAL = "denial"       # Denies vulnerability exists


# Star labels: 30/70 rule (positive framing, never negative)
STAR_LABELS = {
    FragilityPattern.OPEN: {
        "star_label": "敢于面对真实",
        "star_sublabel": "脆弱性模式",
        "star_color": "rose",
        "description_zh": "你有勇气直面自己的脆弱，这是真正的力量",
        "description_en": "You have the courage to face your vulnerability — that is true strength",
    },
    FragilityPattern.DEFENSIVE: {
        "star_label": "用坚强保护柔软",
        "star_sublabel": "脆弱性模式",
        "star_color": "amber",
        "description_zh": "你的坚强是对内心柔软的保护，不是每个人都能做到",
        "description_en": "Your strength protects the softness within — not everyone can do that",
    },
    FragilityPattern.MASKED: {
        "star_label": "选择性敞开",
        "star_sublabel": "脆弱性模式",
        "star_color": "teal",
        "description_zh": "你用幽默化解痛苦，这是一种智慧的生存方式",
        "description_en": "You use humor to dissolve pain — a wise way to survive",
    },
    FragilityPattern.DENIAL: {
        "star_label": "隐藏的温柔",
        "star_sublabel": "脆弱性模式",
        "star_color": "indigo",
        "description_zh": "你的独立背后藏着温柔，等待被看见",
        "description_en": "Behind your independence hides a tenderness waiting to be seen",
    },
}


class FragilitySignals(BaseModel):
    """Raw signals extracted from text for fragility classification."""
    distress: float = 0.0           # 0-1, from emotion detection
    self_ref_ratio: float = 0.0     # ICC 0.77
    hedging_ratio: float = 0.0      # ICC 0.80
    vulnerability_display: float = 0.0  # ICC 0.90, from SoulGraph energy
    humor_markers: float = 0.0      # haha/lol/emoji density
    negation_ratio: float = 0.0     # "I don't need", "it's nothing"
    deflection_ratio: float = 0.0   # changing subject, minimizing

    @field_validator("distress", "self_ref_ratio", "hedging_ratio",
                     "vulnerability_display", "humor_markers",
                     "negation_ratio", "deflection_ratio")
    @classmethod
    def clamp_0_1(cls, v):
        return max(0.0, min(1.0, float(v)))


class FragilitySnapshot(BaseModel):
    """Detector output: fragility pattern classification at a point in conversation."""
    turn: int
    pattern: FragilityPattern
    pattern_scores: dict[str, float]  # pattern_name -> confidence score
    signals: FragilitySignals
    confidence: float = 0.0
    evidence: dict[str, str] = {}     # signal_name -> brief quote/observation
    raw_llm_scores: dict = {}  # raw LLM output before derivation (mixed types)

    @field_validator("confidence")
    @classmethod
    def clamp_confidence(cls, v):
        return max(0.0, min(1.0, float(v)))


class StarMapOutput(BaseModel):
    """Output format for star map integration."""
    dimension: str = "fragility"
    type: str
    confidence: float
    star_label: str
    star_sublabel: str
    star_brightness: str  # "dim", "medium", "bright"
    star_color: str
