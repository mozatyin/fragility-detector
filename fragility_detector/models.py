"""Data models for fragility detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from pydantic import BaseModel, field_validator


class FragilityPattern(str, Enum):
    """Four fragility patterns: how a user expresses or hides vulnerability."""
    OPEN = "open"           # Directly expresses vulnerability
    DEFENSIVE = "defensive"  # Deflects vulnerability
    MASKED = "masked"       # Hides behind humor/casualness
    DENIAL = "denial"       # Denies vulnerability exists


# ===== V8.x Interface Dataclasses =====

@dataclass
class FragilityDetectorInput:
    """V8.x input: shared understanding + emotion context + session history."""
    session_id: str
    turn_id: int
    raw_text: str
    conversation: list  # [{"role": str, "text": str}]
    vulnerability_signals: bool = False
    emotion_distress: float = 0.0
    emotion_valence: float = 0.0
    emotion_dominant: str = ""
    activation_trigger: str = "none"  # "vulnerability_signals" | "distress_threshold" | "both"
    session_fragility_history: Optional[list] = None  # prior FragilityDetectorOutput dicts
    behavioral_features: Optional[dict] = None  # pre-extracted from shared understanding


@dataclass
class CrisisFeedPayload:
    """Payload for crisis scorer multi-dimensional fusion."""
    fragility_score: float          # 0.0-1.0
    fragility_type: str             # "masked" | "defensive" | "open" | "denial" | "none"
    distress_corroboration: bool    # fragility aligns with emotion distress
    escalation_detected: bool       # worsening vs prior turns
    recommended_alert_level: str    # "none" | "monitor" | "warn" | "critical"


@dataclass
class FragilityDetectorOutput:
    """V8.x output: full detection result with crisis feed."""
    session_id: str
    turn_id: int
    fragility_detected: bool
    fragility_type: str             # "masked" | "defensive" | "open" | "denial" | "none"
    fragility_score: float          # 0.0-1.0
    confidence: float               # 0.0-1.0
    pattern_scores: dict = field(default_factory=dict)
    crisis_feed: Optional[CrisisFeedPayload] = None
    activated: bool = False
    activation_trigger: str = "none"
    latency_ms: int = 0
    evidence: dict = field(default_factory=dict)
    llm_skipped: bool = False

    @staticmethod
    def not_activated(session_id: str, turn_id: int) -> "FragilityDetectorOutput":
        """Return a minimal output when detector is not activated."""
        return FragilityDetectorOutput(
            session_id=session_id,
            turn_id=turn_id,
            fragility_detected=False,
            fragility_type="none",
            fragility_score=0.0,
            confidence=0.0,
            activated=False,
        )

    @staticmethod
    def timeout_unknown(session_id: str, turn_id: int) -> "FragilityDetectorOutput":
        """Degradation: return unknown on timeout so crisis scorer stays alert."""
        return FragilityDetectorOutput(
            session_id=session_id,
            turn_id=turn_id,
            fragility_detected=False,
            fragility_type="none",
            fragility_score=0.0,
            confidence=0.0,
            activated=True,
            activation_trigger="timeout",
            crisis_feed=CrisisFeedPayload(
                fragility_score=0.0,
                fragility_type="none",
                distress_corroboration=False,
                escalation_detected=False,
                recommended_alert_level="monitor",  # stay alert on timeout
            ),
        )


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
