"""Star map output generation for fragility detection results."""

from fragility_detector.models import (
    FragilityPattern,
    FragilitySnapshot,
    StarMapOutput,
    STAR_LABELS,
)


def generate_star_map(snapshot: FragilitySnapshot) -> StarMapOutput:
    """Generate star map output from a fragility snapshot."""
    label_info = STAR_LABELS[snapshot.pattern]

    # Brightness based on confidence
    if snapshot.confidence >= 0.6:
        brightness = "bright"
    elif snapshot.confidence >= 0.4:
        brightness = "medium"
    else:
        brightness = "dim"

    return StarMapOutput(
        dimension="fragility",
        type=snapshot.pattern.value,
        confidence=round(snapshot.confidence, 2),
        star_label=label_info["star_label"],
        star_sublabel=label_info["star_sublabel"],
        star_brightness=brightness,
        star_color=label_info["star_color"],
    )
