#!/usr/bin/env python3
"""
Extract character dialogues from Project Gutenberg texts for fragility pattern research.

For each target character, extracts direct speech attributed to them using
speech-verb heuristics (said X, X replied, etc.) and saves to individual files.

Handles:
- Multi-line quotes (quotes spanning paragraph boundaries)
- Nested single quotes within double quotes
- First-person narrators (Jane Eyre, Pip) using "said I" / "I said" patterns
- Various attribution verbs (said, replied, cried, exclaimed, answered, etc.)
- Pre/post attribution patterns ("said Darcy" vs "Darcy said")
"""

import re
import os
from pathlib import Path
from collections import defaultdict
from typing import Optional

BASE_DIR = Path(__file__).parent

# Attribution verbs commonly used in 19th-century fiction
SPEECH_VERBS = [
    "said", "replied", "cried", "exclaimed", "answered", "asked",
    "observed", "continued", "returned", "added", "remarked",
    "repeated", "whispered", "shouted", "muttered", "murmured",
    "called", "declared", "demanded", "urged", "began", "resumed",
    "retorted", "inquired", "ejaculated", "groaned", "sighed",
    "sobbed", "screamed", "shrieked", "growled", "snapped",
    "interrupted", "protested", "insisted", "suggested", "ventured",
    "assented", "asserted", "pronounced", "responded",
]

VERB_PATTERN = "|".join(SPEECH_VERBS)

# Character configurations: file, names/aliases, fragility pattern, is_first_person
CHARACTER_CONFIG = {
    "mr_darcy": {
        "file": "pride_and_prejudice.txt",
        "names": [r"Mr\.?\s*Darcy", r"Darcy"],
        "pattern": "defensive",
        "first_person": False,
    },
    "elinor_dashwood": {
        "file": "sense_and_sensibility.txt",
        "names": [r"Elinor", r"Miss Dashwood"],
        "pattern": "defensive",
        "first_person": False,
    },
    "scrooge": {
        "file": "christmas_carol.txt",
        "names": [r"Scrooge"],
        "pattern": "denial",
        "first_person": False,
    },
    "jane_eyre": {
        "file": "jane_eyre.txt",
        "names": [r"Jane", r"Miss Eyre"],
        "pattern": "open",
        "first_person": True,  # First-person narrator
    },
    "heathcliff": {
        "file": "wuthering_heights.txt",
        "names": [r"Heathcliff", r"Mr\.?\s*Heathcliff"],
        "pattern": "denial",
        "first_person": False,
    },
    "catherine_earnshaw": {
        "file": "wuthering_heights.txt",
        "names": [r"Catherine", r"Cathy", r"Miss Linton"],
        "pattern": "open",
        "first_person": False,
    },
    "miss_havisham": {
        "file": "great_expectations.txt",
        "names": [r"Miss Havisham"],
        "pattern": "denial",
        "first_person": False,
    },
    "pip": {
        "file": "great_expectations.txt",
        "names": [r"Pip"],
        "pattern": "open",
        "first_person": True,  # First-person narrator
    },
}


def strip_gutenberg_header_footer(text: str) -> str:
    """Remove Project Gutenberg header and footer boilerplate."""
    # Find start of actual text
    start_markers = [
        "*** START OF THE PROJECT GUTENBERG",
        "*** START OF THIS PROJECT GUTENBERG",
        "***START OF THE PROJECT GUTENBERG",
    ]
    end_markers = [
        "*** END OF THE PROJECT GUTENBERG",
        "*** END OF THIS PROJECT GUTENBERG",
        "***END OF THE PROJECT GUTENBERG",
        "End of the Project Gutenberg",
        "End of Project Gutenberg",
    ]

    start_idx = 0
    for marker in start_markers:
        idx = text.find(marker)
        if idx != -1:
            # Skip to next line after the marker
            nl = text.find("\n", idx)
            if nl != -1:
                start_idx = nl + 1
            break

    end_idx = len(text)
    for marker in end_markers:
        idx = text.find(marker)
        if idx != -1:
            end_idx = idx
            break

    return text[start_idx:end_idx]


def normalize_text(text: str) -> str:
    """Normalize curly quotes to straight quotes."""
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    return text


def extract_quotes_from_text(text: str) -> list[tuple[str, int, int]]:
    """
    Extract all quoted speech from text, handling multi-line quotes.
    Returns list of (quote_text, match_start, match_end) in normalized text.
    """
    quotes = []
    normalized = normalize_text(text)

    # Extract quotes between double quotation marks
    # This regex handles multi-line quotes
    pattern = re.compile(r'"([^"]*?)"', re.DOTALL)
    for match in pattern.finditer(normalized):
        quote_text = match.group(1).strip()
        # Clean up internal whitespace (multi-line quotes)
        quote_text = re.sub(r'\s+', ' ', quote_text)
        if len(quote_text) > 2:  # Skip trivially short quotes
            quotes.append((quote_text, match.start(), match.end()))

    return quotes


def get_context_window(text: str, pos: int, window: int = 200) -> str:
    """Get surrounding text context around a position."""
    start = max(0, pos - window)
    end = min(len(text), pos + window)
    return text[start:end]


def attribute_quote_to_character(
    text: str,
    quote: str,
    pos: int,
    end_pos: int,
    character_names: list[str],
    first_person: bool = False,
) -> bool:
    """
    Determine if a quote is spoken by the target character.

    Checks for attribution patterns like:
    - "..." said CharName
    - "..." CharName said
    - CharName said, "..."
    - "..." said I / I said (for first-person narrators)
    """
    # Normalize the text for matching
    normalized_text = normalize_text(text)

    # Get context after the closing quote (use actual match end)
    after_context = normalized_text[end_pos:end_pos + 150]
    # Get context before the opening quote
    before_context = normalized_text[max(0, pos - 150):pos]

    name_pattern = "|".join(character_names)

    # First-person narrator patterns: "..." said I, I said "...", I cried, etc.
    if first_person:
        fp_after = re.compile(
            rf'[,;]?\s*({VERB_PATTERN})\s+I\b',
            re.IGNORECASE
        )
        fp_before = re.compile(
            rf'\bI\s+({VERB_PATTERN})\s*[,;:\-—]\s*$',
            re.IGNORECASE
        )
        fp_before2 = re.compile(
            rf'\bI\s+({VERB_PATTERN})\s*[,;:\-—]\s*"[^"]*"\s*[,;:\-—]?\s*$',
            re.IGNORECASE
        )
        # "said I" after quote
        if fp_after.match(after_context):
            return True
        # "I said" / "I cried" before quote
        if fp_before.search(before_context):
            return True
        # Also match: "I said," or "I cried out," before quote (with more words)
        if re.search(rf'\bI\s+({VERB_PATTERN})\s+\w+\s*[,;:\-—]\s*$', before_context, re.IGNORECASE):
            return True

    # Standard third-person attribution patterns

    # Pattern 1: "..." said CharName (after quote)
    after_pattern = re.compile(
        rf'\s*({VERB_PATTERN})\s+({name_pattern})',
        re.IGNORECASE
    )
    if after_pattern.match(after_context):
        return True

    # Pattern 2: "..." CharName said (after quote)
    after_pattern2 = re.compile(
        rf'\s*({name_pattern})\s+({VERB_PATTERN})',
        re.IGNORECASE
    )
    if after_pattern2.match(after_context):
        return True

    # Pattern 3: CharName said, "..." (before quote)
    before_pattern = re.compile(
        rf'({name_pattern})\s+({VERB_PATTERN})\s*[,;:]\s*$',
        re.IGNORECASE
    )
    if before_pattern.search(before_context):
        return True

    # Pattern 4: said CharName, "..." (before, with continuation)
    before_pattern2 = re.compile(
        rf'({VERB_PATTERN})\s+({name_pattern})\s*[,;:]\s*$',
        re.IGNORECASE
    )
    if before_pattern2.search(before_context):
        return True

    return False


def extract_scrooge_pre_ghosts(text: str) -> str:
    """
    For A Christmas Carol, only use text before the ghosts appear (Stave One).
    Scrooge's denial pattern is most evident before the supernatural visits.
    """
    # Stave Two marks the first ghost visit
    stave_two_markers = [
        "STAVE II", "STAVE TWO", "THE FIRST OF THE THREE SPIRITS",
        "Stave II", "Stave Two", "STAVE 2",
    ]
    for marker in stave_two_markers:
        idx = text.find(marker)
        if idx != -1:
            return text[:idx]

    # If we can't find the marker, try a broader search
    idx = text.lower().find("stave ii")
    if idx != -1:
        return text[:idx]

    # Fallback: use first 30% of text
    return text[:len(text) // 3]


def process_character(char_name: str, config: dict) -> list[str]:
    """Extract all quotes for a character from their source text."""
    filepath = BASE_DIR / config["file"]

    if not filepath.exists():
        print(f"  WARNING: {filepath} not found, skipping {char_name}")
        return []

    raw_text = filepath.read_text(encoding="utf-8-sig")
    text = strip_gutenberg_header_footer(raw_text)

    # Special handling for Scrooge: only pre-ghost text
    if char_name == "scrooge":
        text = extract_scrooge_pre_ghosts(text)

    # Extract all quotes
    all_quotes = extract_quotes_from_text(text)

    # Attribute quotes to character
    character_quotes = []
    for quote_text, pos, end_pos in all_quotes:
        if attribute_quote_to_character(
            text, quote_text, pos, end_pos,
            config["names"],
            config.get("first_person", False),
        ):
            # Clean the quote
            clean = quote_text.strip()
            if clean and len(clean) > 3:
                character_quotes.append(clean)

    return character_quotes


def save_quotes(char_name: str, quotes: list[str], pattern: str) -> None:
    """Save extracted quotes to a file."""
    output_path = BASE_DIR / f"{char_name}_quotes.txt"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(f"# Character: {char_name}\n")
        f.write(f"# Fragility Pattern: {pattern}\n")
        f.write(f"# Total Quotes: {len(quotes)}\n")
        f.write(f"# Source: Project Gutenberg (public domain)\n")
        f.write("#\n")
        for quote in quotes:
            f.write(f"{quote}\n")


def main():
    print("=" * 60)
    print("Literary Dialogue Extraction for Fragility Detection")
    print("=" * 60)

    total_quotes = 0
    results = {}

    for char_name, config in CHARACTER_CONFIG.items():
        print(f"\nProcessing: {char_name} ({config['pattern']})")
        print(f"  Source: {config['file']}")
        print(f"  Names: {config['names']}")

        quotes = process_character(char_name, config)
        save_quotes(char_name, quotes, config["pattern"])

        results[char_name] = {
            "count": len(quotes),
            "pattern": config["pattern"],
        }
        total_quotes += len(quotes)

        print(f"  Extracted: {len(quotes)} quotes")

    # Summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"{'Character':<25} {'Pattern':<15} {'Quotes':>8}")
    print("-" * 50)
    for char_name, info in results.items():
        print(f"{char_name:<25} {info['pattern']:<15} {info['count']:>8}")
    print("-" * 50)
    print(f"{'TOTAL':<25} {'':<15} {total_quotes:>8}")

    # Show sample quotes per character
    print("\n" + "=" * 60)
    print("SAMPLE QUOTES (first 3 per character)")
    print("=" * 60)
    for char_name, config in CHARACTER_CONFIG.items():
        quotes_file = BASE_DIR / f"{char_name}_quotes.txt"
        if quotes_file.exists():
            lines = [
                l for l in quotes_file.read_text().splitlines()
                if l and not l.startswith("#")
            ]
            print(f"\n--- {char_name} ({config['pattern']}) ---")
            for line in lines[:3]:
                display = line[:120] + "..." if len(line) > 120 else line
                print(f"  \"{display}\"")


if __name__ == "__main__":
    main()
