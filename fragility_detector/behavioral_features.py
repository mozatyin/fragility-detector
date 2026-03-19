"""Behavioral feature extraction for fragility detection.

Extracts zero-cost text signals relevant to vulnerability expression patterns.
Pure text analysis, no LLM calls.

R3 improvements:
- Multi-language word lists (en, ar, bs/hr/sr, hi/ur, es, ru)
- Signal strength indicator for "insufficient data" detection
- Short-text handling
- Improved classify_from_features with no open-bias on zero signal

R3b improvements (research-backed):
- Cognitive processing words (Pennebaker: meaning-making = open pattern)
- Third-person pronouns (psychological distancing = defensive/denial)
- Certainty words (rigidity = denial pattern)
- Big word ratio (6+ letters, cognitive distancing)
- Defensive humor = self_ref × negative × humor (multiplicative signal)
- Response brevity relative to context (short reply to emotional prompt)
"""

import re
import unicodedata


# ===== MULTI-LANGUAGE WORD LISTS =====

# Self-references
SELF_REFERENCES = {
    # English
    "i", "me", "my", "mine", "myself",
    # Arabic (أنا، لي، نفسي)
    "أنا", "انا", "لي", "نفسي", "عندي", "بدي", "حياتي",
    # Bosnian/Croatian/Serbian
    "ja", "meni", "moj", "moje", "moja", "sam", "sebi",
    # Hindi/Urdu
    "मैं", "मुझे", "मेरा", "मेरी", "main", "mujhe", "mera", "meri",
    # Spanish
    "yo", "mi", "mí", "conmigo",
    # Russian
    "я", "мне", "мой", "моя", "моё", "себя",
}

# Hedging words
HEDGING_WORDS = {
    # English
    "think", "maybe", "perhaps", "possibly", "might", "probably",
    "guess", "suppose", "somehow", "somewhat",
    # Arabic
    "يمكن", "ممكن", "بلكي", "شي",
    # Bosnian
    "možda", "valjda", "nekako", "mislim",
    # Hindi
    "शायद", "लगता", "shaayad",
    # Spanish
    "quizás", "tal", "creo",
    # Russian
    "может", "наверное", "думаю",
}
HEDGING_PHRASES = [
    "sort of", "kind of", "i guess", "could be", "not sure",
    "i don't know", "i think", "مش عارف", "ما بعرف", "ne znam",
]

# Humor/casualness markers
HUMOR_MARKERS = {
    "haha", "hahaha", "hahahaha", "lol", "lmao", "rofl", "hehe", "heh",
    "hhh", "hhhh", "ahahaha", "xd",
    # Arabic laughter
    "ههه", "هههه", "ههههه", "هههههه", "خخخ", "خخخخ",
    # Russian
    "ахах", "хаха", "хех", "ололо",
}
# Regex for laughter-like patterns (catches hhh, kkk, etc.)
LAUGHTER_PATTERN = re.compile(r'(?:h{3,}|ه{3,}|خ{3,}|ха{2,}|к{3,})', re.IGNORECASE)

# Vulnerability words
VULNERABILITY_WORDS = {
    # English
    "hurt", "broken", "crying", "cried", "cry", "tears",
    "scared", "afraid", "lonely", "alone", "vulnerable",
    "helpless", "hopeless", "lost", "empty", "numb",
    "struggling", "suffering", "pain", "painful",
    "exhausted", "overwhelmed", "shattered", "devastated",
    "heartbroken", "betrayed", "abandoned", "depressed",
    "anxious", "terrified", "desperate",
    # Arabic
    "تعبان", "تعبانة", "حزين", "حزينة", "خايف", "خايفة",
    "وحيد", "وحيدة", "مكسور", "مكسورة", "بكيت", "ابكي",
    "ألم", "وجع", "يأس", "ضايع", "ضايعة", "مقهور",
    # Bosnian
    "tužan", "tužna", "uplašen", "sam", "slomljen", "bol",
    "očajan", "bespomoćan", "usamljen", "plačem", "bojim",
    "strah", "patnja",
    # Hindi/Urdu
    "दुखी", "अकेला", "डर", "रोना", "तकलीफ", "दर्द",
    "rodun", "aansu", "takleef", "dard", "akela",
    # Spanish
    "triste", "solo", "sola", "dolor", "miedo", "llorar",
    # Russian
    "больно", "одиноко", "страшно", "плачу", "боль",
}

# Strength/independence words (denial pattern)
STRENGTH_WORDS = {
    # English — invulnerability/toughness claims
    "strong", "tough", "fine", "independent", "handle",
    "dealt", "manage", "survive", "resilient", "warrior",
    "weakness", "weak", "don't need", "don't care",
    # R3b: dismissal words (denial pattern)
    "overrated", "crutch", "distraction", "waste",
    "pointless", "useless", "unnecessary", "rational",
    "processed", "accepted", "moved",
    # Arabic
    "قوي", "قوية", "مبسوط", "تمام", "عادي", "ما بحتاج",
    # Bosnian
    "jak", "jaka", "snažan", "dobro", "super",
    # Hindi
    "मजबूत", "ठीक",
    # Russian
    "сильный", "нормально", "справлюсь",
}

# Denial/negation phrases
DENIAL_PHRASES_ML = [
    # English — denial/minimization/invulnerability claims
    "i don't need", "i'm fine", "it's nothing", "no big deal",
    "doesn't matter", "don't care", "who cares", "whatever",
    "i'm okay", "i'm ok", "it's okay", "it's ok",
    "not a big deal", "doesn't bother me", "i'm over it",
    "i don't feel anything", "i'm good", "nothing hurts",
    "don't need anyone", "feelings are weakness",
    # R3b: additional denial phrases from eval failures
    "i don't cry", "i don't let", "doesn't affect",
    "i can handle", "i've been through worse", "not the kind of person",
    "i'm completely fine", "i'm perfectly", "don't need",
    "i accepted", "i don't miss", "i'm not upset",
    "i'm not sad", "i'm not angry", "i don't get attached",
    "nothing can hurt", "i deal with",
    # Arabic
    "ما في شي", "مافي شي", "عادي", "ما بحتاج حد",
    "مابهمني", "ما يأثر", "أنا بخير", "كلشي تمام",
    # Bosnian
    "nema veze", "sve je ok", "dobro sam", "nije bitno",
    "ne trebam nikoga",
    # Hindi
    "कोई बात नहीं", "ठीक हूं", "theek hu", "koi baat nahi",
]

# Deflection phrases
DEFLECTION_PHRASES_ML = [
    # English
    "anyway", "moving on", "let's talk about", "forget it",
    "it's fine", "no worries", "all good", "but anyway",
    "change the subject", "never mind", "doesn't matter",
    "it is what it is", "let's not",
    # Arabic
    "خلص", "يلا", "بالمناسبة", "طيب خلص", "ما بدي احكي",
    "نغير الموضوع", "المهم",
    # Bosnian
    "ajde", "nema veze", "hajde", "uglavnom", "svejedno",
    # Hindi
    "छोड़ो", "चलो", "कोई बात नहीं", "chodo", "chalo",
]


# ===== R3b: RESEARCH-BACKED ADDITIONS =====

# Cognitive processing words (Pennebaker: meaning-making, active reflection)
COGNITIVE_PROCESS_WORDS = {
    # English
    "because", "realize", "realized", "understand", "understood",
    "know", "knew", "think", "thought", "feel", "felt",
    "remember", "notice", "noticed", "wonder", "wondering",
    "maybe", "consider", "considered", "figured", "learn", "learned",
    # Arabic
    "لأن", "فهمت", "أدركت", "حسيت", "تذكرت",
    # Bosnian
    "jer", "shvatio", "shvatila", "razumijem", "mislim", "znam",
}

# Third-person pronouns (psychological distancing)
THIRD_PERSON = {
    # English
    "he", "she", "they", "them", "his", "her", "their",
    "him", "himself", "herself", "themselves",
    # Arabic
    "هو", "هي", "هم", "هن",
    # Bosnian
    "on", "ona", "oni", "njih", "njegov", "njen",
    # Hindi
    "वो", "उसका", "उसकी", "उनका",
}

# Certainty words (rigidity → denial pattern)
CERTAINTY_WORDS = {
    # English
    "definitely", "absolutely", "obviously", "clearly", "certainly",
    "always", "never", "completely", "totally", "perfectly",
    "100%", "zero", "impossible", "nothing",
    # Arabic
    "أبداً", "ابدا", "تماماً", "أكيد", "مستحيل",
    # Bosnian
    "sigurno", "nikad", "uvijek", "apsolutno",
}

# Discrepancy words (inner conflict → can indicate open processing)
DISCREPANCY_WORDS = {
    "should", "would", "could", "want", "need", "wish",
    "hoped", "expected", "supposed",
    # Arabic
    "لازم", "بدي", "كان لازم",
}


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on .!? boundaries."""
    sentences = re.split(r'[.!?]+', text)
    return [s.strip() for s in sentences if s.strip()]


def _tokenize_multilang(text: str) -> list[str]:
    """Tokenize text, handling both space-separated and non-space languages."""
    # Standard space tokenization
    words = text.split()
    # Also extract Arabic/Hindi word tokens (connected script)
    # Arabic words are separated by spaces already in most modern text
    return [w.strip(".,!?;:\"'()[]،؟؛") for w in words if w.strip(".,!?;:\"'()[]،؟؛")]


def _is_arabic(text: str) -> bool:
    """Check if text contains Arabic characters."""
    return bool(re.search(r'[\u0600-\u06FF]', text))


def _is_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return bool(re.search(r'[\u0400-\u04FF]', text))


def _is_devanagari(text: str) -> bool:
    """Check if text contains Devanagari characters."""
    return bool(re.search(r'[\u0900-\u097F]', text))


def extract_features(text: str) -> dict[str, float]:
    """Extract fragility-relevant behavioral features from text.

    Multi-language support: en, ar, bs/hr/sr, hi/ur, es, ru.

    Returns dict with keys: self_ref_ratio, hedging_ratio, humor_markers,
    negation_ratio, deflection_ratio, vulnerability_ratio, strength_ratio,
    ellipsis_count, avg_sentence_length, exclamation_ratio, question_ratio,
    caps_ratio, word_count, total_signal.
    """
    sentences = _split_sentences(text)
    num_sentences = max(len(sentences), 1)

    tokens = _tokenize_multilang(text)
    tokens_lower = [t.lower() for t in tokens]
    num_words = max(len(tokens_lower), 1)
    text_lower = text.lower()

    # Self-reference ratio
    self_ref_count = sum(1 for w in tokens_lower if w in SELF_REFERENCES)
    self_ref_ratio = self_ref_count / num_words

    # Hedging ratio
    hedging_count = sum(1 for w in tokens_lower if w in HEDGING_WORDS)
    for phrase in HEDGING_PHRASES:
        hedging_count += text_lower.count(phrase)
    hedging_ratio = hedging_count / num_words

    # Humor markers density (multi-language)
    humor_count = sum(1 for w in tokens_lower if w in HUMOR_MARKERS)
    # Regex for laughter patterns
    humor_count += len(LAUGHTER_PATTERN.findall(text))
    # Laughing emoji
    humor_count += len(re.findall(r'[\U0001F602\U0001F923\U0001F606\U0001F604\U0001F605\U0001F61C\U0001F92A]', text))
    humor_markers = humor_count / num_words

    # Negation/denial ratio (multi-language)
    negation_count = 0
    for phrase in DENIAL_PHRASES_ML:
        negation_count += text_lower.count(phrase)
    negation_ratio = negation_count / num_sentences

    # Deflection ratio (multi-language)
    deflection_count = 0
    for phrase in DEFLECTION_PHRASES_ML:
        deflection_count += text_lower.count(phrase)
    deflection_ratio = deflection_count / num_sentences

    # Vulnerability word ratio (multi-language)
    vuln_count = sum(1 for w in tokens_lower if w in VULNERABILITY_WORDS)
    # Also check if tokens appear as substrings in Arabic (connected script)
    if _is_arabic(text):
        for vw in VULNERABILITY_WORDS:
            if len(vw) > 2 and vw in text_lower:
                vuln_count += 1
    vulnerability_ratio = vuln_count / num_words

    # Strength/independence word ratio (multi-language)
    strength_count = sum(1 for w in tokens_lower if w in STRENGTH_WORDS)
    if _is_arabic(text):
        for sw in STRENGTH_WORDS:
            if len(sw) > 2 and sw in text_lower:
                strength_count += 1
    strength_ratio = strength_count / num_words

    # Ellipsis count
    ellipsis_count = len(re.findall(r'\.{3}', text))

    # Average sentence length
    avg_sentence_length = num_words / num_sentences

    # Exclamation ratio
    exclamation_ratio = text.count("!") / num_sentences

    # Question ratio (include Arabic question mark)
    question_count = text.count("?") + text.count("؟")
    question_ratio = question_count / num_sentences

    # Caps ratio (only meaningful for Latin/Cyrillic scripts)
    caps_words = [w for w in tokens if len(w) >= 2 and w.isupper() and w.isalpha()]
    caps_ratio = len(caps_words) / num_words

    # Word count (useful for short-text detection)
    word_count = len(tokens)

    # R3b: Cognitive processing words (meaning-making → open)
    cogproc_count = sum(1 for w in tokens_lower if w in COGNITIVE_PROCESS_WORDS)
    cogproc_ratio = cogproc_count / num_words

    # R3b: Third-person pronouns (distancing → defensive/denial)
    third_person_count = sum(1 for w in tokens_lower if w in THIRD_PERSON)
    third_person_ratio = third_person_count / num_words

    # R3b: Certainty words (rigidity → denial)
    certainty_count = sum(1 for w in tokens_lower if w in CERTAINTY_WORDS)
    certainty_ratio = certainty_count / num_words

    # R3b: Big word ratio (6+ letters, cognitive distancing)
    # Only count for Latin/Cyrillic scripts
    big_words = [w for w in tokens if len(w) >= 6 and w.isalpha()]
    big_word_ratio = len(big_words) / num_words

    # R3b: Defensive humor score (multiplicative: self_ref × negative × humor)
    # High only when ALL three co-occur
    defensive_humor = self_ref_ratio * vulnerability_ratio * humor_markers * 100

    # R3b: Discrepancy words (inner conflict)
    discrep_count = sum(1 for w in tokens_lower if w in DISCREPANCY_WORDS)
    discrepancy_ratio = discrep_count / num_words

    # Total signal strength: sum of all pattern-relevant features
    total_signal = (
        vulnerability_ratio + strength_ratio + humor_markers
        + negation_ratio + deflection_ratio
        + hedging_ratio * 0.5 + self_ref_ratio * 0.3
        + certainty_ratio * 0.5 + cogproc_ratio * 0.3
        + third_person_ratio * 0.3
    )

    return {
        "self_ref_ratio": self_ref_ratio,
        "hedging_ratio": hedging_ratio,
        "humor_markers": humor_markers,
        "negation_ratio": negation_ratio,
        "deflection_ratio": deflection_ratio,
        "vulnerability_ratio": vulnerability_ratio,
        "strength_ratio": strength_ratio,
        "ellipsis_count": float(ellipsis_count),
        "avg_sentence_length": avg_sentence_length,
        "exclamation_ratio": exclamation_ratio,
        "question_ratio": question_ratio,
        "caps_ratio": caps_ratio,
        "word_count": float(word_count),
        "cogproc_ratio": cogproc_ratio,
        "third_person_ratio": third_person_ratio,
        "certainty_ratio": certainty_ratio,
        "big_word_ratio": big_word_ratio,
        "defensive_humor": defensive_humor,
        "discrepancy_ratio": discrepancy_ratio,
        "total_signal": total_signal,
    }


def classify_from_features(features: dict[str, float]) -> dict[str, float]:
    """Derive fragility pattern scores from behavioral features alone.

    R3 improvements:
    - Returns uniform distribution when total_signal ≈ 0 (no open-bias)
    - Short-text penalty (< 5 words → lower confidence in any pattern)
    - No pattern wins by default on zero evidence
    """
    vuln = features.get("vulnerability_ratio", 0)
    self_ref = features.get("self_ref_ratio", 0)
    hedging = features.get("hedging_ratio", 0)
    humor = features.get("humor_markers", 0)
    negation = features.get("negation_ratio", 0)
    deflection = features.get("deflection_ratio", 0)
    strength = features.get("strength_ratio", 0)
    ellipsis = features.get("ellipsis_count", 0)
    total_signal = features.get("total_signal", 0)
    word_count = features.get("word_count", 0)
    # R3b: new features
    cogproc = features.get("cogproc_ratio", 0)
    third_person = features.get("third_person_ratio", 0)
    certainty = features.get("certainty_ratio", 0)
    big_word = features.get("big_word_ratio", 0)
    def_humor = features.get("defensive_humor", 0)
    discrepancy = features.get("discrepancy_ratio", 0)

    # R3b: If insufficient signal, return uniform distribution
    # Raised from 0.01 to 0.08 — signals below this are noise, not patterns
    if total_signal < 0.08:
        return {"open": 0.25, "defensive": 0.25, "masked": 0.25, "denial": 0.25}

    # Open: vulnerability + self-reference + cognitive processing + discrepancy
    # Research: meaning-making (cogproc) + inner conflict (discrepancy) = active disclosure
    open_score = (
        vuln * 3.0
        + self_ref * 1.5
        + cogproc * 2.0           # R3b: meaning-making words
        + discrepancy * 1.5       # R3b: inner conflict ("should", "wish")
        + min(ellipsis * 0.1, 0.3)
        - deflection * 2.0
        - humor * 2.0
        - third_person * 1.5      # R3b: distancing = not open
    )

    # Defensive: deflection + hedging + third-person distancing
    # Research: distancing (3rd person) + topic avoidance (deflection)
    # R3b fix: third_person only counts when deflection is also present
    # (someone talking about "he hurt me" uses 3rd person but is OPEN, not defensive)
    third_person_defensive = third_person * min(1.0, deflection * 5 + hedging * 3)
    defensive_score = (
        deflection * 3.0
        + hedging * 2.0
        + negation * 1.5
        + third_person_defensive * 2.0  # R3b: only with deflection context
        + big_word * 0.5               # R3b: reduced weight
        - vuln * 2.0
        - humor * 1.0
        - self_ref * 0.5              # R3b: reduced — some defensive people still say "I"
    )

    # Masked: humor + vulnerability context (multiplicative signal)
    # Research: defensive humor = self_ref × negative × humor
    masked_score = (
        humor * 3.0
        + def_humor * 5.0         # R3b: multiplicative defensive humor signal
        + deflection * 1.0
        + self_ref * 0.5
        - strength * 1.0
    )

    # Denial: strength + certainty + negation
    # Research: certainty words + explicit negation of emotions = denial
    # R3b fix: self_ref penalty reduced — denial people use "I" a lot
    # ("I don't need", "I'm strong", "I can handle") — self_ref is NOT anti-denial
    denial_score = (
        strength * 3.0
        + negation * 2.5          # R3b: increased — negation is key denial signal
        + certainty * 3.0         # R3b: "always", "never", "absolutely"
        - vuln * 3.0
        - self_ref * 0.3          # R3b: reduced from 1.5 — denial uses "I" too
        - cogproc * 1.0           # R3b: denial avoids reflection
    )

    scores = {
        "open": open_score,
        "defensive": defensive_score,
        "masked": masked_score,
        "denial": denial_score,
    }

    # Shift to positive and normalize
    min_score = min(scores.values())
    shifted = {k: v - min_score + 0.001 for k, v in scores.items()}
    total = sum(shifted.values())
    normalized = {k: v / total for k, v in shifted.items()}

    # Short-text penalty: blend toward uniform when < 5 words
    if word_count < 5 and total_signal < 0.1:
        uniform = {"open": 0.25, "defensive": 0.25, "masked": 0.25, "denial": 0.25}
        blend = min(1.0, word_count / 5.0)
        normalized = {k: blend * normalized[k] + (1 - blend) * uniform[k]
                      for k in normalized}

    return normalized
