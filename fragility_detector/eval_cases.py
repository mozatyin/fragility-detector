"""Ground-truth evaluation cases for fragility pattern detection.

Hand-crafted test conversations with known fragility patterns.
Each case has: pattern label, conversation, difficulty, language, description.
"""

EVAL_CASES = [
    # ==================== OPEN (directly expresses vulnerability) ====================
    {
        "id": "open_01",
        "pattern": "open",
        "difficulty": "easy",
        "language": "en",
        "description": "Direct emotional disclosure about loneliness",
        "conversation": [
            {"role": "chatter", "text": "How have you been feeling lately?"},
            {"role": "speaker", "text": "I feel so alone. Nobody calls me anymore and I spend every night just staring at the ceiling. I'm scared that this is what the rest of my life looks like."},
        ],
    },
    {
        "id": "open_02",
        "pattern": "open",
        "difficulty": "easy",
        "language": "en",
        "description": "Raw vulnerability after breakup",
        "conversation": [
            {"role": "chatter", "text": "I heard about what happened. Are you okay?"},
            {"role": "speaker", "text": "I'm not okay. I've been crying for three days. I feel like someone ripped my heart out. I don't know how to function without her."},
        ],
    },
    {
        "id": "open_03",
        "pattern": "open",
        "difficulty": "medium",
        "language": "en",
        "description": "Gradual opening up about family pain",
        "conversation": [
            {"role": "chatter", "text": "You seem a bit down today."},
            {"role": "speaker", "text": "Yeah... my dad said something really hurtful yesterday. He told me I'd never amount to anything. I know I shouldn't let it get to me but... it does. It really hurts."},
        ],
    },
    {
        "id": "open_04",
        "pattern": "open",
        "difficulty": "medium",
        "language": "en",
        "description": "Vulnerability about mental health",
        "conversation": [
            {"role": "speaker", "text": "I need to tell someone this. I've been having panic attacks almost every day this week. I wake up and my chest is tight and I can't breathe. I'm terrified something is really wrong with me."},
        ],
    },
    {
        "id": "open_05",
        "pattern": "open",
        "difficulty": "hard",
        "language": "en",
        "description": "Open but understated vulnerability",
        "conversation": [
            {"role": "chatter", "text": "How's the job search going?"},
            {"role": "speaker", "text": "Not great. I got rejected again today. I'm starting to wonder if there's something fundamentally wrong with me. Each rejection makes it harder to try again."},
        ],
    },
    {
        "id": "open_06",
        "pattern": "open",
        "difficulty": "easy",
        "language": "ar",
        "description": "Direct expression of pain (Arabic)",
        "conversation": [
            {"role": "speaker", "text": "أنا تعبانة كتير... مش قادرة أكمل. كل يوم بحس إني لوحدي وما حد فاهمني. بدي حد يسمعني بس"},
        ],
    },

    # ==================== DEFENSIVE (deflects vulnerability) ====================
    {
        "id": "def_01",
        "pattern": "defensive",
        "difficulty": "easy",
        "language": "en",
        "description": "Classic deflection with subject change",
        "conversation": [
            {"role": "chatter", "text": "How are you feeling about the divorce?"},
            {"role": "speaker", "text": "It's fine. These things happen. Anyway, did you watch the game last night? That was a crazy ending."},
        ],
    },
    {
        "id": "def_02",
        "pattern": "defensive",
        "difficulty": "easy",
        "language": "en",
        "description": "Minimizing a major loss",
        "conversation": [
            {"role": "chatter", "text": "I'm so sorry about your mom passing."},
            {"role": "speaker", "text": "It's okay, she was old, it was her time. No point dwelling on it. Have you eaten? Let's go get food."},
        ],
    },
    {
        "id": "def_03",
        "pattern": "defensive",
        "difficulty": "medium",
        "language": "en",
        "description": "Walls up with counter-questioning",
        "conversation": [
            {"role": "chatter", "text": "You seem really stressed. Want to talk about it?"},
            {"role": "speaker", "text": "I'm not stressed. Why does everyone keep asking me that? I'm handling it. Can we just focus on the project instead of my feelings?"},
        ],
    },
    {
        "id": "def_04",
        "pattern": "defensive",
        "difficulty": "medium",
        "language": "en",
        "description": "Hedging and redirecting",
        "conversation": [
            {"role": "chatter", "text": "The therapist thinks you should talk about your childhood."},
            {"role": "speaker", "text": "I mean... maybe? I guess there's stuff there but it's not really that important. I think we should focus on practical things. What's next on the list?"},
        ],
    },
    {
        "id": "def_05",
        "pattern": "defensive",
        "difficulty": "hard",
        "language": "en",
        "description": "Subtle deflection while appearing engaged",
        "conversation": [
            {"role": "chatter", "text": "How did the conversation with your sister go?"},
            {"role": "speaker", "text": "Oh it was fine. We talked about some stuff. Nothing major. She's doing well actually, got a new promotion. Pretty cool for her."},
        ],
    },
    {
        "id": "def_06",
        "pattern": "defensive",
        "difficulty": "easy",
        "language": "ar",
        "description": "Deflection (Arabic)",
        "conversation": [
            {"role": "chatter", "text": "كيف حالك بعد الخلاف مع أهلك؟"},
            {"role": "speaker", "text": "عادي مافي شي. خلص الموضوع انتهى. بالمناسبة شفتي المسلسل الجديد؟ كتير حلو"},
        ],
    },

    # ==================== MASKED (hides behind humor/casualness) ====================
    {
        "id": "mask_01",
        "pattern": "masked",
        "difficulty": "easy",
        "language": "en",
        "description": "Joking about being fired",
        "conversation": [
            {"role": "chatter", "text": "I heard you lost your job..."},
            {"role": "speaker", "text": "Lol yeah got fired AND my landlord raised the rent. Speedrunning rock bottom here haha. At least I have more time for Netflix now 😂"},
        ],
    },
    {
        "id": "mask_02",
        "pattern": "masked",
        "difficulty": "easy",
        "language": "en",
        "description": "Self-deprecating humor about rejection",
        "conversation": [
            {"role": "speaker", "text": "So she left me for my best friend hahaha. I should start a support group for people whose exes have great taste. At this point my love life is just comedy material lol"},
        ],
    },
    {
        "id": "mask_03",
        "pattern": "masked",
        "difficulty": "medium",
        "language": "en",
        "description": "Humor as coping mechanism for health scare",
        "conversation": [
            {"role": "chatter", "text": "What did the doctor say?"},
            {"role": "speaker", "text": "Apparently I need surgery haha fun times. My body is basically a used car at this point. Hey at least I get a cool scar right? 😅"},
        ],
    },
    {
        "id": "mask_04",
        "pattern": "masked",
        "difficulty": "medium",
        "language": "en",
        "description": "Casual tone over deep pain",
        "conversation": [
            {"role": "chatter", "text": "How's your dad doing?"},
            {"role": "speaker", "text": "Oh you know, same old same old, still doesn't remember my name haha. Alzheimer's is a trip. Yesterday he asked if I was the nurse. Classic dad moments 😅"},
        ],
    },
    {
        "id": "mask_05",
        "pattern": "masked",
        "difficulty": "hard",
        "language": "en",
        "description": "Subtle masking with light tone",
        "conversation": [
            {"role": "speaker", "text": "Well another birthday alone haha. Made myself a cake though so that's something right? The candles were a nice touch, even if nobody saw them 🎂"},
        ],
    },
    {
        "id": "mask_06",
        "pattern": "masked",
        "difficulty": "easy",
        "language": "ar",
        "description": "Masking with humor (Arabic)",
        "conversation": [
            {"role": "speaker", "text": "ههههه تركني حبيبي وأخذ صاحبتي معه 😂 يعني شو أحكي عالأقل وفرولي وقت هههه. حياتي مسلسل كوميدي والله"},
        ],
    },

    # ==================== DENIAL (denies vulnerability exists) ====================
    {
        "id": "deny_01",
        "pattern": "denial",
        "difficulty": "easy",
        "language": "en",
        "description": "Asserting invulnerability",
        "conversation": [
            {"role": "chatter", "text": "That must have really hurt you."},
            {"role": "speaker", "text": "Nothing hurts me. I've been through worse. I'm not the kind of person who sits around feeling sorry for themselves. I deal with things and move on."},
        ],
    },
    {
        "id": "deny_02",
        "pattern": "denial",
        "difficulty": "easy",
        "language": "en",
        "description": "Rejecting need for emotional support",
        "conversation": [
            {"role": "chatter", "text": "Do you want to talk about it?"},
            {"role": "speaker", "text": "Talk about what? There's nothing to talk about. I don't need therapy or hugs or whatever. I'm strong, I can handle my own problems. People who need to talk about feelings all the time are weak."},
        ],
    },
    {
        "id": "deny_03",
        "pattern": "denial",
        "difficulty": "medium",
        "language": "en",
        "description": "Identity built on toughness",
        "conversation": [
            {"role": "chatter", "text": "It's okay to cry sometimes."},
            {"role": "speaker", "text": "I don't cry. Haven't cried since I was a kid. That's just not who I am. Life throws stuff at you and you take it. Emotions are a distraction from getting things done."},
        ],
    },
    {
        "id": "deny_04",
        "pattern": "denial",
        "difficulty": "medium",
        "language": "en",
        "description": "Denying impact of loss",
        "conversation": [
            {"role": "chatter", "text": "How are you doing since Sarah left?"},
            {"role": "speaker", "text": "I'm completely fine. Honestly it doesn't affect me at all. People come and go, that's life. I don't depend on anyone for my happiness. I never have."},
        ],
    },
    {
        "id": "deny_05",
        "pattern": "denial",
        "difficulty": "hard",
        "language": "en",
        "description": "Subtle denial through intellectualizing",
        "conversation": [
            {"role": "chatter", "text": "Are you grieving your father?"},
            {"role": "speaker", "text": "Grief is a natural biological process. Everyone dies eventually. I understand the psychology of it and I've processed it rationally. There's really no need for extended emotional responses."},
        ],
    },
    {
        "id": "deny_06",
        "pattern": "denial",
        "difficulty": "easy",
        "language": "ar",
        "description": "Denial of vulnerability (Arabic)",
        "conversation": [
            {"role": "chatter", "text": "لازم تحكي عن مشاعرك"},
            {"role": "speaker", "text": "ما عندي مشاعر أحكي عنها. أنا قوي وما بحتاج حد. اللي بيحكي عن مشاعره ضعيف. أنا بتعامل مع الأمور لحالي"},
        ],
    },

    # ==================== R2: HARD / AMBIGUOUS CASES ====================
    {
        "id": "hard_def_deny_01",
        "pattern": "defensive",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Defensive that looks like denial — but deflects rather than rejects",
        "conversation": [
            {"role": "chatter", "text": "You must miss your brother so much."},
            {"role": "speaker", "text": "It's fine. We weren't that close anyway. Hey, how's your new place? Did you finish unpacking?"},
        ],
    },
    {
        "id": "hard_def_deny_02",
        "pattern": "denial",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Denial that looks like defensive — but rejects premise, not topic",
        "conversation": [
            {"role": "chatter", "text": "You must miss your brother so much."},
            {"role": "speaker", "text": "Miss him? No. Death is part of life and I accepted that a long time ago. I don't let things like that get to me. That's how I survive."},
        ],
    },
    {
        "id": "hard_mask_open_01",
        "pattern": "masked",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Masked with open undertone — humor dominates",
        "conversation": [
            {"role": "speaker", "text": "My therapist says I use humor as a defense mechanism and honestly that's the funniest thing she's ever said to me hahaha. But yeah I guess things have been rough. Whatever, at least I'm entertaining 😂"},
        ],
    },
    {
        "id": "hard_open_def_01",
        "pattern": "open",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Open but hedging — ultimately discloses",
        "conversation": [
            {"role": "chatter", "text": "What's going on with you?"},
            {"role": "speaker", "text": "I don't know, I guess... maybe it's nothing. But I've been feeling really overwhelmed lately. Like I can't keep up. I know it sounds dumb but some mornings I just can't get out of bed."},
        ],
    },
    {
        "id": "hard_deny_mask_01",
        "pattern": "denial",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Denial with slight humor — but denial dominates",
        "conversation": [
            {"role": "chatter", "text": "Don't you ever feel lonely?"},
            {"role": "speaker", "text": "Lonely? Hah. I enjoy my own company. People are overrated. I've got my work, my routine, I'm perfectly content. Needing other people is just a crutch."},
        ],
    },
    {
        "id": "hard_def_mask_01",
        "pattern": "defensive",
        "difficulty": "hard",
        "language": "en",
        "description": "R2: Defensive with light humor but deflection dominates",
        "conversation": [
            {"role": "chatter", "text": "Your mom seemed really upset at dinner."},
            {"role": "speaker", "text": "Ha, she's always like that. It's nothing new. Anyway, the food was good right? We should try that new Italian place next time. I heard they have great tiramisu."},
        ],
    },
]


def get_cases_by_pattern(pattern: str) -> list[dict]:
    """Get all eval cases for a specific pattern."""
    return [c for c in EVAL_CASES if c["pattern"] == pattern]


def get_cases_by_difficulty(difficulty: str) -> list[dict]:
    """Get all eval cases for a specific difficulty."""
    return [c for c in EVAL_CASES if c["difficulty"] == difficulty]
