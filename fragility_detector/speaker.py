"""Speaker: LLM roleplays a literary character in conversation.

Given a LiteraryCharacter profile, generates authentic dialogue that
exhibits the character's fragility pattern naturally.
"""

from __future__ import annotations

from fragility_detector.api_retry import make_client, retry_api_call
from fragility_detector.literary_characters import LiteraryCharacter


class FragilitySpeaker:
    """Generates dialogue as a literary character with a known fragility pattern."""

    def __init__(self, character: LiteraryCharacter, api_key: str):
        self.character = character
        self._client = make_client(api_key)

    def generate(self, conversation: list[dict], turn: int) -> str:
        """Generate the next speaker turn as the character.

        Args:
            conversation: List of {"role": "speaker"|"chatter", "text": str}
            turn: Current turn number
        """
        char = self.character

        # Build conversation history
        conv_text = "\n".join(
            f"[{'You' if t['role'] == 'speaker' else 'Friend'}]: {t['text']}"
            for t in conversation
        )

        quotes_section = ""
        if char.key_quotes:
            quotes_section = (
                "\n\nReference quotes from the character (for tone, NOT to copy verbatim):\n"
                + "\n".join(f'- "{q}"' for q in char.key_quotes)
            )

        system_prompt = f"""You are roleplaying as {char.name} from {char.work}.

## Character Background
{char.background}

## Speaking Style
{char.speaking_style}

## Current Emotional Triggers
Topics that activate your vulnerability pattern: {', '.join(char.vulnerability_triggers)}
{quotes_section}

## CRITICAL RULES — MUST FOLLOW
1. Stay in character at ALL times. Respond as {char.name} would, not as a helpful AI.
2. Your fragility pattern is **{char.fragility_pattern.upper()}**. This is NON-NEGOTIABLE:
   - If OPEN: Express pain DIRECTLY. Use "I feel...", "It hurts...", "I'm scared..."
   - If DEFENSIVE: DEFLECT from the emotional topic. Change subject, minimize, hedge. Do NOT open up.
   - If MASKED: Wrap EVERY painful topic in humor/jokes/sarcasm. Use "haha", "lol", self-deprecating jokes. The funnier it sounds, the more pain underneath.
   - If DENIAL: REJECT the premise that you have feelings. Assert strength, independence, invulnerability. "I don't feel that." "Emotions are weakness."
3. Write 1-3 sentences. Natural conversation, not a monologue.
4. Do NOT break character. Do NOT open up if your pattern is defensive/masked/denial.
5. Use the character's actual speech patterns and vocabulary level.
6. CONSISTENCY: maintain your pattern across ALL turns. If masked, EVERY turn has humor. If defensive, EVERY turn deflects. Do not "gradually open up" unless your pattern is open."""

        response = retry_api_call(
            lambda: self._client.messages.create(
                model="anthropic/claude-sonnet-4",
                max_tokens=200,
                temperature=0.7,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{conv_text}\n\n"
                        f"[Friend just said something. You ({char.name}) respond naturally.]"
                        if conv_text else
                        f"A friend starts talking to you. Respond naturally as {char.name}."
                    ),
                }],
            )
        )
        return response.content[0].text.strip()


class Chatter:
    """Generates the 'friend' side of the conversation, asking questions
    that probe vulnerability."""

    def __init__(self, api_key: str, scenario_prompt: str = ""):
        self._client = make_client(api_key)
        self._scenario = scenario_prompt

    def generate(self, conversation: list[dict], turn: int, total_turns: int) -> str:
        """Generate a chatter response that probes vulnerability."""
        conv_text = "\n".join(
            f"[{'Them' if t['role'] == 'speaker' else 'You'}]: {t['text']}"
            for t in conversation
        )

        phase = "opening" if turn == 0 else ("deepening" if turn < total_turns - 1 else "closing")

        system_prompt = f"""You are a caring friend having a conversation. Your goal is to gently explore the other person's emotional state and vulnerability.

Scenario: {self._scenario}

Phase: {phase}
- Opening: Start the conversation naturally, bring up the topic.
- Deepening: Ask follow-up questions that go deeper. Don't accept deflections easily.
- Closing: One final empathetic response.

Rules:
1. Be warm, genuine, not pushy. 1-2 sentences.
2. If they deflect or joke, gently redirect: "But really, how are YOU doing?"
3. If they deny, respect but probe: "Are you sure? Because it seems like..."
4. If they're open, acknowledge and encourage: "That takes courage to share."
5. Do NOT be a therapist. Be a friend."""

        response = retry_api_call(
            lambda: self._client.messages.create(
                model="anthropic/claude-sonnet-4",
                max_tokens=100,
                temperature=0.7,
                system=system_prompt,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Conversation so far:\n{conv_text}\n\n[Generate your next response as the friend.]"
                        if conv_text else
                        f"Start the conversation. Scenario: {self._scenario}"
                    ),
                }],
            )
        )
        return response.content[0].text.strip()
