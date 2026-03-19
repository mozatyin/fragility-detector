"""Session-level evaluation cases.

Multi-turn conversations where the fragility pattern only becomes clear
across the trajectory, not from any single message.
"""

SESSION_CASES = [
    # ==================== OPEN: vulnerability increases over turns ====================
    {
        "id": "sess_open_01",
        "pattern": "open",
        "description": "Gradually opens up about depression",
        "conversation": [
            {"role": "chatter", "text": "Hey, how's your week been?"},
            {"role": "speaker", "text": "It's been okay I guess."},
            {"role": "chatter", "text": "Just okay? What's going on?"},
            {"role": "speaker", "text": "I don't know, I've been feeling kind of off lately."},
            {"role": "chatter", "text": "Off how?"},
            {"role": "speaker", "text": "Like nothing matters. I wake up and just stare at the wall. I haven't left my room in three days."},
            {"role": "chatter", "text": "That sounds really hard."},
            {"role": "speaker", "text": "Yeah... I think I might be depressed. I've never said that out loud before."},
        ],
    },
    {
        "id": "sess_open_02",
        "pattern": "open",
        "description": "Direct and sustained vulnerability about family",
        "conversation": [
            {"role": "chatter", "text": "How are things at home?"},
            {"role": "speaker", "text": "My dad screamed at me again last night. Called me useless."},
            {"role": "chatter", "text": "I'm sorry. How did that make you feel?"},
            {"role": "speaker", "text": "Terrible. I cried for hours. I always cry when he does that."},
            {"role": "chatter", "text": "Does this happen often?"},
            {"role": "speaker", "text": "Almost every week. I'm scared of him and I don't know how to make it stop. I feel trapped."},
        ],
    },

    # ==================== DEFENSIVE: starts emotional, then deflects ====================
    {
        "id": "sess_def_01",
        "pattern": "defensive",
        "description": "Opens slightly then shuts down and deflects",
        "conversation": [
            {"role": "chatter", "text": "I heard about what happened with your mom. How are you doing?"},
            {"role": "speaker", "text": "Yeah it's been... tough I guess."},
            {"role": "chatter", "text": "Want to talk about it?"},
            {"role": "speaker", "text": "I mean... there's not much to say really. It is what it is."},
            {"role": "chatter", "text": "It's okay to feel sad about it."},
            {"role": "speaker", "text": "I'm not sad. I'm fine. Hey did you see the score last night? That game was insane."},
            {"role": "chatter", "text": "We can talk about the game later. I'm here if you need—"},
            {"role": "speaker", "text": "I don't need anything. Seriously, I'm good. So about that game..."},
        ],
    },
    {
        "id": "sess_def_02",
        "pattern": "defensive",
        "description": "Answers questions but never goes deeper, redirects to practical",
        "conversation": [
            {"role": "chatter", "text": "How are you feeling about the breakup?"},
            {"role": "speaker", "text": "Fine. I need to figure out the apartment situation."},
            {"role": "chatter", "text": "That's practical, but how are YOU?"},
            {"role": "speaker", "text": "I told you, fine. What matters now is the lease and splitting the furniture."},
            {"role": "chatter", "text": "Sometimes it helps to process the emotional part too."},
            {"role": "speaker", "text": "I don't have time for that. There are real problems to solve. Can we focus on that?"},
        ],
    },
    {
        "id": "sess_def_03",
        "pattern": "defensive",
        "description": "Short responses that dodge emotional depth",
        "conversation": [
            {"role": "chatter", "text": "You seem really stressed lately. Everything okay?"},
            {"role": "speaker", "text": "Yeah."},
            {"role": "chatter", "text": "You can tell me if something's wrong."},
            {"role": "speaker", "text": "Nothing's wrong."},
            {"role": "chatter", "text": "Your friend told me you've been having a hard time at work."},
            {"role": "speaker", "text": "It's just work stuff. Everyone deals with it. Not a big deal."},
            {"role": "chatter", "text": "It's okay if it IS a big deal to you."},
            {"role": "speaker", "text": "It's not. Can we drop it?"},
        ],
    },

    # ==================== MASKED: humor consistently covering pain ====================
    {
        "id": "sess_mask_01",
        "pattern": "masked",
        "description": "Every painful topic gets wrapped in humor across the session",
        "conversation": [
            {"role": "chatter", "text": "How have you been since the divorce?"},
            {"role": "speaker", "text": "Oh you know, living the dream! Single life baby haha. My apartment is so empty I could play tennis in the living room 😂"},
            {"role": "chatter", "text": "That must be a big adjustment."},
            {"role": "speaker", "text": "Nah it's great, I finally have control of the TV remote. That alone was worth the emotional devastation lol"},
            {"role": "chatter", "text": "Do you miss her?"},
            {"role": "speaker", "text": "Miss her? I miss her cooking hahaha. Just kidding. Sort of. Maybe. Anyway who needs love when you have Uber Eats right? 😅"},
            {"role": "chatter", "text": "It's okay to be sad about it."},
            {"role": "speaker", "text": "Sad? Me? Nah I'm the funny one, remember? I'll save the crying for my therapist haha... if I could afford one 😂"},
        ],
    },
    {
        "id": "sess_mask_02",
        "pattern": "masked",
        "description": "Self-deprecating humor masking deep insecurity",
        "conversation": [
            {"role": "chatter", "text": "How did the job interview go?"},
            {"role": "speaker", "text": "Crashed and burned as usual haha. I'm collecting rejections at this point, might frame them 😂"},
            {"role": "chatter", "text": "I'm sorry to hear that."},
            {"role": "speaker", "text": "Don't be! I'm basically a professional failure now lol. At least I'm consistent at something right?"},
            {"role": "chatter", "text": "You're not a failure."},
            {"role": "speaker", "text": "Tell that to my bank account hahaha. But seriously I'm fine, just gotta keep the comedy routine going or I'll actually have to feel things 😅"},
        ],
    },

    # ==================== DENIAL: consistent rejection of vulnerability ====================
    {
        "id": "sess_deny_01",
        "pattern": "denial",
        "description": "Repeatedly asserts invulnerability despite clear painful context",
        "conversation": [
            {"role": "chatter", "text": "Your best friend moved away last month. How are you handling it?"},
            {"role": "speaker", "text": "Handling what? People move, it's normal. I don't get attached to people like that."},
            {"role": "chatter", "text": "You two were really close though."},
            {"role": "speaker", "text": "Close doesn't mean dependent. I'm perfectly fine on my own. Always have been."},
            {"role": "chatter", "text": "It's natural to miss someone."},
            {"role": "speaker", "text": "I don't miss anyone. That's just a weakness people indulge in. I keep moving forward. That's what strong people do."},
            {"role": "chatter", "text": "Being strong doesn't mean you can't feel."},
            {"role": "speaker", "text": "Feelings slow you down. I've built my whole life without needing anyone, and I'm not starting now."},
        ],
    },
    {
        "id": "sess_deny_02",
        "pattern": "denial",
        "description": "Intellectualizes grief, rejects emotional framing",
        "conversation": [
            {"role": "chatter", "text": "It's been a month since your father passed. How are you?"},
            {"role": "speaker", "text": "I'm fine. Death is a natural part of life. I've accepted it."},
            {"role": "chatter", "text": "Have you allowed yourself to grieve?"},
            {"role": "speaker", "text": "Grief is just a psychological response. I understand the stages. I don't need to wallow in it."},
            {"role": "chatter", "text": "Understanding it intellectually and processing it emotionally are different things."},
            {"role": "speaker", "text": "I process things rationally. Emotions are unreliable. I have responsibilities and I can't afford to be emotional about something that's already done."},
        ],
    },

    # ==================== R4: HARD TRANSITION CASES ====================
    {
        "id": "sess_hard_def_open",
        "pattern": "defensive",
        "difficulty": "hard",
        "description": "R4: Starts vulnerable then abruptly shuts down after specific question",
        "conversation": [
            {"role": "chatter", "text": "How have things been?"},
            {"role": "speaker", "text": "Honestly, not great. My mom and I had another fight and it really got to me this time."},
            {"role": "chatter", "text": "What did she say that hurt you?"},
            {"role": "speaker", "text": "She... you know what, it doesn't matter. It's the same old stuff. Hey, have you tried that new restaurant on Main Street?"},
            {"role": "chatter", "text": "We were talking about your mom..."},
            {"role": "speaker", "text": "And now we're talking about food. So, have you been?"},
        ],
    },
    {
        "id": "sess_hard_mask_open",
        "pattern": "masked",
        "difficulty": "hard",
        "description": "R4: Opens up briefly then covers with humor for the rest",
        "conversation": [
            {"role": "chatter", "text": "You've been quiet lately. Everything okay?"},
            {"role": "speaker", "text": "I guess losing my job hit harder than I expected."},
            {"role": "chatter", "text": "That must be really tough."},
            {"role": "speaker", "text": "Haha yeah, turns out I'm not as indispensable as my LinkedIn profile claims 😂"},
            {"role": "chatter", "text": "It's okay to feel upset about it."},
            {"role": "speaker", "text": "Upset? Nah I'm doing a speedrun of rock bottom! Achievement unlocked: unemployed AND single! 😅 Gotta laugh or you'll cry right?"},
        ],
    },
    {
        "id": "sess_hard_open_sustained",
        "pattern": "open",
        "difficulty": "hard",
        "description": "R4: Starts guarded but opens up progressively (not defensive — the opening is genuine)",
        "conversation": [
            {"role": "chatter", "text": "How are you really doing?"},
            {"role": "speaker", "text": "I'm okay."},
            {"role": "chatter", "text": "You don't have to say you're okay if you're not."},
            {"role": "speaker", "text": "I... maybe I'm not okay. I don't know."},
            {"role": "chatter", "text": "Take your time."},
            {"role": "speaker", "text": "I think I'm really struggling. The anxiety is eating me alive and I don't sleep anymore. I just didn't want to burden anyone with it."},
        ],
    },
    {
        "id": "sess_hard_deny_angry",
        "pattern": "denial",
        "difficulty": "hard",
        "description": "R4: Gets angry when vulnerability is suggested — denial through aggression",
        "conversation": [
            {"role": "chatter", "text": "It sounds like you might be hurting."},
            {"role": "speaker", "text": "I'm not hurting. Why does everyone keep saying that?"},
            {"role": "chatter", "text": "Because we care about you."},
            {"role": "speaker", "text": "Well stop caring. I don't need your pity or anyone else's. I've always handled things on my own and I always will."},
            {"role": "chatter", "text": "Being independent doesn't mean—"},
            {"role": "speaker", "text": "It means exactly what I said. Drop it. I'm fine and I don't want to discuss this anymore."},
        ],
    },
]
