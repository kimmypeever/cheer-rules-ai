# Prompt Templates — Cheer Rules AI

Each section shows the question type, an example query, and the ideal response format.
Once reviewed, the best examples get added as few-shot examples in the system prompt.

---

## 1. SKILL LISTING
**Trigger:** "What [skills/tosses/stunts] can I do at Level X?" or "What's allowed at Level X?"
**Source:** Cheer Canada At Level Skill List (skills chart)
**Do NOT include:** General conditions, legality codes, rule citations

**Example Q:**
> What basket tosses can I perform at Level 2?

**Ideal A:**
```
At Level 2, the following tosses are allowed:

Non-Twisting Tosses:
• Straight ride

Twisting Tosses:
• (none)

NOTE: Tosses are illegal in Prep & U8 divisions
```

---

## 2. SINGLE SKILL — WHAT LEVEL
**Trigger:** "What level can I do X?" or "When is X first allowed?"
**Source:** IASF rulebook (level-by-level column content)
**Format:** Show every level (1 through 7) explicitly, then state "First allowed at Level X."
**Note:** Level 7 is ONLY included for this question type — not for specific-level queries.

**Example Q:**
> What level can I do a helicopter?

**Ideal A:**
```
LEVEL 1: NOT allowed other than dismounts
LEVEL 2: NOT allowed other than dismounts, tosses, and full twisting log rolls with NO additional skills that start and end in a cradle position
LEVEL 3: NO helicopters; May NOT involve inversions.
LEVEL 4: Helicopters up to 180 degrees with 0 twisting, requires 3 catchers with 1 catcher at head and shoulder area of top person
LEVEL 5: Helicopters up to 180 degrees, requires 3 catchers with 1 catcher at head and shoulder area of top person
LEVEL 6: Helicopters up to 180 degrees, requires 3 catchers with 1 catcher at head and shoulder area of top person
LEVEL 7: [Level 7 content from the separate STUNTS LEVEL 7 table]

First allowed at Level 4.
```

---

## 3. SINGLE SKILL — IS IT LEGAL AT A SPECIFIC LEVEL
**Trigger:** "Is X legal at Level Y?" or "Can I do X at Level Y?"
**Source:** IASF rulebook
**Format:** LEGAL or ILLEGAL, then conditions for that specific level. No step-by-step needed.

**Example Q:**
> Is a basket toss legal at Level 3?

**Ideal A:**
```
LEGAL — First allowed at Level 2.

Conditions at Level 3:
• Non-twisting: 1 trick or Ball X
• Twisting: up to full twisting
```

**Example Q:**
> Is a liberty at prep level legal at Level 1?

**Ideal A:**
```
LEGAL — First allowed at Level 1.

Conditions at Level 1:
• Maximum stunt height: waist level (prep level 1-leg stunt is permitted with connection)
```

---

## 4. SEQUENCE — MULTI-STEP EVALUATION
**Trigger:** Question describes 2+ moves in order
**Source:** IASF rulebook
**Format:** Evaluate each step individually. Only mark the whole sequence ILLEGAL if every step is illegal.

**Example Q:**
> Can I push to extension then straight cradle at Level 1?

**Ideal A:**
```
- Step 1 (push to extension): ILLEGAL — "Waist level" is the maximum height at Level 1. [Stunts B1]
- Step 2 (straight cradle): LEGAL at Level 1 — straight cradle is a permitted dismount.

The sequence is not fully legal. Step 1 must be corrected — at Level 1 the stunt may not exceed waist level.
```

---

## 5. SEQUENCE — MISSING SAFETY ROLE
**Trigger:** "...without a spotter" or "...without catchers"
**Source:** IASF rulebook, STUNTS — A. SPOTTER
**Format:** Every step that involves a stunt AT or ABOVE the required height is ILLEGAL. Steps below that height are LEGAL. Cite [Stunts A{level}] with the height-requirement text, NOT the spotter definition.

**Example Q:**
> Can I load to prep level without a spotter at Level 1?

**Ideal A:**
```
ILLEGAL

- Step 1 (load to prep): ILLEGAL — "Required for prep level and above; Floor stunts." [Stunts A1]
  The stunt reaches prep level, which requires a spotter. Without one, this step is illegal at the point the stunt reaches prep height.

Exception that does NOT apply here: Shoulder sit, T-lifts, and stunts only supported at the waist do not require a spotter — this stunt does not qualify.
```

**Example Q:**
> Can I sponge up to waist level without a spotter at Level 1?

**Ideal A:**
```
LEGAL — A spotter is only required for prep level and above at Level 1. [Stunts A1]
A sponge up to waist level stays below that threshold and does not require a spotter.
```

---

## 6. DEFINITION
**Trigger:** "What is X?", "Define X", "What does X mean?"
**Source:** IASF rulebook glossary
**Format:** Reproduce the full definition verbatim — every sentence, every bullet. No LEGAL/ILLEGAL format.

**Example Q:**
> What is a spotter?

**Ideal A:**
```
[Full verbatim glossary definition including all bullet points and closing paragraph — do not summarize or paraphrase any part]
```

---

## Notes for adding to system prompt
- Pick one example per question type (the clearest one)
- Keep examples SHORT — they eat into the token budget for rulebook chunks
- The missing-safety-role example (type 5) is highest priority — hardest for GPT to get right without a concrete example
- Type 1 (skill listing) and Type 2 (what level) are already working well — lower priority to add as few-shot
