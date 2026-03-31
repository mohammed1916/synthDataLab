"""
few_shot_examples.py — Gold-standard few-shot examples for each task type.

These examples are injected into prompts to demonstrate schema adherence,
output structure, and the level of detail expected.  Providing diverse
examples reduces hallucination and improves consistency.
"""
from __future__ import annotations

# Keys must match TaskType string values: "qa", "extraction", "reasoning"
FEW_SHOT_EXAMPLES: dict = {

    # ─────────────────────────────────────────────────────────────────────────
    "qa": [
        {
            "input": (
                "The human brain contains approximately 86 billion neurons. "
                "These neurons communicate via electrochemical signals across "
                "synaptic junctions.  The cerebral cortex, the outermost layer, "
                "is responsible for higher cognitive functions including language, "
                "reasoning, and consciousness."
            ),
            "output": {
                "question": "How many neurons does the human brain contain?",
                "answer": (
                    "The human brain contains approximately 86 billion neurons."
                ),
                "evidence": (
                    "The human brain contains approximately 86 billion neurons."
                ),
            },
        },
        {
            "input": (
                "Photosynthesis is the process by which plants convert sunlight "
                "into chemical energy stored as glucose.  The reaction requires "
                "carbon dioxide, water, and light energy, and releases oxygen "
                "as a by-product."
            ),
            "output": {
                "question": (
                    "What are the inputs and outputs of photosynthesis?"
                ),
                "answer": (
                    "Photosynthesis requires carbon dioxide, water, and light "
                    "energy as inputs, and produces glucose (stored chemical "
                    "energy) while releasing oxygen as a by-product."
                ),
                "evidence": (
                    "The reaction requires carbon dioxide, water, and light "
                    "energy, and releases oxygen as a by-product."
                ),
            },
        },
    ],

    # ─────────────────────────────────────────────────────────────────────────
    "extraction": [
        {
            "input": (
                "SpaceX, founded by Elon Musk in 2002, launched its Falcon 9 "
                "rocket from Cape Canaveral on May 30, 2020, carrying NASA "
                "astronauts Doug Hurley and Bob Behnken to the International "
                "Space Station."
            ),
            "output": {
                "entities": [
                    {"text": "SpaceX", "type": "ORGANIZATION"},
                    {"text": "Elon Musk", "type": "PERSON"},
                    {"text": "Falcon 9", "type": "PRODUCT"},
                    {"text": "Cape Canaveral", "type": "LOCATION"},
                    {"text": "NASA", "type": "ORGANIZATION"},
                    {"text": "Doug Hurley", "type": "PERSON"},
                    {"text": "Bob Behnken", "type": "PERSON"},
                    {"text": "International Space Station", "type": "FACILITY"},
                ],
                "relations": [
                    {
                        "subject": "SpaceX",
                        "predicate": "founded_by",
                        "object": "Elon Musk",
                    },
                    {
                        "subject": "Falcon 9",
                        "predicate": "launched_from",
                        "object": "Cape Canaveral",
                    },
                ],
                "key_facts": [
                    "SpaceX was founded in 2002.",
                    "The launch occurred on May 30, 2020.",
                    "The mission carried astronauts to the ISS.",
                ],
            },
        },
    ],

    # ─────────────────────────────────────────────────────────────────────────
    "reasoning": [
        {
            "input": (
                "A battery-electric vehicle (BEV) produces zero tailpipe "
                "emissions during operation.  However, the electricity used "
                "to charge BEVs may come from fossil-fuel power plants.  In "
                "regions with a high share of renewable energy in the grid, "
                "BEVs have significantly lower lifecycle carbon emissions than "
                "internal combustion engine (ICE) vehicles."
            ),
            "output": {
                "reasoning_steps": [
                    "Step 1 — Identify the claim: BEVs are cleaner than ICE vehicles.",
                    (
                        "Step 2 — Examine the caveat: BEV cleanliness depends on "
                        "electricity grid composition."
                    ),
                    (
                        "Step 3 — Apply conditional logic: If the grid is "
                        "renewable-heavy, BEV lifecycle emissions are lower; if "
                        "coal-heavy, the advantage diminishes."
                    ),
                    (
                        "Step 4 — Synthesise: The environmental benefit of BEVs is "
                        "context-dependent and increases as renewable penetration rises."
                    ),
                ],
                "conclusion": (
                    "BEVs are environmentally superior to ICE vehicles in regions "
                    "with high renewable energy shares, but the advantage varies "
                    "with grid carbon intensity."
                ),
                "confidence_explanation": (
                    "High confidence — the reasoning is directly supported by "
                    "explicit text statements without extrapolation."
                ),
            },
        },
    ],

    # ─────────────────────────────────────────────────────────────────────────
    "reasoning_trace": [
        {
            "input": (
                "Antibiotics are compounds that kill or inhibit bacteria. "
                "They are ineffective against viruses. Overuse of antibiotics "
                "leads to antibiotic-resistant bacteria, sometimes called "
                "'superbugs'. In 2019, WHO designated antimicrobial resistance "
                "as one of the top 10 global public health threats."
            ),
            "output": {
                "think": (
                    "<think>\n"
                    "Let me work through this carefully.\n\n"
                    "The question is about what conclusion we can draw from this passage.\n\n"
                    "Initial read: The passage says antibiotics kill bacteria but not viruses, "
                    "and overuse causes resistance. WHO flagged it as a top-10 global threat.\n\n"
                    "Wait — I should check: is the WHO statement from the passage or am I "
                    "adding external knowledge? Let me re-read... yes, '2019, WHO designated' "
                    "is explicitly in the passage. Good, I'm grounded.\n\n"
                    "Is there a logical conclusion here? The chain is:\n"
                    "  1. Antibiotics only work on bacteria.\n"
                    "  2. Overuse → resistance.\n"
                    "  3. WHO severity rating confirms this is a major issue.\n\n"
                    "I was initially going to say 'antibiotics are dangerous' but that's "
                    "too broad. The more precise conclusion is about the consequence of overuse. "
                    "Let me refine: the main conclusion is that antibiotic overuse is creating "
                    "a global public health crisis through resistant superbugs.\n\n"
                    "Verification: Does this follow from the text? Yes — directly stated.\n"
                    "</think>"
                ),
                "answer": (
                    "Antibiotic overuse drives the emergence of resistant bacteria (superbugs), "
                    "which the WHO identified as a top-10 global public health threat in 2019. "
                    "This makes responsible antibiotic stewardship critical."
                ),
                "verification": (
                    "Confirmed: both the mechanism (overuse → resistance) and the WHO severity "
                    "classification are explicitly stated in the passage."
                ),
                "confidence": 0.96,
            },
        },
    ],

    # ─────────────────────────────────────────────────────────────────────────
    "preference": [
        {
            "input": (
                "mRNA vaccines work by delivering messenger RNA into cells. "
                "The cell's ribosomes read the mRNA and produce the target antigen "
                "(e.g., the spike protein of SARS-CoV-2). The immune system then "
                "mounts a response, creating memory cells for future protection. "
                "The mRNA does not enter the cell nucleus and cannot alter DNA."
            ),
            "output": {
                "prompt": "How do mRNA vaccines work and can they change your DNA?",
                "chosen": {
                    "response": (
                        "mRNA vaccines deliver messenger RNA into your cells. Your ribosomes "
                        "use this mRNA as a blueprint to produce the target antigen — for "
                        "COVID-19 vaccines, that is the spike protein. Your immune system "
                        "recognises the antigen, mounts a response, and creates memory cells "
                        "for long-term protection. Crucially, the mRNA never enters the "
                        "cell nucleus and cannot interact with or alter your DNA."
                    ),
                    "quality_score": 0.95,
                },
                "rejected": {
                    "response": (
                        "mRNA vaccines basically reprogram your cells to fight viruses. "
                        "They inject genetic material that teaches your body to produce "
                        "antibodies. There is some debate about whether they can affect "
                        "your DNA, but scientists say the effect is minimal."
                    ),
                    "quality_score": 0.28,
                },
                "preference_margin": 0.67,
            },
        },
    ],
}
