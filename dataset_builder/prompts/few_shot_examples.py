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
}
