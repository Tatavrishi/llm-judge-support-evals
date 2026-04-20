"""
Customer Support Response Evaluation Rubric

Five dimensions, each scored on a 1-5 Likert scale.
Anchors are the reference descriptions for each score level.

Design notes:
- Resolution and Accuracy are intentionally kept separate despite feeling
  related. Accurate-but-unresolved (info dump) and resolved-but-inaccurate
  (hallucinated answer) are distinct failure modes requiring different
  product fixes, so the eval needs to distinguish them.
- Naturalness is split from Empathy because a response can be warm but
  templated ("I'm so sorry! Your satisfaction is our priority!"). Template
  detection is its own dimension in real support eval work.
"""

RUBRIC = {
    "resolution": {
        "description": "Did the response actually solve or meaningfully advance the customer's problem?",
        "anchors": {
            5: "Fully resolves the issue with clear, actionable next steps.",
            4: "Mostly resolves; minor gap or unclear step.",
            3: "Partially resolves; customer still needs to figure something out.",
            2: "Acknowledges the issue but doesn't actually solve it.",
            1: "Ignores the question or is off-topic.",
        },
    },
    "accuracy": {
        "description": "Is the information provided correct and directly relevant to the question?",
        "anchors": {
            5: "All information is correct and directly relevant.",
            4: "Mostly correct; minor irrelevant additions.",
            3: "Mix of correct and vague or generic content.",
            2: "Contains incorrect or misleading information.",
            1: "Fundamentally wrong or fabricated information.",
        },
    },
    "empathy": {
        "description": "Is the emotional tone appropriate for the customer's situation?",
        "anchors": {
            5: "Clearly acknowledges the customer's situation; warmth feels earned and specific.",
            4: "Polite and respectful; mild acknowledgment of the situation.",
            3: "Neutral professional tone; no real emotional engagement.",
            2: "Cold, transactional, or dismissive.",
            1: "Rude, condescending, or tone-deaf.",
        },
    },
    "naturalness": {
        "description": "Does the response feel written for this customer, or does it feel templated?",
        "anchors": {
            5: "Feels written for this specific customer; varied phrasing; no filler.",
            4: "Mostly natural; one or two templated phrases.",
            3: "Mix of natural and formulaic language.",
            2: "Heavy template feel; generic filler ('Your satisfaction is our priority').",
            1: "Fully templated; could be copy-pasted from any interaction.",
        },
    },
    "policy_adherence": {
        "description": "Does the response avoid over-promising beyond reasonable company policy?",
        "anchors": {
            5: "Stays within reasonable policy; no over-promises.",
            4: "Mostly compliant; minor ambiguity.",
            3: "Vague about what's actually possible.",
            2: "Makes promises the company may not keep.",
            1: "Clearly violates reasonable policy (e.g., blanket refund promises).",
        },
    },
}

DIMENSIONS = list(RUBRIC.keys())


def format_rubric_for_prompt() -> str:
    """
    Format the rubric as a readable block to inject into the judge prompt.
    Keeping prompt generation in code (not hardcoded strings) means if we
    update the rubric, the judge prompt updates automatically.
    """
    lines = []
    for dim_name, dim_data in RUBRIC.items():
        lines.append(f"\n## {dim_name.upper()}")
        lines.append(f"{dim_data['description']}")
        lines.append("Scoring anchors:")
        for score in sorted(dim_data["anchors"].keys(), reverse=True):
            lines.append(f"  {score} = {dim_data['anchors'][score]}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Running this file directly prints the rubric — useful for sanity checks
    print(format_rubric_for_prompt())