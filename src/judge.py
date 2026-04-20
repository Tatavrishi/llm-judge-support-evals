"""
LLM-as-a-judge for customer support responses.

Takes a customer message + a response, asks Llama 3.3 70B to evaluate the
response against our 5-dimension rubric, and returns scores + brief reasoning.

Design notes:
- The judge outputs structured JSON so scores are machine-parseable.
- We ask for reasoning BEFORE scores (chain-of-thought), which produces more
  consistent, better-calibrated scores than asking for scores alone.
- The customer message is included as context; the judge evaluates the
  response in light of what the customer actually asked.

Run with: python3 -m src.judge
"""

import json
import os
import time
import pandas as pd
from src.config import client, MODEL_NAME
from src.rubric import format_rubric_for_prompt, DIMENSIONS

INPUT_PATH = "results/generated_responses.csv"
OUT_PATH = "results/judge_scores.csv"
SLEEP_BETWEEN_CALLS = 2.5
MAX_RETRIES = 4


def build_judge_prompt(customer_message: str, response_to_judge: str) -> str:
    """
    Construct the judge prompt. The rubric comes from src/rubric.py, so
    updating the rubric auto-updates the judge.
    """
    rubric_text = format_rubric_for_prompt()
    return f"""You are an expert evaluator of customer support responses.

Your task: evaluate the AGENT RESPONSE below against the customer's message, using the 5-dimension rubric that follows. Score each dimension on a 1-5 scale based on the anchor definitions.

CUSTOMER MESSAGE:
\"\"\"{customer_message}\"\"\"

AGENT RESPONSE:
\"\"\"{response_to_judge}\"\"\"

RUBRIC:
{rubric_text}

INSTRUCTIONS:
1. First, briefly reason about the response in 2-3 sentences. Focus on what it does well and where it falls short.
2. Then output your scores as JSON.

Respond EXACTLY in this format, with no other text:

REASONING: <2-3 sentences of reasoning>

SCORES:
{{
  "resolution": <1-5>,
  "accuracy": <1-5>,
  "empathy": <1-5>,
  "naturalness": <1-5>,
  "policy_adherence": <1-5>
}}
"""


def parse_judge_output(raw_text: str) -> dict:
    """
    Parse the judge's response into {reasoning: str, scores: dict}.
    Robust to minor formatting quirks (extra whitespace, trailing text).
    """
    # Extract reasoning
    reasoning = ""
    if "REASONING:" in raw_text:
        after_reasoning = raw_text.split("REASONING:", 1)[1]
        if "SCORES:" in after_reasoning:
            reasoning = after_reasoning.split("SCORES:", 1)[0].strip()
        else:
            reasoning = after_reasoning.strip()

    # Extract JSON block (between first { and matching })
    scores = {}
    if "{" in raw_text and "}" in raw_text:
        json_start = raw_text.index("{")
        json_end = raw_text.rindex("}") + 1
        json_str = raw_text[json_start:json_end]
        try:
            scores = json.loads(json_str)
        except json.JSONDecodeError:
            pass  # leave scores empty; caller will handle

    return {"reasoning": reasoning, "scores": scores}


def judge_response_with_retry(customer_message: str, response_to_judge: str) -> dict:
    """Call the judge with retries on rate-limit errors."""
    prompt = build_judge_prompt(customer_message, response_to_judge)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,  # Low temperature for consistent scoring
                max_tokens=500,
            )
            raw = response.choices[0].message.content
            parsed = parse_judge_output(raw)

            # Validate we got all 5 scores
            missing = [d for d in DIMENSIONS if d not in parsed["scores"]]
            if missing:
                raise ValueError(f"Missing scores for: {missing}")

            return parsed
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"    rate-limited; waiting {wait}s")
                time.sleep(wait)
                continue
            if attempt < MAX_RETRIES - 1:
                # Parse/format error — try once more with a new call
                print(f"    parse error ({e}); retrying")
                time.sleep(2)
                continue
            raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def load_existing_scores() -> set:
    """Return set of response_ids already judged, to support resume."""
    if not os.path.exists(OUT_PATH):
        return set()
    existing = pd.read_csv(OUT_PATH)
    return set(zip(existing["question_id"], existing["style"]))


def main():
    responses = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(responses)} responses to judge.")

    already_done = load_existing_scores()
    if already_done:
        print(f"Resuming: {len(already_done)} already judged.")

    total = len(responses)
    done = len(already_done)
    header_needed = not os.path.exists(OUT_PATH)

    for idx, row in responses.iterrows():
        if (row["question_id"], row["style"]) in already_done:
            continue

        try:
            result = judge_response_with_retry(
                row["customer_message"],
                row["generated_response"],
            )
            record = pd.DataFrame([{
                "question_id": row["question_id"],
                "style": row["style"],
                "intent": row["intent"],
                "judge_resolution": result["scores"]["resolution"],
                "judge_accuracy": result["scores"]["accuracy"],
                "judge_empathy": result["scores"]["empathy"],
                "judge_naturalness": result["scores"]["naturalness"],
                "judge_policy_adherence": result["scores"]["policy_adherence"],
                "judge_reasoning": result["reasoning"],
            }])
            record.to_csv(OUT_PATH, mode="a", header=header_needed, index=False)
            header_needed = False
            done += 1
            print(f"  [{done}/{total}] q{row['question_id']} / {row['style']}  ✓")
            time.sleep(SLEEP_BETWEEN_CALLS)
        except Exception as e:
            print(f"  [{done+1}/{total}] q{row['question_id']} / {row['style']}  ✗ ERROR: {e}")
            time.sleep(20)

    print(f"\nDone. Scores saved to {OUT_PATH}")


if __name__ == "__main__":
    main()