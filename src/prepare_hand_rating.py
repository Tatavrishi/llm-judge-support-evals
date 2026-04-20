"""
Prepare a sample of 25 responses for blind hand-rating.

Crucial design choice: the judge's scores are NOT included in the rating sheet.
If the rater sees the judge's scores, they'll anchor on them and the human
ratings will no longer be independent ground truth.

Output: results/hand_rating_sheet.csv
  - Has columns: response_id, customer_message, generated_response, + 5 empty score columns
  - 25 responses, stratified across the 3 styles (roughly 8-9 each)

Run with: python3 -m src.prepare_hand_rating
"""

import pandas as pd

RESPONSES_PATH = "results/generated_responses.csv"
SCORES_PATH = "results/judge_scores.csv"
OUT_PATH = "results/hand_rating_sheet.csv"
N_TO_RATE = 25


def main():
    responses = pd.read_csv(RESPONSES_PATH)
    judged = pd.read_csv(SCORES_PATH)

    # Only keep responses that were also judged (so agreement analysis works later)
    merged = responses.merge(
        judged[["question_id", "style"]],
        on=["question_id", "style"],
        how="inner",
    )
    print(f"Responses both generated AND judged: {len(merged)}")

    # Stratified sample: ~equal from each style
    per_style = N_TO_RATE // 3 + 1  # 9 each = 27, we'll trim to 25
    samples = []
    for style in ["professional", "rushed", "over_eager"]:
        style_df = merged[merged["style"] == style]
        n = min(per_style, len(style_df))
        samples.append(style_df.sample(n, random_state=7))
    sample = pd.concat(samples).sample(N_TO_RATE, random_state=7).reset_index(drop=True)

    # Build the rating sheet with empty score columns
    sheet = pd.DataFrame({
        "rating_id": range(len(sample)),
        "question_id": sample["question_id"],
        "style": sample["style"],   # kept visible so we can analyze by style later
        "intent": sample["intent"],
        "customer_message": sample["customer_message"],
        "generated_response": sample["generated_response"],
        "human_resolution": "",
        "human_accuracy": "",
        "human_empathy": "",
        "human_naturalness": "",
        "human_policy_adherence": "",
        "human_notes": "",
    })

    sheet.to_csv(OUT_PATH, index=False)
    print(f"\nSaved {len(sheet)} responses for hand-rating to {OUT_PATH}")
    print(f"Distribution across styles:")
    print(sheet["style"].value_counts())


if __name__ == "__main__":
    main()