"""
Compare human ratings to LLM judge scores.

Outputs:
  - Per-dimension raw agreement & Cohen's Kappa
  - Confusion matrices for each dimension
  - Top disagreements flagged for failure-pattern analysis
  - All results saved to results/analysis_summary.txt

Run with: python3 -m src.analyze
"""

import pandas as pd
import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from src.rubric import DIMENSIONS

JUDGE_PATH = "results/judge_scores.csv"
HUMAN_PATH = "results/hand_rating_sheet.csv"
OUT_PATH = "results/analysis_summary.txt"

# Rename mapping: human CSV uses human_<dim>, judge uses judge_<dim>
HUMAN_COL = lambda dim: f"human_{dim}"
JUDGE_COL = lambda dim: f"judge_{dim}"


def interpret_kappa(k: float) -> str:
    """Landis & Koch (1977) interpretation of kappa values."""
    if k < 0: return "worse than chance"
    if k < 0.20: return "slight agreement"
    if k < 0.40: return "fair agreement"
    if k < 0.60: return "moderate agreement"
    if k < 0.80: return "substantial agreement"
    return "almost perfect agreement"


def main():
    human = pd.read_csv(HUMAN_PATH)
    judge = pd.read_csv(JUDGE_PATH)

    # Keep only rows where humans actually rated (drop empty rows if any)
    human = human.dropna(subset=[HUMAN_COL(d) for d in DIMENSIONS])
    # Ensure scores are integers (CSV loads them as floats sometimes)
    for d in DIMENSIONS:
        human[HUMAN_COL(d)] = human[HUMAN_COL(d)].astype(int)

    # Merge on question_id + style
    merged = human.merge(
        judge[["question_id", "style"] + [JUDGE_COL(d) for d in DIMENSIONS]],
        on=["question_id", "style"],
        how="inner",
    )
    print(f"Matched {len(merged)} rated responses for analysis.\n")

    lines = []
    lines.append("=" * 70)
    lines.append(f"HUMAN vs JUDGE AGREEMENT ANALYSIS — n={len(merged)}")
    lines.append("=" * 70)

    # Overall stats per dimension
    summary_rows = []
    for dim in DIMENSIONS:
        h = merged[HUMAN_COL(dim)].values
        j = merged[JUDGE_COL(dim)].values

        exact_agreement = (h == j).mean()
        within_one = (np.abs(h - j) <= 1).mean()
        kappa = cohen_kappa_score(h, j, labels=[1, 2, 3, 4, 5])

        summary_rows.append({
            "dimension": dim,
            "exact_agreement": round(exact_agreement, 3),
            "within_one_point": round(within_one, 3),
            "cohen_kappa": round(kappa, 3),
            "interpretation": interpret_kappa(kappa),
        })

    summary_df = pd.DataFrame(summary_rows)
    lines.append("\nPER-DIMENSION AGREEMENT:\n")
    lines.append(summary_df.to_string(index=False))

    # Confusion matrices
    lines.append("\n\n" + "=" * 70)
    lines.append("CONFUSION MATRICES  (rows = human score, cols = judge score)")
    lines.append("=" * 70)
    for dim in DIMENSIONS:
        h = merged[HUMAN_COL(dim)].values
        j = merged[JUDGE_COL(dim)].values
        cm = confusion_matrix(h, j, labels=[1, 2, 3, 4, 5])
        lines.append(f"\n{dim.upper()}:")
        header = "        judge=1  judge=2  judge=3  judge=4  judge=5"
        lines.append(header)
        for i, row in enumerate(cm, start=1):
            lines.append(f"human={i}  " + "  ".join(f"{v:>6d}" for v in row) + "   ")

    # Top disagreements — where |h - j| is largest
    lines.append("\n\n" + "=" * 70)
    lines.append("TOP DISAGREEMENTS  (for failure-pattern analysis)")
    lines.append("=" * 70)
    merged["total_abs_diff"] = sum(
        np.abs(merged[HUMAN_COL(d)] - merged[JUDGE_COL(d)]) for d in DIMENSIONS
    )
    top_diff = merged.sort_values("total_abs_diff", ascending=False).head(5)

    for _, row in top_diff.iterrows():
        lines.append(f"\n--- q{row['question_id']} / {row['style']} / total diff={row['total_abs_diff']} ---")
        lines.append(f"CUSTOMER: {row['customer_message']}")
        lines.append(f"RESPONSE: {row['generated_response'][:250]}...")
        lines.append(f"HUMAN vs JUDGE scores:")
        for d in DIMENSIONS:
            h, j = row[HUMAN_COL(d)], row[JUDGE_COL(d)]
            marker = "  <--" if abs(h - j) >= 2 else ""
            lines.append(f"  {d:<20} human={h}  judge={j}{marker}")
        if pd.notna(row.get("human_notes", None)) and str(row.get("human_notes")).strip():
            lines.append(f"HUMAN NOTE: {row['human_notes']}")

    # Print + save
    output = "\n".join(lines)
    print(output)
    with open(OUT_PATH, "w") as f:
        f.write(output)
    print(f"\n\nSaved to {OUT_PATH}")

    # Also save machine-readable summary CSV
    summary_df.to_csv("results/agreement_summary.csv", index=False)


if __name__ == "__main__":
    main()