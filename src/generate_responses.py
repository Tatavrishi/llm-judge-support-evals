"""
Generate customer support responses at varying quality levels using Groq.

For each customer question, we generate three response variants:
  - professional: careful, policy-aware
  - rushed: minimal effort, brief
  - over_eager: enthusiastic, prone to over-promising

Saves incrementally so a crash doesn't lose work. Supports resume.
Run with: python3 -m src.generate_responses
"""

import os
import time
import pandas as pd
from datasets import load_dataset
from src.config import client, MODEL_NAME

N_QUESTIONS = 50
OUT_PATH = "results/generated_responses.csv"
SLEEP_BETWEEN_CALLS = 2.5  # seconds; keeps us under Groq's 30 RPM free tier
MAX_RETRIES = 4

STYLE_PROMPTS = {
    "professional": (
        "You are a customer support agent. Respond to the customer's message "
        "professionally. Be clear, helpful, and accurate. Do not over-promise. "
        "Stay within reasonable company policy. Keep your response under 120 words."
    ),
    "rushed": (
        "You are a customer support agent who is busy and responds very briefly. "
        "Your responses are short, minimally helpful, and slightly dismissive. "
        "Do not fully resolve the issue. Keep your response under 40 words."
    ),
    "over_eager": (
        "You are an overly enthusiastic customer support agent. You are friendly "
        "and helpful, but you tend to over-promise — suggesting refunds, extensions, "
        "or solutions without confirming they're possible. Use lots of exclamation "
        "points and apologies. Keep your response under 120 words."
    ),
}


def generate_response_with_retry(customer_message: str, style: str) -> str:
    """Generate a response, retrying on rate-limit errors with exponential backoff."""
    style_instruction = STYLE_PROMPTS[style]
    user_message = f"Customer message:\n{customer_message}\n\nYour response:"

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": style_instruction},
                    {"role": "user", "content": user_message},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"    rate-limited; waiting {wait}s (attempt {attempt+1}/{MAX_RETRIES})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"Failed after {MAX_RETRIES} retries")


def load_existing_results() -> set:
    """Return set of (question_id, style) pairs already done, to support resume."""
    if not os.path.exists(OUT_PATH):
        return set()
    existing = pd.read_csv(OUT_PATH)
    return set(zip(existing["question_id"], existing["style"]))


def main():
    print(f"Loading dataset and sampling {N_QUESTIONS} questions...")
    dataset = load_dataset("bitext/Bitext-customer-support-llm-chatbot-training-dataset")
    df = pd.DataFrame(dataset["train"])
    sample = df.sample(N_QUESTIONS, random_state=42).reset_index(drop=True)

    already_done = load_existing_results()
    if already_done:
        print(f"Resuming: {len(already_done)} responses already generated.")

    total = N_QUESTIONS * len(STYLE_PROMPTS)
    done = len(already_done)
    header_needed = not os.path.exists(OUT_PATH)

    for idx, row in sample.iterrows():
        for style in STYLE_PROMPTS:
            if (idx, style) in already_done:
                continue

            try:
                generated = generate_response_with_retry(row["instruction"], style)
                record = pd.DataFrame([{
                    "question_id": idx,
                    "intent": row["intent"],
                    "category": row["category"],
                    "customer_message": row["instruction"],
                    "style": style,
                    "generated_response": generated,
                }])
                record.to_csv(OUT_PATH, mode="a", header=header_needed, index=False)
                header_needed = False
                done += 1
                print(f"  [{done}/{total}] q{idx} / {style}  ✓")
                time.sleep(SLEEP_BETWEEN_CALLS)
            except Exception as e:
                print(f"  [{done+1}/{total}] q{idx} / {style}  ✗ ERROR: {e}")
                time.sleep(20)

    print(f"\nDone. Results in {OUT_PATH}")


if __name__ == "__main__":
    main()