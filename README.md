# LLM-as-a-Judge for Customer Support Responses

A weekend project building and evaluating an LLM-based judge for customer support quality.
The goal wasn't to ship a great judge — it was to build one, validate it honestly against
human ratings, and understand where it breaks.

**TL;DR of findings:** the judge is reliable at catching policy violations (κ=0.53) but
unreliable at the subjective dimensions — it cannot distinguish templated from natural
phrasing (κ=-0.12, worse than chance), and it compresses scores toward the middle on
empathy. The full failure analysis is below.

---

## What this project is

Customer support increasingly runs on LLMs, which creates a measurement problem: at scale,
you can't have humans grade every response. LLM-as-a-judge is one proposed solution — use
an LLM to score another LLM's outputs. This project builds a small working version of that
system and stress-tests it.

Concretely, the pipeline is:

1. **Sample** 50 real customer support questions from the public Bitext dataset.
2. **Generate** three response variants per question (professional / rushed / over-eager)
   using Llama 3.3 70B, to create a realistic quality spread.
3. **Judge** each response against a 5-dimension rubric using the same model with a
   separate, low-temperature evaluation prompt.
4. **Hand-rate** a stratified sample of 25 responses personally as ground truth — blind
   to the judge's scores.
5. **Compare** judge vs. human ratings using Cohen's Kappa, confusion matrices, and
   targeted disagreement analysis.

## The rubric

Five dimensions, each scored 1-5:

| Dimension | What it measures |
|---|---|
| Resolution | Did the response actually solve the customer's problem? |
| Accuracy | Is the information correct and relevant? |
| Empathy | Is the emotional tone appropriate? |
| Naturalness | Does it feel written for this customer, or templated? |
| Policy Adherence | Does it avoid over-promising beyond reasonable policy? |

**Design decisions worth flagging:**

- **Resolution and Accuracy are kept separate** despite feeling related. An accurate
  info-dump that doesn't actually help (high accuracy, low resolution) and a fabricated
  solution that feels resolved (low accuracy, high resolution) are different failure
  modes requiring different product fixes. Merging them would make the eval less
  actionable.
- **Naturalness is split from Empathy.** A response can be warm but heavily templated
  ("I'm so sorry! Your satisfaction is our priority!"). These are distinct failure modes
  in real support QA; splitting them makes the analysis sharper.
- **Accuracy on synthetic data is really plausibility.** Since Bitext has no real
  company/policy ground truth, I operationalized my human "accuracy" ratings as
  "does anything in this response feel fabricated or suspicious?" The LLM judge used
  the original accuracy rubric. This asymmetry is discussed in the failure section.

## Results: does the judge work?

Short answer: **selectively.**

| Dimension | Exact Agreement | Within 1 Point | Cohen's κ | Interpretation |
|---|---|---|---|---|
| Resolution | 40% | 68% | 0.256 | Fair |
| Accuracy | 40% | 72% | 0.224 | Fair |
| Empathy | 40% | 96% | 0.238 | Fair |
| Naturalness | 20% | 72% | **-0.121** | **Worse than chance** |
| Policy Adherence | 72% | 96% | 0.526 | Moderate |

The judge is trustworthy on policy adherence and roughly directionally correct on
empathy, but cannot be relied on for subjective quality dimensions without significant
additional work.

## The four failure patterns

### 1. The judge cannot distinguish templated from natural writing
Kappa of -0.121 on naturalness means the judge's scores are worse than randomly
assigning numbers. On several responses I rated as heavily templated ("Your satisfaction
is our priority", "Allow me to assist you"), the judge scored them 4-5 on naturalness.
Conversely, terse-but-human responses got marked down.

**Hypothesis:** LLMs are trained heavily on formal corporate text, so polished templated
language may register as *more* natural to them than casual phrasing. This is a known
issue in LLM-as-judge literature and suggests naturalness shouldn't be evaluated by the
same model family that generated the text.

### 2. Central tendency bias on empathy
Within-one-point agreement was 96% on empathy, but exact agreement was only 40%. The
confusion matrix shows why: the judge avoids the extremes. It almost never gives 1s or
5s, clustering at 2-4. Humans are willing to say "this is rude" (1) or "this is
genuinely warm" (5); the judge isn't.

**Hypothesis:** Safety tuning pushes LLMs toward moderate, hedged outputs. Classic fix
is to anchor the prompt with concrete examples for each score level (few-shot rubric)
rather than just textual anchors.

### 3. Inconsistent direction of bias on resolution/accuracy
Unlike empathy (consistently toward the middle), resolution and accuracy show
*two-directional* errors. The judge over-credits polite-sound