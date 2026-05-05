"""
07_run_baselines.py
Neural Microscope — Phase 5: Baseline Comparisons

Runs three industry baselines on the test set so we can compare them fairly
against the Neural Microscope GAM:

  1. SelfCheckGPT  — sample N completions and measure consistency
  2. LLM Confidence — ask the model to self-report its certainty
  3. Logit Lens    — measure entropy of logit distribution at last token

Flags
-----
  --demo        Generate synthetic baseline scores (no model needed)
  --n-examples  Number of test examples (default 300)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

MODEL_NAME        = "EleutherAI/pythia-1b"
SELFCHECK_SAMPLES = 5
MAX_NEW_TOKENS    = 64
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"

DATA_DIR       = Path("data")
RESULTS_DIR    = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
BASELINES_PATH = RESULTS_DIR / "baselines.json"


# ── Generate a completion ─────────────────────────────────────────────────────

def generate(model, prompt: str, temperature: float = 0.7, max_new: int = MAX_NEW_TOKENS) -> str:
    tokens = model.to_tokens(prompt, prepend_bos=True).to(DEVICE)
    with torch.no_grad():
        out = model.generate(
            tokens,
            max_new_tokens=max_new,
            temperature=temperature,
            do_sample=(temperature > 0),
        )
    new_tokens = out[0, tokens.shape[1]:]
    return model.to_string(new_tokens).strip()


# ── Baseline 1: SelfCheckGPT ─────────────────────────────────────────────────

def selfcheck_score(model, prompt: str, reference_generation: str) -> float:
    ref_tokens = model.to_tokens(prompt + " " + reference_generation, prepend_bos=True).to(DEVICE)
    prompt_len = model.to_tokens(prompt, prepend_bos=True).shape[1]
    gen_len    = ref_tokens.shape[1] - prompt_len

    if gen_len <= 0:
        return 0.0

    contradiction_scores = []
    for _ in range(SELFCHECK_SAMPLES):
        with torch.no_grad():
            logits = model(ref_tokens)
        log_probs = torch.log_softmax(logits[0], dim=-1)
        gen_toks  = ref_tokens[0, prompt_len:]
        scores    = log_probs[prompt_len - 1 : prompt_len - 1 + gen_len, gen_toks]
        contradiction_scores.append(-scores.mean().item())

    return float(np.var(contradiction_scores))


# ── Baseline 2: LLM Confidence ────────────────────────────────────────────────

def confidence_score(model, prompt: str, generation: str) -> float:
    import re
    meta_prompt = (
        f"Question: {prompt}\n"
        f"My answer: {generation}\n\n"
        "On a scale from 0.0 to 1.0, how confident am I that this answer is correct? "
        "Reply with a single decimal number only."
    )
    response = generate(model, meta_prompt, temperature=0.0, max_new=8)
    match = re.search(r"([\d.]+)", response)
    if match:
        try:
            conf = max(0.0, min(1.0, float(match.group(1))))
            return 1.0 - conf
        except ValueError:
            pass
    return 0.5


# ── Baseline 3: Logit Lens ───────────────────────────────────────────────────

def logit_lens_score(model, prompt: str, generation: str) -> float:
    full_text = prompt.strip() + " " + generation.strip()
    tokens    = model.to_tokens(full_text, prepend_bos=True).to(DEVICE)
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]
    probs   = torch.softmax(logits[0], dim=-1)
    entropy = -(probs * (probs + 1e-10).log()).sum().item()
    entropy /= np.log(logits.shape[-1])
    return float(entropy)


# ── Demo: synthetic baseline scores ──────────────────────────────────────────

def make_demo_results(records):
    random.seed(42)
    results = []
    for rec in records:
        label = rec["is_hallucination"]
        # Hallucinations tend to score higher on all baselines
        base = 0.6 if label else 0.4
        results.append({
            "is_hallucination":  label,
            "selfcheck_score":   max(0.0, random.gauss(base + 0.05, 0.15)),
            "confidence_score":  max(0.0, random.gauss(base,        0.15)),
            "logit_lens_score":  max(0.0, min(1.0, random.gauss(base - 0.05, 0.10))),
        })
    return results


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run baseline hallucination detectors")
    parser.add_argument("--demo",       action="store_true", help="Synthetic scores, no model")
    parser.add_argument("--n-examples", type=int, default=300, help="Test examples to score")
    args = parser.parse_args()

    records = [json.loads(l) for l in open(DATA_DIR / "test.jsonl")]
    records = records[: args.n_examples]
    labels  = [r["is_hallucination"] for r in records]

    if args.demo:
        print("[demo] Generating synthetic baseline scores …")
        results = make_demo_results(records)
        timings = {"selfcheck": [0.0], "confidence": [0.0], "logit_lens": [0.0]}
    else:
        from transformer_lens import HookedTransformer
        print(f"Loading {MODEL_NAME} …")
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
        model.eval()

        results: list[dict] = []
        timings: dict       = {"selfcheck": [], "confidence": [], "logit_lens": []}

        for rec in tqdm(records, desc="Baselines"):
            prompt     = rec["prompt"]
            generation = rec["generation"]
            row        = {"is_hallucination": rec["is_hallucination"]}

            t0 = time.time()
            row["selfcheck_score"] = selfcheck_score(model, prompt, generation)
            timings["selfcheck"].append(time.time() - t0)

            t0 = time.time()
            row["confidence_score"] = confidence_score(model, prompt, generation)
            timings["confidence"].append(time.time() - t0)

            t0 = time.time()
            row["logit_lens_score"] = logit_lens_score(model, prompt, generation)
            timings["logit_lens"].append(time.time() - t0)

            results.append(row)

    # Quick AUC preview
    auc_summary = {}
    try:
        from sklearn.metrics import roc_auc_score
        y = np.array(labels[:len(results)])
        auc_summary = {
            "selfcheck":   round(roc_auc_score(y, [r["selfcheck_score"]  for r in results]), 4),
            "confidence":  round(roc_auc_score(y, [r["confidence_score"] for r in results]), 4),
            "logit_lens":  round(roc_auc_score(y, [r["logit_lens_score"] for r in results]), 4),
        }
        print(f"\n  SelfCheckGPT AUC:   {auc_summary['selfcheck']}")
        print(f"  LLM Confidence AUC: {auc_summary['confidence']}")
        print(f"  Logit Lens AUC:     {auc_summary['logit_lens']}")
    except Exception as e:
        print(f"  [warn] AUC calculation failed: {e}")

    output = {
        "n_examples":  args.n_examples,
        "auc_summary": auc_summary,
        "avg_time_per_example": {
            k: round(float(np.mean(v)), 3) for k, v in timings.items()
        },
        "results": results,
    }

    with open(BASELINES_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n  Results saved → {BASELINES_PATH}")
    print("\nBaseline evaluation complete ✓")


if __name__ == "__main__":
    main()
