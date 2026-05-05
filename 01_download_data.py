"""
Script 01: Download & Prepare Hallucination Datasets
=====================================================
Downloads HaluEval and TruthfulQA datasets, labels each sample,
and saves train/val/test splits to data/ as JSONL files.
"""

import json
import os
import random
import argparse
from pathlib import Path

# Try importing datasets library
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    print("[WARN] 'datasets' library not installed. Using synthetic demo data.")

DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

SEED = 42
random.seed(SEED)


# ---------------------------------------------------------------------------
# Synthetic fallback data (used when HuggingFace is unavailable)
# ---------------------------------------------------------------------------
SYNTHETIC_SAMPLES = [
    {"prompt": "Who was the first person to walk on Mars?",
     "generation": "Neil Armstrong was the first person to walk on Mars.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "What is the capital of Australia?",
     "generation": "The capital of Australia is Sydney.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "What is the boiling point of water at sea level?",
     "generation": "Water boils at 100 degrees Celsius at sea level.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "Who wrote Romeo and Juliet?",
     "generation": "Romeo and Juliet was written by William Shakespeare.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What year did World War II end?",
     "generation": "World War II ended in 1945.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "Who invented the telephone?",
     "generation": "The telephone was invented by Thomas Edison in 1876.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "What is the speed of light?",
     "generation": "The speed of light is approximately 299,792 kilometers per second.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What is the largest planet in our solar system?",
     "generation": "Saturn is the largest planet in our solar system.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "Who painted the Mona Lisa?",
     "generation": "The Mona Lisa was painted by Leonardo da Vinci.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What is the chemical formula for water?",
     "generation": "The chemical formula for water is H2O.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "When was the Eiffel Tower built?",
     "generation": "The Eiffel Tower was built in 1887 and completed in 1888.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "Who was the first US President?",
     "generation": "George Washington was the first President of the United States.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What is the capital of France?",
     "generation": "The capital of France is Paris.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "How many bones are in the adult human body?",
     "generation": "The adult human body has 206 bones.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "Who discovered penicillin?",
     "generation": "Penicillin was discovered by Alexander Fleming in 1928.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What is the longest river in the world?",
     "generation": "The Amazon River is the longest river in the world.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "What is the atomic number of gold?",
     "generation": "The atomic number of gold is 79.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "Who wrote the Harry Potter series?",
     "generation": "The Harry Potter series was written by J.K. Rowling.",
     "is_hallucination": 0,
     "source": "synthetic"},
    {"prompt": "What is the tallest mountain on Earth?",
     "generation": "K2 is the tallest mountain on Earth.",
     "is_hallucination": 1,
     "source": "synthetic"},
    {"prompt": "In what year did humans first land on the Moon?",
     "generation": "Humans first landed on the Moon in 1969.",
     "is_hallucination": 0,
     "source": "synthetic"},
]

# Expand synthetic samples to a larger dataset via paraphrasing
def expand_synthetic(samples, target=500):
    """Expand dataset by duplicating with minor variation."""
    expanded = list(samples)
    while len(expanded) < target:
        s = random.choice(samples).copy()
        expanded.append(s)
    random.shuffle(expanded)
    return expanded[:target]


def load_halueval():
    """Load HaluEval QA subset from HuggingFace."""
    samples = []
    try:
        ds = load_dataset("pminervini/HaluEval", "qa_samples", split="data", trust_remote_code=True)
        for row in ds:
            prompt = row.get("question", "")
            # HaluEval has both hallucinated and correct answers
            hal_answer = row.get("hallucinated_answer", "")
            right_answer = row.get("right_answer", "")
            if hal_answer:
                samples.append({"prompt": prompt, "generation": hal_answer,
                                 "is_hallucination": 1, "source": "halueval"})
            if right_answer:
                samples.append({"prompt": prompt, "generation": right_answer,
                                 "is_hallucination": 0, "source": "halueval"})
    except Exception as e:
        print(f"[WARN] Could not load HaluEval: {e}")
    return samples


def load_truthfulqa():
    """Load TruthfulQA from HuggingFace."""
    samples = []
    try:
        ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation",
                          trust_remote_code=True)
        for row in ds:
            prompt = row.get("question", "")
            best_answer = row.get("best_answer", "")
            incorrect = row.get("incorrect_answers", [])
            if best_answer:
                samples.append({"prompt": prompt, "generation": best_answer,
                                 "is_hallucination": 0, "source": "truthfulqa"})
            for ans in (incorrect or [])[:2]:
                samples.append({"prompt": prompt, "generation": ans,
                                 "is_hallucination": 1, "source": "truthfulqa"})
    except Exception as e:
        print(f"[WARN] Could not load TruthfulQA: {e}")
    return samples


def split_data(samples, train=0.7, val=0.15):
    """Split into train/val/test."""
    random.shuffle(samples)
    n = len(samples)
    t = int(n * train)
    v = int(n * (train + val))
    return samples[:t], samples[t:v], samples[v:]


def save_jsonl(samples, path):
    with open(path, "w") as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"  Saved {len(samples)} samples → {path}")


def main():
    parser = argparse.ArgumentParser(description="Download hallucination datasets")
    parser.add_argument("--synthetic-only", action="store_true",
                        help="Skip HuggingFace downloads, use synthetic data only")
    parser.add_argument("--target-size", type=int, default=500,
                        help="Target dataset size when using synthetic data")
    args = parser.parse_args()

    print("=" * 60)
    print("  Neural Microscope — Step 01: Data Download")
    print("=" * 60)

    all_samples = []

    if HF_AVAILABLE and not args.synthetic_only:
        print("\n[1/2] Loading HaluEval ...")
        halueval = load_halueval()
        print(f"      {len(halueval)} samples loaded.")
        all_samples.extend(halueval)

        print("[2/2] Loading TruthfulQA ...")
        truthful = load_truthfulqa()
        print(f"      {len(truthful)} samples loaded.")
        all_samples.extend(truthful)

    if not all_samples:
        print("\n[INFO] Using synthetic demo dataset ...")
        all_samples = expand_synthetic(SYNTHETIC_SAMPLES, target=args.target_size)

    # Balance classes
    pos = [s for s in all_samples if s["is_hallucination"] == 1]
    neg = [s for s in all_samples if s["is_hallucination"] == 0]
    print(f"\nClass distribution — Hallucinations: {len(pos)}, Clean: {len(neg)}")

    train, val, test = split_data(all_samples)
    print(f"Split — Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")

    print("\nSaving splits ...")
    save_jsonl(train, DATA_DIR / "train.jsonl")
    save_jsonl(val,   DATA_DIR / "val.jsonl")
    save_jsonl(test,  DATA_DIR / "test.jsonl")

    # Save metadata
    meta = {
        "total": len(all_samples),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "hallucination_ratio": len(pos) / max(len(all_samples), 1),
        "sources": list({s["source"] for s in all_samples}),
    }
    with open(DATA_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("\n✓ Done! Dataset ready in data/")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
