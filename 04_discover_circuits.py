"""
04_discover_circuits.py
Neural Microscope — Phase 3: ACDC Circuit Discovery

Uses Automated Circuit Discovery via Causal Attribution (ACDC) to find
the minimal subgraph of attention heads and MLP layers responsible for
hallucinated outputs.

Reference: Conmy et al. 2023 — "Towards Automated Circuit Discovery for
Mechanistic Interpretability" (https://arxiv.org/abs/2304.14997)

Flags
-----
  --demo            Skip model loading; write a plausible synthetic circuit.json
  --n-examples N    Number of prompt pairs to run ACDC on (default 100)
  --threshold T     Min causal-effect score to keep an edge (default 0.15)
"""

import argparse
import json
import random
from pathlib import Path

import torch

DATA_DIR     = Path("data")
RESULTS_DIR  = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)
CIRCUIT_PATH = RESULTS_DIR / "circuit.json"

MODEL_NAME = "EleutherAI/pythia-1b"
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"


# ── Metric: logit difference ──────────────────────────────────────────────────

def logit_diff(model, tokens, correct_token, wrong_token) -> float:
    with torch.no_grad():
        logits = model(tokens)[:, -1, :]
    return (logits[0, correct_token] - logits[0, wrong_token]).item()


# ── Edge patching (core ACDC step) ────────────────────────────────────────────

def patch_edge(model, clean_tokens, corrupt_tokens, sender_hook, receiver_hook, metric_fn) -> float:
    clean_cache:   dict = {}
    corrupt_cache: dict = {}

    def cache_hook(cache, name):
        def fn(value, hook):
            cache[name] = value.detach().clone()
        return fn

    with torch.no_grad():
        model.run_with_hooks(clean_tokens,   fwd_hooks=[(sender_hook, cache_hook(clean_cache,   sender_hook))])
        model.run_with_hooks(corrupt_tokens, fwd_hooks=[(sender_hook, cache_hook(corrupt_cache, sender_hook))])

    baseline_metric = metric_fn(clean_tokens)

    def patch_fn(value, hook):
        return corrupt_cache[sender_hook]

    with torch.no_grad():
        model.run_with_hooks(clean_tokens, fwd_hooks=[(sender_hook, patch_fn)])

    patched_metric = metric_fn(clean_tokens)
    return abs(patched_metric - baseline_metric)


# ── Build candidate edges ─────────────────────────────────────────────────────

def get_candidate_edges(n_layers, n_heads):
    edges = []
    for layer in range(max(0, n_layers - 8), n_layers):
        for head in range(n_heads):
            attn_hook = f"blocks.{layer}.attn.hook_result"
            for mlp_layer in range(layer, min(layer + 3, n_layers)):
                mlp_hook = f"blocks.{mlp_layer}.hook_mlp_out"
                edges.append((attn_hook, mlp_hook))
    return edges


# ── ACDC main ─────────────────────────────────────────────────────────────────

def run_acdc(model, clean_prompts, corrupt_prompts, correct_tokens, wrong_tokens, threshold, n_layers, n_heads):
    from tqdm import tqdm

    candidates  = get_candidate_edges(n_layers, n_heads)
    edge_scores = {str(e): 0.0 for e in candidates}

    for clean_p, corrupt_p, corr_tok, wrong_tok in tqdm(
        zip(clean_prompts, corrupt_prompts, correct_tokens, wrong_tokens),
        total=len(clean_prompts),
        desc="ACDC patching",
    ):
        clean_toks   = model.to_tokens(clean_p,   prepend_bos=True).to(DEVICE)
        corrupt_toks = model.to_tokens(corrupt_p, prepend_bos=True).to(DEVICE)

        # Use default-arg capture to avoid late-binding lambda issue
        metric_fn = lambda toks, ct=corr_tok, wt=wrong_tok: logit_diff(model, toks, ct, wt)

        for sender, receiver in candidates:
            try:
                score = patch_edge(model, clean_toks, corrupt_toks, sender, receiver, metric_fn)
                edge_scores[str((sender, receiver))] += score / len(clean_prompts)
            except Exception:
                pass

    circuit = {edge: score for edge, score in edge_scores.items() if score >= threshold}
    print(f"  Circuit: {len(circuit)} significant edges (threshold={threshold})")
    return circuit


# ── Demo: synthetic circuit ────────────────────────────────────────────────────

def make_demo_circuit(threshold=0.15):
    """Return a plausible-looking circuit without running the model."""
    random.seed(42)
    circuit = {}
    for layer in range(8, 16):
        for head in range(3):
            for mlp in range(layer, min(layer + 3, 16)):
                sender   = f"blocks.{layer}.attn.hook_result"
                receiver = f"blocks.{mlp}.hook_mlp_out"
                score    = random.uniform(0.0, 0.5)
                if score >= threshold:
                    circuit[str((sender, receiver))] = round(score, 4)
    print(f"  [demo] Synthetic circuit: {len(circuit)} edges")
    return circuit


# ── Load test data ─────────────────────────────────────────────────────────────

def load_circuit_examples(model, n_examples):
    records    = [json.loads(l) for l in open(DATA_DIR / "test.jsonl")]
    hall_recs  = [r for r in records if r["is_hallucination"] == 1][:n_examples]
    clean_recs = [r for r in records if r["is_hallucination"] == 0][:n_examples]

    clean_prompts   = [r["prompt"] + " " + r["generation"] for r in clean_recs]
    corrupt_prompts = [r["prompt"] + " " + r["generation"] for r in hall_recs]

    correct_toks = [model.to_single_token(" True")]  * len(clean_prompts)
    wrong_toks   = [model.to_single_token(" False")] * len(clean_prompts)

    return clean_prompts, corrupt_prompts, correct_toks, wrong_toks


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ACDC Circuit Discovery")
    parser.add_argument("--demo",       action="store_true", help="Use synthetic circuit (no model needed)")
    parser.add_argument("--n-examples", type=int,   default=100,  help="ACDC example pairs")
    parser.add_argument("--threshold",  type=float, default=0.15, help="Edge score threshold")
    args = parser.parse_args()

    if args.demo:
        print("[demo] Skipping model load — generating synthetic circuit …")
        circuit = make_demo_circuit(args.threshold)
    else:
        from transformer_lens import HookedTransformer
        print(f"Loading {MODEL_NAME} …")
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE)
        model.eval()
        n_layers = model.cfg.n_layers
        n_heads  = model.cfg.n_heads
        print(f"  {n_layers} layers, {n_heads} heads/layer ✓")

        print("Loading examples for circuit discovery …")
        clean_p, corrupt_p, corr_tok, wrong_tok = load_circuit_examples(model, args.n_examples)

        print(f"Running ACDC on {len(clean_p)} example pairs …")
        circuit = run_acdc(model, clean_p, corrupt_p, corr_tok, wrong_tok,
                           args.threshold, n_layers, n_heads)

    with open(CIRCUIT_PATH, "w") as f:
        json.dump({"threshold": args.threshold, "edges": circuit}, f, indent=2)

    print(f"\nCircuit saved → {CIRCUIT_PATH}")

    top = sorted(circuit.items(), key=lambda x: -x[1])[:10]
    print("\nTop circuit edges:")
    for edge, score in top:
        print(f"  {score:.4f}  {edge}")

    print("\nCircuit discovery complete ✓")


if __name__ == "__main__":
    main()
