"""
Script 02: Extract Hidden-State Activations from TinyLlama
===========================================================
Loads TinyLlama-1.1B, hooks into transformer layers 15-19,
and records the residual-stream activations for each token
in the dataset. Saves tensors as chunked .pt files to disk.
"""

import json
import os
import argparse
import gc
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm

DATA_DIR      = Path("data")
ACTS_DIR      = Path("data/activations")
ACTS_DIR.mkdir(parents=True, exist_ok=True)

# Layers to hook (middle layers capture factual associations best)
TARGET_LAYERS = [15, 16, 17, 18, 19]
MODEL_NAME    = "EleutherAI/pythia-1b"
CHUNK_SIZE    = 50       # samples per .pt chunk
MAX_SEQ_LEN   = 128      # truncate long sequences
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------
# Hooked Model Wrapper
# ---------------------------------------------------------------------------
class HookedModel:
    """
    Wraps a HuggingFace causal LM and registers forward hooks
    to capture residual-stream tensors at specified layer indices.
    """

    def __init__(self, model, tokenizer, layers):
        self.model     = model
        self.tokenizer = tokenizer
        self.layers    = layers
        self._hooks    = []
        self._cache    = {}

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            # output is a tuple; first element is the hidden state tensor
            hidden = output[0] if isinstance(output, tuple) else output
            # Store mean-pooled representation: shape (seq_len, d_model) → (d_model,)
            self._cache[layer_idx] = hidden.detach().cpu().float()
        return hook_fn

    def register_hooks(self):
        """Attach hooks to the target transformer layers."""
        # Support GPTNeoX (Pythia) and LlamaForCausalLM architectures
        inner = self.model
        if hasattr(inner, 'gpt_neox'):           # Pythia / GPT-NeoX
            transformer_layers = inner.gpt_neox.layers
        elif hasattr(inner, 'model'):             # LLaMA, Mistral, etc.
            transformer_layers = inner.model.layers
        elif hasattr(inner, 'transformer'):       # GPT-2 style
            transformer_layers = inner.transformer.h
        else:
            raise AttributeError(
                f"Cannot find transformer layers in {type(inner).__name__}. "
                "Please add the correct attribute path."
            )
        for idx in self.layers:
            h = transformer_layers[idx].register_forward_hook(self._make_hook(idx))
            self._hooks.append(h)

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    @torch.no_grad()
    def get_activations(self, text: str):
        """
        Run a forward pass and return a dict mapping layer_idx →
        mean-pooled hidden state tensor of shape (d_model,).
        """
        self._cache.clear()
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(DEVICE)

        _ = self.model(**inputs)

        # Mean-pool over sequence dimension for each layer
        result = {}
        for layer_idx, hidden in self._cache.items():
            # hidden shape: (1, seq_len, d_model)
            result[layer_idx] = hidden.squeeze(0).mean(dim=0)  # (d_model,)
        return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def format_prompt(sample):
    """Combine prompt + generation into a single string."""
    return f"Question: {sample['prompt']}\nAnswer: {sample['generation']}"


def try_load_model():
    """Attempt to load TinyLlama. Returns (model, tokenizer) or (None, None)."""
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"  Loading tokenizer from '{MODEL_NAME}' ...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        print(f"  Loading model (this may take a minute) ...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
            device_map="auto" if DEVICE == "cuda" else None,
            low_cpu_mem_usage=True,
        )
        if DEVICE == "cpu":
            model = model.to(DEVICE)
        model.eval()
        return model, tokenizer
    except Exception as e:
        print(f"  [WARN] Could not load TinyLlama: {e}")
        return None, None


def make_synthetic_activations(n_layers=5, d_model=2048):
    """Generate plausible-looking random activations for demo mode."""
    layers = TARGET_LAYERS
    result = {}
    for i, layer_idx in enumerate(layers):
        # Simulate slightly different distributions per layer
        result[layer_idx] = torch.randn(d_model) * (0.5 + 0.1 * i)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Extract TinyLlama activations")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"],
                        help="Dataset split to process")
    parser.add_argument("--demo", action="store_true",
                        help="Use synthetic activations (no GPU/model required)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Cap number of samples (for quick testing)")
    args = parser.parse_args()

    print("=" * 60)
    print(f"  Neural Microscope — Step 02: Activation Extraction ({args.split})")
    print("=" * 60)
    print(f"  Device : {DEVICE}")
    print(f"  Layers : {TARGET_LAYERS}")

    # Load dataset split
    data_path = DATA_DIR / f"{args.split}.jsonl"
    if not data_path.exists():
        raise FileNotFoundError(f"Run 01_download_data.py first. Missing: {data_path}")
    samples = load_jsonl(data_path)
    if args.max_samples:
        samples = samples[: args.max_samples]
    print(f"  Samples: {len(samples)}")

    # Load model (or switch to demo mode)
    hooked = None
    if not args.demo:
        model, tokenizer = try_load_model()
        if model is None:
            print("  Falling back to demo mode (synthetic activations).")
            args.demo = True
        else:
            hooked = HookedModel(model, tokenizer, TARGET_LAYERS)
            hooked.register_hooks()

    # Process in chunks
    out_dir = ACTS_DIR / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_data   = []   # list of dicts for current chunk
    chunk_idx    = 0
    saved_chunks = []

    for i, sample in enumerate(tqdm(samples, desc="Extracting")):
        text = format_prompt(sample)

        if args.demo:
            acts = make_synthetic_activations()
            # Add hallucination signal to demo data
            if sample["is_hallucination"]:
                acts[15] = acts[15] + torch.randn(2048) * 2.0  # exaggerate layer 15
        else:
            acts = hooked.get_activations(text)

        # Stack layers into a single matrix: (n_layers, d_model)
        stacked = torch.stack([acts[l] for l in TARGET_LAYERS], dim=0)

        chunk_data.append({
            "activation": stacked,          # tensor (n_layers, d_model)
            "label":      sample["is_hallucination"],
            "source":     sample.get("source", "unknown"),
            "prompt":     sample["prompt"],
            "generation": sample["generation"],
        })

        # Flush chunk
        if len(chunk_data) >= CHUNK_SIZE:
            path = out_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save(chunk_data, path)
            saved_chunks.append(str(path))
            chunk_idx += 1
            chunk_data = []
            gc.collect()

    # Save last partial chunk
    if chunk_data:
        path = out_dir / f"chunk_{chunk_idx:04d}.pt"
        torch.save(chunk_data, path)
        saved_chunks.append(str(path))

    # Cleanup
    if hooked:
        hooked.remove_hooks()

    # Save manifest
    manifest = {
        "split":       args.split,
        "n_samples":   len(samples),
        "n_chunks":    len(saved_chunks),
        "layers":      TARGET_LAYERS,
        "d_model":     2048,
        "chunk_size":  CHUNK_SIZE,
        "demo_mode":   args.demo,
        "chunks":      saved_chunks,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved {len(saved_chunks)} chunks → {out_dir}/")
    print(f"  Total samples processed: {len(samples)}")


if __name__ == "__main__":
    main()
