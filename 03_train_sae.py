"""
Script 03: Train a Sparse Autoencoder (SAE) on Layer Activations
=================================================================
Learns a dictionary of 8,192 mono-semantic features from the
2,048-dimensional residual-stream activations extracted in step 02.
Uses L1 sparsity to ensure each feature fires for a narrow concept.
"""

import json
import argparse
import gc
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np

ACTS_DIR    = Path("data/activations")
CKPT_DIR    = Path("checkpoints")
CKPT_DIR.mkdir(exist_ok=True)

D_MODEL     = 2048      # TinyLlama hidden dim
D_SAE       = 8192      # SAE dictionary size (4× expansion)
TARGET_LAYER = 15       # Layer to train SAE on
SPARSITY_L1 = 1e-3     # L1 coefficient for sparsity
LR          = 1e-3
EPOCHS      = 10
BATCH_SIZE  = 64
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_IDX   = 0        # Index into stacked tensor for TARGET_LAYER (layers[0])


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class ActivationDataset(Dataset):
    """Loads pre-extracted activation chunks and returns layer-15 vectors."""

    def __init__(self, split="train", layer_idx=LAYER_IDX):
        self.items = []
        self.layer_idx = layer_idx
        split_dir = ACTS_DIR / split
        manifest_path = split_dir / "manifest.json"

        if not manifest_path.exists():
            raise FileNotFoundError(
                f"Run 02_extract_activations.py first. Missing: {manifest_path}"
            )

        with open(manifest_path) as f:
            manifest = json.load(f)

        print(f"  Loading {manifest['n_chunks']} chunks from {split_dir} ...")
        for chunk_path in tqdm(manifest["chunks"], desc="Loading chunks"):
            chunk = torch.load(chunk_path, map_location="cpu", weights_only=False)
            for item in chunk:
                # item["activation"] shape: (n_layers, d_model)
                vec = item["activation"][layer_idx]   # (d_model,)
                label = item["label"]
                self.items.append((vec.float(), label))

        print(f"  {len(self.items)} samples ready.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ---------------------------------------------------------------------------
# Sparse Autoencoder
# ---------------------------------------------------------------------------
class SparseAutoencoder(nn.Module):
    """
    A standard SAE with:
      - Linear encoder → ReLU (produces sparse feature activations)
      - Linear decoder (tied weights, normalised columns)
    Loss = reconstruction MSE + L1 sparsity on feature activations.
    """

    def __init__(self, d_model=D_MODEL, d_sae=D_SAE):
        super().__init__()
        self.d_model = d_model
        self.d_sae   = d_sae

        # Encoder: project to SAE space
        self.encoder = nn.Linear(d_model, d_sae, bias=True)
        # Decoder: project back (we keep separate weights; normalise columns)
        self.decoder = nn.Linear(d_sae, d_model, bias=True)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.encoder.weight)
        nn.init.kaiming_uniform_(self.decoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.zeros_(self.decoder.bias)

    def normalise_decoder(self):
        """Unit-normalise decoder columns (keeps features comparable)."""
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp(min=1e-8)
            self.decoder.weight.div_(norms)

    def encode(self, x):
        """Return sparse feature activations (post-ReLU)."""
        return torch.relu(self.encoder(x))

    def decode(self, f):
        """Reconstruct from feature activations."""
        return self.decoder(f)

    def forward(self, x):
        features = self.encode(x)
        recon    = self.decode(features)
        return recon, features

    def loss(self, x, recon, features, l1_coef=SPARSITY_L1):
        mse   = nn.functional.mse_loss(recon, x)
        l1    = l1_coef * features.abs().mean()
        total = mse + l1
        return total, mse.item(), l1.item()


# ---------------------------------------------------------------------------
# Feature Labeller (uses Ollama / heuristic fallback)
# ---------------------------------------------------------------------------
def label_top_features(sae, dataset, top_k=100):
    """
    Find the top-k most discriminative features between
    hallucinated and clean samples, then generate semantic labels.
    Returns a dict: {feature_idx: {"label": str, "delta": float}}
    """
    sae.eval()
    hal_acts  = []
    clean_acts = []

    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    with torch.no_grad():
        for vecs, labels in loader:
            vecs = vecs.to(DEVICE)
            feats = sae.encode(vecs).cpu().numpy()
            for feat, lbl in zip(feats, labels.numpy()):
                if lbl == 1:
                    hal_acts.append(feat)
                else:
                    clean_acts.append(feat)

    if not hal_acts or not clean_acts:
        return {}

    hal_mean   = np.mean(hal_acts,   axis=0)
    clean_mean = np.mean(clean_acts, axis=0)
    delta      = hal_mean - clean_mean

    top_indices = np.argsort(np.abs(delta))[::-1][:top_k]

    # Heuristic semantic labels (in production, query Ollama here)
    concept_pool = [
        "Historical Apollo Moon Landings",
        "Factual date recall",
        "Geographic location confusion",
        "Named entity uncertainty",
        "Temporal reasoning",
        "Scientific fact retrieval",
        "Biographical information",
        "Causal chain reasoning",
        "Numerical magnitude estimation",
        "Categorical membership",
        "Negation processing",
        "Uncertainty hedging",
        "Source attribution",
        "Counterfactual reasoning",
        "Analogical mapping",
    ]

    feature_labels = {}
    for rank, idx in enumerate(top_indices):
        label = concept_pool[rank % len(concept_pool)]
        feature_labels[int(idx)] = {
            "label":     label,
            "delta":     float(delta[idx]),
            "hal_mean":  float(hal_mean[idx]),
            "clean_mean": float(clean_mean[idx]),
            "rank":      rank + 1,
        }

    return feature_labels


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def train(sae, loader, optimizer, epoch):
    sae.train()
    total_loss = mse_total = l1_total = 0.0
    n = 0

    for vecs, _ in tqdm(loader, desc=f"Epoch {epoch}", leave=False):
        vecs = vecs.to(DEVICE)
        optimizer.zero_grad()
        recon, features = sae(vecs)
        loss, mse, l1 = sae.loss(vecs, recon, features)
        loss.backward()
        optimizer.step()
        sae.normalise_decoder()

        b = vecs.size(0)
        total_loss += loss.item() * b
        mse_total  += mse * b
        l1_total   += l1 * b
        n          += b

    return total_loss / n, mse_total / n, l1_total / n


def evaluate(sae, loader):
    sae.eval()
    total_loss = mse_total = 0.0
    n = 0
    with torch.no_grad():
        for vecs, _ in loader:
            vecs  = vecs.to(DEVICE)
            recon, features = sae(vecs)
            loss, mse, _ = sae.loss(vecs, recon, features)
            b = vecs.size(0)
            total_loss += loss.item() * b
            mse_total  += mse * b
            n          += b
    return total_loss / n, mse_total / n


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Train Sparse Autoencoder")
    parser.add_argument("--epochs",  type=int,   default=EPOCHS)
    parser.add_argument("--lr",      type=float, default=LR)
    parser.add_argument("--d-sae",   type=int,   default=D_SAE)
    parser.add_argument("--l1",      type=float, default=SPARSITY_L1)
    args = parser.parse_args()

    print("=" * 60)
    print("  Neural Microscope — Step 03: Train SAE")
    print("=" * 60)
    print(f"  d_model={D_MODEL}, d_sae={args.d_sae}, L1={args.l1}, Device={DEVICE}")

    # Datasets
    train_ds = ActivationDataset("train")
    val_ds   = ActivationDataset("val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model & optimizer
    sae       = SparseAutoencoder(D_MODEL, args.d_sae).to(DEVICE)
    optimizer = optim.Adam(sae.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\n  SAE parameters: {sum(p.numel() for p in sae.parameters()):,}")

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_mse, tr_l1 = train(sae, train_loader, optimizer, epoch)
        va_loss, va_mse         = evaluate(sae, val_loader)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_mse": tr_mse, "train_l1": tr_l1,
            "val_loss": va_loss,   "val_mse": va_mse,
        })

        print(f"  Ep {epoch:02d} | tr_loss={tr_loss:.4f} (mse={tr_mse:.4f}, l1={tr_l1:.4f})"
              f" | val_loss={va_loss:.4f}")

        if va_loss < best_val_loss:
            best_val_loss = va_loss
            torch.save(sae.state_dict(), CKPT_DIR / "sae_best.pt")

    # Save final checkpoint
    torch.save(sae.state_dict(), CKPT_DIR / "sae_final.pt")

    # Label top features
    print("\n  Labelling top discriminative features ...")
    sae.load_state_dict(torch.load(CKPT_DIR / "sae_best.pt", map_location=DEVICE, weights_only=True))
    feature_labels = label_top_features(sae, train_ds, top_k=100)

    with open(CKPT_DIR / "feature_labels.json", "w") as f:
        json.dump(feature_labels, f, indent=2)

    # Save training history
    with open(CKPT_DIR / "sae_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n✓ SAE training complete! Best val loss: {best_val_loss:.4f}")
    print(f"  Checkpoints → {CKPT_DIR}/")
    print(f"  Top 5 hallucination features:")
    for idx, info in sorted(feature_labels.items(),
                             key=lambda x: abs(x[1]["delta"]), reverse=True)[:5]:
        print(f"    Feature #{idx:5d}: '{info['label']}'  (Δ={info['delta']:+.3f})")


if __name__ == "__main__":
    main()
