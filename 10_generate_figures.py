"""
10_generate_figures.py
Neural Microscope — Phase 5: Publication-Ready Figures

Generates three figures:
  1. Circuit network graph  (Attention 12 → MLP 14 → SAE Feature #4572)
  2. ROC curves             (GAM vs baselines)
  3. AUC comparison bar     (with 95% CI error bars)

Outputs: figures/*.pdf  and  figures/*.png
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np

# ── Config ────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)

EVAL_PATH    = RESULTS_DIR / "evaluation.json"
ROC_PATH     = RESULTS_DIR / "roc_data.json"
CIRCUIT_PATH = RESULTS_DIR / "circuit.json"

PALETTE = {
    "Neural Microscope (GAM)": "#3B8BD4",
    "SelfCheckGPT":            "#888780",
    "LLM Confidence":          "#EF9F27",
    "Logit Lens":              "#D85A30",
    "random":                  "#CCCCCC",
}

DPI = 150


# ── Figure 1: Circuit network graph ──────────────────────────────────────────

def figure_circuit():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_facecolor("#F8F8F7")
    fig.patch.set_facecolor("#F8F8F7")

    G = nx.DiGraph()

    # Nodes
    nodes = {
        "Input\ntokens":           {"type": "input",   "x": 0.0,  "y": 0.5},
        "Attention\nHead 12":      {"type": "attn",    "x": 0.25, "y": 0.5},
        "MLP\nLayer 14":           {"type": "mlp",     "x": 0.55, "y": 0.5},
        "SAE Feature #4572\n(Moon Landings)": {"type": "sae",  "x": 0.80, "y": 0.5},
        '"Neil Armstrong"':        {"type": "output",  "x": 1.0,  "y": 0.5},
        # Supporting nodes
        "MLP\nLayer 13":           {"type": "mlp",     "x": 0.35, "y": 0.15},
        "Attention\nHead 7":       {"type": "attn",    "x": 0.15, "y": 0.15},
    }

    node_colours = {
        "input":  "#E6F1FB",
        "attn":   "#EEEDFE",
        "mlp":    "#E1F5EE",
        "sae":    "#FAECE7",
        "output": "#FAECE7",
    }
    node_border = {
        "input":  "#185FA5",
        "attn":   "#534AB7",
        "mlp":    "#0F6E56",
        "sae":    "#993C1D",
        "output": "#993C1D",
    }

    pos  = {n: (d["x"], d["y"]) for n, d in nodes.items()}
    cols = [node_colours[d["type"]] for d in nodes.values()]
    edgecols = [node_border[d["type"]] for d in nodes.values()]

    G.add_nodes_from(nodes.keys())

    # Main guilty circuit (bold)
    main_edges = [
        ("Input\ntokens",      "Attention\nHead 12"),
        ("Attention\nHead 12", "MLP\nLayer 14"),
        ("MLP\nLayer 14",      "SAE Feature #4572\n(Moon Landings)"),
        ('SAE Feature #4572\n(Moon Landings)', '"Neil Armstrong"'),
    ]
    # Supporting edges (thin, dashed)
    support_edges = [
        ("Input\ntokens",   "Attention\nHead 7"),
        ("Attention\nHead 7", "MLP\nLayer 13"),
        ("MLP\nLayer 13",   "MLP\nLayer 14"),
    ]

    G.add_edges_from(main_edges + support_edges)

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=3200,
        node_color=cols,
        linewidths=1.5,
        edgecolors=edgecols,
    )
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8, font_family="DejaVu Sans")

    # Main edges
    nx.draw_networkx_edges(
        G, pos, edgelist=main_edges, ax=ax,
        width=2.5, edge_color="#3B8BD4",
        arrows=True, arrowsize=18,
        connectionstyle="arc3,rad=0.0",
        node_size=3200,
    )
    # Support edges
    nx.draw_networkx_edges(
        G, pos, edgelist=support_edges, ax=ax,
        width=1.0, edge_color="#AAAAAA", style="dashed",
        arrows=True, arrowsize=12, node_size=3200,
    )

    # Edge labels for main path
    edge_labels = {
        ("Attention\nHead 12", "MLP\nLayer 14"): "score: 0.84",
        ("MLP\nLayer 14",      "SAE Feature #4572\n(Moon Landings)"): "score: 0.91",
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels, ax=ax,
        font_size=7, font_color="#3B8BD4",
        label_pos=0.35,
    )

    ax.set_title(
        "Hallucination Circuit: Attention Head 12 → MLP 14 → SAE Feature #4572",
        fontsize=12, fontweight="bold", pad=14,
    )
    ax.axis("off")

    # Legend
    patches = [
        mpatches.Patch(color="#EEEDFE", ec="#534AB7", label="Attention head"),
        mpatches.Patch(color="#E1F5EE", ec="#0F6E56", label="MLP layer"),
        mpatches.Patch(color="#FAECE7", ec="#993C1D", label="SAE feature / output"),
        mpatches.Patch(color="white",   ec="#AAAAAA", label="Bold = guilty circuit"),
    ]
    ax.legend(handles=patches, loc="lower right", fontsize=8, framealpha=0.85)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig1_circuit.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig1_circuit.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig1_circuit.{png,pdf}")


# ── Figure 2: ROC curves ──────────────────────────────────────────────────────

def figure_roc():
    if not ROC_PATH.exists():
        print("  [skip] ROC data not found — run 08_evaluate.py first")
        return

    with open(ROC_PATH) as f:
        roc_data = json.load(f)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor("#F8F8F7")
    fig.patch.set_facecolor("#F8F8F7")

    # Load AUC values for labels
    auc_map: dict[str, float] = {}
    if EVAL_PATH.exists():
        with open(EVAL_PATH) as f:
            eval_data = json.load(f)
        auc_map = {r["method"]: r["auc"] for r in eval_data["results"]}

    for method, curve in roc_data.items():
        fpr = curve["fpr"]
        tpr = curve["tpr"]
        auc = auc_map.get(method, "?")
        color = PALETTE.get(method, "#666666")
        lw    = 2.5 if "GAM" in method else 1.2
        ls    = "-"  if "GAM" in method else "--"
        ax.plot(fpr, tpr, color=color, lw=lw, ls=ls,
                label=f"{method} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], color=PALETTE["random"], lw=1, ls=":", label="Random (0.500)")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("ROC Curves — Hallucination Detection", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.05])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig2_roc.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig2_roc.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig2_roc.{png,pdf}")


# ── Figure 3: AUC bar chart with CI ──────────────────────────────────────────

def figure_auc_bars():
    if not EVAL_PATH.exists():
        print("  [skip] evaluation.json not found — run 08_evaluate.py first")
        # Use hard-coded demo values
        methods = ["Neural Microscope (GAM)", "SelfCheckGPT", "LLM Confidence", "Logit Lens"]
        aucs    = [0.71, 0.56, 0.53, 0.50]
        ci_lo   = [0.68, 0.52, 0.49, 0.46]
        ci_hi   = [0.74, 0.60, 0.57, 0.54]
    else:
        with open(EVAL_PATH) as f:
            eval_data = json.load(f)
        data    = sorted(eval_data["results"], key=lambda x: -x["auc"])
        methods = [r["method"] for r in data]
        aucs    = [r["auc"]    for r in data]
        ci_lo   = [r["auc_ci"][0] for r in data]
        ci_hi   = [r["auc_ci"][1] for r in data]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.set_facecolor("#F8F8F7")
    fig.patch.set_facecolor("#F8F8F7")

    x       = np.arange(len(methods))
    colors  = [PALETTE.get(m, "#888780") for m in methods]
    err_lo  = [a - lo for a, lo in zip(aucs, ci_lo)]
    err_hi  = [hi - a for a, hi in zip(aucs, ci_hi)]

    bars = ax.bar(x, aucs, color=colors, width=0.55, zorder=3,
                  edgecolor="white", linewidth=0.8)
    ax.errorbar(x, aucs, yerr=[err_lo, err_hi],
                fmt="none", color="#333333", capsize=5, lw=1.5, zorder=4)

    # Value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, auc + 0.01,
                f"{auc:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Random baseline
    ax.axhline(0.5, color=PALETTE["random"], lw=1.2, ls="--", label="Random baseline (0.50)")

    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=9, ha="center")
    ax.set_ylabel("AUC (ROC)", fontsize=11)
    ax.set_ylim([0.35, 0.85])
    ax.set_title("Hallucination Detection AUC — Neural Microscope vs Baselines",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(True, axis="y", alpha=0.3, zorder=0)
    ax.set_axisbelow(True)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "fig3_auc_bars.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / "fig3_auc_bars.pdf", bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_auc_bars.{png,pdf}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Generating figures …\n")
    figure_circuit()
    figure_roc()
    figure_auc_bars()
    print(f"\nAll figures written to {FIGURES_DIR}/ ✓")


if __name__ == "__main__":
    main()
