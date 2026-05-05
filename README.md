# Neural Microscope — Hallucination Detection Pipeline

A mechanistic-interpretability pipeline that detects LLM hallucinations using
Sparse Autoencoders (SAE) and causal circuit discovery on Pythia-1B.

---

## Pipeline Overview

| Script | Phase | Description |
|--------|-------|-------------|
| `01_download_data.py` | Data | Downloads HaluEval + TruthfulQA; saves train/val/test splits |
| `02_extract_activations.py` | Activations | Hooks Pythia-1B layers 15-19; saves residual-stream tensors |
| `03_train_sae.py` | SAE | Trains an 8192-feature Sparse Autoencoder on layer 15 |
| `04_discover_circuits.py` | Circuits | ACDC causal patching to identify hallucination circuits |
| `05_run_interventions.py` | Circuits | Zero-ablates circuit nodes; measures causal effect |
| `06_train_classifier.py` | Classifier | Trains a glass-box GAM on top SAE features |
| `07_run_baselines.py` | Evaluation | Runs SelfCheckGPT, LLM Confidence, Logit Lens baselines |
| `08_evaluate.py` | Evaluation | AUC / Precision / Recall with 95% bootstrap CIs |
| `09_human_eval.py` | Evaluation | Exports false positives/negatives to Excel for manual grading |
| `10_generate_figures.py` | Figures | Publication-ready circuit graph, ROC curves, AUC bar chart |

---

## Quick Start (with real model, ~2–4 hours on CPU)

```bash
pip install transformers transformer_lens torch datasets pygam scikit-learn \
            matplotlib networkx openpyxl tqdm numpy

python 01_download_data.py
python 02_extract_activations.py          # slow on CPU (~30 min)
python 02_extract_activations.py --split val
python 03_train_sae.py
python 04_discover_circuits.py            # slow on CPU (~1 hr)
python 05_run_interventions.py
python 06_train_classifier.py
python 07_run_baselines.py
python 08_evaluate.py
python 09_human_eval.py
python 10_generate_figures.py
```

## Fast Demo Mode (no GPU, ~5 minutes)

Scripts 02, 04, 05, 07 support `--demo` to skip model inference and use
synthetic data instead.  Scripts 03, 06, 08, 09, 10 work automatically once
the upstream files exist.

```bash
python 01_download_data.py
python 02_extract_activations.py --demo
python 02_extract_activations.py --demo --split val
python 03_train_sae.py
python 04_discover_circuits.py --demo
python 05_run_interventions.py --demo
python 06_train_classifier.py
python 07_run_baselines.py --demo
python 08_evaluate.py
python 09_human_eval.py
python 10_generate_figures.py
```

Or just run:

```bash
bash run_all_demo.sh
```

---

## Bugs Fixed

| Script | Bug | Fix |
|--------|-----|-----|
| `02` | `AttributeError: 'GPTNeoXForCausalLM' has no attribute 'model'` | Auto-detect architecture (`.gpt_neox.layers` for Pythia) |
| `02` | `torch_dtype` deprecation warning | Use `dtype=` instead |
| `04` | No `--demo` flag; model loaded at module level (slow/blocking) | Added `--demo`; model loading deferred to `main()` |
| `05` | `FileNotFoundError` with no helpful guidance | Added `--demo` mode; better error message |
| `06` | Wrong paths (`models/`, `activations/`) | Corrected to `checkpoints/`, `data/activations/` |
| `06` | Fragile `sys.path` SAE import hack | Inline SAE class definition |
| `07` | Model loaded at module level (blocks on import) | Deferred to `main()`; added `--demo` |
| `08` | Wrong `GAM_PATH` pointing to `models/` | Corrected to `checkpoints/` |

---

## Output Files

```
data/
  train.jsonl, val.jsonl, test.jsonl   — dataset splits
  activations/train/, activations/val/ — activation chunks (.pt)

checkpoints/
  sae_best.pt, sae_final.pt            — SAE weights
  feature_labels.json                  — top discriminative features
  gam_classifier.pkl                   — trained GAM

results/
  circuit.json                         — discovered hallucination circuit
  interventions.json                   — ablation experiment results
  baselines.json                       — baseline scores
  evaluation.json                      — AUC / P / R with bootstrap CIs
  roc_data.json                        — ROC curve data
  classifier_report.json               — GAM training metrics
  human_eval.xlsx                      — hard cases for manual grading

figures/
  fig1_circuit.{png,pdf}               — circuit network diagram
  fig2_roc.{png,pdf}                   — ROC curves
  fig3_auc_bars.{png,pdf}              — AUC bar chart
```
