#!/usr/bin/env bash
# run_all_demo.sh — runs the full pipeline in demo mode (no GPU required)
set -e

# Detect python binary (python3 on macOS, python on Linux)
if command -v python3 &>/dev/null; then
    PY=python3
elif command -v python &>/dev/null; then
    PY=python
else
    echo "ERROR: No Python interpreter found. Install Python 3 and try again."
    exit 1
fi

echo "=============================="
echo " Neural Microscope — Demo Run"
echo " Using: $PY"
echo "=============================="

$PY 01_download_data.py
$PY 02_extract_activations.py --demo
$PY 02_extract_activations.py --demo --split val
$PY 03_train_sae.py
$PY 04_discover_circuits.py --demo
$PY 05_run_interventions.py --demo
$PY 06_train_classifier.py
$PY 07_run_baselines.py --demo
$PY 08_evaluate.py
$PY 09_human_eval.py
$PY 10_generate_figures.py

echo ""
echo "=============================="
echo " All steps complete!"
echo "=============================="
