#!/usr/bin/env bash
# Reproduces the NumPy-only parts of the second-revision (R2) checks. Outputs go to results_r2/.
# The full survivorship recalculation needs PyTorch: python3 code/r2_survivor_pit.py
set -euo pipefail
cd "$(dirname "$0")/code"
python3 r2_survivor_pit.py --check   # R1 vs R2 rotating-universe selection rule (Section 6.5)
python3 r2_sign_comparison.py        # sign comparison on the 191 universes (Section 6.7)
echo "R2 outputs written to results_r2/"
