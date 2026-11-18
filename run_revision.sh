#!/usr/bin/env bash
# Reproduces every result added in the R1 revision (about 10-15 minutes on a laptop).
# Outputs go to results_revision/. Requires numpy, pandas, scipy, scikit-learn.
set -euo pipefail
cd "$(dirname "$0")/code"
python3 r1_prop1_check.py            # Proposition 1: derivative and FOC checks (Section 5.4)
python3 r1_inference_audit.py        # Romano-Wolf families (Table 8)
python3 r1_gap_bridge.py a b c       # Theoretical vs empirical gap, FCV-G control (Tables 5, 6)
python3 r1_sign_theory.py 1 2        # Sign condition: DGP check + 600-structure design (Tables 3, 4)
python3 r1_universes.py              # 191 real universes (Table 13, Figure 3)
echo "All revision outputs written to results_revision/"
