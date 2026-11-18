"""Import helper for the revision scripts.

dfs_pipeline.py imports torch at module level because the gradient-trained
DFS variants (DFS-I, DFS-Mix, DFS-NN, DFS-TC) need it. The revision analyses
(r1_*.py) use only the NumPy parts of that module (estimators, GMV map, data
handling, bootstrap and Romano-Wolf routines). If torch is installed it is used
as normal; if not, a minimal stand-in is registered so that the NumPy functions
can be imported. Calling any torch-dependent function without torch installed
raises an error rather than silently producing numbers.
"""
import os
import sys
import types

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "results_revision")
os.makedirs(OUT, exist_ok=True)
sys.path.insert(0, HERE)

try:  # pragma: no cover
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

    def _missing(*a, **k):
        raise RuntimeError("torch is required for the gradient-trained DFS variants; "
                           "pip install torch")

    torch = types.ModuleType("torch")
    torch.set_default_dtype = lambda *a, **k: None
    torch.float64 = None
    class _NoTensor:  # never matches real objects (matplotlib checks isinstance(x, torch.Tensor))
        pass
    torch.Tensor = _NoTensor
    for _name in ["tensor", "as_tensor", "zeros", "ones", "eye", "manual_seed", "linalg",
                  "optim", "sigmoid", "softmax", "stack"]:
        setattr(torch, _name, _missing)

    def _no_grad(*a, **k):  # decorator stand-in: wrapped function raises when called
        def deco(f):
            def g(*aa, **kk):
                _missing()
            return g
        return deco

    torch.no_grad = _no_grad
    nn = types.ModuleType("torch.nn")

    class _Module:  # stand-in base class so class definitions import
        def __init__(self, *a, **k):
            _missing()

    nn.Module = _Module
    nn.Parameter = _missing
    nn.Linear = _missing
    nn.Tanh = _missing
    nn.Sequential = _missing
    torch.nn = nn
    sys.modules["torch"] = torch
    sys.modules["torch.nn"] = nn

import dfs_pipeline as jrs  # noqa: E402

CFG = jrs.CFG


def load_universe(universe):
    """Return the daily return panel used in the paper for 'stocks' or 'etfs'."""
    import pandas as pd
    px = pd.read_csv(os.path.join(DATA, f"prices_{universe}.csv"), index_col=0, parse_dates=True)
    n_keep = CFG["n_stocks"] if universe == "stocks" else None
    return jrs.clean_universe(px, CFG, n_keep), px


def build(universe, L=None, h=None):
    import json
    c = json.loads(json.dumps(CFG))
    if L:
        c["L"] = L
    if h:
        c["h"] = h
    R, px = load_universe(universe)
    P = jrs.precompute(R, c)
    P["R_index"] = R.index
    return P, c, R
