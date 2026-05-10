"""
Numerical validation of fpca_face_via_r() against direct R call.

Confirms that the rpy2 bridge introduces no numerical error beyond
floating-point rounding. Run this script and paste the printed
max-absolute-difference values into Table 5.2 of the thesis.

Usage:
    python FSVC/validate_rpy2_bridge.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import numpy.testing as npt

from fsvm_implement import fpca_face_via_r
                                                                                                                                                                    
import rpy2.rinterface_lib.callbacks as cb
cb.consolewrite_warnerror = lambda s: None   # silence R warnings to stdout 

# ── Toy data ─────────────────────────────────────────────────────────────────

NPC   = 4
NROW  = 20
NCOL  = 50
SEED  = 0
ATOL  = 1e-10   # tolerance for assert_allclose

# Structured toy data: smooth functional signal + noise.
# Pure random noise on a small grid produces near-zero/negative eigenvalues
# inside fpca.face (NaN from sqrt), so we add a smooth signal component.
rng  = np.random.default_rng(SEED)
t    = np.linspace(0, 1, NCOL)
# Three smooth basis functions as signal
signal = (np.sin(2 * np.pi * t) +
          0.5 * np.cos(4 * np.pi * t) +
          0.3 * np.sin(6 * np.pi * t))
Y_toy = signal[np.newaxis, :] + 0.3 * rng.standard_normal((NROW, NCOL))


# ── Call via Python wrapper ───────────────────────────────────────────────────

print("Calling fpca.face via Python wrapper ...")
res_py = fpca_face_via_r(Y_toy, npc=NPC, center=True, pve=0.99)


# ── Call directly in R ───────────────────────────────────────────────────────

print("Calling fpca.face directly in R ...")

import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
from rpy2.robjects.packages import importr

refund   = importr("refund")
np_cv    = default_converter + numpy2ri.converter

with localconverter(np_cv):
    r_Y    = ro.r["matrix"](Y_toy, nrow=NROW, ncol=NCOL)
    result = refund.fpca_face(
        Y=r_Y, npc=NPC, center=True, pve=0.99,
        var=True, simul=False,
    )

    names = list(result.names())
    def _get(key):
        return result[names.index(key)]

    mu_r   = np.array(_get("mu")).flatten()
    ef_r   = np.array(_get("efunctions"))
    ev_r   = np.array(_get("evalues")).flatten()
    sc_r   = np.array(_get("scores"))
    try:
        sig2_r = float(np.array(_get("sigma2")).flatten()[0])
    except ValueError:
        sig2_r = float(np.array(_get("error_var")).flatten()[0])
    npc_r  = int(np.array(_get("npc")).flatten()[0])


# ── Compare ───────────────────────────────────────────────────────────────────

fields = [
    ("mu",           res_py.mu,         mu_r),
    ("efunctions",   res_py.efunctions, ef_r),
    ("evalues",      res_py.evalues,    ev_r),
    ("scores",       res_py.scores,     sc_r),
    ("sigma2",       np.array([res_py.sigma2]), np.array([sig2_r])),
    ("npc",          np.array([res_py.npc]),    np.array([npc_r])),
]

print()
print(f"{'Output':<20}  {'Shape (wrapper)':<20}  {'Shape (R direct)':<20}  {'Max |diff|'}")
print("-" * 80)

all_passed = True
for name, py_val, r_val in fields:
    diff = np.max(np.abs(py_val - r_val))
    shape_match = "✓" if py_val.shape == r_val.shape else "✗ SHAPE MISMATCH"
    print(f"{name:<20}  {str(py_val.shape):<20}  {str(r_val.shape):<20}  {diff:.3e}  {shape_match}")

    try:
        npt.assert_allclose(py_val, r_val, atol=ATOL)
    except AssertionError as e:
        print(f"  FAILED: {e}")
        all_passed = False

print()
if all_passed:
    print(f"All outputs agree to atol={ATOL:.0e}. Bridge is numerically exact.")
else:
    print("WARNING: one or more outputs exceed tolerance. Investigate before use.")

print()
print("── Values for Table 5.2 ─────────────────────────────────────────────")
for name, py_val, r_val in fields[:4]:   # mu, efunctions, evalues, scores
    diff = np.max(np.abs(py_val - r_val))
    print(f"  {name:<20}: {diff:.3e}")
