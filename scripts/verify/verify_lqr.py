"""
Regression self-tests for the LQR design (mirrors verify_segway / verify_controller).

Checks, all printed as residuals that should be ~0 (or PASS):
    1. analytic (A, B) == central finite differences of f_nl at (x*=0, u*=0),
    2. CARE residual  A^T P + P A - P B R^-1 B^T P + Q  ~ 0,
    3. closed-loop A - B K is Hurwitz (all Re(eig) < 0),
    4. P is symmetric positive definite,
    5. closed-loop Lyapunov identity  Acl^T P + P Acl == -(Q + K^T R K),
    6. nonlinear Vdot < 0 in a small neighbourhood of the origin.
"""

from __future__ import annotations

import numpy as np

# scripts/ live two levels below the project root; make `src` importable
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams, SegwayDynamics
from src.lqr import LQRController, LQRWeights, linearize_upright

p = SegwayParams()
dyn = SegwayDynamics(p)
ctrl = LQRController(p, LQRWeights())

# 1. linearization vs finite differences
A, B = linearize_upright(p)
x0, u0, eps = np.zeros(4), 0.0, 1e-6
Afd = np.zeros((4, 4))
for j in range(4):
    e = np.zeros(4); e[j] = eps
    Afd[:, j] = (dyn.rhs(x0 + e, u0) - dyn.rhs(x0 - e, u0)) / (2 * eps)
Bfd = ((dyn.rhs(x0, u0 + eps) - dyn.rhs(x0, u0 - eps)) / (2 * eps)).reshape(4, 1)
print("1. max|A - A_fd| =", f"{np.abs(A - Afd).max():.2e}",
      "  max|B - B_fd| =", f"{np.abs(B - Bfd).max():.2e}")

# 2. CARE residual
Q, R = ctrl.w.Q, np.atleast_2d(ctrl.w.R)
P = ctrl.P
care = A.T @ P + P @ A - P @ B @ np.linalg.inv(R) @ B.T @ P + Q
print("2. max|CARE residual| =", f"{np.abs(care).max():.2e}")

# 3. closed-loop Hurwitz
ev = np.linalg.eigvals(ctrl.Acl)
print("3. max Re(eig Acl) =", f"{ev.real.max():.4f}",
      "-> Hurwitz" if ev.real.max() < 0 else "-> NOT Hurwitz")

# 4. P SPD
sym = np.abs(P - P.T).max()
mineig = np.linalg.eigvalsh(P).min()
print("4. P asymmetry =", f"{sym:.2e}", "  min eig(P) =", f"{mineig:.4e}",
      "-> SPD" if (sym < 1e-9 and mineig > 0) else "-> NOT SPD")

# 5. closed-loop Lyapunov identity
lhs = ctrl.Acl.T @ P + P @ ctrl.Acl
rhs = -(Q + ctrl.K.T @ R @ ctrl.K)
print("5. max|Acl^T P + P Acl + (Q+K^T R K)| =", f"{np.abs(lhs - rhs).max():.2e}")

# 6. nonlinear Vdot < 0 near the origin
rng = np.random.default_rng(1)
bad = 0
for _ in range(2000):
    x = rng.standard_normal(4) * 0.01
    if ctrl.Vdot(x) >= 0:
        bad += 1
print("6. nonlinear Vdot>=0 among 2000 tiny states:", bad,
      "-> PASS" if bad == 0 else "-> FAIL")
