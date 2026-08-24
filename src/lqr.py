"""
LQR controller for the segway, designed on the linearization about the
upright equilibrium x* = (theta, psi, dtheta, dpsi) = 0, u* = 0.

This module is the basis for the "where does LQR fail" study (see roa_lqr.py
and verify_lqr_roa.py).  It provides

    * the analytic linearization (A, B) of SegwayDynamics.rhs at the origin,
    * a numpy-only continuous-time algebraic Riccati (CARE) solver
      (Hamiltonian eigen-decomposition; scipy is not available here),
    * an LQRController with the same .control(state) / .diagnostics(state)
      interface as LyapunovController, so it drops straight into simulate(),
    * the linearization residual g(x) = f_nl(x, -K x) - A_cl x and the
      Lyapunov rate Vdot(x) = 2 x^T P f_nl(x, -K x) used to certify the
      region of attraction.

Sign/notation matches CLAUDE.md and src/system.py.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .system import SegwayParams, SegwayDynamics


# ---------------------------------------------------------------------------
# CARE solver (numpy only)
# ---------------------------------------------------------------------------
def solve_care(A, B, Q, R):
    """Solve A^T P + P A - P B R^-1 B^T P + Q = 0 for the stabilizing P.

    Uses the Hamiltonian matrix
        H = [[A, -B R^-1 B^T], [-Q, -A^T]]
    whose stable invariant subspace [X1; X2] gives P = X2 X1^-1.
    Returns the symmetric positive-definite stabilizing solution.
    """
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    Q = np.asarray(Q, float)
    R = np.atleast_2d(np.asarray(R, float))
    n = A.shape[0]
    Rinv = np.linalg.inv(R)
    G = B @ Rinv @ B.T
    H = np.block([[A, -G], [-Q, -A.T]])

    w, V = np.linalg.eig(H)
    # Select the n eigenvectors with strictly-negative real-part eigenvalues
    # (the stable invariant subspace).
    order = np.argsort(w.real)
    stable = order[:n]
    if not np.all(w.real[stable] < 0):
        raise ValueError("CARE: could not find n stable eigenvalues "
                         "(is (A, B) stabilizable?)")
    U = V[:, stable]
    X1 = U[:n, :]
    X2 = U[n:, :]
    P = X2 @ np.linalg.inv(X1)
    P = P.real
    P = 0.5 * (P + P.T)        # symmetrize
    return P


# ---------------------------------------------------------------------------
# Linearization of the plant about the upright equilibrium
# ---------------------------------------------------------------------------
def linearize_upright(p: SegwayParams):
    """Return (A, B) of dx/dt = A x + B u linearized at x* = 0, u* = 0.

    Derivation (CLAUDE.md sec. 2.3, evaluated at theta = 0):
        a0 = alpha + 2 beta + gamma,  b0 = gamma + beta,  c0 = gamma,
        det = a0 c0 - b0^2 = alpha gamma - beta^2 = Delta(0).
    The quadratic velocity terms (beta sin(theta) dtheta^2) are second order
    and drop out, leaving constant mass matrix M0 = [[a0, b0], [b0, c0]] and
    forcing (D theta, -u):
        ddtheta = ( gamma D / det) theta + ( (gamma+beta) / det) u
        ddpsi   = (-(gamma+beta) D / det) theta + (-(alpha+2 beta+gamma)/det) u
    """
    a0 = p.alpha + 2.0 * p.beta + p.gamma
    b0 = p.gamma + p.beta
    c0 = p.gamma
    det = a0 * c0 - b0 ** 2            # == alpha*gamma - beta**2 == Delta(0)

    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0] = c0 * p.D / det           # gamma D / det
    A[3, 0] = -b0 * p.D / det          # -(gamma+beta) D / det

    B = np.zeros((4, 1))
    B[2, 0] = b0 / det                 # (gamma+beta)/det
    B[3, 0] = -a0 / det                # -(alpha+2 beta+gamma)/det
    return A, B


# ---------------------------------------------------------------------------
# LQR controller
# ---------------------------------------------------------------------------
@dataclass
class LQRWeights:
    """State and input weights for the LQR cost int (x^T Q x + u^T R u) dt.

    Default Q penalizes tilt heavily, then tilt rate, with light weights on
    the cyclic wheel coordinate (psi, dpsi) which need not be regulated hard.
    """
    Q: np.ndarray = field(default_factory=lambda: np.diag([50.0, 0.1, 5.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))


class LQRController:
    """Full-state-feedback LQR u = -K x for the segway.

    Interface mirrors LyapunovController so simulate() can use it unchanged.
    """

    def __init__(self, params: SegwayParams,
                 weights: Optional[LQRWeights] = None):
        self.p = params
        self.w = weights if weights is not None else LQRWeights()
        self.dyn = SegwayDynamics(params)

        self.A, self.B = linearize_upright(params)
        self.P = solve_care(self.A, self.B, self.w.Q, self.w.R)
        Rinv = np.linalg.inv(np.atleast_2d(self.w.R))
        self.K = Rinv @ self.B.T @ self.P          # (1 x 4)
        self.Acl = self.A - self.B @ self.K

        # Closed-loop Lyapunov decay matrix: Acl^T P + P Acl = -(Q + K^T R K)
        self.Qcl = self.w.Q + self.K.T @ np.atleast_2d(self.w.R) @ self.K

    # ---- control law -------------------------------------------------------
    def control(self, state) -> float:
        x = np.asarray(state, float)
        return float(-(self.K @ x)[0])

    # ---- Lyapunov / residual diagnostics -----------------------------------
    def V(self, state) -> float:
        x = np.asarray(state, float)
        return float(x @ self.P @ x)

    def f_closed(self, state):
        """Nonlinear closed-loop vector field f_nl(x, -K x)."""
        x = np.asarray(state, float)
        return self.dyn.rhs(x, self.control(x))

    def residual(self, state):
        """g(x) = f_nl(x, -K x) - Acl x  (the linearization error)."""
        x = np.asarray(state, float)
        return self.f_closed(x) - self.Acl @ x

    def Vdot(self, state) -> float:
        """Vdot along the *nonlinear* closed loop = 2 x^T P f_nl(x, -K x).

        Equivalently -x^T Qcl x + 2 x^T P g(x): the nominal LQR decay minus
        the projected linearization error.  Vdot >= 0 marks states where the
        error has overwhelmed the nominal decay (LQR no longer certified).
        """
        x = np.asarray(state, float)
        return float(2.0 * x @ self.P @ self.f_closed(x))


    def f_batch(self, X, u_max: float = 0.0):
        """Vectorized nonlinear closed-loop field f_nl(x, -K x), X of shape (N, 4).

        Optional torque saturation |u| <= u_max (u_max <= 0 means unsaturated).
        Diverging batch cells can overflow intermediates before the caller
        latches them as fallen; that is benign, so float errors are silenced
        locally rather than warned about."""
        X = np.atleast_2d(np.asarray(X, float))
        p = self.p
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            th, ps, dth, dps = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
            ct, st = np.cos(th), np.sin(th)
            a = p.alpha + 2.0 * p.beta * ct + p.gamma
            b = p.gamma + p.beta * ct
            c = p.gamma
            det2 = a * c - b * b
            K = self.K.ravel()
            u = -(K[0] * th + K[1] * ps + K[2] * dth + K[3] * dps)
            if u_max and u_max > 0.0:
                u = np.clip(u, -u_max, u_max)
            r1 = p.beta * st * dth ** 2 + p.D * st
            r2 = p.beta * st * dth ** 2 - u
            ddth = (c * r1 - b * r2) / det2
            ddps = (-b * r1 + a * r2) / det2
        return np.stack([dth, dps, ddth, ddps], axis=1)

    def diagnostics(self, state):
        x = np.asarray(state, float)
        u = self.control(x)
        return dict(u=u, v=0.0, xi=0.0, dxi=0.0,
                    V=self.V(x), Vdot=self.Vdot(x))

    def __call__(self, state) -> float:
        return self.control(state)
