"""
LQR design in the ORIGINAL physical coordinates  x = (theta, phi, dtheta, dphi),
with NO psi = phi - theta substitution.

The wheel angle phi is kept as-is, so every plot made from this module has a
genuine dphi (wheel-velocity) axis.  The tilt physics is identical to the
psi-based model (theta-acceleration is phi-independent); only the second
coordinate and the input distribution differ.

EOM (CLAUDE.md sec. 2.3), with motor torque tau = (u, -u):
    M_phi(theta) (ddtheta, ddphi)^T = ( D sin th + u ,  beta sin th dth^2 - u )^T
    M_phi = [[alpha, beta cos th], [beta cos th, gamma]],
    Delta(theta) = alpha gamma - beta^2 cos^2 th.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .system import SegwayParams
from .lqr import solve_care


# --------------------------------------------------------------------------
# Nonlinear dynamics in (theta, phi)
# --------------------------------------------------------------------------
class SegwayDynamicsPhys:
    def __init__(self, params: SegwayParams):
        self.p = params

    def rhs(self, state, u: float):
        th, phi, dth, dphi = state
        p = self.p
        m12 = p.beta * np.cos(th)
        Delta = p.alpha * p.gamma - p.beta ** 2 * np.cos(th) ** 2
        F1 = p.D * np.sin(th) + u
        F2 = p.beta * np.sin(th) * dth ** 2 - u
        ddth = (p.gamma * F1 - m12 * F2) / Delta
        ddphi = (-m12 * F1 + p.alpha * F2) / Delta
        return np.array([dth, dphi, ddth, ddphi])


def linearize_upright_phys(p: SegwayParams):
    """(A, B) of dx/dt = A x + B u at x*=0, u*=0 in (theta, phi) coords."""
    Delta0 = p.alpha * p.gamma - p.beta ** 2          # Delta(0) = alpha gamma - beta^2
    A = np.zeros((4, 4))
    A[0, 2] = 1.0
    A[1, 3] = 1.0
    A[2, 0] = p.gamma * p.D / Delta0                  # ddtheta wrt theta
    A[3, 0] = -p.beta * p.D / Delta0                  # ddphi   wrt theta
    B = np.zeros((4, 1))
    B[2, 0] = (p.gamma + p.beta) / Delta0             # ddtheta wrt u
    B[3, 0] = -(p.alpha + p.beta) / Delta0            # ddphi   wrt u
    return A, B


@dataclass
class LQRWeightsPhys:
    # weights on (theta, phi, dtheta, dphi)
    Q: np.ndarray = field(default_factory=lambda: np.diag([50.0, 0.1, 5.0, 0.1]))
    R: np.ndarray = field(default_factory=lambda: np.array([[1.0]]))


class LQRControllerPhys:
    def __init__(self, params: SegwayParams, weights: LQRWeightsPhys = None):
        self.p = params
        self.w = weights if weights is not None else LQRWeightsPhys()
        self.dyn = SegwayDynamicsPhys(params)
        self.A, self.B = linearize_upright_phys(params)
        self.P = solve_care(self.A, self.B, self.w.Q, self.w.R)
        Rinv = np.linalg.inv(np.atleast_2d(self.w.R))
        self.K = Rinv @ self.B.T @ self.P
        self.Acl = self.A - self.B @ self.K
        self.Qcl = self.w.Q + self.K.T @ np.atleast_2d(self.w.R) @ self.K

    def control(self, state) -> float:
        return float(-(self.K @ np.asarray(state, float))[0])

    def V(self, state) -> float:
        x = np.asarray(state, float)
        return float(x @ self.P @ x)

    def f_closed(self, state):
        x = np.asarray(state, float)
        return self.dyn.rhs(x, self.control(x))

    def Vdot(self, state) -> float:
        x = np.asarray(state, float)
        return float(2.0 * x @ self.P @ self.f_closed(x))

    def f_batch(self, X, u_max: float = 0.0):
        """Vectorized nonlinear closed-loop field f_nl(x, -K x) in
        (theta, phi, dtheta, dphi) coordinates, X of shape (N, 4).
        Optional saturation |u| <= u_max (u_max <= 0 means unsaturated)."""
        X = np.atleast_2d(np.asarray(X, float))
        p = self.p
        th, phi, dth, dphi = X[:, 0], X[:, 1], X[:, 2], X[:, 3]
        ct, st = np.cos(th), np.sin(th)
        m12 = p.beta * ct
        Delta = p.alpha * p.gamma - p.beta ** 2 * ct ** 2
        u = -(X @ self.K.T).ravel()
        if u_max and u_max > 0.0:
            u = np.clip(u, -u_max, u_max)
        F1 = p.D * st + u
        F2 = p.beta * st * dth ** 2 - u
        ddth = (p.gamma * F1 - m12 * F2) / Delta
        ddphi = (-m12 * F1 + p.alpha * F2) / Delta
        return np.stack([dth, dphi, ddth, ddphi], axis=1)
