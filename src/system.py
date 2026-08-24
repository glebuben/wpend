"""
Segway plant model.

State convention used throughout the package:
    x = (theta, psi, dtheta, dpsi),
where psi = phi - theta is the wheel angle relative to the body
(eq. 9 of Segway_Lyapunov_Controller.docx).  The motor torque is u,
and tau = (u, -u)^T in the original (theta, phi) coordinates.

The equations of motion are
    M_tilde(theta) * (ddtheta, ddpsi)^T + (-beta sin(theta) dtheta^2 - D sin(theta),
                                            -beta sin(theta) dtheta^2)^T = (0, -u)^T
which is the report's (12a)-(12b).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class SegwayParams:
    """Physical parameters of the segway plant.

    Defaults match cell 49 of AIDA_project.ipynb (slender rod body, small wheel)."""
    mb: float = 10.0   # body mass            [kg]
    mw: float = 1.0    # wheel mass           [kg]
    l:  float = 0.30   # body CG offset from wheel axle [m]
    r:  float = 0.05   # wheel radius         [m]
    g:  float = 9.8    # gravitational accel  [m/s^2]

    # Centroidal inertia of the body (slender rod about its CG)
    def _Ib(self) -> float: return self.mb * self.l**2 / 3.0
    # Wheel inertia about its centre
    def _Iw(self) -> float: return 0.7 * self.mw * self.r**2

    @property
    def Ib(self) -> float: return self._Ib()
    @property
    def Iw(self) -> float: return self._Iw()

    # Lumped parameters (eq. 4 of the report)
    @property
    def alpha(self) -> float: return self.Ib + self.mb * self.l**2
    @property
    def beta(self)  -> float: return self.mb * self.r * self.l
    @property
    def gamma(self) -> float: return self.Iw + (self.mw + self.mb) * self.r**2
    @property
    def D(self)     -> float: return self.mb * self.g * self.l


class SegwayDynamics:
    """Right-hand side of the segway ODE."""

    def __init__(self, params: Optional[SegwayParams] = None):
        self.p = params if params is not None else SegwayParams()

    def coefs(self, theta: float):
        """Configuration-dependent scalars (a(theta), b(theta), c) of eq. (11)."""
        p = self.p
        a = p.alpha + 2.0 * p.beta * np.cos(theta) + p.gamma
        b = p.gamma + p.beta * np.cos(theta)
        c = p.gamma
        return a, b, c

    def Delta(self, theta: float) -> float:
        """det M_tilde, eq. (13)."""
        p = self.p
        return p.alpha * p.gamma - p.beta**2 * np.cos(theta)**2

    def rhs(self, state, u: float):
        """ODE right-hand side dx/dt = f(x, u)."""
        theta, psi, dth, dps = state
        p = self.p
        a, b, c = self.coefs(theta)
        # Solve M_tilde * (ddtheta, ddpsi) = rhs_vec
        rhs_vec = np.array([
            p.beta * np.sin(theta) * dth**2 + p.D * np.sin(theta),   # eq. (12a) rearranged
            p.beta * np.sin(theta) * dth**2 - u,                     # eq. (12b) rearranged
        ])
        M = np.array([[a, b], [b, c]])
        ddth, ddps = np.linalg.solve(M, rhs_vec)
        return np.array([dth, dps, ddth, ddps])
