"""
Lyapunov-based controller for the segway, derived in
Segway_Lyapunov_Controller.docx, §4-6.

The closed-loop satisfies V_dot = -dxi^2 (eq. 24).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .system import SegwayParams


@dataclass
class ControllerGains:
    """Positive design gains.  Must satisfy the gain condition (27)."""
    kp: float = 100.0
    ki: float = 1.0
    kl: float = 0.5


class LyapunovController:
    def __init__(self, params: SegwayParams,
                 gains: Optional[ControllerGains] = None,
                 *, check_gains: bool = True):
        self.p = params
        self.g = gains if gains is not None else ControllerGains()
        if check_gains:
            self.check_gain_condition()

    # ---------- Gain & domain bookkeeping ----------
    def check_gain_condition(self) -> None:
        """Raise ValueError if gain condition (27) is violated."""
        p, g = self.p, self.g
        lhs = g.kp * (p.gamma + p.beta) ** 2
        rhs = (1.0 + g.kl) * (p.alpha + 2.0 * p.beta + p.gamma)
        if lhs <= rhs:
            raise ValueError(
                f"Gain condition (27) violated: kp(gamma+beta)^2 = {lhs:.4g} "
                f"<= (1+kl)(alpha+2 beta+gamma) = {rhs:.4g}"
            )

    def gain_condition_value(self) -> Tuple[float, float]:
        p, g = self.p, self.g
        return (g.kp * (p.gamma + p.beta) ** 2,
                (1.0 + g.kl) * (p.alpha + 2.0 * p.beta + p.gamma))

    def critical_angle(self) -> float:
        """Smallest |theta|>0 such that delta(theta)=0, eq. (28)."""
        p, g = self.p, self.g
        ths = np.linspace(0.0, np.pi / 2.0, 10000)
        a = p.alpha + 2.0 * p.beta * np.cos(ths) + p.gamma
        b = p.gamma + p.beta * np.cos(ths)
        d = g.kl + 1.0 - g.kp * b**2 / a
        sign_change = np.where(np.diff(np.sign(d)) != 0)[0]
        return float(ths[sign_change[0]]) if len(sign_change) else float(np.pi / 2.0)

    def K_S(self) -> float:
        """Attraction-region threshold K_S = k_l k_p D (1 - cos theta_S), eq. (30)."""
        return float(self.g.kl * self.g.kp * self.p.D * (1.0 - np.cos(self.critical_angle())))

    # ---------- Local quantities ----------
    def _coefs(self, theta: float):
        p = self.p
        a = p.alpha + 2.0 * p.beta * np.cos(theta) + p.gamma
        b = p.gamma + p.beta * np.cos(theta)
        return a, b, p.gamma

    def diagnostics(self, state):
        """Return dict with u, v, xi, dxi, eta, delta, V."""
        theta, psi, dth, dps = state
        p, g = self.p, self.g
        a, b, _ = self._coefs(theta)
        Delta = p.alpha * p.gamma - p.beta**2 * np.cos(theta)**2
        delta = g.kl + 1.0 - g.kp * b**2 / a

        # eq. (16)
        xi  = psi + g.kp * (p.gamma * theta + p.beta * np.sin(theta))
        dxi = dps + g.kp * b * dth
        # eq. (21)
        eta = g.ki * xi + (g.kp * np.sin(theta) / a) * (
            p.D * b - p.beta * dth**2 * (p.alpha + p.beta * np.cos(theta))
        )
        # eq. (23) and (25)
        v = -(dxi + eta) / delta
        u = p.beta * np.sin(theta) * dth**2 - (
            b * np.sin(theta) * (p.D + p.beta * dth**2) + Delta * v
        ) / a
        # Lyapunov function (eq. 17, 19)
        Phi = g.kp * p.D * (1.0 - np.cos(theta)) - 0.5 * g.kp * a * dth**2 + 0.5 * dps**2
        V = 0.5 * g.ki * xi**2 + 0.5 * dxi**2 + g.kl * Phi
        return dict(u=u, v=v, xi=xi, dxi=dxi, eta=eta, delta=delta, V=V, Phi=Phi)

    def control(self, state) -> float:
        """Return the motor torque u for the given state."""
        return self.diagnostics(state)["u"]

    def V(self, state) -> float:
        """Lyapunov function value at the given state."""
        return self.diagnostics(state)["V"]

    def __call__(self, state) -> float:
        return self.control(state)


class SaturatedController:
    """Thin wrapper that clips the applied torque to ``|u| <= u_max``.

    Used to make the interactive animation physically consistent with the
    recoverable-set map (which is computed under the same torque limit).

    NOTE: the closed-loop identity ``V_dot = -dxi^2`` (eq. 24) only holds for
    the *unsaturated* Lyapunov law.  Once the torque is clipped the logged
    ``Vdot`` (still reported as ``-dxi^2`` for reference) is no longer exact;
    the numerical ``dV/dt`` is the truthful one.  All other diagnostics
    (xi, dxi, V, eta, delta) are unaffected -- only the applied ``u`` changes.
    """

    def __init__(self, base: LyapunovController, u_max: float):
        if u_max is None or u_max <= 0.0:
            raise ValueError("SaturatedController requires u_max > 0")
        self.base = base
        self.u_max = float(u_max)

    def _clip(self, u: float) -> float:
        return float(np.clip(u, -self.u_max, self.u_max))

    def control(self, state) -> float:
        return self._clip(self.base.control(state))

    def diagnostics(self, state) -> dict:
        d = dict(self.base.diagnostics(state))
        d["u"] = self._clip(d["u"])
        return d

    def V(self, state) -> float:
        return self.base.V(state)

    def __getattr__(self, name):
        # Delegate critical_angle, K_S, gain_condition_value,
        # check_gain_condition, p, g, ... to the wrapped controller.
        return getattr(self.base, name)
