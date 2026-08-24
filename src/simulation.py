"""
Closed-loop simulator: integrates SegwayDynamics under a feedback controller.

Default integrator is fixed-step RK4 (the dynamics are smooth, no stiffness),
chosen because scipy is not always available.
"""

from __future__ import annotations

import numpy as np

from .system import SegwayDynamics
from .controller import LyapunovController


def rk4_step(f, t, y, h):
    """One classical Runge-Kutta-4 step."""
    k1 = f(t, y)
    k2 = f(t + h / 2.0, y + h * k1 / 2.0)
    k3 = f(t + h / 2.0, y + h * k2 / 2.0)
    k4 = f(t + h,       y + h * k3)
    return y + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def simulate(dynamics: SegwayDynamics,
             controller: LyapunovController,
             state0,
             T: float,
             *,
             n_out: int = 1200,
             inner_dt: float = 5e-4) -> dict:
    """Closed-loop simulation.

    Parameters
    ----------
    dynamics    : SegwayDynamics instance
    controller  : LyapunovController instance
    state0      : iterable [theta0, psi0, dtheta0, dpsi0]
    T           : final time (seconds)
    n_out       : number of output samples (uniform grid 0..T)
    inner_dt    : RK4 inner step size (s)

    Returns
    -------
    dict with keys
        t, theta, psi, phi, dtheta, dpsi, dphi, u, v, xi, dxi, V, Vdot
    """
    t_eval = np.linspace(0.0, T, n_out)
    Y      = np.zeros((4, n_out))

    state = np.asarray(state0, dtype=float)
    d0 = controller.diagnostics(state)
    # Some controllers (e.g. LQR) report their own analytic Vdot; the Lyapunov
    # controller does not and we fall back to the closed-form -dxi^2 below.
    has_vdot = "Vdot" in d0
    diag_keys = ("u", "v", "xi", "dxi", "V") + (("Vdot",) if has_vdot else ())
    Diag   = {k: np.zeros(n_out) for k in diag_keys}

    Y[:, 0] = state
    for k in Diag:
        Diag[k][0] = d0[k]

    def rhs(t, y):
        u = controller.control(y)
        return dynamics.rhs(y, u)

    t_prev = 0.0
    for i in range(1, n_out):
        t_next = t_eval[i]
        h_macro = t_next - t_prev
        n_inner = max(1, int(np.ceil(h_macro / inner_dt)))
        h = h_macro / n_inner
        tt = t_prev
        for _ in range(n_inner):
            state = rk4_step(rhs, tt, state, h)
            tt += h
        Y[:, i] = state
        di = controller.diagnostics(state)
        for k in Diag:
            Diag[k][i] = di[k]
        t_prev = t_next

    theta, psi, dth, dps = Y
    # Vdot: prefer the controller's own analytic value (LQR), otherwise use the
    # Lyapunov closed form -dxi^2 (eq. 24).
    Vdot = Diag["Vdot"] if has_vdot else -Diag["dxi"]**2
    return dict(
        t      = t_eval,
        theta  = theta,
        psi    = psi,
        phi    = psi + theta,           # eq. 9 inverted
        dtheta = dth,
        dpsi   = dps,
        dphi   = dps + dth,
        u      = Diag["u"],
        v      = Diag["v"],
        xi     = Diag["xi"],
        dxi    = Diag["dxi"],
        V      = Diag["V"],
        Vdot   = Vdot,
    )
