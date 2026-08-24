"""
Region of attraction for an LQR designed on the *pure 2-D tilt subsystem*.

Motivation (chat / CLAUDE.md sec. 2.3 and 9.x): falling is a theta-only
phenomenon -- the reduced tilt ODE

    ddtheta = ( b(theta) u + gamma D sin(theta)
                - beta^2 sin(theta) cos(theta) dtheta^2 ) / Delta(theta)

does not involve psi or dpsi at all (the wheel coordinate is cyclic).  So the
honest, un-sliced picture of "where does LQR keep the body up" lives in the
2-D state (theta, dtheta).  Here we:

    1. design an LQR on the 2x2 linearization A2, B2 of the tilt subsystem,
    2. take V(x) = x^T P x (P from the 2-D CARE) as the Lyapunov function,
    3. compute Vdot along the *nonlinear* 2-D closed loop u = -K2 x,
    4. certify the largest sublevel set {V <= c*} inside {Vdot < 0},
    5. simulate the true 2-D basin from a grid for ground truth,
    6. draw the explicit Vdot = 0 curve.

Writes figures/fig_lqr_roa_tilt.png and prints soundness + conservatism.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

# scripts/ live two levels below the project root; make `src` importable
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams
from src.lqr import solve_care
from src.recoverable import theta_ddot as _theta_ddot

THETA_MAX = np.pi / 2.0
THETA_FALL = 1.45


# --------------------------------------------------------------------------
# Reduced 2-D tilt dynamics  (theta, dtheta), input u
# --------------------------------------------------------------------------
def linearize_tilt(p):
    """2x2 (A2, B2) of the tilt subsystem about theta = 0."""
    Delta0 = p.alpha * p.gamma - p.beta ** 2          # Delta(0)
    A2 = np.array([[0.0, 1.0],
                   [p.gamma * p.D / Delta0, 0.0]])
    B2 = np.array([[0.0],
                   [(p.gamma + p.beta) / Delta0]])
    return A2, B2


class TiltLQR:
    """LQR on the 2-D tilt subsystem; u = -K2 (theta, dtheta)."""

    def __init__(self, p, Q=None, R=None):
        self.p = p
        self.Q = np.diag([50.0, 5.0]) if Q is None else np.asarray(Q, float)
        self.R = np.array([[1.0]]) if R is None else np.atleast_2d(R)
        self.A, self.B = linearize_tilt(p)
        self.P = solve_care(self.A, self.B, self.Q, self.R)
        self.K = np.linalg.inv(self.R) @ self.B.T @ self.P    # (1 x 2)
        self.Acl = self.A - self.B @ self.K

    def u(self, theta, dtheta):
        return -(self.K[0, 0] * theta + self.K[0, 1] * dtheta)

    def vdot_grid(self, TH, DTH):
        """Vdot = 2 x^T P f_nl over a (theta, dtheta) mesh."""
        u = self.u(TH, DTH)
        ddth = _theta_ddot(TH, DTH, u, self.p)
        P = self.P
        # 2 (x^T P) . f,  with x = (TH, DTH), f = (DTH, ddth)
        px0 = P[0, 0] * TH + P[0, 1] * DTH
        px1 = P[1, 0] * TH + P[1, 1] * DTH
        return 2.0 * (px0 * DTH + px1 * ddth)


# --------------------------------------------------------------------------
# Certified sublevel set in 2-D (dense angular line search)
# --------------------------------------------------------------------------
def certified_level_2d(ctrl, *, n_dirs=4000, s_max=16.0, ds=2e-3):
    angs = np.linspace(0.0, 2 * np.pi, n_dirs, endpoint=False)
    D = np.stack([np.cos(angs), np.sin(angs)], axis=1)       # (n,2)
    P = ctrl.P
    dPd = np.einsum("ij,jk,ik->i", D, P, D)
    c_dir = np.full(n_dirs, np.inf)
    active = np.ones(n_dirs, bool)
    s = ds
    while s <= s_max and active.any():
        idx = np.where(active)[0]
        TH = s * D[idx, 0]
        DTH = s * D[idx, 1]
        left = np.abs(TH) >= THETA_MAX
        vd = ctrl.vdot_grid(TH, DTH)
        cross = (vd >= 0.0) & ~left
        c_dir[idx[cross]] = s * s * dPd[idx[cross]]
        active[idx[cross | left]] = False
        s += ds
    c_star = float(c_dir.min())
    return c_star


def roa_ellipse_2d(P, c, th_fine):
    """dtheta(theta) on ellipse P00 th^2 + 2 P01 th dth + P11 dth^2 = c."""
    a = P[1, 1]
    b = 2.0 * P[0, 1] * th_fine
    cc = P[0, 0] * th_fine ** 2 - c
    disc = b ** 2 - 4.0 * a * cc
    m = disc >= 0
    sq = np.sqrt(np.where(m, disc, 0.0))
    upper = np.full_like(th_fine, np.nan)
    lower = np.full_like(th_fine, np.nan)
    upper[m] = (-b[m] + sq[m]) / (2.0 * a)
    lower[m] = (-b[m] - sq[m]) / (2.0 * a)
    return upper, lower, m


# --------------------------------------------------------------------------
# True 2-D basin by simulation
# --------------------------------------------------------------------------
def simulate_basin(ctrl, TH0, DTH0, *, T=6.0, dt=2.5e-3):
    th = TH0.astype(float).copy()
    dth = DTH0.astype(float).copy()
    fallen = np.zeros_like(th, dtype=bool)
    n = int(T / dt)

    def deriv(th, dth):
        return dth, _theta_ddot(th, dth, ctrl.u(th, dth), ctrl.p)

    for _ in range(n):
        fallen |= np.abs(th) >= THETA_FALL
        act = ~fallen
        if not act.any():
            break
        k1 = deriv(th, dth)
        k2 = deriv(th + 0.5 * dt * k1[0], dth + 0.5 * dt * k1[1])
        k3 = deriv(th + 0.5 * dt * k2[0], dth + 0.5 * dt * k2[1])
        k4 = deriv(th + dt * k3[0], dth + dt * k3[1])
        th_n = th + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        dth_n = dth + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        th = np.where(act, th_n, th)
        dth = np.where(act, dth_n, dth)
    return ((~fallen) & (np.abs(th) < 0.05) & (np.abs(dth) < 0.1)).astype(int)


def main():
    p = SegwayParams()
    ctrl = TiltLQR(p)
    print("2-D tilt-subsystem LQR")
    print("  A2 =", ctrl.A.tolist())
    print("  B2 =", ctrl.B.ravel().tolist())
    print("  K2 =", np.array2string(ctrl.K.ravel(), precision=4))
    print("  closed-loop poles =",
          np.array2string(np.linalg.eigvals(ctrl.Acl), precision=3))
    print("  P =", np.array2string(ctrl.P, precision=4))

    c_star = certified_level_2d(ctrl)
    th_extent = np.sqrt(c_star / (ctrl.P[0, 0] - ctrl.P[0, 1] ** 2 / ctrl.P[1, 1]))
    print("\nCertified ROA level c* = %.4f" % c_star)
    print("  -> certified tilt extent |theta| <= %.3f rad (%.1f deg)"
          % (th_extent, np.degrees(th_extent)))

    th_b, dth_b, N = 1.55, 14.0, 300
    th_grid = np.linspace(-th_b, th_b, N)
    dth_grid = np.linspace(-dth_b, dth_b, N)
    TH, DTH = np.meshgrid(th_grid, dth_grid)

    import time
    t0 = time.perf_counter()
    conv = simulate_basin(ctrl, TH, DTH)
    print("grid sim wall-time: %.2f s" % (time.perf_counter() - t0))

    Vd = ctrl.vdot_grid(TH, DTH)
    Vslice = (ctrl.P[0, 0] * TH ** 2 + 2 * ctrl.P[0, 1] * TH * DTH
              + ctrl.P[1, 1] * DTH ** 2)
    inside = Vslice <= c_star
    cert = inside.sum()
    cert_conv = (inside & (conv == 1)).sum()
    cell = (th_grid[1] - th_grid[0]) * (dth_grid[1] - dth_grid[0])
    print("certified points converge: %d / %d  (%.2f%%)"
          % (cert_conv, cert, 100.0 * cert_conv / max(cert, 1)))
    print("slice areas: basin = %.3f, certified = %.3f -> captures %.1f%%"
          % (conv.sum() * cell, inside.sum() * cell,
             100.0 * inside.sum() / max(conv.sum(), 1)))

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cmap = ListedColormap([(0.95, 0.6, 0.6), (0.75, 0.9, 0.75)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(conv, extent=[-th_b, th_b, -dth_b, dth_b], origin="lower",
              aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
              alpha=0.80)

    cs = ax.contour(TH, DTH, Vd, levels=[0.0], colors="k", linewidths=1.8,
                    linestyles="--")

    th_fine = np.linspace(-th_b, th_b, 4000)
    up, lo, m = roa_ellipse_2d(ctrl.P, c_star, th_fine)
    ax.plot(th_fine[m], up[m], "b-", lw=2.4)
    ax.plot(th_fine[m], lo[m], "b-", lw=2.4)

    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-th_b, th_b)
    ax.set_ylim(-dth_b, dth_b)
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\dot\theta$  [rad/s]")
    ax.set_title("LQR on the 2-D tilt subsystem: basin, certified ROA, "
                 r"and $\dot V = 0$")

    handles = [
        Patch(color=(0.75, 0.9, 0.75), label="converges (true basin, sim)"),
        Patch(color=(0.95, 0.6, 0.6), label="diverges / falls (sim)"),
        plt.Line2D([], [], color="b", lw=2.4, label=r"certified ROA $x^TPx=c^*$"),
        plt.Line2D([], [], color="k", lw=1.8, ls="--", label=r"$\dot V = 0$ curve"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.93)

    out = ROOT / "figures" / "fig_lqr_roa_tilt.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
