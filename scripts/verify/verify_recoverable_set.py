"""
Numerical verification of the closed-form recoverable-set boundary
H_-(theta, dtheta) = K(u_max).

For each u_max we
    (1) compute the saddle angle theta_eq and the saddle level K,
    (2) sample a (N x N) grid of initial conditions,
    (3) integrate the planar tilt ODE simultaneously from every grid point
        (vectorized RK4) for both u = -u_max and u = +u_max,
    (4) classify each grid point by whether |theta(t)| reaches pi/2 within
        the simulation horizon under each control,
    (5) overlay the closed-form level sets H_+- = K to confirm that the
        analytic curves match the simulation outcomes.

The y-axis range is computed from the boundary at theta = 0 so that the
recoverable region is always fully on screen.
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
from src.recoverable import Delta, H_, find_theta_eq, vectorized_sim

# ---------- plant + dynamics ------------------------------------------------
p = SegwayParams()
alpha, beta_, gamma_, D = p.alpha, p.beta, p.gamma, p.D


def grid_experiment(u_max: float, *, N=60, T=4.5):
    th_eq = find_theta_eq(u_max)
    K = H_(th_eq, 0.0, -u_max)

    # Closed-form bound on dtheta at theta = 0 for the H_- = K level set.
    sq_at_0 = 2.0 * (K - gamma_ * D) / Delta(0.0)
    dth_bound = max(2.0, 1.45 * np.sqrt(max(sq_at_0, 1e-6)))
    th_bound  = 1.50    # always show up to ~ pi/2

    th_grid  = np.linspace(-th_bound, th_bound, N)
    dth_grid = np.linspace(-dth_bound, dth_bound, N)
    TH, DTH  = np.meshgrid(th_grid, dth_grid)

    out_minus = vectorized_sim(TH, DTH, -u_max, T=T)   # max forward brake
    out_plus  = vectorized_sim(TH, DTH, +u_max, T=T)   # max backward brake

    # Combine into 4 categories
    #   0 = survives both                  green
    #   1 = falls fwd under -u_max only    red
    #   2 = falls bwd under +u_max only    blue
    #   3 = falls under both               purple
    sim_code = np.zeros_like(TH, dtype=int)
    fwd_fail = (out_minus ==  1)
    bwd_fail = (out_plus  == -1)
    sim_code[ fwd_fail & ~bwd_fail] = 1
    sim_code[~fwd_fail &  bwd_fail] = 2
    sim_code[ fwd_fail &  bwd_fail] = 3

    # Analytic prediction
    pred_fwd_fail = H_(TH, DTH, -u_max) > K
    pred_bwd_fail = H_(TH, DTH, +u_max) > K
    pred_code = np.zeros_like(TH, dtype=int)
    pred_code[ pred_fwd_fail & ~pred_bwd_fail] = 1
    pred_code[~pred_fwd_fail &  pred_bwd_fail] = 2
    pred_code[ pred_fwd_fail &  pred_bwd_fail] = 3

    match = (sim_code == pred_code).mean()

    print(f"u_max = {u_max:6.2f} N m   "
          f"theta_eq = {np.degrees(th_eq):6.2f} deg   "
          f"K = {K:8.3f}   "
          f"sim/analytic agreement = {100*match:.2f}%   "
          f"y-range = +-{dth_bound:.1f} rad/s")
    return TH, DTH, sim_code, K, th_eq, dth_bound


def plot_experiment(TH, DTH, sim_code, K, th_eq, dth_bound, u_max, ax):
    colors = [(0.78, 0.93, 0.78),    # 0 -- survives
              (0.95, 0.55, 0.55),    # 1 -- falls forward
              (0.55, 0.65, 0.95),    # 2 -- falls backward
              (0.70, 0.50, 0.70)]    # 3 -- both
    cmap = ListedColormap(colors)
    norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(sim_code,
              extent=[TH.min(), TH.max(), DTH.min(), DTH.max()],
              origin="lower", aspect="auto", cmap=cmap, norm=norm,
              interpolation="nearest", alpha=0.78)

    # Closed-form level sets H_- = K  and  H_+ = K
    th_fine = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 6000)
    sq_minus = 2.0 * (K - u_max * (gamma_ * th_fine + beta_ * np.sin(th_fine))
                        - gamma_ * D * np.cos(th_fine)) / Delta(th_fine)
    sq_plus  = 2.0 * (K + u_max * (gamma_ * th_fine + beta_ * np.sin(th_fine))
                        - gamma_ * D * np.cos(th_fine)) / Delta(th_fine)
    m = sq_minus >= 0
    ax.plot(th_fine[m],  np.sqrt(sq_minus[m]), "k-", lw=2,
            label=r"$H_-= K(u_{max})$")
    ax.plot(th_fine[m], -np.sqrt(sq_minus[m]), "k-", lw=2)
    m = sq_plus >= 0
    ax.plot(th_fine[m],  np.sqrt(sq_plus[m]), "k--", lw=2,
            label=r"$H_+ = K(u_{max})$")
    ax.plot(th_fine[m], -np.sqrt(sq_plus[m]), "k--", lw=2)

    ax.scatter([th_eq, -th_eq], [0, 0], s=80, c="black", marker="x", zorder=5,
               label=r"saddles  $(\pm\theta_{eq}, 0)$")
    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.axvline( np.pi / 2, color="red", lw=0.7, ls=":")
    ax.axvline(-np.pi / 2, color="red", lw=0.7, ls=":")
    ax.set_xlim(-1.55, 1.55)
    ax.set_ylim(-dth_bound, dth_bound)
    ax.set_xlabel(r"$\theta_0$  [rad]")
    ax.set_ylabel(r"$\dot\theta_0$  [rad/s]")
    ax.set_title(f"u_max = {u_max:.2f} N m")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)


if __name__ == "__main__":
    import time
    u_maxes = [2.0, 5.0, 15.0, 50.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    t0 = time.perf_counter()
    for ax, um in zip(axes.flat, u_maxes):
        TH, DTH, sim_code, K, th_eq, dth_bound = grid_experiment(um, N=60, T=4.5)
        plot_experiment(TH, DTH, sim_code, K, th_eq, dth_bound, um, ax)
    print(f"\ntotal sim+plot wall-time: {time.perf_counter() - t0:.2f} s")

    legend_handles = [
        Patch(color=(0.78, 0.93, 0.78), label="survives both brake directions"),
        Patch(color=(0.95, 0.55, 0.55), label=r"falls forward under $u=-u_{max}$"),
        Patch(color=(0.55, 0.65, 0.95), label=r"falls backward under $u=+u_{max}$"),
        Patch(color=(0.70, 0.50, 0.70), label="falls under both"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 0.99), frameon=False, fontsize=10)
    fig.suptitle("Grid search verification: simulation outcomes vs. "
                 r"closed-form level sets $H_{\pm} = K(u_{max})$",
                 y=0.945)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = ROOT / "figures" / "fig_grid_verification.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\nwrote {out}")
