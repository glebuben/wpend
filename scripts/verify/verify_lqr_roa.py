"""
Grid verification of the LQR region of attraction.

Ground truth: integrate the *full nonlinear* 4-D closed loop x' = f(x, -K x)
from a grid of initial conditions (theta0, dtheta0) with psi0 = dpsi0 = 0,
and classify each as converged (returns upright) or diverged (falls).

We then overlay
    * the certified Lyapunov sublevel set  x^T P x = c*  (conservative inner
      estimate of the basin),
    * the  Vdot = 0  boundary (where the linearization error overtakes the
      nominal LQR decay),
so all three pictures can be compared on one axis.

Reports:
    * fraction of grid where the true basin agrees with the certified ROA
      (certified points must converge -- soundness check),
    * the conservatism ratio (certified-ellipse area / true-basin area on the
      slice).

Writes figures/fig_lqr_roa_verification.png.
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
from src.lqr import LQRController, LQRWeights
from src.roa import (certified_level, vdot_batch, roa_ellipse,
                     simulate_grid as simulate_grid_full)

def simulate_grid(ctrl, TH0, DTH0, *, T=8.0, dt=2e-3):
    """RK4 the full 4-D closed loop from every grid IC; classify convergence.

    Returns int array: 1 = converged (upright), 0 = diverged.
    """
    code, th, dth = simulate_grid_full(ctrl, TH0, DTH0, T=T, dt=dt)
    converged = (code == 0) & (np.abs(th) < 0.05) & (np.abs(dth) < 0.1)
    return converged.astype(int)


def main():
    p = SegwayParams()
    ctrl = LQRController(p, LQRWeights())
    c_star, _ = certified_level(ctrl)

    th_b, dth_b = 1.2, 6.0
    N = 220
    th_grid = np.linspace(-th_b, th_b, N)
    dth_grid = np.linspace(-dth_b, dth_b, N)
    TH, DTH = np.meshgrid(th_grid, dth_grid)

    import time
    t0 = time.perf_counter()
    conv = simulate_grid(ctrl, TH, DTH)
    print("grid sim wall-time: %.2f s" % (time.perf_counter() - t0))

    # Certified ellipse membership on the slice (psi = dpsi = 0).
    Vslice = (ctrl.P[0, 0] * TH ** 2 + 2 * ctrl.P[0, 2] * TH * DTH
              + ctrl.P[2, 2] * DTH ** 2)
    inside = Vslice <= c_star

    # Soundness: every certified point must actually converge.
    cert_pts = inside.sum()
    cert_conv = (inside & (conv == 1)).sum()
    print("certified-ellipse points: %d, of which converge: %d  (%.2f%%)"
          % (cert_pts, cert_conv, 100.0 * cert_conv / max(cert_pts, 1)))

    cell = (th_grid[1] - th_grid[0]) * (dth_grid[1] - dth_grid[0])
    area_basin = conv.sum() * cell
    area_cert = inside.sum() * cell
    print("slice areas:  true basin = %.3f,  certified ROA = %.3f,  "
          "conservatism = %.1f%% of basin captured"
          % (area_basin, area_cert, 100.0 * area_cert / area_basin))

    # Vdot sign on the slice for the dotted boundary.
    X = np.stack([TH.ravel(), np.zeros(TH.size),
                  DTH.ravel(), np.zeros(TH.size)], axis=1)
    Vd = vdot_batch(ctrl, X).reshape(TH.shape)

    # ---- figure ----
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    cmap = ListedColormap([(0.95, 0.6, 0.6), (0.75, 0.9, 0.75)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(conv, extent=[-th_b, th_b, -dth_b, dth_b], origin="lower",
              aspect="auto", cmap=cmap, norm=norm, interpolation="nearest",
              alpha=0.85)

    ax.contour(TH, DTH, Vd, levels=[0.0], colors="k", linewidths=1.0,
               linestyles=":")

    th_fine = np.linspace(-th_b, th_b, 4000)
    upper, lower, m = roa_ellipse(ctrl, c_star, th_fine)
    ax.plot(th_fine[m], upper[m], "b-", lw=2.4)
    ax.plot(th_fine[m], lower[m], "b-", lw=2.4)

    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-th_b, th_b)
    ax.set_ylim(-dth_b, dth_b)
    ax.set_xlabel(r"$\theta_0$  [rad]")
    ax.set_ylabel(r"$\dot\theta_0$  [rad/s]")
    ax.set_title("LQR basin of attraction: simulation vs certified ROA\n"
                 "(full 4-D nonlinear closed loop, slice psi = dpsi = 0)")

    handles = [
        Patch(color=(0.75, 0.9, 0.75), label="converges (true basin, sim)"),
        Patch(color=(0.95, 0.6, 0.6), label="diverges / falls (sim)"),
        plt.Line2D([], [], color="b", lw=2.4, label=r"certified ROA $x^TPx=c^*$"),
        plt.Line2D([], [], color="k", lw=1.0, ls=":", label=r"$\dot V = 0$ boundary"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.93)

    out = ROOT / "figures" / "fig_lqr_roa_verification.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)


if __name__ == "__main__":
    main()
