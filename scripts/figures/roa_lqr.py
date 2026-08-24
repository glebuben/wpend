"""
Certified region of attraction (ROA) for the LQR controller, and the
"linearization-error" picture that motivates it.

Idea (see CLAUDE.md sec. 9):

    The LQR cost-to-go P gives a quadratic Lyapunov function V(x) = x^T P x.
    Along the *nonlinear* closed loop,
        Vdot(x) = -x^T Qcl x + 2 x^T P g(x),
    where g(x) = f_nl(x, -K x) - Acl x is exactly the linearization error and
    Qcl = Q + K^T R K is the nominal LQR decay.  So the intuition "LQR fails
    where the linearization error is too big" is made precise: the error is
    "too big" precisely where it overwhelms the nominal decay, i.e. where
    Vdot >= 0.

    The largest sublevel set {x : V(x) <= c*} that stays inside {Vdot < 0}
    (except the origin) is a *certified* region of attraction.  We find c* by
    a line search: along many directions, march out from the origin to the
    first Vdot >= 0 crossing and record V there; c* is the minimum.

Outputs:
    * prints c*, the binding direction, and the tilt extent of the ROA,
    * figures/fig_lqr_roa.png  -- (theta, dtheta) slice showing the Vdot sign
      region (the linearization-error map) and the certified ROA ellipse.
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
from src.roa import vdot_batch, certified_level, slice_fields, roa_ellipse

def main():
    p = SegwayParams()
    ctrl = LQRController(p, LQRWeights())

    print("LQR design about the upright equilibrium")
    print("  K   =", np.array2string(ctrl.K.ravel(), precision=4))
    ev = np.linalg.eigvals(ctrl.Acl)
    print("  closed-loop poles =", np.array2string(ev, precision=3))
    print("  P diag =", np.array2string(np.diag(ctrl.P), precision=4))

    c_star, best_dir = certified_level(ctrl)
    print("\nCertified ROA level  c* = %.4f" % c_star)
    print("  binding direction (theta,psi,dtheta,dpsi) =",
          np.array2string(best_dir, precision=3))

    P = ctrl.P
    schur = P[0, 0] - P[0, 2] ** 2 / P[2, 2]
    th_extent = np.sqrt(c_star / schur)
    print("  -> certified upright-tilt extent |theta| <= "
          "%.3f rad (%.1f deg) on the (theta, dtheta) slice"
          % (th_extent, np.degrees(th_extent)))

    th_b = 1.2
    dth_b = 6.0
    th_grid = np.linspace(-th_b, th_b, 320)
    dth_grid = np.linspace(-dth_b, dth_b, 320)
    TH, DTH, Vd = slice_fields(ctrl, th_grid, dth_grid)

    fig, ax = plt.subplots(figsize=(9, 7))
    sign = (Vd >= 0).astype(int)
    cmap = ListedColormap([(0.80, 0.93, 0.80), (0.96, 0.80, 0.80)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(sign, extent=[-th_b, th_b, -dth_b, dth_b], origin="lower",
              aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(TH, DTH, Vd, levels=[0.0], colors="k", linewidths=1.0,
               linestyles=":")

    th_fine = np.linspace(-th_b, th_b, 4000)
    upper, lower, m = roa_ellipse(ctrl, c_star, th_fine)
    ax.plot(th_fine[m], upper[m], "b-", lw=2.2)
    ax.plot(th_fine[m], lower[m], "b-", lw=2.2)

    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlabel(r"$\theta_0$  [rad]")
    ax.set_ylabel(r"$\dot\theta_0$  [rad/s]")
    ax.set_title("LQR linearization-error map and certified region of "
                 "attraction\n(slice psi = dpsi = 0)")

    handles = [
        Patch(color=(0.80, 0.93, 0.80), label=r"$\dot V < 0$ (LQR decreasing)"),
        Patch(color=(0.96, 0.80, 0.80),
              label=r"$\dot V \geq 0$ (linearization error wins)"),
        plt.Line2D([], [], color="b", lw=2.2, label=r"certified ROA $x^TPx=c^*$"),
        plt.Line2D([], [], color="k", lw=1.0, ls=":", label=r"$\dot V = 0$ boundary"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.93)

    out = ROOT / "figures" / "fig_lqr_roa.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    print("\nwrote %s" % out)
    return ctrl, c_star


if __name__ == "__main__":
    main()
