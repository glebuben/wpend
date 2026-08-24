"""
LQR certified ROA and Vdot=0, drawn entirely in the ORIGINAL coordinates
x = (theta, phi, dtheta, dphi)  --  NO psi = phi - theta substitution.

Produces:
  figures/fig_lqr_roa_phys.png         -- (theta, dtheta) slice at phi=dphi=0
  figures/fig_lqr_roa_phys_slices.png  -- slices incl. the real dphi axis
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
from src.lqr_phys import LQRControllerPhys, LQRWeightsPhys
from src.roa import vdot_batch, certified_level, vdot_slice, ellipse_ij

LABELS = [r"$\theta$", r"$\phi$", r"$\dot\theta$", r"$\dot\phi$"]


def panel(ax, ctrl, c, i, j, ri, rj, title, N=240):
    gi = np.linspace(-ri, ri, N); gj = np.linspace(-rj, rj, N)
    Gi, Gj = np.meshgrid(gi, gj)
    Vd = vdot_slice(ctrl, i, j, Gi, Gj, np.zeros(4))
    cmap = ListedColormap([(0.80, 0.93, 0.80), (0.96, 0.80, 0.80)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow((Vd >= 0).astype(int), extent=[-ri, ri, -rj, rj], origin="lower",
              aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(Gi, Gj, Vd, levels=[0.0], colors="k", linewidths=1.0, linestyles=":")
    xi = np.linspace(-ri, ri, 4000)
    up, lo, m = ellipse_ij(ctrl.P, c, i, j, xi)
    ax.plot(xi[m], up[m], "b-", lw=2.2); ax.plot(xi[m], lo[m], "b-", lw=2.2)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel(LABELS[i]); ax.set_ylabel(LABELS[j]); ax.set_title(title, fontsize=10)


def box(P, c, i, j):
    d = P[i, i] * P[j, j] - P[i, j] ** 2
    return np.sqrt(c * P[j, j] / d), np.sqrt(c * P[i, i] / d)


def main():
    p = SegwayParams()
    ctrl = LQRControllerPhys(p, LQRWeightsPhys())
    c, bdir = certified_level(ctrl, s_max=24.0)
    P = ctrl.P
    print("(theta,phi) LQR:  K =", np.round(ctrl.K.ravel(), 4))
    print("c* = %.4f" % c, " binding dir (th,phi,dth,dphi) =", np.round(bdir, 3))
    th_ext = np.sqrt(c / (P[0, 0] - P[0, 2] ** 2 / P[2, 2]))
    print("certified tilt extent |theta| <= %.3f rad (%.1f deg)"
          % (th_ext, np.degrees(th_ext)))

    # ---- figure 1: main (theta, dtheta) slice ----
    fig, ax = plt.subplots(figsize=(9, 7))
    panel(ax, ctrl, c, 0, 2, 1.2, 4.0,
          r"LQR in $(\theta,\phi)$: certified ROA and $\dot V=0$  "
          r"(slice $\phi=\dot\phi=0$)")
    handles = [
        Patch(color=(0.80, 0.93, 0.80), label=r"$\dot V < 0$"),
        Patch(color=(0.96, 0.80, 0.80), label=r"$\dot V \geq 0$"),
        plt.Line2D([], [], color="b", lw=2.2, label=r"certified ROA $x^TPx=c^*$"),
        plt.Line2D([], [], color="k", lw=1.0, ls=":", label=r"$\dot V=0$"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.93)
    out1 = ROOT / "figures" / "fig_lqr_roa_phys.png"
    out1.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(); fig.savefig(out1, dpi=140); print("wrote", out1)

    # ---- figure 2: slices including the real dphi axis ----
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))
    _, dphi_b = box(P, c, 0, 3)
    _, dphi_c = box(P, c, 2, 3)
    panel(axes[0, 0], ctrl, c, 0, 2, 1.2, 4.0,
          r"(a) $(\theta,\dot\theta)$ at $\phi=\dot\phi=0$")
    panel(axes[0, 1], ctrl, c, 0, 3, 1.2, 1.12 * dphi_b,
          r"(b) $(\theta,\dot\phi)$ at $\dot\theta=\phi=0$")
    panel(axes[1, 0], ctrl, c, 2, 3, 6.2, 1.12 * dphi_c,
          r"(c) $(\dot\theta,\dot\phi)$ at $\theta=\phi=0$")
    ax = axes[1, 1]
    ri, rj, N = 1.2, 4.0, 260
    gi = np.linspace(-ri, ri, N); gj = np.linspace(-rj, rj, N)
    Gi, Gj = np.meshgrid(gi, gj)
    for dphi, col in zip([0.0, 5.0, 10.0], ["k", "tab:orange", "tab:red"]):
        Vd = vdot_slice(ctrl, 0, 2, Gi, Gj, np.array([0.0, 0.0, 0.0, dphi]))
        ax.contour(Gi, Gj, Vd, levels=[0.0], colors=col, linewidths=1.8)
        ax.plot([], [], color=col, lw=1.8, label=r"$\dot\phi=%g$" % dphi)
    ax.axhline(0, color="k", lw=0.3); ax.axvline(0, color="k", lw=0.3)
    ax.set_xlim(-ri, ri); ax.set_ylim(-rj, rj)
    ax.set_xlabel(LABELS[0]); ax.set_ylabel(LABELS[2])
    ax.set_title(r"(d) $\dot V=0$ in $(\theta,\dot\theta)$ shifts as $\dot\phi$ grows",
                 fontsize=10)
    ax.legend(loc="upper right", fontsize=9)
    handles = [
        Patch(color=(0.80, 0.93, 0.80), label=r"$\dot V<0$"),
        Patch(color=(0.96, 0.80, 0.80), label=r"$\dot V\geq0$"),
        plt.Line2D([], [], color="b", lw=2.2, label=r"certified ROA (cross-section)"),
        plt.Line2D([], [], color="k", lw=1.0, ls=":", label=r"$\dot V=0$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.suptitle(r"4-D LQR in physical $(\theta,\phi,\dot\theta,\dot\phi)$ "
                 r"-- certified ROA and $\dot V=0$ (no $\psi$ substitution)",
                 y=0.955, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = ROOT / "figures" / "fig_lqr_roa_phys_slices.png"
    fig.savefig(out2, dpi=140); print("wrote", out2)


if __name__ == "__main__":
    main()
