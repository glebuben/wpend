"""
Slices of the 4-D LQR certified ROA and the Vdot = 0 surface, with emphasis on
the wheel velocity.  Recall psi = phi - theta, so dpsi = dphi - dtheta; on a
slice with dtheta = 0 the dpsi axis IS the wheel velocity dphi.

The certified ellipsoid x^T P x = c* is genuinely 4-D.  The (theta, dtheta)
picture is only one slice (psi = dpsi = 0) and is small precisely because the
binding direction lives mostly in dpsi.  Here we show three orthogonal slices
(so the full ellipse is visible along the dpsi axis) and demonstrate that the
Vdot = 0 boundary in (theta, dtheta) moves as dpsi grows.

Writes figures/fig_lqr_roa_phi_slices.png.
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
from src.roa import certified_level, vdot_batch, vdot_slice, ellipse_ij

LABELS = [r"$\theta$", r"$\psi$", r"$\dot\theta$", r"$\dot\psi=\dot\phi-\dot\theta$"]


def panel(ax, ctrl, c, i, j, ri, rj, *, N=240, title=""):
    gi = np.linspace(-ri, ri, N)
    gj = np.linspace(-rj, rj, N)
    Gi, Gj = np.meshgrid(gi, gj)
    Vd = vdot_slice(ctrl, i, j, Gi, Gj, base=np.zeros(4))
    sign = (Vd >= 0).astype(int)
    cmap = ListedColormap([(0.80, 0.93, 0.80), (0.96, 0.80, 0.80)])
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    ax.imshow(sign, extent=[-ri, ri, -rj, rj], origin="lower", aspect="auto",
              cmap=cmap, norm=norm, interpolation="nearest")
    ax.contour(Gi, Gj, Vd, levels=[0.0], colors="k", linewidths=1.0, linestyles=":")
    xi = np.linspace(-ri, ri, 4000)
    up, lo, m = ellipse_ij(ctrl.P, c, i, j, xi)
    ax.plot(xi[m], up[m], "b-", lw=2.2)
    ax.plot(xi[m], lo[m], "b-", lw=2.2)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlabel(LABELS[i])
    ax.set_ylabel(LABELS[j])
    ax.set_title(title, fontsize=10)


def main():
    p = SegwayParams()
    ctrl = LQRController(p, LQRWeights())
    c, _ = certified_level(ctrl)
    print("c* =", round(c, 4))

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 11))

    # (a) (theta, dtheta), psi=dpsi=0  -- the original slice
    panel(axes[0, 0], ctrl, c, 0, 2, 1.2, 4.0,
          title=r"(a) $(\theta,\dot\theta)$ slice at $\psi=\dot\psi=0$")

    # (b) (theta, dpsi=dphi), dtheta=psi=0  -- wheel-velocity axis widened
    panel(axes[0, 1], ctrl, c, 0, 3, 1.2, 28.0,
          title=r"(b) $(\theta,\dot\psi)$ slice at $\dot\theta=\psi=0$  "
                r"($\dot\psi=\dot\phi$ here)")

    # (c) (dtheta, dpsi), theta=psi=0
    panel(axes[1, 0], ctrl, c, 2, 3, 6.2, 56.0,
          title=r"(c) $(\dot\theta,\dot\psi)$ slice at $\theta=\psi=0$")

    # (d) Vdot=0 in (theta,dtheta) for several dpsi -> boundary moves with dphi
    ax = axes[1, 1]
    ri, rj, N = 1.2, 4.0, 260
    gi = np.linspace(-ri, ri, N)
    gj = np.linspace(-rj, rj, N)
    Gi, Gj = np.meshgrid(gi, gj)
    colors = ["k", "tab:orange", "tab:red"]
    for dps, col in zip([0.0, 5.0, 10.0], colors):
        base = np.array([0.0, 0.0, 0.0, dps])
        Vd = vdot_slice(ctrl, 0, 2, Gi, Gj, base=base)
        cs = ax.contour(Gi, Gj, Vd, levels=[0.0], colors=col, linewidths=1.8)
        ax.plot([], [], color=col, lw=1.8, label=r"$\dot\psi=%g$" % dps)
    ax.axhline(0, color="k", lw=0.3)
    ax.axvline(0, color="k", lw=0.3)
    ax.set_xlim(-ri, ri)
    ax.set_ylim(-rj, rj)
    ax.set_xlabel(LABELS[0])
    ax.set_ylabel(LABELS[2])
    ax.set_title(r"(d) $\dot V=0$ in $(\theta,\dot\theta)$ shifts as $\dot\psi$"
                 r"($\approx\dot\phi$) grows", fontsize=10)
    ax.legend(loc="upper right", fontsize=9)

    handles = [
        Patch(color=(0.80, 0.93, 0.80), label=r"$\dot V < 0$"),
        Patch(color=(0.96, 0.80, 0.80), label=r"$\dot V \geq 0$"),
        plt.Line2D([], [], color="b", lw=2.2, label=r"certified ROA $x^TPx=c^*$ (cross-section)"),
        plt.Line2D([], [], color="k", lw=1.0, ls=":", label=r"$\dot V=0$"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, 0.995), fontsize=10)
    fig.suptitle("4-D LQR certified ROA and $\\dot V=0$ across slices "
                 "(wheel-velocity $\\dot\\psi=\\dot\\phi-\\dot\\theta$ included)",
                 y=0.955, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    out = ROOT / "figures" / "fig_lqr_roa_phi_slices.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote", out)


if __name__ == "__main__":
    main()
