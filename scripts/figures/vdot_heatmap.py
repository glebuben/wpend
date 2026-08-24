"""
Heat map of the LQR Lyapunov rate  Vdot = d/dt (x^T P x)  on the (theta, dtheta)
slice (psi = dpsi = 0).

Continuous-valued companion to roa_lqr.py: shows the full scalar field
Vdot = 2 x^T P f_nl(x, -K x) (not just its sign).  Because |Vdot| spans several
orders (tiny near the origin, huge in the far corners), the colour scale uses a
symmetric-log norm so both regimes stay readable.

Overlaid: the black Vdot = 0 contour and the certified ROA ellipse x^T P x = c*.

Usage
-----
    python vdot_heatmap.py                # write figures/fig_lqr_vdot_heatmap.png
    python vdot_heatmap.py --interactive  # open a window; hovering the cursor
                                          # shows the exact Vdot at the pointer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm

# scripts/ live two levels below the project root; make `src` importable
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams
from src.lqr import LQRController, LQRWeights
from src.roa import vdot_batch, certified_level, slice_fields, roa_ellipse


def vdot_at(ctrl, theta, dtheta):
    """Exact Vdot at a single (theta, dtheta) point on the psi=dpsi=0 slice."""
    x = np.array([[theta, 0.0, dtheta, 0.0]], float)
    return float(vdot_batch(ctrl, x)[0])


def build_figure(ctrl, c_star, th_b=1.2, dth_b=6.0, n=400):
    th_grid = np.linspace(-th_b, th_b, n)
    dth_grid = np.linspace(-dth_b, dth_b, n)
    TH, DTH, Vd = slice_fields(ctrl, th_grid, dth_grid)

    # Symmetric-log diverging norm: linear within +/- lin_thresh around 0,
    # logarithmic beyond, so the wide dynamic range of Vdot stays legible.
    vmax = float(np.percentile(np.abs(Vd), 99.5)) + 1e-9
    lin_thresh = max(1.0, 0.01 * vmax)
    norm = SymLogNorm(linthresh=lin_thresh, vmin=-vmax, vmax=vmax, base=10)

    fig, ax = plt.subplots(figsize=(9.5, 7.2))
    pcm = ax.pcolormesh(TH, DTH, Vd, cmap="RdBu_r", norm=norm, shading="auto")
    cb = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\dot V = \frac{d}{dt}\,x^{T}Px$  (symlog scale)")

    ax.contour(TH, DTH, Vd, levels=[0.0], colors="k", linewidths=1.4)

    th_fine = np.linspace(-th_b, th_b, 4000)
    upper, lower, m = roa_ellipse(ctrl, c_star, th_fine)
    ax.plot(th_fine[m], upper[m], color="lime", lw=2.2)
    ax.plot(th_fine[m], lower[m], color="lime", lw=2.2)

    ax.axhline(0, color="k", lw=0.4)
    ax.axvline(0, color="k", lw=0.4)
    ax.set_xlim(-th_b, th_b)
    ax.set_ylim(-dth_b, dth_b)
    ax.set_xlabel(r"$\theta$  [rad]")
    ax.set_ylabel(r"$\dot\theta$  [rad/s]")
    ax.set_title(r"LQR Lyapunov-rate heat map  $\dot V = 2\,x^{T}P\,f_{\rm nl}(x,-Kx)$"
                 "\n(slice $\\psi=\\dot\\psi=0$;  blue $\\dot V<0$, red $\\dot V>0$)")

    handles = [
        plt.Line2D([], [], color="k", lw=1.4, label=r"$\dot V = 0$ boundary"),
        plt.Line2D([], [], color="lime", lw=2.2,
                   label=r"certified ROA $x^{T}Px=c^*$"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.93)
    fig.tight_layout()
    return fig, ax


def add_hover_readout(fig, ax, ctrl):
    """Live Vdot at the cursor: toolbar readout + a moving annotation box."""
    ann = ax.annotate(
        "", xy=(0, 0), xytext=(12, 12), textcoords="offset points",
        fontsize=10, ha="left", va="bottom",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.4", alpha=0.92),
    )
    ann.set_visible(False)

    def format_coord(x, y):
        return "theta=%.3f  dtheta=%.3f   Vdot=%.4g" % (x, y, vdot_at(ctrl, x, y))
    ax.format_coord = format_coord

    def on_move(event):
        if event.inaxes is ax and event.xdata is not None:
            vd = vdot_at(ctrl, event.xdata, event.ydata)
            ann.xy = (event.xdata, event.ydata)
            ann.set_text("Vdot = %.4g\ntheta=%.3f, dtheta=%.3f"
                         % (vd, event.xdata, event.ydata))
            ann.set_visible(True)
            fig.canvas.draw_idle()
        elif ann.get_visible():
            ann.set_visible(False)
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", on_move)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--interactive", action="store_true",
                    help="open a window with a live Vdot-at-cursor readout")
    args = ap.parse_args()

    p = SegwayParams()
    ctrl = LQRController(p, LQRWeights())

    print("LQR design about the upright equilibrium")
    print("  K   =", np.array2string(ctrl.K.ravel(), precision=4))
    print("  P diag =", np.array2string(np.diag(ctrl.P), precision=4))

    c_star, _ = certified_level(ctrl)
    print("Certified ROA level  c* = %.4f" % c_star)

    fig, ax = build_figure(ctrl, c_star)

    out = ROOT / "figures" / "fig_lqr_vdot_heatmap.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print("wrote %s" % out)

    if args.interactive:
        add_hover_readout(fig, ax, ctrl)
        print("hover the cursor over the map to read Vdot; close window to exit.")
        plt.show()

    return ctrl, c_star


if __name__ == "__main__":
    main()
