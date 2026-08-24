"""
Lyapunov rate Vdot on phase portraits of the LQR closed loop, in physical
coordinates x = (theta, phi, dtheta, dphi).

Three figures:
  fig_vdot_2d_portraits.png   -- Vdot scalar field on ALL 6 coordinate-pair
                                 slices (other two = 0), with the Vdot=0 curve
                                 and the certified-ROA ellipse cross-section.
  fig_vdot_3d_surfaces.png    -- the same Vdot as a 3-D height surface over
                                 each of the 6 pairs, with the z=0 plane.
  fig_vdot_isosurface.png     -- the Vdot=0 surface in the 3-D slice
                                 (theta, dtheta, dphi) at phi=0, obtained by
                                 solving the (quadratic-in-dphi) equation
                                 Vdot=0, with the nested certified ellipsoid.
"""

from __future__ import annotations

from pathlib import Path
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# scripts/ live two levels below the project root; make `src` importable
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams
from src.lqr_phys import LQRControllerPhys, LQRWeightsPhys
from src.roa import vdot_batch, certified_level, ellipse_ij

LABELS = [r"$\theta$", r"$\phi$", r"$\dot\theta$", r"$\dot\phi$"]
RANGE = {0: 1.2, 1: 10.0, 2: 4.0, 3: 14.0}
PAIRS = list(combinations(range(4), 2))


def vd_grid(ctrl, i, j, ri, rj, N=220, base=None):
    base = np.zeros(4) if base is None else np.asarray(base, float)
    gi = np.linspace(-ri, ri, N)
    gj = np.linspace(-rj, rj, N)
    Gi, Gj = np.meshgrid(gi, gj)
    X = np.tile(base, (Gi.size, 1))
    X[:, i] = Gi.ravel()
    X[:, j] = Gj.ravel()
    Vd = vdot_batch(ctrl, X).reshape(Gi.shape)
    return Gi, Gj, Vd


def fig_2d(ctrl, c):
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    for ax, (i, j) in zip(axes.flat, PAIRS):
        ri, rj = RANGE[i], RANGE[j]
        Gi, Gj, Vd = vd_grid(ctrl, i, j, ri, rj)
        vmax = np.percentile(np.abs(Vd), 99) + 1e-9
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        cf = ax.contourf(Gi, Gj, Vd, levels=40, cmap="RdBu_r", norm=norm)
        ax.contour(Gi, Gj, Vd, levels=[0.0], colors="k", linewidths=1.6)
        xi = np.linspace(-ri, ri, 3000)
        up, lo, m = ellipse_ij(ctrl.P, c, i, j, xi)
        ax.plot(xi[m], up[m], color="lime", lw=2.0)
        ax.plot(xi[m], lo[m], color="lime", lw=2.0)
        ax.axhline(0, color="k", lw=0.3)
        ax.axvline(0, color="k", lw=0.3)
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel(LABELS[j])
        ax.set_title(r"%s vs %s  (others $=0$)" % (LABELS[j], LABELS[i]), fontsize=10)
        fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04, label=r"$\dot V$")
    fig.suptitle(r"$\dot V$ on all 6 phase-plane slices  "
                 r"(black: $\dot V=0$,  green: certified ROA $x^TPx=c^*$)",
                 y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "figures" / "fig_vdot_2d_portraits.png"
    fig.savefig(out, dpi=135)
    print("wrote", out)


def fig_3d_surfaces(ctrl):
    fig = plt.figure(figsize=(18, 10))
    for k, (i, j) in enumerate(PAIRS, 1):
        ri, rj = RANGE[i], RANGE[j]
        Gi, Gj, Vd = vd_grid(ctrl, i, j, ri, rj, N=120)
        ax = fig.add_subplot(2, 3, k, projection="3d")
        vmax = np.percentile(np.abs(Vd), 99) + 1e-9
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
        ax.plot_surface(Gi, Gj, Vd, cmap="RdBu_r", norm=norm,
                        linewidth=0, antialiased=True, alpha=0.95)
        ax.plot_surface(Gi, Gj, np.zeros_like(Vd), color="gray", alpha=0.18,
                        linewidth=0)
        ax.set_xlabel(LABELS[i])
        ax.set_ylabel(LABELS[j])
        ax.set_zlabel(r"$\dot V$")
        ax.set_title(r"$\dot V(%s,%s)$" % (LABELS[i], LABELS[j]), fontsize=10)
        ax.view_init(elev=28, azim=-60)
    fig.suptitle(r"$\dot V$ as a height surface over each phase plane "
                 r"(gray = $\dot V=0$ plane)", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = ROOT / "figures" / "fig_vdot_3d_surfaces.png"
    fig.savefig(out, dpi=130)
    print("wrote", out)


def solve_quadratic_in_dphi(ctrl, c, TH, DTH):
    """Roots in dphi of Vdot=0 (quadratic, fit from 3 samples) and of x^T P x=c."""
    def vd_at(dphi):
        X = np.stack([TH.ravel(), np.zeros(TH.size), DTH.ravel(),
                      np.full(TH.size, dphi)], axis=1)
        return vdot_batch(ctrl, X).reshape(TH.shape)
    Vm, V0, Vp = vd_at(-1.0), vd_at(0.0), vd_at(1.0)
    A = 0.5 * (Vp + Vm) - V0
    B = 0.5 * (Vp - Vm)
    C = V0
    disc = B * B - 4 * A * C
    mvd = disc >= 0
    sq = np.sqrt(np.where(mvd, disc, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        vd_hi = np.where(mvd, (-B + sq) / (2 * A), np.nan)
        vd_lo = np.where(mvd, (-B - sq) / (2 * A), np.nan)

    P = ctrl.P
    a = P[3, 3]
    b = 2.0 * (P[0, 3] * TH + P[2, 3] * DTH)
    cc = P[0, 0] * TH ** 2 + 2 * P[0, 2] * TH * DTH + P[2, 2] * DTH ** 2 - c
    dell = b * b - 4 * a * cc
    mel = dell >= 0
    sqe = np.sqrt(np.where(mel, dell, 0.0))
    el_hi = np.where(mel, (-b + sqe) / (2 * a), np.nan)
    el_lo = np.where(mel, (-b - sqe) / (2 * a), np.nan)
    return (vd_hi, vd_lo), (el_hi, el_lo)


def fig_isosurface(ctrl, c):
    th_b, dth_b, N = 0.8, 2.6, 200
    TH, DTH = np.meshgrid(np.linspace(-th_b, th_b, N),
                          np.linspace(-dth_b, dth_b, N))
    (vd_hi, vd_lo), (el_hi, el_lo) = solve_quadratic_in_dphi(ctrl, c, TH, DTH)
    clip = 15.0
    for S in (vd_hi, vd_lo, el_hi, el_lo):
        S[np.abs(S) > clip] = np.nan

    def draw(ax):
        ax.plot_surface(TH, DTH, vd_hi, color="indianred", alpha=0.5,
                        rstride=2, cstride=2, linewidth=0)
        ax.plot_surface(TH, DTH, vd_lo, color="salmon", alpha=0.5,
                        rstride=2, cstride=2, linewidth=0)
        ax.plot_surface(TH, DTH, el_hi, color="royalblue", alpha=0.55,
                        rstride=2, cstride=2, linewidth=0)
        ax.plot_surface(TH, DTH, el_lo, color="royalblue", alpha=0.55,
                        rstride=2, cstride=2, linewidth=0)
        ax.set_xlabel(LABELS[0]); ax.set_ylabel(LABELS[2]); ax.set_zlabel(LABELS[3])
        ax.set_zlim(-clip, clip)

    fig = plt.figure(figsize=(15, 6.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d"); draw(ax)
    ax.view_init(elev=16, azim=-65)
    ax2 = fig.add_subplot(1, 2, 2, projection="3d"); draw(ax2)
    ax2.view_init(elev=22, azim=115)

    proxy = [plt.Line2D([], [], color="indianred", lw=6, alpha=0.6,
                        label=r"$\dot V=0$ surface (two sheets)"),
             plt.Line2D([], [], color="royalblue", lw=6, alpha=0.6,
                        label=r"certified ellipsoid $x^TPx=c^*$")]
    fig.legend(handles=proxy, loc="lower center", ncol=2, frameon=False, fontsize=11)
    fig.suptitle(r"3-D $\dot V=0$ isosurface and certified ROA in physical "
                 r"$(\theta,\dot\theta,\dot\phi)$ at $\phi=0$  "
                 r"(ellipsoid lies inside $\dot V<0$)", y=0.99, fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out = ROOT / "figures" / "fig_vdot_isosurface.png"
    fig.savefig(out, dpi=140)
    print("wrote", out)


def main():
    ctrl = LQRControllerPhys(SegwayParams(), LQRWeightsPhys())
    c, _ = certified_level(ctrl, s_max=24.0)
    print("c* =", round(c, 4))
    fig_2d(ctrl, c)
    fig_3d_surfaces(ctrl)
    fig_isosurface(ctrl, c)


if __name__ == "__main__":
    main()
