"""
Interactive LQR recoverable-set map.

A clickable (theta0, theta_dot0) map of the *LQR controller's own* recoverable
set -- i.e. its true basin of attraction on the nonlinear plant -- for an LQR
designed with the two-parameter weighting

    Q = lambda1 * I_4 ,    R = lambda2 * I_1 .

Unlike ``recoverable_map.py`` (which draws the controller-independent bang-bang
recoverable set H_+- = K), the region shown here *depends on the controller*:
each grid initial condition (theta0, 0, theta_dot0, 0) is integrated under the
full 4-D nonlinear closed loop  x' = f(x, -K x)  and classified as

    * RECOVERABLE  -- returns upright,
    * falls FORWARD / falls BACKWARD -- the body topples (|theta| >= theta_fall).

The two lambdas are the knobs: change lambda1 / lambda2 and the whole basin is
recomputed (re-solving the CARE, re-simulating the grid).  Overlaid for context:

    * the certified quadratic ROA  x^T P x = c*  (conservative inner estimate),
    * the  Vdot = 0  curve on the (theta, dtheta) slice (where the linearization
      error overtakes the nominal LQR decay).

Two-window design (identical to recoverable_map.py)
---------------------------------------------------
*  This map runs in its own matplotlib window.
*  Each click launches the animation in a **separate OS process**
   (``python main.py --animate ... --controller lqr --lam1 .. --lam2 ..``),
   so several animations can run at once and the map stays responsive.

Controls
--------
*  Hover over the map               -> the cell under the cursor is highlighted.
*  Left-click on a coloured cell    -> open an LQR animation from that IC.
*  lambda1 / lambda2 text boxes     -> redesign the LQR and recompute the basin.
*  "u_max" slider + text box (left) -> torque limit; 0 = unsaturated LQR.  Drag
                                       the slider (recomputes on release) or type
                                       an exact value.  The basin is simulated
                                       with this same limit, and it is passed to
                                       the spawned animation.  The recoverable-set
                                       overlay is only drawn when u_max > 0.
*  Close the window to quit (running animations keep going).

Run
---
    python lqr_map.py
    python main.py --lqr-map          # equivalent entry point
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm, SymLogNorm
from matplotlib.widgets import RadioButtons, TextBox, Slider
from matplotlib.patches import Patch, Rectangle

# scripts/maps/ lives two levels below the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.system import SegwayParams
from src.lqr import LQRController, LQRWeights
from src.roa import (closed_loop_accel, vdot_batch,
                     simulate_grid as _simulate_grid,
                     certified_level as _certified_level)
from src.recoverable import (Delta as recov_Delta,
                             find_theta_eq as recov_theta_eq,
                             K_level as recov_K,
                             boundary_curves as recov_curves,
                             ceiling_floor as recov_ceiling_floor,
                             recoverable_mask as recov_mask)

# --------------------------------------------------------------------------
# Plant
# --------------------------------------------------------------------------
P = SegwayParams()
ALPHA, BETA, GAMMA, D = P.alpha, P.beta, P.gamma, P.D

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "main.py"



# --------------------------------------------------------------------------
# Controller-independent bang-bang recoverable set (same math as
# recoverable_map.py / CLAUDE.md sec. 6).  Used as an overlay to compare the
# LQR basin against "what *any* controller could recover" at a torque limit.
# --------------------------------------------------------------------------
# Cell colours: 0 recoverable (green), 1 fwd fall (red), 2 bwd fall (blue).
CELL_COLORS = [(0.78, 0.93, 0.78),     # 0 recoverable
               (0.95, 0.55, 0.55),     # 1 falls forward
               (0.55, 0.65, 0.95)]     # 2 falls backward
CELL_LABEL = {0: "RECOVERABLE (LQR returns it upright)",
              1: "falls FORWARD (LQR cannot recover)",
              2: "falls BACKWARD (LQR cannot recover)"}

U_MAX_SLIDER_MAX = 60.0                              # slider/text upper bound
DEFAULT_U_MAX = 0.0                                  # 0 = unsaturated (no limit)
DEFAULT_LAM1 = 50.0
DEFAULT_LAM2 = 1.0


# --------------------------------------------------------------------------
# LQR design (full 4-D, matching src/animation.py) and basin simulation
# --------------------------------------------------------------------------
class LQRDesign(LQRController):
    """4-D LQR with Q = lam1 * I, R = lam2 * I (same as the animation)."""

    def __init__(self, lam1: float, lam2: float):
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        super().__init__(P, LQRWeights(Q=self.lam1 * np.eye(4),
                                       R=self.lam2 * np.eye(1)))
        self.Pm = self.P          # legacy alias used throughout the map
        self.K = self.K.ravel()   # the map code expects a flat (4,) gain


def simulate_grid(des, TH0, DTH0, u_max, *, dphi0=0.0, T=6.0, dt=2.5e-3):
    """Basin codes on the grid (see src.roa.simulate_grid; map convention:
    the wheel starts at phi_dot0 = dphi0, so dpsi0 = dphi0 - dtheta0)."""
    return _simulate_grid(des, TH0, DTH0, u_max=u_max, dphi0=dphi0,
                          T=T, dt=dt)[0]


def vdot_slice(des, TH, DTH, u_max, dphi0=0.0):
    """Vdot = 2 x^T P f_nl on the slice psi = 0, dpsi = dphi0 - dtheta."""
    X = np.stack([TH.ravel(), np.zeros(TH.size), DTH.ravel(),
                  (dphi0 - DTH).ravel()], axis=1)
    return vdot_batch(des, X, u_max).reshape(TH.shape)


def roa_ellipse(Pm, c, th_fine, dphi0=0.0):
    """theta_dot(theta) on the certified-ellipse cross-section through the slice
    psi = 0, dpsi = dphi0 - theta_dot.  Substituting dpsi = dphi0 - dth into
    x^T P x = c leaves a quadratic A*dth^2 + B*dth + C = 0."""
    d = dphi0
    A = Pm[2, 2] - 2.0 * Pm[2, 3] + Pm[3, 3]
    B = (2.0 * (Pm[0, 2] - Pm[0, 3]) * th_fine
         + 2.0 * d * (Pm[2, 3] - Pm[3, 3]))
    C = Pm[0, 0] * th_fine ** 2 + 2.0 * Pm[0, 3] * th_fine * d + Pm[3, 3] * d ** 2 - c
    disc = B ** 2 - 4.0 * A * C
    m = disc >= 0
    sq = np.sqrt(np.where(m, disc, 0.0))
    upper = np.full_like(th_fine, np.nan)
    lower = np.full_like(th_fine, np.nan)
    upper[m] = (-B[m] + sq[m]) / (2.0 * A)
    lower[m] = (-B[m] - sq[m]) / (2.0 * A)
    return upper, lower, m


def certified_level(des, u_max, *, n_dirs=4000, s_max=12.0, ds=8e-3, seed=0):
    """Largest c with {x^T P x <= c} subset {Vdot < 0}, honouring |u| <= u_max."""
    return _certified_level(des, n_dirs=n_dirs, s_max=s_max, ds=ds, seed=seed,
                            u_max=u_max)[0]


def estimate_dth_bound(des, u_max, dphi0=0.0):
    """Rough vertical framing: largest recoverable |dtheta| at theta = 0."""
    dth_test = np.concatenate([np.linspace(-18.0, 0.0, 60),
                               np.linspace(0.0, 18.0, 60)])
    th0 = np.zeros_like(dth_test)
    code = simulate_grid(des, th0, dth_test, u_max, dphi0=dphi0, T=5.0, dt=3e-3)
    rec = np.abs(dth_test[code == 0])
    edge = rec.max() if rec.size else 4.0
    return float(np.clip(1.25 * edge, 4.0, 16.0))


def classify_grid(des, u_max, *, N=65, dphi0=0.0):
    """Simulate the N x N basin grid plus everything the map needs to draw."""
    dth_bound = estimate_dth_bound(des, u_max, dphi0)
    th_bound = 1.50
    th_grid = np.linspace(-th_bound, th_bound, N)
    dth_grid = np.linspace(-dth_bound, dth_bound, N)
    TH, DTH = np.meshgrid(th_grid, dth_grid)

    code = simulate_grid(des, TH, DTH, u_max, dphi0=dphi0)
    try:
        c_star = certified_level(des, u_max)
    except Exception:
        c_star = np.nan

    return dict(th_grid=th_grid, dth_grid=dth_grid, code=code,
                dth_bound=dth_bound, c_star=c_star, Pm=des.Pm, K=des.K,
                u_max=u_max, N=N, dphi0=dphi0)


# --------------------------------------------------------------------------
# Interactive map application
# --------------------------------------------------------------------------
class LQRMap:
    OVERLAY_OPTIONS = ["off", "H± lines", "mask", "V̇ heatmap", "|H| mask"]
    # Short labels for the |H|-mask norm selector  (label -> hessian.NORMS key).
    HNORM_CHOICES = [("Frob", "frobenius"), ("Spec", "spectral"),
                     ("Nucl", "nuclear"), ("Max", "maxabs")]

    def __init__(self, *, N: int = 65, lam1: float = DEFAULT_LAM1,
                 lam2: float = DEFAULT_LAM2, u_max: float = DEFAULT_U_MAX,
                 overlay: str = "off", dphi0: float = 0.0):
        self.N = N
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)
        self.u_max = float(u_max)
        self.dphi0 = float(dphi0)     # initial wheel angular velocity phi_dot0
        self._dphi_dirty = False      # set on slider drag, cleared on release
        self.overlay = overlay if overlay in self.OVERLAY_OPTIONS else "off"
        self.des = None
        self.grid = None
        self.procs = []

        # hover/blitting state
        self.hover_rect = None
        self._bg = None
        self.marker = None
        self.im = None

        self.fig = plt.figure(figsize=(11.0, 8.0))
        try:
            self.fig.canvas.manager.set_window_title(
                "LQR recoverable set  --  hover to highlight, click to launch")
        except Exception:
            pass

        self.ax = self.fig.add_axes((0.32, 0.10, 0.57, 0.82))

        # Vertical threshold slider next to the V̇ colorbar.  It is created once
        # and kept permanently responsive (toggling a widget axis' visibility can
        # leave the Slider dead); it simply has no effect unless the heatmap
        # overlay is active.  A label shows the current threshold value.
        self.ax_vthr = self.fig.add_axes((0.905, 0.14, 0.020, 0.70))
        self.ax_vthr.set_title("keep\n≥ thr", fontsize=7)
        self.s_vthr = Slider(self.ax_vthr, "", 0.0, 1.0, valinit=0.0,
                             orientation="vertical")
        self.s_vthr.valtext.set_visible(False)
        self.s_vthr.on_changed(self._on_vthresh)

        self.cax = self.fig.add_axes((0.95, 0.14, 0.016, 0.70))
        self.cax.set_visible(False)
        self._vdot_im = None
        self._vdot_Vd = None            # cached field, for threshold masking
        self._vdot_norm = None          # current symlog norm (for frac -> value)
        self._vdot_vmin = None          # current colorbar data range
        self._vdot_vmax = None
        self.vdot_frac = 0.0            # 0 = show whole mask; raise to keep high V̇
        self._thresh_line = None        # threshold marker drawn on the colorbar
        self._vdot_alpha = 0.40         # base opacity of the heatmap mask

        # |H| (Hessian-norm) nonlinearity mask -- a second heatmap overlay that
        # shares the same vertical threshold slider as the V̇ heatmap.
        self._hess_im = None
        self._hess_field = None
        self._hess_norm = None
        self._hess_vmin = None
        self._hess_vmax = None
        self.hess_norm = "frobenius"    # which tensor norm the |H| mask uses

        # ---- side column widgets ----------------------------------------
        # Torque limit as a continuous slider + a text box for exact entry
        # (0 = no limit / unsaturated LQR).  Both are kept in sync; the basin
        # re-simulation is deferred to slider *release* (see _on_release) since
        # each solve is ~2-3 s, while a typed value recomputes immediately.
        self._umax_dirty = False        # slider dragged, awaiting release
        self._syncing_umax = False      # guard against slider<->textbox feedback
        self.fig.text(0.04, 0.83, "u_max [Nm]  (0 = none)",
                      fontsize=9, weight="bold")
        self.ax_umax = self.fig.add_axes((0.075, 0.785, 0.125, 0.03))
        self.s_umax = Slider(self.ax_umax, "", 0.0, U_MAX_SLIDER_MAX,
                             valinit=self.u_max, valfmt="%.1f")
        self.s_umax.on_changed(self._on_umax_slider)
        self.ax_umax_tb = self.fig.add_axes((0.13, 0.725, 0.06, 0.038))
        self.tb_umax = TextBox(self.ax_umax_tb, "set ", initial=f"{self.u_max:g}")
        self.tb_umax.on_submit(self._on_umax_text)

        self.fig.text(0.04, 0.52, "LQR weights:", fontsize=10, weight="bold")
        self.ax_l1 = self.fig.add_axes((0.13, 0.47, 0.08, 0.038))
        self.tb_l1 = TextBox(self.ax_l1, "lam1 (Q=l1*I) ", initial=f"{self.lam1:g}")
        self.tb_l1.on_submit(self._on_lam1)
        self.ax_l2 = self.fig.add_axes((0.13, 0.42, 0.08, 0.038))
        self.tb_l2 = TextBox(self.ax_l2, "lam2 (R=l2*I) ", initial=f"{self.lam2:g}")
        self.tb_l2.on_submit(self._on_lam2)

        # Overlay of the controller-independent recoverable set (H+- = K).
        self.ax_ovl = self.fig.add_axes((0.04, 0.22, 0.18, 0.185),
                                        facecolor=(0.96, 0.96, 0.97))
        self.ax_ovl.set_title("overlay", fontsize=9)
        o_active = self.OVERLAY_OPTIONS.index(self.overlay)
        self.radio_ovl = RadioButtons(self.ax_ovl, self.OVERLAY_OPTIONS,
                                      active=o_active)
        self.radio_ovl.on_clicked(self._on_overlay)

        # Tensor-norm selector for the |H| mask (sits just right of the overlay
        # radio).  Only affects the "|H| mask" overlay; harmless otherwise.
        self.ax_hnorm = self.fig.add_axes((0.225, 0.225, 0.085, 0.16),
                                          facecolor=(0.96, 0.96, 0.97))
        self.ax_hnorm.set_title("‖H‖ norm", fontsize=8)
        hn_labels = [lbl for lbl, _ in self.HNORM_CHOICES]
        hn_keys = [k for _, k in self.HNORM_CHOICES]
        hn_active = hn_keys.index(self.hess_norm) if self.hess_norm in hn_keys else 0
        self.radio_hnorm = RadioButtons(self.ax_hnorm, hn_labels, active=hn_active)
        self.radio_hnorm.on_clicked(self._on_hess_norm)

        # Initial wheel velocity phi_dot0 (enters the LQR through dpsi).
        self.ax_dphi = self.fig.add_axes((0.06, 0.17, 0.14, 0.03))
        self.s_dphi = Slider(self.ax_dphi, r"$\dot\varphi_0$", -15.0, 15.0,
                             valinit=self.dphi0, valfmt="%+.1f")
        self.s_dphi.on_changed(self._on_dphi_changed)

        self._draw_side_text()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        # Recompute the basin only when the dphi0 slider is *released* (dragging
        # fires many events; a full re-simulation per event would be too slow).
        self.fig.canvas.mpl_connect("button_release_event", self._on_release)
        self._recompute_and_draw()

    # ----- side panel ------------------------------------------------------
    def _draw_side_text(self):
        handles = [Patch(color=CELL_COLORS[i], label=CELL_LABEL[i].split(" (")[0])
                   for i in range(3)]
        handles += [
            plt.Line2D([], [], color="b", lw=2.2, label="certified ROA  xᵀPx=c*"),
            plt.Line2D([], [], color="k", lw=1.4, ls="--", label="V̇ = 0 curve"),
            plt.Line2D([], [], color="magenta", lw=1.8, label="H₋ = K  (upper, θ̇>0)"),
            plt.Line2D([], [], color="magenta", lw=1.8, ls="--", label="H₊ = K  (lower, θ̇<0)"),
            Patch(color=(0.78, 0.30, 0.30), label="V̇ heatmap  (red V̇>0 / blue V̇<0)"),
            Patch(color=(0.55, 0.10, 0.45), label="|H| mask  (Hessian-norm nonlinearity)"),
        ]
        self.fig.legend(handles=handles, loc="lower left",
                        bbox_to_anchor=(0.02, 0.02), fontsize=7.0,
                        frameon=True, title="legend")

    # ----- compute + render ------------------------------------------------
    def _recompute_and_draw(self):
        self.ax.set_title("computing LQR basin ...", fontsize=11)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        try:
            self.des = LQRDesign(self.lam1, self.lam2)
        except Exception as exc:
            self.ax.set_title(f"CARE failed for these weights:\n{exc}",
                              fontsize=10, color="red")
            self.fig.canvas.draw_idle()
            return

        self.grid = classify_grid(self.des, self.u_max, N=self.N,
                                  dphi0=self.dphi0)
        self._render()

    def _redraw_from_cache(self):
        """Re-render using the cached basin (for overlay-only toggles)."""
        self.ax.set_title("redrawing ...", fontsize=11)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        self._render()

    def _render(self):
        g = self.grid
        th_grid, dth_grid = g["th_grid"], g["dth_grid"]
        dth_bound = g["dth_bound"]

        self.ax.clear()
        cmap = ListedColormap(CELL_COLORS)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)
        self.im = self.ax.imshow(
            g["code"],
            extent=[th_grid.min(), th_grid.max(), dth_grid.min(), dth_grid.max()],
            origin="lower", aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest", alpha=0.85)

        # certified ROA ellipse cross-section through the current slice
        th_fine = np.linspace(-1.55, 1.55, 4000)
        if np.isfinite(g["c_star"]):
            up, lo, m = roa_ellipse(g["Pm"], g["c_star"], th_fine, self.dphi0)
            self.ax.plot(th_fine[m], up[m], "b-", lw=2.2,
                         label=r"certified ROA $x^TPx=c^*$")
            self.ax.plot(th_fine[m], lo[m], "b-", lw=2.2)

        # Vdot = 0 curve on the slice psi = 0, dpsi = dphi0 - dtheta
        TH, DTH = np.meshgrid(th_fine, np.linspace(-dth_bound, dth_bound, 600))
        Vd = vdot_slice(self.des, TH, DTH, self.u_max, self.dphi0)
        self.ax.contour(TH, DTH, Vd, levels=[0.0], colors="k",
                        linewidths=1.4, linestyles="--")

        # Optional overlay: V̇ heatmap mask, or the controller-independent set.
        self._draw_overlay(th_fine, dth_bound)

        self.ax.axhline(0, color="k", lw=0.4)
        self.ax.axvline(0, color="k", lw=0.4)
        self.ax.axvline(np.pi / 2, color="red", lw=0.7, ls=":")
        self.ax.axvline(-np.pi / 2, color="red", lw=0.7, ls=":")
        self.ax.set_xlim(-1.55, 1.55)
        self.ax.set_ylim(-dth_bound, dth_bound)
        self.ax.set_xlabel(r"$\theta_0$  [rad]")
        self.ax.set_ylabel(r"$\dot\theta_0$  [rad/s]")
        self.ax.set_title(self._title())

        # (re)create the animated hover rectangle
        self.marker = None
        self.hover_rect = Rectangle((0, 0), 0, 0, fill=False, ec="black",
                                    lw=2.0, zorder=7, animated=True,
                                    visible=False)
        self.ax.add_patch(self.hover_rect)
        self._bg = None
        self.fig.canvas.draw_idle()

    def _hide_vdot_cbar(self):
        """Hide the heatmap colorbar + threshold marker when no mask is active."""
        if getattr(self, "cax", None) is not None:
            self.cax.cla()
            self.cax.set_visible(False)
        self._vdot_im = None
        self._vdot_Vd = None
        self._vdot_norm = None
        self._hess_im = None
        self._hess_field = None
        self._hess_norm = None
        self._thresh_line = None

    def _draw_vdot_heatmap(self, dth_bound):
        """Live heat map of V̇ = 2 xᵀP f_nl(x,-Kx) on the (θ, θ̇) slice as a
        semi-transparent mask over the basin.  Recomputed every render, so it
        tracks lambda1/lambda2/u_max/phi_dot0 just like the basin itself.

        Symmetric-log colour scale: |V̇| spans several orders (tiny near upright,
        huge in the corners), so a linear scale would wash everything out."""
        th_lim, ng = 1.55, 260
        thg = np.linspace(-th_lim, th_lim, ng)
        dthg = np.linspace(-dth_bound, dth_bound, ng)
        TH, DTH = np.meshgrid(thg, dthg)
        Vd = vdot_slice(self.des, TH, DTH, self.u_max, self.dphi0)

        finite = np.isfinite(Vd)
        if finite.any():
            vmax = float(np.percentile(np.abs(Vd[finite]), 99.0)) + 1e-9
        else:
            vmax = 1.0
        lin = max(1.0, 0.01 * vmax)
        norm = SymLogNorm(linthresh=lin, vmin=-vmax, vmax=vmax, base=10)

        # cache the field + range + norm so the threshold slider can mask it
        self._vdot_Vd = Vd
        self._vdot_norm = norm
        self._vdot_vmin, self._vdot_vmax = -vmax, vmax

        # Masked-array image: cells below the threshold are masked -> rendered
        # with the cmap's "bad" colour, which we make fully transparent.  This
        # repaints reliably via set_data (unlike a per-pixel alpha array) and
        # keeps a uniform opacity for the visible cells.
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))
        self._vdot_im = self.ax.imshow(
            self._mask_below_frac(Vd, norm),
            extent=[-th_lim, th_lim, -dth_bound, dth_bound],
            origin="lower", aspect="auto", cmap=cmap, norm=norm,
            alpha=self._vdot_alpha, zorder=1.5, interpolation="nearest")

        self.cax.cla()
        self.cax.set_visible(True)
        cb = self.fig.colorbar(self._vdot_im, cax=self.cax)
        cb.set_label(r"$\dot V$  (symlog)", fontsize=8)
        self.cax.tick_params(labelsize=7)
        self._thresh_line = None
        self._update_thresh_line()        # honour the slider across re-renders

    def _draw_hessian_heatmap(self, dth_bound):
        """Semi-transparent mask of the open-loop Hessian norm |H(x)| over the
        (theta, dtheta) slice -- a controller-independent nonlinearity map (see
        src/hessian.py).  Drawn *under* the basin (low zorder + alpha) so the
        recoverable set still shows through; the vertical slider hides every cell
        whose |H| is below the chosen value (slide up to keep only the strongly
        nonlinear region).

        |H| spans orders of magnitude (~0 near upright, ~1e3 near +-pi/2), so a
        logarithmic colour scale is used."""
        from matplotlib.colors import LogNorm
        from src.hessian import norm_grids

        key = self.hess_norm
        ng = 200
        grids = norm_grids(P, theta_max=1.55, dtheta_max=dth_bound, N=ng,
                           u=0.0, norms=[key])
        Hf = np.asarray(grids[key])

        finite = np.isfinite(Hf) & (Hf > 0)
        if finite.any():
            vmin = float(np.percentile(Hf[finite], 2.0))
            vmax = float(np.percentile(Hf[finite], 99.5))
        else:
            vmin, vmax = 1e-3, 1.0
        vmin = max(vmin, 1e-6)
        vmax = max(vmax, vmin * 10.0)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        # cache field + range + norm so the threshold slider can re-mask cheaply
        self._hess_field = Hf
        self._hess_norm = norm
        self._hess_vmin, self._hess_vmax = vmin, vmax

        cmap = plt.get_cmap("magma").copy()
        cmap.set_bad((0.0, 0.0, 0.0, 0.0))      # below-threshold cells -> clear
        self._hess_im = self.ax.imshow(
            self._mask_below_frac(Hf, norm),
            extent=[-1.55, 1.55, -dth_bound, dth_bound],
            origin="lower", aspect="auto", cmap=cmap, norm=norm,
            alpha=self._vdot_alpha, zorder=1.5, interpolation="nearest")

        self.cax.cla()
        self.cax.set_visible(True)
        cb = self.fig.colorbar(self._hess_im, cax=self.cax)
        nice = {"frobenius": r"\|H\|_F", "spectral": r"\|H\|_\sigma",
                "nuclear": r"\|H\|_*", "maxabs": r"\max|H_{ijk}|"}.get(key, r"\|H\|")
        cb.set_label(rf"${nice}$  (log)", fontsize=8)
        self.cax.tick_params(labelsize=7)
        self._thresh_line = None
        self._update_thresh_line()        # honour the slider across re-renders

    def _active_heatmap(self):
        """Return the heatmap mask the threshold slider currently drives
        (V̇ or |H|), or None.  Both masks share the one vertical slider."""
        if self.overlay == "V̇ heatmap" and self._vdot_im is not None:
            return dict(im=self._vdot_im, field=self._vdot_Vd,
                        norm=self._vdot_norm,
                        base_label=r"$\dot V$  (symlog)",
                        thr_fmt=r"$\dot V \geq %.3g$")
        if self.overlay == "|H| mask" and self._hess_im is not None:
            return dict(im=self._hess_im, field=self._hess_field,
                        norm=self._hess_norm,
                        base_label=r"$\|H\|$  (log)",
                        thr_fmt=r"$\|H\| \geq %.3g$")
        return None

    def _mask_below_frac(self, field, norm):
        """field with every cell below the current slider threshold masked out."""
        if norm is None:
            return np.ma.masked_array(field)
        thr = float(norm.inverse(np.clip(self.vdot_frac, 0.0, 1.0)))
        return np.ma.masked_less(field, thr)

    def _thresh_value(self):
        """Slider fraction (0..1 along the colour bar) -> data threshold for the
        active mask.  frac=0 -> vmin (keep everything); frac=1 -> vmax."""
        hm = self._active_heatmap()
        if hm is None or hm["norm"] is None:
            return None
        return float(hm["norm"].inverse(np.clip(self.vdot_frac, 0.0, 1.0)))

    def _update_thresh_line(self):
        """Redraw the threshold marker + label on the colorbar for the active mask."""
        if self.cax is None:
            return
        hm = self._active_heatmap()
        thr = self._thresh_value()
        if hm is None or thr is None:
            return
        if self._thresh_line is None:
            self._thresh_line = self.cax.axhline(thr, color="k", lw=2.0,
                                                 solid_capstyle="butt", zorder=5)
        else:
            self._thresh_line.set_ydata([thr, thr])
        active = self.vdot_frac > 1e-6
        self._thresh_line.set_visible(active)
        lab = (hm["thr_fmt"] % thr) if active else hm["base_label"]
        try:
            self.cax.set_ylabel(lab, fontsize=8)
        except Exception:
            pass

    def _on_vthresh(self, val):
        """Threshold slider moved: re-mask the active heatmap (cached, cheap)."""
        self.vdot_frac = float(val)
        hm = self._active_heatmap()
        if hm is None:
            return
        thr = self._thresh_value()
        hm["im"].set_data(np.ma.masked_less(hm["field"], thr))
        self._update_thresh_line()
        self._bg = None                  # mask changed -> invalidate blit cache
        self.fig.canvas.draw_idle()

    def _draw_overlay(self, th_fine, dth_bound):
        """Superimpose one of: the V̇ heatmap mask, the |H| nonlinearity mask, or
        the controller-independent bang-bang recoverable set (H_-+ = K)."""
        if self.overlay == "V̇ heatmap":
            self._hess_im = None
            self._draw_vdot_heatmap(dth_bound)
            return
        if self.overlay == "|H| mask":
            self._vdot_im = None
            self._draw_hessian_heatmap(dth_bound)
            return
        self._hide_vdot_cbar()
        if self.overlay == "off":
            return
        um = self.u_max
        # The bang-bang recoverable set (H_-+ = K) is only defined with a finite
        # torque limit; without one (u_max = 0) the saddle level degenerates and
        # the H = K curve is meaningless (formally only the origin is
        # recoverable).  Skip the overlay and tell the user to set u_max > 0.
        if um <= 0.0:
            self.ax.text(0.5, 0.02,
                         "recoverable-set overlay needs a torque limit "
                         "(set u_max > 0)",
                         transform=self.ax.transAxes, ha="center", va="bottom",
                         fontsize=8, color="magenta",
                         bbox=dict(boxstyle="round", fc="white", ec="magenta",
                                   alpha=0.85), zorder=8)
            return
        ceiling, floor, sq_minus, sq_plus, th_eq = recov_ceiling_floor(um, th_fine)
        mneg = sq_minus >= 0
        mpos = sq_plus >= 0

        if self.overlay == "mask":
            ng = 320
            thg = np.linspace(-1.55, 1.55, ng)
            dthg = np.linspace(-dth_bound, dth_bound, ng)
            TH, DTH = np.meshgrid(thg, dthg)
            rec = recov_mask(um, TH, DTH)
            # Dim every state that is *not* recoverable by any controller at
            # this torque limit, so the bright cells are basin AND recoverable.
            rgba = np.zeros((ng, ng, 4))
            rgba[..., :3] = 0.20
            rgba[..., 3] = np.where(rec, 0.0, 0.45)
            self.ax.imshow(rgba, extent=[-1.55, 1.55, -dth_bound, dth_bound],
                           origin="lower", aspect="auto", zorder=2,
                           interpolation="nearest")

        # The recoverable-set boundary is the two saddle stable manifolds, each
        # a continuous curve that flips theta_dot sign at its saddle.  The
        # UPPER bound is the H_- manifold (solid), the LOWER bound the H_+
        # manifold (dashed); together with their post-saddle continuations they
        # make the 4 boundary pieces.
        self.ax.plot(th_fine[mneg], ceiling[mneg], color="magenta",
                     lw=1.9, zorder=5)                       # H_- stable manifold
        self.ax.plot(th_fine[mpos], floor[mpos], color="magenta",
                     lw=1.9, ls="--", zorder=5)              # H_+ stable manifold
        if th_eq > 1e-6:
            self.ax.scatter([th_eq, -th_eq], [0, 0], s=60, c="magenta",
                            marker="x", zorder=6)

    def _title(self, extra: str = ""):
        g = self.grid
        K = g["K"]
        um = "none" if self.u_max == 0 else f"{self.u_max:.1f} Nm"
        cs = "n/a" if not np.isfinite(g["c_star"]) else f"{g['c_star']:.2f}"
        head = (f"LQR basin   Q={self.lam1:g}·I, R={self.lam2:g}·I,  u_max={um}, "
                f"φ̇₀={self.dphi0:+.1f}\n"
                f"K=[{K[0]:.1f}, {K[1]:.2f}, {K[2]:.2f}, {K[3]:.2f}]   c*={cs}")
        return head + ("\n" + extra if extra else "   --  hover, then click")

    # ----- helpers ---------------------------------------------------------
    def _snap(self, value, grid):
        idx = int(np.clip(np.argmin(np.abs(grid - value)), 0, len(grid) - 1))
        return grid[idx], idx

    def _blit(self):
        if self._bg is None:
            self.fig.canvas.draw_idle()
            return
        self.fig.canvas.restore_region(self._bg)
        self.ax.draw_artist(self.hover_rect)
        self.fig.canvas.blit(self.fig.bbox)

    # ----- events ----------------------------------------------------------
    def _on_draw(self, event):
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)

    def _on_motion(self, event):
        if self.grid is None or self.hover_rect is None:
            return
        if event.inaxes is not self.ax or event.xdata is None:
            if self.hover_rect.get_visible():
                self.hover_rect.set_visible(False)
                self._blit()
            return
        th0, _ = self._snap(event.xdata, self.grid["th_grid"])
        dth0, _ = self._snap(event.ydata, self.grid["dth_grid"])
        tg, dg = self.grid["th_grid"], self.grid["dth_grid"]
        dx = tg[1] - tg[0]
        dy = dg[1] - dg[0]
        self.hover_rect.set_bounds(th0 - dx / 2, dth0 - dy / 2, dx, dy)
        self.hover_rect.set_visible(True)
        self._blit()

    def _on_umax_slider(self, val):
        """u_max slider moved: record the value + mirror it into the text box,
        but defer the (expensive) basin re-simulation to release."""
        self.u_max = float(val)
        if not self._syncing_umax:
            self._syncing_umax = True
            self.tb_umax.set_val(f"{self.u_max:g}")
            self._syncing_umax = False
        self._umax_dirty = True

    def _on_umax_text(self, text):
        """Exact u_max entry: parse, clamp to [0, U_MAX_SLIDER_MAX], sync the
        slider, and recompute immediately (a typed value is a deliberate set)."""
        try:
            val = float(text)
            if val < 0:
                raise ValueError
        except ValueError:
            self.tb_umax.set_val(f"{self.u_max:g}")
            return
        val = float(np.clip(val, 0.0, U_MAX_SLIDER_MAX))
        self.u_max = val
        self._syncing_umax = True
        self.s_umax.set_val(val)          # mirror into slider (guarded)
        self.tb_umax.set_val(f"{val:g}")  # normalise the displayed text
        self._syncing_umax = False
        self._umax_dirty = False
        self._recompute_and_draw()

    def _on_overlay(self, label):
        self.overlay = label
        # The overlay is purely visual; reuse the cached basin if we have one.
        if self.grid is not None and self.des is not None:
            self._redraw_from_cache()
        else:
            self._recompute_and_draw()

    def _on_hess_norm(self, label):
        """Switch the tensor norm used by the |H| mask."""
        key = dict(self.HNORM_CHOICES).get(label, "frobenius")
        if key == self.hess_norm:
            return
        self.hess_norm = key
        # Only the |H| mask depends on the norm; recompute it from the cached
        # basin (no re-simulation of the LQR closed loop needed).
        if self.overlay == "|H| mask" and self.grid is not None and self.des is not None:
            self._redraw_from_cache()

    def _on_dphi_changed(self, val):
        # Cheap: just record the new wheel velocity; recompute on release.
        self.dphi0 = float(val)
        self._dphi_dirty = True

    def _on_release(self, event):
        # Recompute the basin once a slider drag ends (dphi0 or u_max).  One
        # recompute covers both, since it reads the current self.* state.
        if self._dphi_dirty or self._umax_dirty:
            self._dphi_dirty = False
            self._umax_dirty = False
            self._recompute_and_draw()

    def _on_lam1(self, text):
        try:
            val = float(text)
            if val <= 0:
                raise ValueError
            self.lam1 = val
            self._recompute_and_draw()
        except ValueError:
            self.tb_l1.set_val(f"{self.lam1:g}")

    def _on_lam2(self, text):
        try:
            val = float(text)
            if val <= 0:
                raise ValueError
            self.lam2 = val
            self._recompute_and_draw()
        except ValueError:
            self.tb_l2.set_val(f"{self.lam2:g}")

    def _on_click(self, event):
        if event.inaxes is not self.ax or event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return
        g = self.grid
        th0, j = self._snap(event.xdata, g["th_grid"])
        dth0, i = self._snap(event.ydata, g["dth_grid"])
        code = int(g["code"][i, j])

        if self.marker is not None:
            self.marker.remove()
        (self.marker,) = self.ax.plot([th0], [dth0], marker="o", ms=12,
                                      mfc="none", mec="yellow", mew=2.5,
                                      zorder=8)
        self.ax.set_title(self._title(
            f"launching theta0={th0:+.3f}, theta_dot0={dth0:+.3f}  ->  "
            f"{CELL_LABEL[code].split(' (')[0]}"))
        self._bg = None
        self.fig.canvas.draw_idle()

        self._launch_animation(th0, dth0)

    def _launch_animation(self, th0, dth0):
        dpsi0 = self.dphi0 - dth0          # so the animation starts at phi_dot0
        cmd = [sys.executable, str(MAIN), "--animate",
               "--theta0", f"{th0:.6f}", "--psi0", "0",
               "--dtheta0", f"{dth0:.6f}", "--dpsi0", f"{dpsi0:.6f}",
               "--u-max", f"{self.u_max:.6f}",
               "--controller", "lqr",
               "--lam1", f"{self.lam1:.6f}", "--lam2", f"{self.lam2:.6f}",
               "--autostart"]
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
            self.procs = [p for p in self.procs if p.poll() is None]
            self.procs.append(proc)
            print(f"[lqr-map] launched LQR animation  "
                  f"theta0={th0:+.3f}  dtheta0={dth0:+.3f}  "
                  f"lam1={self.lam1:g}  lam2={self.lam2:g}  "
                  f"u_max={self.u_max:.1f}  (pid {proc.pid})")
        except Exception as exc:                                    # pragma:         except Exception as exc:                                    # pragma: no cover
            print(f"[lqr-map] failed to launch animation: {exc}")


def main(lam1: float = DEFAULT_LAM1, lam2: float = DEFAULT_LAM2,
         u_max: float = DEFAULT_U_MAX, dphi0: float = 0.0):
    app = LQRMap(lam1=lam1, lam2=lam2, u_max=u_max, dphi0=dphi0)
    plt.show()
    return app


if __name__ == "__main__":
    main()
