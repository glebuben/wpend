"""
Interactive recoverable-set map  (window 1 of 2).

A clickable (theta0, theta_dot0) map of the *simulated* recoverable set under
a torque limit |u| <= u_max (the same classification used by
``verify_recoverable_set.py``).  Click any cell and an independent segway
animation window opens, initialised at that (theta0, 0, theta_dot0, 0) and
running the Lyapunov controller saturated at the selected u_max.

Two-window design
-----------------
*  This map runs in its own matplotlib window.
*  Each click launches the animation in a **separate OS process**
   (``python main.py --animate ...``), so the two windows are fully
   independent: you can open several animations at once, keep clicking while
   they play, and the animation can also be started on its own
   (``python main.py --animate``) without this map at all.

Controls
--------
*  Hover over the map               -> the cell under the cursor is highlighted.
*  Left-click on a coloured cell    -> open an animation from that IC.
*  "u_max" radio (left)             -> switch torque limit and recompute the map.
*  "controller" radio + lambda1/2   -> pick the controller used by the spawned
                                       animation: the original Lyapunov law, or a
                                       simple LQR with Q = lambda1*I, R = lambda2*I.
*  Close the window to quit (running animations keep going).

Run
---
    python recoverable_map.py
    python main.py --map            # equivalent entry point
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.widgets import RadioButtons, TextBox
from matplotlib.patches import Patch, Rectangle

# scripts/maps/ lives two levels below the project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.system import SegwayParams
from src.recoverable import Delta, H_, find_theta_eq, vectorized_sim

# --------------------------------------------------------------------------
# Plant + reduced-tilt dynamics (identical to verify_recoverable_set.py)
# --------------------------------------------------------------------------
P = SegwayParams()
ALPHA, BETA, GAMMA, D = P.alpha, P.beta, P.gamma, P.D

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "main.py"

# The four-way classification colours (survives / fwd / bwd / both).
CELL_COLORS = [(0.78, 0.93, 0.78),    # 0 survives both
               (0.95, 0.55, 0.55),    # 1 falls forward  (u = -u_max)
               (0.55, 0.65, 0.95),    # 2 falls backward (u = +u_max)
               (0.70, 0.50, 0.70)]    # 3 falls under both
CELL_LABEL = {0: "RECOVERABLE (survives both brake directions)",
              1: "falls FORWARD under u = -u_max",
              2: "falls BACKWARD under u = +u_max",
              3: "falls under BOTH brake directions"}

U_MAX_OPTIONS = [2.0, 5.0, 10.0, 15.0, 30.0, 50.0]
DEFAULT_U_MAX = 15.0


def classify_grid(u_max: float, *, N=70, T=4.5):
    """Simulate the N x N grid and return everything the map needs."""
    th_eq = find_theta_eq(u_max)
    K = H_(th_eq, 0.0, -u_max)

    sq_at_0 = 2.0 * (K - GAMMA * D) / Delta(0.0)
    dth_bound = max(2.0, 1.45 * np.sqrt(max(sq_at_0, 1e-6)))
    th_bound = 1.50

    th_grid = np.linspace(-th_bound, th_bound, N)
    dth_grid = np.linspace(-dth_bound, dth_bound, N)
    TH, DTH = np.meshgrid(th_grid, dth_grid)

    out_minus = vectorized_sim(TH, DTH, -u_max, T=T)
    out_plus = vectorized_sim(TH, DTH, +u_max, T=T)

    sim_code = np.zeros_like(TH, dtype=int)
    fwd = (out_minus == 1)
    bwd = (out_plus == -1)
    sim_code[fwd & ~bwd] = 1
    sim_code[~fwd & bwd] = 2
    sim_code[fwd & bwd] = 3

    return dict(th_grid=th_grid, dth_grid=dth_grid, sim_code=sim_code,
                K=K, th_eq=th_eq, dth_bound=dth_bound, u_max=u_max, N=N)


# --------------------------------------------------------------------------
# Interactive map application
# --------------------------------------------------------------------------
class RecoverableMap:
    def __init__(self, u_max: float = DEFAULT_U_MAX, *, N: int = 70,
                 controller: str = "lyapunov", lam1: float = 50.0,
                 lam2: float = 1.0):
        self.N = N
        self.u_max = float(u_max)
        self.grid = None
        self.procs = []          # keep Popen handles alive

        # controller passed to the spawned animation
        self.controller_kind = "lqr" if str(controller).lower() == "lqr" else "lyapunov"
        self.lam1 = float(lam1)
        self.lam2 = float(lam2)

        # hover/blitting state
        self.hover_rect = None
        self._bg = None
        self.marker = None
        self.im = None

        self.fig = plt.figure(figsize=(11.0, 8.0))
        try:
            self.fig.canvas.manager.set_window_title(
                "Recoverable set  --  hover to highlight, click to launch an animation")
        except Exception:
            pass

        # main map axes
        self.ax = self.fig.add_axes((0.32, 0.10, 0.64, 0.82))

        # ---- side column widgets ----------------------------------------
        self.ax_umax = self.fig.add_axes((0.04, 0.56, 0.18, 0.30),
                                         facecolor=(0.96, 0.96, 0.97))
        self.ax_umax.set_title("u_max [Nm]", fontsize=10)
        u_labels = [f"{u:.0f}" for u in U_MAX_OPTIONS]
        u_active = U_MAX_OPTIONS.index(self.u_max) if self.u_max in U_MAX_OPTIONS else 0
        self.radio_umax = RadioButtons(self.ax_umax, u_labels, active=u_active)
        self.radio_umax.on_clicked(self._on_umax)

        self.ax_ctrl = self.fig.add_axes((0.04, 0.40, 0.18, 0.12),
                                         facecolor=(0.96, 0.96, 0.97))
        self.ax_ctrl.set_title("controller", fontsize=10)
        c_active = 1 if self.controller_kind == "lqr" else 0
        self.radio_ctrl = RadioButtons(self.ax_ctrl, ["Lyapunov", "LQR"],
                                       active=c_active)
        self.radio_ctrl.on_clicked(self._on_ctrl)

        self.ax_l1 = self.fig.add_axes((0.12, 0.34, 0.09, 0.035))
        self.tb_l1 = TextBox(self.ax_l1, "lam1 ", initial=f"{self.lam1:g}")
        self.tb_l1.on_submit(self._on_lam1)
        self.ax_l2 = self.fig.add_axes((0.12, 0.29, 0.09, 0.035))
        self.tb_l2 = TextBox(self.ax_l2, "lam2 ", initial=f"{self.lam2:g}")
        self.tb_l2.on_submit(self._on_lam2)

        # legend / instructions in the side column
        self._draw_side_text()

        self.fig.canvas.mpl_connect("button_press_event", self._on_click)
        self.fig.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.fig.canvas.mpl_connect("draw_event", self._on_draw)
        self._recompute_and_draw()

    # ----- side panel ------------------------------------------------------
    def _draw_side_text(self):
        handles = [Patch(color=CELL_COLORS[i], label=CELL_LABEL[i].split(" (")[0])
                   for i in range(4)]
        self.fig.legend(handles=handles, loc="lower left",
                        bbox_to_anchor=(0.02, 0.02), fontsize=8,
                        frameon=True, title="cell colour")
        self.fig.text(0.04, 0.95,
                      "Hover -> highlight cell.\n"
                      "Click -> open animation at\n"
                      "(theta0, 0, theta_dot0, 0)\n"
                      "with |u| <= u_max, using the\n"
                      "selected controller.",
                      fontsize=8.5, va="top")

    # ----- compute + render ------------------------------------------------
    def _recompute_and_draw(self):
        self.ax.set_title("computing ...", fontsize=11)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

        self.grid = classify_grid(self.u_max, N=self.N)
        g = self.grid
        th_grid, dth_grid = g["th_grid"], g["dth_grid"]
        dth_bound = g["dth_bound"]

        self.ax.clear()
        cmap = ListedColormap(CELL_COLORS)
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N)
        self.im = self.ax.imshow(
            g["sim_code"],
            extent=[th_grid.min(), th_grid.max(), dth_grid.min(), dth_grid.max()],
            origin="lower", aspect="auto", cmap=cmap, norm=norm,
            interpolation="nearest", alpha=0.85)

        # closed-form level sets H_+- = K for reference
        th_fine = np.linspace(-np.pi / 2 + 1e-3, np.pi / 2 - 1e-3, 4000)
        K, um = g["K"], self.u_max
        sq_minus = 2.0 * (K - um * (GAMMA * th_fine + BETA * np.sin(th_fine))
                          - GAMMA * D * np.cos(th_fine)) / Delta(th_fine)
        sq_plus = 2.0 * (K + um * (GAMMA * th_fine + BETA * np.sin(th_fine))
                         - GAMMA * D * np.cos(th_fine)) / Delta(th_fine)
        m = sq_minus >= 0
        self.ax.plot(th_fine[m], np.sqrt(sq_minus[m]), "k-", lw=1.6,
                     label=r"$H_-=K$")
        self.ax.plot(th_fine[m], -np.sqrt(sq_minus[m]), "k-", lw=1.6)
        m = sq_plus >= 0
        self.ax.plot(th_fine[m], np.sqrt(sq_plus[m]), "k--", lw=1.6,
                     label=r"$H_+=K$")
        self.ax.plot(th_fine[m], -np.sqrt(sq_plus[m]), "k--", lw=1.6)
        self.ax.scatter([g["th_eq"], -g["th_eq"]], [0, 0], s=70, c="black",
                        marker="x", zorder=6, label=r"saddles")

        self.ax.axhline(0, color="k", lw=0.4)
        self.ax.axvline(0, color="k", lw=0.4)
        self.ax.axvline(np.pi / 2, color="red", lw=0.7, ls=":")
        self.ax.axvline(-np.pi / 2, color="red", lw=0.7, ls=":")
        self.ax.set_xlim(-1.55, 1.55)
        self.ax.set_ylim(-dth_bound, dth_bound)
        self.ax.set_xlabel(r"$\theta_0$  [rad]")
        self.ax.set_ylabel(r"$\dot\theta_0$  [rad/s]")
        self.ax.set_title(self._title())
        self.ax.legend(loc="upper right", fontsize=8, framealpha=0.92)

        # (re)create the animated hover rectangle (ax.clear() dropped it)
        self.marker = None
        self.hover_rect = Rectangle((0, 0), 0, 0, fill=False, ec="black",
                                    lw=2.0, zorder=7, animated=True,
                                    visible=False)
        self.ax.add_patch(self.hover_rect)
        self._bg = None          # force background re-capture on next draw
        self.fig.canvas.draw_idle()

    def _title(self, extra: str = ""):
        nice = "LQR" if self.controller_kind == "lqr" else "Lyapunov"
        g = self.grid
        head = (f"u_max = {self.u_max:.0f} N m   "
                f"(theta_eq = {np.degrees(g['th_eq']):.1f} deg)   "
                f"ctrl = {nice}")
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
        # capture everything except the animated hover rectangle
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

    def _on_umax(self, label):
        self.u_max = float(label)
        self._recompute_and_draw()

    def _on_ctrl(self, label):
        self.controller_kind = "lqr" if label == "LQR" else "lyapunov"
        # controller choice does not change the recoverable set; just relabel
        self.ax.set_title(self._title())
        self.fig.canvas.draw_idle()

    def _on_lam1(self, text):
        try:
            self.lam1 = float(text)
        except ValueError:
            self.tb_l1.set_val(f"{self.lam1:g}")

    def _on_lam2(self, text):
        try:
            self.lam2 = float(text)
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
        code = int(g["sim_code"][i, j])

        # mark the chosen cell
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
        cmd = [sys.executable, str(MAIN), "--animate",
               "--theta0", f"{th0:.6f}", "--psi0", "0",
               "--dtheta0", f"{dth0:.6f}", "--dpsi0", "0",
               "--u-max", f"{self.u_max:.6f}",
               "--controller", self.controller_kind,
               "--lam1", f"{self.lam1:.6f}", "--lam2", f"{self.lam2:.6f}",
               "--autostart"]
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
            self.procs = [p for p in self.procs if p.poll() is None]
            self.procs.append(proc)
            print(f"[map] launched {self.controller_kind} animation  "
                  f"theta0={th0:+.3f}  dtheta0={dth0:+.3f}  "
                  f"u_max={self.u_max:.1f}  (pid {proc.pid})")
        except Exception as exc:                                    # pragma: no cover
            print(f"[map] failed to launch animation: {exc}")


def main(u_max: float = DEFAULT_U_MAX):
    app = RecoverableMap(u_max=u_max)
    plt.show()
    return app


if __name__ == "__main__":
    main()
