"""
Interactive Hessian-norm heatmap (pygame).

A single heatmap of the open-loop field's second-derivative tensor H(x) over
the (theta, dtheta) slice (see src/hessian.py for the math), shown one tensor
norm at a time.  Use the "Norm:" button (or number keys 1-4) to switch which
norm is displayed -- each one highlights where the plant is most nonlinear in a
different way.

The window auto-sizes to fit your screen (capped at ~90% of the desktop).

Live knobs (right panel):
    * u          -- torque at which the field f(x,u) is frozen (default 0)
    * theta_max  -- half-width of the theta axis [rad]
    * dtheta_max -- half-width of the dtheta axis [rad/s]
    * N          -- grid resolution (N x N)
    * h          -- finite-difference step for the Hessian

Buttons / keys:
    * Norm: <name>    (or 1-4)  -- choose which tensor norm to display
    * Recompute       (R)       -- rebuild with the current sliders
    * Color: lin/log  (L)       -- linear vs logarithmic color scale
    * Save PNG        (S)       -- write figures/fig_hessian_norms.png
    * Quit            (Esc)

Switching the norm or the color scale reuses the already-computed Hessian
field, so it is instant; only changing u / the ranges / N / h recomputes H.
Heavy work runs on a background thread so the GUI stays responsive.

Install:  pip install pygame   (matplotlib is already a project dependency)
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import pygame
except ImportError as exc:                                           # pragma: no cover
    raise RuntimeError(
        "pygame is not installed.  Install it with:  pip install pygame"
    ) from exc

import matplotlib
matplotlib.use("Agg")                       # headless rendering into a buffer
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

from .system import SegwayParams
from .hessian import norm_grids, make_figure, NORMS, DEFAULT_NORMS
# Reuse the widget toolkit and palette from the animation module.
from .animation import Slider, Button, COL


PANEL_W = 330            # width of the right-hand control column
# Preferred window size; clamped to the desktop in __init__.
PREF_W, PREF_H = 1060, 720


# Short button labels for the norm cycle (the figure title carries the math).
_NORM_SHORT = {
    "frobenius": "Frobenius",
    "spectral":  "Spectral",
    "nuclear":   "Nuclear",
    "maxabs":    "Max-abs",
}


class App:
    def __init__(self, *, u: float = 0.0, theta_max: float = 1.45,
                 dtheta_max: float = 12.0, N: int = 160,
                 log_scale: bool = True, norm: str = "frobenius",
                 fig_dir: Optional[Path] = None):
        pygame.init()
        pygame.display.set_caption("Hessian-norm heatmap -- nonlinearity map")

        # Fit the window to the screen (cap at ~90% of the desktop).
        info = pygame.display.Info()
        sw, sh = info.current_w, info.current_h
        self.W = min(PREF_W, int(sw * 0.92)) if sw > 0 else PREF_W
        self.H = min(PREF_H, int(sh * 0.90)) if sh > 0 else PREF_H
        self.W = max(self.W, 760)
        self.H = max(self.H, 520)
        self.right_x = self.W - PANEL_W
        self.fig_rect = pygame.Rect(12, 12, self.right_x - 24, self.H - 24)

        self.screen = pygame.display.set_mode((self.W, self.H))
        self.clock = pygame.time.Clock()
        self.fps = 60

        self.fig_dir = (fig_dir if fig_dir is not None
                        else Path(__file__).resolve().parent.parent / "figures")
        self.fig_dir.mkdir(parents=True, exist_ok=True)

        self.font = pygame.font.SysFont("dejavusans,arial", 14)
        self.small_font = pygame.font.SysFont("dejavusans,arial", 11)
        self.header_font = pygame.font.SysFont("dejavusans,arial", 13, bold=True)

        self.params = SegwayParams()
        self.log_scale = bool(log_scale)
        order = list(DEFAULT_NORMS)
        self.norm_key = norm if norm in order else order[0]

        # Rendered-figure handoff between worker and main thread.
        self._lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._pending = None            # (rgba_bytes, w, h)
        self._stat = None
        self._busy_done = False
        self._fig: Optional[Figure] = None   # last Figure (for Save PNG)
        # Cache of the computed Hessian field, keyed by the field signature, so
        # switching norm / color scale does not recompute H.
        self._cache_sig = None
        self._cache_grids = None

        self._busy = False
        self._dots = 0
        self.surf: Optional[pygame.Surface] = None
        self._map = None                # pixel<->data mapping for hover readout
        self.status = "Computing the first heatmap ..."
        self.status_color = COL["info"]

        self._build_ui(u, theta_max, dtheta_max, N)
        self._request_compute()         # kick off the initial render

    # ----- UI ---------------------------------------------------------------
    def _build_ui(self, u, theta_max, dtheta_max, N):
        x = self.right_x + 20
        w = PANEL_W - 40
        self.sections = [(16, "Field & slice")]
        y = 50
        self.s_u = Slider((x, y, w, 6), "u  [Nm]  (frozen torque)", -30.0, 30.0, u, "%+.1f")
        y += 44
        self.s_thmax = Slider((x, y, w, 6), "theta_max  [rad]", 0.30, 1.55, theta_max, "%.2f")
        y += 44
        self.s_dthmax = Slider((x, y, w, 6), "dtheta_max  [rad/s]", 2.0, 20.0, dtheta_max, "%.1f")
        y += 50

        self.sections.append((y - 8, "Numerics"))
        y += 16
        self.s_N = Slider((x, y, w, 6), "N  (grid is N x N)", 40, 240, N, "%.0f")
        y += 44
        self.s_h = Slider((x, y, w, 6), "h  (finite-diff step, x1e-3)", 0.2, 5.0, 1.0, "%.2f")
        y += 52

        self.sections.append((y - 8, "Display & actions"))
        y += 14
        bh, gap = 30, 8
        self.b_norm = Button((x, y, w, bh), self._norm_label(), self._cycle_norm)
        y += bh + gap
        self.b_recompute = Button((x, y, w, bh), "Recompute", self._request_compute)
        y += bh + gap
        self.b_scale = Button((x, y, w, bh), self._scale_label(), self._toggle_scale)
        y += bh + gap
        # Save / Quit share a row.
        half = (w - gap) // 2
        self.b_save = Button((x, y, half, bh), "Save PNG", self._on_save, style="save")
        self.b_quit = Button((x + half + gap, y, w - half - gap, bh), "Quit", self._on_quit)
        y += bh + 12
        self.status_y = y

        self.sliders = [self.s_u, self.s_thmax, self.s_dthmax, self.s_N, self.s_h]
        self.buttons = [self.b_norm, self.b_recompute, self.b_scale,
                        self.b_save, self.b_quit]
        self._running = True

    def _norm_label(self) -> str:
        return f"Norm: {_NORM_SHORT[self.norm_key]}  (click / 1-4)"

    def _scale_label(self) -> str:
        return f"Color: {'log' if self.log_scale else 'linear'}  (click / L)"

    # ----- Callbacks --------------------------------------------------------
    def _set_norm(self, key: str):
        if key == self.norm_key or key not in NORMS:
            return
        self.norm_key = key
        self.b_norm.label = self._norm_label()
        self._request_compute()         # reuses cached H -> fast

    def _cycle_norm(self):
        order = list(DEFAULT_NORMS)
        i = order.index(self.norm_key)
        self._set_norm(order[(i + 1) % len(order)])

    def _toggle_scale(self):
        self.log_scale = not self.log_scale
        self.b_scale.label = self._scale_label()
        self._request_compute()         # reuses cached H -> fast

    def _on_quit(self):
        self._running = False

    def _on_save(self):
        if self._fig is None:
            self.status = "Nothing to save yet -- wait for the first render."
            self.status_color = COL["warn"]
            return
        path = self.fig_dir / "fig_hessian_norms.png"
        try:
            self._fig.savefig(str(path), dpi=150)
            self.status = f"Saved {path.name} to {self.fig_dir}"
            self.status_color = COL["ok"]
        except Exception as exc:                                       # pragma: no cover
            self.status = f"Save failed: {exc}"
            self.status_color = COL["warn"]

    def _request_compute(self):
        if self._busy:
            return
        snap = dict(
            u=float(self.s_u.value),
            theta_max=float(self.s_thmax.value),
            dtheta_max=float(self.s_dthmax.value),
            N=int(round(self.s_N.value)),
            h=float(self.s_h.value) * 1.0e-3,
            log_scale=self.log_scale,
            norm=self.norm_key,
        )
        self._busy = True
        self._dots = 0
        self.status = "Computing ..."
        self.status_color = COL["info"]
        self._thread = threading.Thread(target=self._worker, args=(snap,), daemon=True)
        self._thread.start()

    # ----- Worker (compute + render off the GUI thread) ---------------------
    def _worker(self, s: dict):
        t0 = time.perf_counter()
        try:
            sig = (s["u"], s["theta_max"], s["dtheta_max"], s["N"], s["h"])
            reused = (self._cache_sig == sig and self._cache_grids is not None)
            if reused:
                grids = self._cache_grids
            else:
                grids = norm_grids(self.params, theta_max=s["theta_max"],
                                   dtheta_max=s["dtheta_max"], N=s["N"],
                                   u=s["u"], h=s["h"], norms=[s["norm"]])
                self._cache_sig = sig
                self._cache_grids = grids
            # Make sure the requested norm field exists in the cache.
            key = s["norm"]
            if key not in grids:
                grids[key] = NORMS[key][1](grids["H"])
            grids["keys"] = [key]

            dpi = 100
            fig = Figure(figsize=(self.fig_rect.width / dpi,
                                  self.fig_rect.height / dpi), dpi=dpi)
            make_figure(grids, log_scale=s["log_scale"], fig=fig)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = bytes(canvas.buffer_rgba())

            # Pixel<->data mapping for the hover readout.  fig.axes[0] is the
            # heatmap (the colorbar is fig.axes[1]); its position is finalized
            # after tight_layout + draw.  Matplotlib measures y from the bottom,
            # so flip to top-origin pixels to match pygame.
            pos = fig.axes[0].get_position()
            mapinfo = dict(
                x0=pos.x0 * w, x1=pos.x1 * w,
                ytop=(1.0 - pos.y1) * h, ybot=(1.0 - pos.y0) * h,
                th0=float(grids["theta"][0]), th1=float(grids["theta"][-1]),
                dth0=float(grids["dtheta"][0]), dth1=float(grids["dtheta"][-1]),
                Z=np.asarray(grids[key]), key=key,
            )
            wall = time.perf_counter() - t0
            with self._lock:
                self._pending = (buf, w, h, mapinfo)
                self._fig = fig
                self._stat = (True, s, wall, reused)
        except Exception as exc:                                       # pragma: no cover
            with self._lock:
                self._stat = (False, s, exc, False)
        finally:
            with self._lock:
                self._busy_done = True

    def _poll(self):
        with self._lock:
            done = self._busy_done
            pending = self._pending
            self._pending = None
            stat = self._stat
            if done:
                self._busy_done = False
                self._busy = False
                self._stat = None
        if pending is not None:
            buf, w, h, mapinfo = pending
            self.surf = pygame.image.frombuffer(buf, (w, h), "RGBA").convert_alpha()
            self._map = mapinfo
        if stat is not None:
            ok, s, info, reused = stat
            if ok:
                tag = "norm switch" if reused else "recomputed H"
                self.status = (f"{_NORM_SHORT[s['norm']]}   {s['N']}x{s['N']}   "
                               f"u = {s['u']:+.1f} Nm   ({tag}, {info:.2f} s)")
                self.status_color = COL["ok"]
            else:
                self.status = f"Compute failed: {info}"
                self.status_color = COL["warn"]

    # ----- Main loop --------------------------------------------------------
    def run(self):
        while self._running:
            released = False
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._running = False
                elif ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        self._running = False
                    elif ev.key == pygame.K_r:
                        self._request_compute()
                    elif ev.key == pygame.K_l:
                        self._toggle_scale()
                    elif ev.key == pygame.K_s:
                        self._on_save()
                    elif ev.key in (pygame.K_1, pygame.K_2, pygame.K_3, pygame.K_4):
                        idx = ev.key - pygame.K_1
                        order = list(DEFAULT_NORMS)
                        if idx < len(order):
                            self._set_norm(order[idx])
                for sld in self.sliders:
                    was = sld.dragging
                    if sld.handle(ev) and was and not sld.dragging:
                        released = True       # slider just released -> recompute
                for b in self.buttons:
                    b.handle(ev)
            if released:
                self._request_compute()

            self._poll()
            if self._busy:
                self._dots = (self._dots + 1) % 90

            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)
        pygame.quit()

    # ----- Drawing ----------------------------------------------------------
    def _draw(self):
        scr = self.screen
        scr.fill(COL["bg"])

        # Figure panel
        pygame.draw.rect(scr, COL["plot_bg"], self.fig_rect)
        pygame.draw.rect(scr, COL["axis"], self.fig_rect, 1)
        if self.surf is not None:
            scr.blit(self.surf, self.fig_rect.topleft)
            if not self._busy:
                self._draw_hover(scr)
        if self._busy:
            dots = "." * (1 + self._dots // 30)
            t = self.header_font.render("Computing " + dots, True, COL["info"])
            bg = pygame.Surface((t.get_width() + 16, t.get_height() + 8), pygame.SRCALPHA)
            bg.fill((255, 255, 255, 220))
            scr.blit(bg, (self.fig_rect.left + 10, self.fig_rect.top + 10))
            scr.blit(t, (self.fig_rect.left + 18, self.fig_rect.top + 14))

        # Right control panel
        pygame.draw.rect(scr, COL["panel_bg"],
                         pygame.Rect(self.right_x, 0, PANEL_W, self.H))
        for y, label in self.sections:
            scr.blit(self.header_font.render(label, True, COL["section"]),
                     (self.right_x + 20, y))
        for sld in self.sliders:
            sld.draw(scr, self.font)
        for b in self.buttons:
            b.draw(scr, self.font)

        for i, line in enumerate(self._wrap(self.status, 40)):
            scr.blit(self.small_font.render(line, True, self.status_color),
                     (self.right_x + 20, self.status_y + i * 16))

        hint = self.small_font.render(
            "1-4 norm  R recompute  L lin/log  S save  Esc quit",
            True, COL["sub_text"])
        scr.blit(hint, (self.right_x + 20, self.H - 22))

    def _draw_hover(self, scr):
        """Show theta, dtheta and the exact norm value under the cursor."""
        m = self._map
        if m is None:
            return
        mx, my = pygame.mouse.get_pos()
        relx = mx - self.fig_rect.left
        rely = my - self.fig_rect.top
        if not (m["x0"] <= relx <= m["x1"] and m["ytop"] <= rely <= m["ybot"]):
            return

        fx = (relx - m["x0"]) / (m["x1"] - m["x0"])
        fy = (m["ybot"] - rely) / (m["ybot"] - m["ytop"])   # 0 at bottom, 1 at top
        theta = m["th0"] + fx * (m["th1"] - m["th0"])
        dtheta = m["dth0"] + fy * (m["dth1"] - m["dth0"])

        Z = m["Z"]
        nr, nc = Z.shape
        c = int(round(fx * (nc - 1)))
        r = int(round(fy * (nr - 1)))
        c = min(max(c, 0), nc - 1)
        r = min(max(r, 0), nr - 1)
        val = float(Z[r, c])

        # Crosshair at the cursor, inside the axes.
        pygame.draw.line(scr, COL["cursor"], (m["x0"] + self.fig_rect.left, my),
                         (m["x1"] + self.fig_rect.left, my), 1)
        pygame.draw.line(scr, COL["cursor"], (mx, m["ytop"] + self.fig_rect.top),
                         (mx, m["ybot"] + self.fig_rect.top), 1)

        lines = [
            f"theta = {theta:+.3f} rad",
            f"dtheta = {dtheta:+.2f} rad/s",
            f"{_NORM_SHORT[m['key']]} = {val:.4g}",
        ]
        tw = max(self.small_font.size(s)[0] for s in lines)
        pad, lh = 6, self.small_font.get_height() + 2
        bw, bh = tw + 2 * pad, lh * len(lines) + 2 * pad
        # Place the box near the cursor, flipping so it stays on-screen.
        bx = mx + 14
        by = my + 14
        if bx + bw > self.fig_rect.right:
            bx = mx - 14 - bw
        if by + bh > self.fig_rect.bottom:
            by = my - 14 - bh
        bg = pygame.Surface((bw, bh), pygame.SRCALPHA)
        bg.fill((255, 255, 255, 235))
        scr.blit(bg, (bx, by))
        pygame.draw.rect(scr, COL["axis"], pygame.Rect(bx, by, bw, bh), 1)
        for i, s in enumerate(lines):
            scr.blit(self.small_font.render(s, True, COL["text"]),
                     (bx + pad, by + pad + i * lh))

    @staticmethod
    def _wrap(text, width):
        words, lines, cur = text.split(), [], ""
        for wd in words:
            if len(cur) + len(wd) + 1 <= width:
                cur = (cur + " " + wd).strip()
            else:
                lines.append(cur); cur = wd
        if cur:
            lines.append(cur)
        return lines or [""]


# ---------------------------------------------------------------------------
def build_hessian_map(*, u: float = 0.0, theta_max: float = 1.45,
                      dtheta_max: float = 12.0, N: int = 160,
                      log_scale: bool = True, norm: str = "frobenius",
                      fig_dir=None):
    """Open the interactive single-panel Hessian-norm heatmap window."""
    app = App(u=u, theta_max=theta_max, dtheta_max=dtheta_max, N=N,
              log_scale=log_scale, norm=norm, fig_dir=fig_dir)
    app.run()
    return app


if __name__ == "__main__":
    build_hessian_map()
