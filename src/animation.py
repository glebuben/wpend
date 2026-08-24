"""
Interactive pygame-based animation of the closed-loop segway.

How it works
------------
* Window opens with sliders for initial conditions, controller gains and
  the simulation horizon.  Nothing happens until you press **Start**.
* **Start**    runs the simulation with the current slider values
                (recomputes the controller, re-checks the gain condition)
                and plays the animation.
* **Replay**   restarts playback of the last simulation.
* **Pause**    toggles play/pause.
* **Save GIF** writes the current trajectory as an animated GIF into
                the ``animations/`` directory next to ``main.py``.

Hotkeys: R = Start, Space = Pause/Resume, G = Save GIF, Esc = Quit.

Install
-------
    pip install pygame pillow
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np

try:
    import pygame
except ImportError as exc:                                           # pragma: no cover
    raise RuntimeError(
        "pygame is not installed.  Install it with:  pip install pygame"
    ) from exc

from .system import SegwayParams, SegwayDynamics
from .controller import LyapunovController, ControllerGains, SaturatedController
from .lqr import LQRController, LQRWeights
from .simulation import simulate, rk4_step

# ---------------------------------------------------------------------------
# Layout & colours
# ---------------------------------------------------------------------------
WIDTH, HEIGHT = 1320, 960
RIGHT_PANEL_X = 920
SEGWAY_RECT   = pygame.Rect(15, 15, 890, 340)
STATE_RECT    = pygame.Rect(15, 370, 890, 215)
LYAP_RECT     = pygame.Rect(15, 600, 890, 210)

COL = {
    "bg":         (245, 247, 250),
    "panel_bg":   (228, 234, 242),
    "plot_bg":    (255, 255, 255),
    "text":       ( 25,  30,  40),
    "sub_text":   ( 95, 105, 120),
    "axis":       (110, 115, 125),
    "grid":       (225, 230, 235),
    "cursor":     (215,  50,  50),
    "theta":      ( 31, 119, 180),
    "psi":        ( 44, 160,  44),
    "u":          (148, 103, 189),
    "V":          ( 31, 119, 180),
    "Vdot":       (214,  39,  40),
    "KS":         (214,  39,  40),
    "btn":        (211, 224, 242),
    "btn_hover":  (180, 205, 232),
    "btn_press":  (150, 188, 222),
    "btn_save":   (220, 240, 220),
    "btn_save_h": (190, 225, 190),
    "wheel":      ( 31,  91, 180),
    "body":       (200,  44,  44),
    "com":        (200,  44,  44),
    "ground":     (120, 130, 140),
    "tick":       (170, 180, 195),
    "slider_track":         (200, 210, 220),
    "slider_track_active":  (170, 185, 200),
    "slider_knob":          ( 31,  56, 100),
    "slider_knob_hover":    ( 60,  90, 140),
    "section":    ( 31,  56, 100),
    "warn":       (200,  50,  50),
    "ok":         ( 30, 130,  50),
    "info":       ( 60,  80, 130),
}


# ---------------------------------------------------------------------------
# UI widgets
# ---------------------------------------------------------------------------
class Slider:
    def __init__(self, rect, label, vmin, vmax, value, fmt="%.2f"):
        self.rect   = pygame.Rect(rect)
        self.label  = label
        self.vmin   = float(vmin)
        self.vmax   = float(vmax)
        self.value  = float(value)
        self.fmt    = fmt
        self.dragging = False
        self.hover    = False

    @property
    def knob_pos(self):
        span = self.vmax - self.vmin
        frac = 0.0 if span == 0 else (self.value - self.vmin) / span
        return (int(self.rect.left + frac * self.rect.width), self.rect.centery)

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            kx, ky = self.knob_pos
            in_knob  = (event.pos[0] - kx) ** 2 + (event.pos[1] - ky) ** 2 < 169
            in_track = self.rect.inflate(0, 18).collidepoint(event.pos)
            if in_knob or in_track:
                self.dragging = True
                self._set_from_x(event.pos[0])
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            return True
        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.inflate(0, 18).collidepoint(event.pos)
            if self.dragging:
                self._set_from_x(event.pos[0])
                return True
        return False

    def _set_from_x(self, x):
        frac = (x - self.rect.left) / self.rect.width
        frac = max(0.0, min(1.0, frac))
        self.value = self.vmin + frac * (self.vmax - self.vmin)

    def draw(self, surf, font):
        lbl_text = font.render(self.label, True, COL["text"])
        val_text = font.render(self.fmt % self.value, True, COL["sub_text"])
        surf.blit(lbl_text, (self.rect.left, self.rect.top - 20))
        surf.blit(val_text, (self.rect.right - val_text.get_width(), self.rect.top - 20))

        track_col = COL["slider_track_active"] if self.dragging else COL["slider_track"]
        pygame.draw.rect(surf, track_col, self.rect, border_radius=4)
        kx, ky = self.knob_pos
        kcol = COL["slider_knob_hover"] if (self.dragging or self.hover) else COL["slider_knob"]
        pygame.draw.circle(surf, kcol, (kx, ky), 9)
        pygame.draw.circle(surf, (255, 255, 255), (kx, ky), 3)


class Button:
    def __init__(self, rect, label, on_click, *, style="default"):
        self.rect = pygame.Rect(rect)
        self.label = label
        self.on_click = on_click
        self.style = style
        self.hover = False
        self.press = False

    def handle(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.press = True
                return True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            was_press = self.press
            self.press = False
            if was_press and self.rect.collidepoint(event.pos):
                self.on_click()
                return True
        elif event.type == pygame.MOUSEMOTION:
            self.hover = self.rect.collidepoint(event.pos)
        return False

    def draw(self, surf, font):
        if self.style == "save":
            base, hov = COL["btn_save"], COL["btn_save_h"]
        else:
            base, hov = COL["btn"], COL["btn_hover"]
        if self.press:    col = COL["btn_press"]
        elif self.hover:  col = hov
        else:             col = base
        pygame.draw.rect(surf, col, self.rect, border_radius=6)
        pygame.draw.rect(surf, COL["axis"], self.rect, 1, border_radius=6)
        t = font.render(self.label, True, COL["text"])
        surf.blit(t, (self.rect.centerx - t.get_width() // 2,
                       self.rect.centery - t.get_height() // 2))


# ---------------------------------------------------------------------------
# Plot rendering
# ---------------------------------------------------------------------------
def draw_plot(surf, font, small_font, area, series, t_range, y_range,
              cursor_t=None, title="", hlines=None, legend=True):
    margin_top    = 22 if title else 6
    margin_bottom = 22
    margin_left   = 56
    margin_right  = 12
    plot_rect = pygame.Rect(area.left + margin_left,
                            area.top + margin_top,
                            area.width - margin_left - margin_right,
                            area.height - margin_top - margin_bottom)
    pygame.draw.rect(surf, COL["plot_bg"], plot_rect)

    if title:
        ts = font.render(title, True, COL["text"])
        surf.blit(ts, (plot_rect.centerx - ts.get_width() // 2, area.top + 2))

    t0, t1 = t_range
    y0, y1 = y_range
    if t1 <= t0: t1 = t0 + 1.0
    if y1 <= y0: y1 = y0 + 1.0

    def to_px(t, y):
        px = plot_rect.left + (t - t0) / (t1 - t0) * plot_rect.width
        py = plot_rect.bottom - (y - y0) / (y1 - y0) * plot_rect.height
        return (int(px), int(py))

    n_y, n_x = 4, 6
    for i in range(n_y + 1):
        y_val = y0 + (y1 - y0) * i / n_y
        py = int(plot_rect.bottom - i * plot_rect.height / n_y)
        pygame.draw.line(surf, COL["grid"], (plot_rect.left, py), (plot_rect.right, py))
        lbl = small_font.render(f"{y_val:g}", True, COL["axis"])
        surf.blit(lbl, (plot_rect.left - lbl.get_width() - 4, py - lbl.get_height() // 2))
    for i in range(n_x + 1):
        t_val = t0 + (t1 - t0) * i / n_x
        px = int(plot_rect.left + i * plot_rect.width / n_x)
        pygame.draw.line(surf, COL["grid"], (px, plot_rect.top), (px, plot_rect.bottom))
        lbl = small_font.render(f"{t_val:.1f}", True, COL["axis"])
        surf.blit(lbl, (px - lbl.get_width() // 2, plot_rect.bottom + 3))

    if y0 < 0 < y1:
        z = to_px(t0, 0)[1]
        pygame.draw.line(surf, COL["axis"], (plot_rect.left, z), (plot_rect.right, z), 1)

    if hlines:
        for y_val, color, _lbl in hlines:
            if y0 <= y_val <= y1:
                z = to_px(t0, y_val)[1]
                for x in range(plot_rect.left, plot_rect.right, 14):
                    x_end = min(x + 8, plot_rect.right)
                    pygame.draw.line(surf, color, (x, z), (x_end, z), 2)

    pygame.draw.rect(surf, COL["axis"], plot_rect, 1)

    for t_arr, y_arr, color, _label in series:
        if len(t_arr) < 2: continue
        pts = []
        for ti, yi in zip(t_arr, y_arr):
            if not np.isfinite(yi):
                if len(pts) >= 2:
                    pygame.draw.lines(surf, color, False, pts, 2)
                pts = []
                continue
            pts.append(to_px(ti, yi))
        if len(pts) >= 2:
            pygame.draw.lines(surf, color, False, pts, 2)

    if cursor_t is not None and t0 <= cursor_t <= t1:
        cx = to_px(cursor_t, y0)[0]
        pygame.draw.line(surf, COL["cursor"], (cx, plot_rect.top), (cx, plot_rect.bottom), 2)

    if legend:
        items = [(c, l) for (_, _, c, l) in series if l]
        if hlines:
            items += [(c, l) for (_, c, l) in hlines if l]
        if items:
            lx = plot_rect.right - 8
            ly = plot_rect.top + 4
            for color, lbl in items:
                t = small_font.render(lbl, True, COL["text"])
                w = t.get_width() + 26
                box = pygame.Rect(lx - w, ly, w, t.get_height() + 2)
                bg = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
                bg.fill((255, 255, 255, 215))
                surf.blit(bg, box.topleft)
                cy = ly + t.get_height() // 2 + 1
                pygame.draw.line(surf, color, (lx - w + 4, cy), (lx - w + 20, cy), 3)
                surf.blit(t, (lx - w + 23, ly + 1))
                ly += t.get_height() + 4


def draw_segway(surf, font, mono_font, area, params, theta, phi, psi,
                t_now, V_now, u_now, K_S, theta_S_deg):
    ts = font.render("Segway view", True, COL["text"])
    surf.blit(ts, (area.centerx - ts.get_width() // 2, area.top + 4))

    inset = pygame.Rect(area.left + 18, area.top + 28,
                        area.width - 36, area.height - 46)
    pygame.draw.rect(surf, COL["plot_bg"], inset)
    pygame.draw.rect(surf, COL["axis"], inset, 1)

    body_len_world = 2.4 * params.l
    height_world   = body_len_world + params.r + 0.05
    width_world    = 1.6
    px_per_m = min((inset.width - 40) / width_world,
                   (inset.height - 60) / height_world)
    ground_y = inset.bottom - 30
    # Guard against a non-finite / runaway state (toppled body) so int() below
    # never raises and kills the window.
    if not np.isfinite(theta): theta = 1.7
    if not np.isfinite(phi):   phi = 0.0
    if not np.isfinite(psi):   psi = 0.0
    x_w_world = params.r * phi
    cam_x = x_w_world

    def to_px(xw, yw):
        px = inset.centerx + (xw - cam_x) * px_per_m
        py = ground_y - yw * px_per_m
        px = int(np.clip(px, -32000, 32000)) if np.isfinite(px) else inset.centerx
        py = int(np.clip(py, -32000, 32000)) if np.isfinite(py) else ground_y
        return (px, py)

    pygame.draw.line(surf, COL["ground"], (inset.left, ground_y), (inset.right, ground_y), 2)
    base_tx = round(cam_x / 0.1) * 0.1
    for i in range(-15, 16):
        tx_w = base_tx + i * 0.1
        tx_px = int(inset.centerx + (tx_w - cam_x) * px_per_m)
        if inset.left + 4 <= tx_px <= inset.right - 4:
            pygame.draw.line(surf, COL["tick"], (tx_px, ground_y), (tx_px, ground_y + 6), 1)

    wc = to_px(x_w_world, params.r)
    wheel_r_px = max(10, int(params.r * px_per_m))
    pygame.draw.circle(surf, (255, 255, 255), wc, wheel_r_px)
    pygame.draw.circle(surf, COL["wheel"], wc, wheel_r_px, 3)
    sx_w = x_w_world + params.r * np.sin(-phi)
    sy_w = params.r   + params.r * np.cos(-phi)
    pygame.draw.line(surf, COL["wheel"], wc, to_px(sx_w, sy_w), 2)
    pygame.draw.circle(surf, COL["wheel"], wc, 3)

    bx_w = x_w_world      + body_len_world * np.sin(theta)
    by_w = params.r       + body_len_world * np.cos(theta)
    pygame.draw.line(surf, COL["body"], wc, to_px(bx_w, by_w), 6)

    cx_w = x_w_world + params.l * np.sin(theta)
    cy_w = params.r  + params.l * np.cos(theta)
    pygame.draw.circle(surf, COL["com"], to_px(cx_w, cy_w), 7)

    eq_top_px = to_px(x_w_world, params.r + body_len_world)
    for y in range(wc[1], eq_top_px[1], -10):
        pygame.draw.line(surf, COL["tick"], (wc[0], y), (wc[0], max(eq_top_px[1], y - 6)), 1)

    lines = [
        f"t  = {t_now:6.2f} s",
        f"theta = {np.degrees(theta):+7.2f} deg",
        f"psi   = {np.degrees(psi):+7.2f} deg",
        f"u  = {u_now:+7.3f} Nm",
        f"V  = {V_now:8.3f}",
    ]
    if K_S is not None:
        lines.append(f"KS = {K_S:8.2f}")
    if theta_S_deg is not None:
        lines.append(f"theta_S = {theta_S_deg:6.2f} deg")

    pad = 6
    line_h = mono_font.get_height() + 1
    box_w = 200
    box_h = line_h * len(lines) + 2 * pad
    bg = pygame.Surface((box_w, box_h), pygame.SRCALPHA)
    bg.fill((255, 255, 255, 220))
    surf.blit(bg, (inset.left + 6, inset.top + 6))
    ox, oy = inset.left + 6 + pad, inset.top + 6 + pad
    for line in lines:
        surf.blit(mono_font.render(line, True, COL["text"]), (ox, oy))
        oy += line_h


# ---------------------------------------------------------------------------
# GIF capture helper
# ---------------------------------------------------------------------------
def _save_gif(frames, path, fps):
    """Save a list of PIL Images to an animated GIF (lazy Pillow import)."""
    try:
        from PIL import Image                                          # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed.  Install it with:  pip install pillow"
        ) from exc
    duration_ms = max(20, int(1000 / fps))
    frames[0].save(str(path), save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True, disposal=2)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
class App:
    """Interactive segway demo built on pygame.

    Pressing **Start** runs the simulation with the *current* slider values
    (the controller is rebuilt from scratch each time), then plays the
    animation.  The window starts in a 'ready' state without an active
    simulation -- adjust the sliders, then press Start.
    """

    def __init__(self, *, T_sim: float = 8.0, fps: int = 60,
                 playback_speed: float = 1.0,
                 state0=(0.30, 0.0, 0.0, 0.0),
                 u_max: float = 0.0,
                 autostart: bool = False,
                 controller: str = "lyapunov",
                 lam1: float = 50.0,
                 lam2: float = 1.0,
                 gif_fps: int = 25,
                 anim_dir: Optional[Path] = None):
        pygame.init()
        pygame.display.set_caption("Segway Lyapunov controller -- interactive")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()
        self.fps    = fps
        self.playback_speed = playback_speed
        self.gif_fps = gif_fps

        self.anim_dir = (anim_dir if anim_dir is not None
                         else Path(__file__).resolve().parent.parent / "animations")
        self.anim_dir.mkdir(parents=True, exist_ok=True)

        # Fonts (DejaVu Sans usually has Greek + degree sign on Windows)
        self.font        = pygame.font.SysFont("dejavusans,arial", 14)
        self.small_font  = pygame.font.SysFont("dejavusans,arial", 11)
        self.title_font  = pygame.font.SysFont("dejavusans,arial", 14, bold=True)
        self.header_font = pygame.font.SysFont("dejavusans,arial", 13, bold=True)
        self.mono_font   = pygame.font.SysFont("dejavusansmono,consolas,courier", 13)

        self.sim          = None
        self.idx          = 0
        self.stride       = 1
        self.playing      = False
        self.params       = SegwayParams()
        self.gains        = ControllerGains()
        self.dyn          = SegwayDynamics(self.params)
        self.ctrl         = None
        self.theta_S      = 0.0
        self.K_S          = 0.0

        # Live ("manual") vs precomputed ("playback") integration.
        self.mode         = "playback"
        self.manual_x     = None       # current live state vector
        self.manual_t     = 0.0
        self.manual_T     = 8.0
        self.manual_umax  = 0.0        # torque magnitude applied by the keys
        self.manual_u     = 0.0
        self.fell         = False
        self._hist        = None       # growing dict of lists for the live run
        self.status_lines = [
            "Ready.  Adjust the sliders, then press Start.",
            "(The controller is rebuilt for the values you choose.)",
        ]
        self.status_color = COL["info"]

        # Threading: run simulate() off the GUI thread so the window
        # stays responsive even when T is large.
        self._sim_thread: Optional[threading.Thread] = None
        self._sim_lock = threading.Lock()
        self._sim_result: Optional[dict] = None
        self._sim_dots = 0  # animated "Simulating..." indicator

        # u_max == 0 means "unbounded torque" (pure Lyapunov law).
        self.u_max_init = float(u_max)
        self._autostart_pending = bool(autostart)

        # Controller selection: "lyapunov" (original) or "lqr".
        self.controller_kind = "lqr" if str(controller).lower() == "lqr" else "lyapunov"

        self._build_ui(T_sim, state0, self.u_max_init, lam1, lam2)

    # ----- UI build ---------------------------------------------------------
    def _build_ui(self, T_sim, state0, u_max0=0.0, lam1=50.0, lam2=1.0):
        x = RIGHT_PANEL_X + 30
        w = 340
        section_y = []

        section_y.append((22, "Initial conditions"))
        y = 60
        self.s_th0  = Slider((x, y, w, 6), "theta0  [rad]",   -1.55, 1.55, state0[0], "%+.3f")
        y += 42
        self.s_ps0  = Slider((x, y, w, 6), "psi0  [rad]",     -3.14, 3.14, state0[1], "%+.3f")
        y += 42
        self.s_dth0 = Slider((x, y, w, 6), "dtheta0  [rad/s]", -15.0, 15.0, state0[2], "%+.3f")
        y += 42
        self.s_dps0 = Slider((x, y, w, 6), "dpsi0  [rad/s]",   -15.0, 15.0, state0[3], "%+.3f")
        y += 46

        # ---- Controller selector (toggle) --------------------------------
        section_y.append((y - 4, "Controller"))
        y += 18
        self.b_ctrl = Button((x, y, w, 28), self._ctrl_label(), self._on_toggle_ctrl)
        y += 28 + 24

        # Lyapunov gains
        self.s_kp = Slider((x, y, w, 6), "kp  (Lyapunov)", 50.0, 400.0, 100.0, "%.0f")
        y += 42
        self.s_ki = Slider((x, y, w, 6), "ki  (Lyapunov)", 0.05, 5.0, 1.0, "%.2f")
        y += 42
        self.s_kl = Slider((x, y, w, 6), "kL  (Lyapunov)", 0.05, 2.0, 0.5, "%.2f")
        y += 42
        # LQR weights:  Q = lam1 * I,  R = lam2 * I
        self.s_lam1 = Slider((x, y, w, 6), "lambda1  (LQR: Q = l1*I)", 1.0, 200.0, lam1, "%.1f")
        y += 42
        self.s_lam2 = Slider((x, y, w, 6), "lambda2  (LQR: R = l2*I)", 0.1, 20.0, lam2, "%.2f")
        y += 50

        section_y.append((y - 18, "Horizon, torque & speed"))
        self.s_T = Slider((x, y + 12, w, 6), "T  [s]", 2.0, 20.0, T_sim, "%.1f")
        y += 54
        self.s_umax = Slider((x, y + 12, w, 6), "u_max [Nm]  (0 = unbounded)",
                             0.0, 60.0, u_max0, "%.1f")
        y += 54
        self.s_speed = Slider((x, y + 12, w, 6),
                              "speed x  (lower = slow-mo, for manual)",
                              0.05, 1.50, 0.30, "%.2f")
        y += 66

        # Buttons (2 x 2 grid)
        bw, bh, gap = 165, 34, 10
        row1_y = y
        row2_y = y + bh + gap
        self.b_start  = Button((x,           row1_y, bw, bh), "Start",   self._on_start)
        self.b_replay = Button((x + bw + gap, row1_y, bw, bh), "Replay",  self._on_replay)
        self.b_pause  = Button((x,           row2_y, bw, bh), "Pause",   self._on_pause)
        self.b_save   = Button((x + bw + gap, row2_y, bw, bh), "Save GIF",
                                self._on_save_gif, style="save")
        y = row2_y + bh + 16
        self.status_y = y

        self.sliders = [self.s_th0, self.s_ps0, self.s_dth0, self.s_dps0,
                        self.s_kp,  self.s_ki,  self.s_kl,
                        self.s_lam1, self.s_lam2,
                        self.s_T,   self.s_umax, self.s_speed]
        self.buttons = [self.b_ctrl, self.b_start, self.b_replay,
                        self.b_pause, self.b_save]
        self.section_labels = section_y

    _CTRL_ORDER = ("lyapunov", "lqr", "manual")
    _CTRL_NICE  = {"lyapunov": "Lyapunov", "lqr": "LQR",
                   "manual": "Manual (arrow keys)"}

    def _ctrl_label(self) -> str:
        return f"Controller: {self._CTRL_NICE[self.controller_kind]}  (click to switch)"

    def _on_toggle_ctrl(self):
        i = self._CTRL_ORDER.index(self.controller_kind)
        self.controller_kind = self._CTRL_ORDER[(i + 1) % len(self._CTRL_ORDER)]
        self.b_ctrl.label = self._ctrl_label()

    # ----- Callbacks --------------------------------------------------------
    def _on_start(self):
        if self._sim_thread is not None and self._sim_thread.is_alive():
            return  # already running -- ignore the click
        # Snapshot slider values now; the user might keep dragging while
        # the worker is integrating.
        snapshot = dict(
            th0=self.s_th0.value, ps0=self.s_ps0.value,
            dth0=self.s_dth0.value, dps0=self.s_dps0.value,
            kp=self.s_kp.value, ki=self.s_ki.value, kl=self.s_kl.value,
            lam1=self.s_lam1.value, lam2=self.s_lam2.value,
            kind=self.controller_kind,
            T=self.s_T.value, umax=self.s_umax.value,
        )

        if self.controller_kind == "manual":
            self._start_manual(snapshot)
            return

        self.mode = "playback"
        self.status_lines = ["Simulating ..."]
        self.status_color = COL["info"]
        self.playing = False
        self._sim_dots = 0
        with self._sim_lock:
            self._sim_result = None
        self._sim_thread = threading.Thread(target=self._sim_worker,
                                            args=(snapshot,), daemon=True)
        self._sim_thread.start()

    def _on_replay(self):
        if self.sim is None:
            return
        # Replay the recorded trajectory (manual runs are replayed, not re-driven).
        self.mode = "playback"
        self.idx = 0
        self.playing = True
        self.b_pause.label = "Pause"

    def _on_pause(self):
        if self.sim is None:
            return
        self.playing = not self.playing
        self.b_pause.label = "Resume" if not self.playing else "Pause"

    # ----- Manual (live keyboard-driven) integration -----------------------
    def _start_manual(self, s: dict) -> None:
        """Begin a live run where the torque is supplied by the arrow keys.

        Lets the user try to keep the body up by hand under |u| <= u_max, to
        check whether a chosen IC is really inside the recoverable set.
        """
        self.mode = "live"
        self.params = SegwayParams()
        self.dyn    = SegwayDynamics(self.params)
        self.ctrl   = None
        self.controller_kind = "manual"
        self.theta_S = None
        self.K_S     = None

        um = float(s.get("umax", 0.0))
        self.manual_umax = um if um > 0.0 else 29.4   # default ~ D if unbounded
        self.u_max       = self.manual_umax
        self.manual_T    = float(s["T"])
        x0 = np.array([s["th0"], s["ps0"], s["dth0"], s["dps0"]], dtype=float)
        self.manual_x = x0.copy()
        self.manual_t = 0.0
        self.manual_u = 0.0
        self.fell     = False

        th, ps, dth, dps = x0
        self._hist = dict(
            t=[0.0], theta=[th], psi=[ps], phi=[ps + th],
            dtheta=[dth], dpsi=[dps], dphi=[dps + dth],
            u=[0.0], v=[0.0], xi=[0.0], dxi=[0.0], V=[0.0], Vdot=[0.0])
        self.sim = {k: np.asarray(v, dtype=float) for k, v in self._hist.items()}
        self.idx = 0
        self.playing = True
        self.b_pause.label = "Pause"
        self.status_lines = [
            "MANUAL mode -- you drive the torque.",
            f"Hold LEFT / RIGHT  =  -/+ {self.manual_umax:.1f} Nm.",
            "Keep |theta| < 90 deg for the whole horizon.",
            f"Tip: lower the 'speed x' slider (now {self.s_speed.value:.2f}x) for slow-mo.",
        ]
        self.status_color = COL["info"]

    def _live_step(self) -> None:
        """Advance the live manual simulation by one display frame."""
        keys = pygame.key.get_pressed()
        um = self.manual_umax
        u = 0.0
        if keys[pygame.K_LEFT]:
            u -= um
        if keys[pygame.K_RIGHT]:
            u += um
        u = float(np.clip(u, -um, um))
        self.manual_u = u

        # Sim-seconds advanced per real frame.  speed < 1 => slow motion, so the
        # fast (~0.1 s) pendulum dynamics become hand-controllable.  Read live so
        # the slider can be dragged mid-run.
        speed = float(self.s_speed.value)
        dt_frame = speed / self.fps
        n_inner = max(1, int(np.ceil(dt_frame / 5e-4)))
        h = dt_frame / n_inner
        x = self.manual_x
        rhs = lambda t, y: self.dyn.rhs(y, u)
        for _ in range(n_inner):
            x = rk4_step(rhs, 0.0, x, h)
        self.manual_x = x
        self.manual_t += dt_frame

        th, ps, dth, dps = x
        H = self._hist
        H["t"].append(self.manual_t); H["theta"].append(th); H["psi"].append(ps)
        H["phi"].append(ps + th);     H["dtheta"].append(dth); H["dpsi"].append(dps)
        H["dphi"].append(dps + dth);  H["u"].append(u);        H["v"].append(0.0)
        H["xi"].append(0.0); H["dxi"].append(0.0); H["V"].append(0.0); H["Vdot"].append(0.0)
        self.sim = {k: np.asarray(v, dtype=float) for k, v in H.items()}
        self.idx = len(H["t"]) - 1

        if abs(th) >= np.pi / 2.0:
            self.fell = True
            self.playing = False
            self.status_lines = [
                f"FELL at t = {self.manual_t:.2f} s  (|theta| reached 90 deg).",
                "You could not hold it -> evidence the IC is NOT recoverable",
                "with this torque budget.  Press Start to retry.",
            ]
            self.status_color = COL["warn"]
        elif self.manual_t >= self.manual_T:
            self.playing = False
            self.status_lines = [
                f"SURVIVED the full {self.manual_T:.0f} s  (|theta| < 90 deg).",
                "Consistent with the IC being recoverable.",
                "Press Start to retry, or Replay to watch it again.",
            ]
            self.status_color = COL["ok"]
        else:
            self.status_lines = [
                f"MANUAL  t = {self.manual_t:5.2f} / {self.manual_T:.0f} s    u = {u:+6.1f} Nm",
                f"theta = {np.degrees(th):+6.1f} deg     (fail at +-90)",
                f"Hold LEFT / RIGHT = -/+ {um:.1f} Nm    speed = {speed:.2f}x",
            ]
            self.status_color = COL["info"]

    def _on_save_gif(self):
        if self.sim is None:
            self.status_lines = ["No simulation yet -- press Start first."]
            self.status_color = COL["warn"]
            return
        # Tell the user something is happening
        self.status_lines = ["Saving GIF ..."]
        self.status_color = COL["info"]
        self._draw(); pygame.display.flip()

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        path = self.anim_dir / f"segway_{timestamp}.gif"
        try:
            n = self._render_gif(path)
            self.status_lines = [
                f"Saved {n} frames to:",
                str(path),
            ]
            self.status_color = COL["ok"]
        except Exception as exc:                                       # pragma: no cover
            self.status_lines = ["GIF save failed:", str(exc)]
            self.status_color = COL["warn"]

    # ----- Simulation (runs in background) ---------------------------------
    def _sim_worker(self, s: dict) -> None:
        """Run simulate() on a worker thread; main thread polls the result."""
        t0 = time.perf_counter()
        kind = s.get("kind", "lyapunov")
        try:
            params = SegwayParams()
            gains  = ControllerGains(kp=s["kp"], ki=s["ki"], kl=s["kl"])
            if kind == "lqr":
                lam1 = float(s["lam1"]); lam2 = float(s["lam2"])
                weights = LQRWeights(Q=lam1 * np.eye(4), R=lam2 * np.eye(1))
                base = LQRController(params, weights)
            else:
                base = LyapunovController(params, gains)
            umax   = float(s.get("umax", 0.0))
            ctrl   = SaturatedController(base, umax) if umax > 0.0 else base
            dyn    = SegwayDynamics(params)
            T = s["T"]
            x0 = [s["th0"], s["ps0"], s["dth0"], s["dps0"]]
            n_out = max(400, int(T * 150))
            sim = simulate(dyn, ctrl, x0, T, n_out=n_out, inner_dt=5e-4)

            # A non-recoverable IC (a red/blue cell on the map) makes the body
            # topple: the trajectory diverges to huge / non-finite values.  The
            # renderer converts state to pixels with int(), and int(NaN)/int(inf)
            # raises -- which, on the unguarded main loop, would close the window
            # the instant such a frame is drawn.  So truncate the trajectory at
            # the moment it falls (|theta| > 1.7 rad, just past horizontal) and
            # scrub any remaining non-finite values.
            th = np.asarray(sim["theta"], float)
            bad = ~np.isfinite(th) | (np.abs(th) > 1.7)
            if bad.any():
                cut = max(int(np.argmax(bad)), 1)
                sim = {k: np.asarray(v, float)[:cut + 1] for k, v in sim.items()}
            sim = {k: np.nan_to_num(np.asarray(v, float),
                                    nan=0.0, posinf=1.0e6, neginf=-1.0e6)
                   for k, v in sim.items()}
            wall = time.perf_counter() - t0
            print(f"[animation] simulation done in {wall:.2f} s "
                  f"(kind={kind}, T={T} s, n_out={n_out}, u_max={umax:.1f})")
            with self._sim_lock:
                self._sim_result = dict(ok=True, params=params, gains=gains,
                                        ctrl=ctrl, dyn=dyn, sim=sim, T=T,
                                        wall=wall, umax=umax, kind=kind,
                                        lam1=s.get("lam1"), lam2=s.get("lam2"))
        except Exception as exc:                                       # pragma: no cover
            with self._sim_lock:
                self._sim_result = dict(ok=False, error=exc, kind=kind)

    def _poll_simulation(self) -> None:
        """If the worker thread produced a result, install it."""
        with self._sim_lock:
            res = self._sim_result
            self._sim_result = None
        if res is None:
            return
        if not res["ok"]:
            err = res["error"]
            kind = res.get("kind", "lyapunov")
            if isinstance(err, ValueError) and kind == "lyapunov":
                self.status_lines = ["Gain condition (27) violated:",
                                     str(err),
                                     "Increase kp or decrease kL."]
            else:
                self.status_lines = [f"{kind.upper()} setup failed:", str(err)]
            self.status_color = COL["warn"]
            self.sim = None
            return
        self.params  = res["params"]
        self.gains   = res["gains"]
        self.ctrl    = res["ctrl"]
        self.dyn     = res["dyn"]
        self.sim     = res["sim"]
        T            = res["T"]
        self.u_max   = float(res.get("umax", 0.0))
        self.controller_kind = res.get("kind", "lyapunov")
        self.idx     = 0
        self.stride  = max(1, int(len(self.sim["t"])
                                  / (T * self.fps / self.playback_speed)))
        self.playing = True
        self.b_pause.label = "Pause"

        u_peak = float(np.max(np.abs(self.sim["u"])))
        final_th = np.degrees(self.sim["theta"][-1])
        converged = abs(final_th) < 1.0

        if self.controller_kind == "lqr":
            # LQR has no Lyapunov-specific attraction threshold.
            self.theta_S = None
            self.K_S     = None
            poles = np.linalg.eigvals(self.ctrl.Acl)
            slow = np.max(poles.real)            # least-stable closed-loop pole
            K = np.ravel(self.ctrl.K)            # recomputed gains for this Q,R
            self.status_lines = [
                f"controller : LQR   Q = {res['lam1']:.1f} I,  R = {res['lam2']:.2f} I",
                f"K = [{K[0]:.2f}, {K[1]:.2f}, {K[2]:.2f}, {K[3]:.2f}]   (recomputed)",
                f"max Re(pole) = {slow:+.2f}   (3 poles weight-invariant here)",
                f"V(x0)={self.sim['V'][0]:.2f} -> V(T)={self.sim['V'][-1]:.2e}  (P from CARE)",
                f"final theta: {final_th:+.4f} deg "
                f"({'converged' if converged else 'NOT converged'})",
            ]
            self.status_color = COL["ok"] if converged else COL["warn"]
        else:
            self.theta_S = self.ctrl.critical_angle()
            self.K_S     = self.ctrl.K_S()
            lhs, rhs_ = self.ctrl.gain_condition_value()
            in_region = self.sim["V"][0] < self.K_S
            self.status_lines = [
                "controller : Lyapunov",
                f"gain check : {lhs:.2f}  >  {rhs_:.2f}",
                f"theta_S = {np.degrees(self.theta_S):.2f} deg   K_S = {self.K_S:.2f}",
                f"V(x0)={self.sim['V'][0]:.2f} -> V(T)={self.sim['V'][-1]:.2e}  "
                f"(in region: {'yes' if in_region else 'NO'})",
                f"final theta: {final_th:+.4f} deg",
            ]
            self.status_color = COL["ok"] if in_region else COL["warn"]

        if self.u_max > 0.0:
            hit = "  (LIMIT HIT)" if u_peak >= 0.999 * self.u_max else ""
            self.status_lines.append(
                f"u_max      : {self.u_max:.1f} Nm  peak|u| {u_peak:.1f}{hit}")
            self.status_lines.append(
                "saturated -> analytic dV/dt is approximate (use numeric dV/dt)")
        else:
            self.status_lines.append(
                f"u_max      : unbounded   peak|u| {u_peak:.1f} Nm")

    # ----- GIF rendering ----------------------------------------------------
    def _render_gif(self, path: Path) -> int:
        from PIL import Image                                          # lazy import
        T = self.s_T.value
        n_frames = max(20, int(T * self.gif_fps))
        N = len(self.sim["t"])
        # Render the LEFT side of the layout (no slider panel) into an
        # offscreen surface so the GIF doesn't include the controls.
        gif_w, gif_h = RIGHT_PANEL_X, HEIGHT
        canvas = pygame.Surface((gif_w, gif_h))
        # Target frame size (downscale to keep file size manageable)
        out_w = 700
        out_h = int(out_w * gif_h / gif_w)

        saved_idx = self.idx
        frames = []
        for k in range(n_frames):
            self.idx = int(round(k * (N - 1) / max(1, n_frames - 1)))
            canvas.fill(COL["bg"])
            self._draw_panels(canvas)
            arr = pygame.surfarray.array3d(canvas)        # (W, H, 3)
            arr = arr.swapaxes(0, 1)                      # (H, W, 3)
            img = Image.fromarray(arr.astype("uint8"))
            img = img.resize((out_w, out_h), Image.LANCZOS)
            frames.append(img)
        self.idx = saved_idx
        _save_gif(frames, path, fps=self.gif_fps)
        return len(frames)

    # ----- Main loop --------------------------------------------------------
    def run(self):
        running = True
        while running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    running = False
                if ev.type == pygame.KEYDOWN:
                    if ev.key == pygame.K_ESCAPE:
                        running = False
                    elif ev.key == pygame.K_SPACE:
                        self._on_pause()
                    elif ev.key == pygame.K_r:
                        self._on_start()
                    elif ev.key == pygame.K_g:
                        self._on_save_gif()
                for w in self.sliders + self.buttons:
                    w.handle(ev)

            # Auto-start once (used when launched from the recoverable-set map
            # with a chosen initial condition).
            if self._autostart_pending and self.sim is None \
                    and (self._sim_thread is None or not self._sim_thread.is_alive()):
                self._autostart_pending = False
                self._on_start()

            # Background simulation -- non-blocking.
            self._poll_simulation()
            if self._sim_thread is not None and self._sim_thread.is_alive():
                self._sim_dots = (self._sim_dots + 1) % 90
                dots = "." * (1 + self._sim_dots // 30)
                self.status_lines = [f"Simulating {dots}"]
                self.status_color = COL["info"]

            if self.mode == "live" and self.playing:
                # Live manual integration: advance one frame of real dynamics.
                self._live_step()
            elif self.sim is not None and self.playing:
                N = len(self.sim["t"])
                self.idx = min(self.idx + self.stride, N - 1)
                if self.idx >= N - 1:
                    self.playing = False
                    self.b_pause.label = "Resume"

            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    # ----- Drawing ----------------------------------------------------------
    def _draw_panels(self, surf):
        """Draw the three left-side panels (segway view + two plots).

        Used both for live rendering and for GIF capture.
        """
        if self.sim is None:
            # placeholder text inside SEGWAY_RECT
            pygame.draw.rect(surf, COL["plot_bg"], SEGWAY_RECT.inflate(-4, -4))
            pygame.draw.rect(surf, COL["axis"],    SEGWAY_RECT.inflate(-4, -4), 1)
            msg1 = self.title_font.render("No simulation yet", True, COL["text"])
            msg2 = self.font.render("Adjust the sliders on the right, then press Start.",
                                    True, COL["sub_text"])
            surf.blit(msg1, (SEGWAY_RECT.centerx - msg1.get_width() // 2,
                              SEGWAY_RECT.centery - msg1.get_height()))
            surf.blit(msg2, (SEGWAY_RECT.centerx - msg2.get_width() // 2,
                              SEGWAY_RECT.centery + 4))
            for r in (STATE_RECT, LYAP_RECT):
                pygame.draw.rect(surf, COL["plot_bg"], r.inflate(-4, -4))
                pygame.draw.rect(surf, COL["axis"],    r.inflate(-4, -4), 1)
            return

        sim = self.sim
        i   = self.idx

        theta_S_deg = (np.degrees(self.theta_S)
                       if self.theta_S is not None else None)
        draw_segway(surf, self.font, self.mono_font,
                    SEGWAY_RECT, self.params,
                    sim["theta"][i], sim["phi"][i], sim["psi"][i],
                    sim["t"][i], sim["V"][i], sim["u"][i],
                    self.K_S, theta_S_deg)

        t      = sim["t"]
        th_deg = np.degrees(sim["theta"])
        ps_deg = np.degrees(sim["psi"])
        u_arr  = sim["u"]
        y_lo = float(min(th_deg.min(), ps_deg.min(), u_arr.min()))
        y_hi = float(max(th_deg.max(), ps_deg.max(), u_arr.max()))
        pad  = 0.07 * (y_hi - y_lo + 1.0)
        idx_slice = slice(0, i + 1)
        draw_plot(surf, self.font, self.small_font, STATE_RECT,
                  series=[
                      (t[idx_slice], th_deg[idx_slice], COL["theta"], "theta [deg]"),
                      (t[idx_slice], ps_deg[idx_slice], COL["psi"],   "psi   [deg]"),
                      (t[idx_slice], u_arr[idx_slice],  COL["u"],     "u     [Nm]"),
                  ],
                  t_range=(0.0, float(t[-1])),
                  y_range=(y_lo - pad, y_hi + pad),
                  cursor_t=float(t[i]),
                  title="States and control input")

        if self.controller_kind == "manual":
            # Bottom panel: body angle vs the +-90 deg failure envelope.
            th_deg_full = np.degrees(sim["theta"])
            lo = min(-95.0, float(th_deg_full.min()) * 1.1)
            hi = max(95.0, float(th_deg_full.max()) * 1.1)
            draw_plot(surf, self.font, self.small_font, LYAP_RECT,
                      series=[(t[idx_slice], th_deg_full[idx_slice],
                               COL["theta"], "theta [deg]")],
                      t_range=(0.0, float(t[-1])),
                      y_range=(lo, hi),
                      cursor_t=float(t[i]),
                      title="Manual mode -- keep theta within +-90 deg",
                      hlines=[( 90.0, COL["KS"], "+90 deg (fall)"),
                              (-90.0, COL["KS"], "-90 deg (fall)")])
            return

        V    = sim["V"]
        Vdot = sim["Vdot"]
        is_lqr = (self.controller_kind == "lqr")
        y_lo2 = float(min(V.min(), Vdot.min(), 0.0)) - 1.0
        top   = V.max() if self.K_S is None else max(V.max(), self.K_S)
        y_hi2 = float(top) * 1.05 + 1.0
        vdot_lbl = "dV/dt = 2 x^T P f(x)" if is_lqr else "dV/dt = -dxi^2"
        hlines = None if self.K_S is None else \
            [(self.K_S, COL["KS"], "K_S (attraction threshold)")]
        draw_plot(surf, self.font, self.small_font, LYAP_RECT,
                  series=[
                      (t[idx_slice], V[idx_slice],    COL["V"],    "V(x(t))"),
                      (t[idx_slice], Vdot[idx_slice], COL["Vdot"], vdot_lbl),
                  ],
                  t_range=(0.0, float(t[-1])),
                  y_range=(y_lo2, y_hi2),
                  cursor_t=float(t[i]),
                  title=("LQR value function  V = x^T P x" if is_lqr
                         else "Lyapunov function"),
                  hlines=hlines)

    def _draw(self):
        scr = self.screen
        scr.fill(COL["bg"])

        # Left panel (data view + plots)
        self._draw_panels(scr)

        # Right panel: sliders + buttons + status
        pygame.draw.rect(scr, COL["panel_bg"],
                         pygame.Rect(RIGHT_PANEL_X, 0, WIDTH - RIGHT_PANEL_X, HEIGHT))

        for y, label in self.section_labels:
            t = self.header_font.render(label, True, COL["section"])
            scr.blit(t, (RIGHT_PANEL_X + 30, y))

        for s in self.sliders:
            s.draw(scr, self.font)
        for b in self.buttons:
            b.draw(scr, self.font)

        for i, line in enumerate(self.status_lines):
            scr.blit(self.small_font.render(line, True, self.status_color),
                     (RIGHT_PANEL_X + 30, self.status_y + i * 16))

        hint = self.small_font.render(
            "Keys:  R = Start   Space = Pause/Resume   G = Save GIF   Esc = Quit"
            "    |   Manual mode: hold LEFT / RIGHT for -/+ u_max",
            True, COL["sub_text"])
        scr.blit(hint, (RIGHT_PANEL_X + 30, HEIGHT - 22))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def build_animation(*, T_sim: float = 8.0,
                    state0=(0.30, 0.0, 0.0, 0.0),
                    u_max: float = 0.0,
                    autostart: bool = False,
                    controller: str = "lyapunov",
                    lam1: float = 50.0,
                    lam2: float = 1.0,
                    playback_speed: float = 1.0,
                    fps: int = 60,
                    gif_fps: int = 25,
                    anim_dir = None):
    """Open the pygame animation window and run the event loop until quit.

    Parameters
    ----------
    state0     : (theta0, psi0, dtheta0, dpsi0) initial condition to preload.
    u_max      : torque limit [Nm]; 0 means the unbounded control law.
    autostart  : if True, immediately simulate ``state0`` on open (used when
                 the window is spawned by clicking a cell of the recoverable map).
    controller : "lyapunov" (original) or "lqr".
    lam1, lam2 : LQR weights, used as Q = lam1 * I (4x4), R = lam2 * I (1x1).
    """
    app = App(T_sim=T_sim, state0=state0, u_max=u_max, autostart=autostart,
              controller=controller, lam1=lam1, lam2=lam2,
              playback_speed=playback_speed, fps=fps, gif_fps=gif_fps,
              anim_dir=anim_dir)
    app.run()
    return app


if __name__ == "__main__":
    build_animation()
