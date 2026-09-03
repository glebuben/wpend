"""Окно исследователя: карта начальных условий слева, робот справа.

    python -m wpend.viz.explorer                 # сетка 21x21 считается заранее
    python -m wpend.viz.explorer --grid 41
    python -m wpend.viz.explorer --no-precompute # считать по одной по клику

Управление: клик по клетке -- проиграть её траекторию; ПРОБЕЛ -- пауза;
R -- сначала; Esc -- выход.

Модуль читает только Trajectory / TrajectoryBatch: ни одной формулы динамики
здесь нет.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ..controller import LinearFeedbackController
from ..integrator import RK4Integrator
from ..models import WheeledPendulum
from ..rollout import rollout_many
from .grid import FELL_BACKWARD, FELL_FORWARD, HELD, GridSpec, classify

# ЛКР для линеаризации в верхнем положении, Q = diag(100, 1, 10, 1), R = 1.
K_LQR = np.array([[95.122221, 1.0, 19.807871, 1.449624]])

I_THETA, I_PHI, I_DTHETA, I_DPHI = 0, 1, 2, 3

W, H = 1280, 760
MAP = (56, 78, 568, 600)            # x, y, w, h
ANIM = (656, 78, 600, 336)
PLOT = (656, 434, 600, 244)

BG = (22, 24, 28)
PANEL = (33, 36, 42)
GRID_LINE = (52, 56, 64)
TEXT = (216, 219, 224)
DIM = (140, 146, 156)
ACCENT = (240, 198, 92)
CURVE = (245, 247, 250)

OUTCOME_COLOR = {
    HELD: (58, 132, 196),
    FELL_FORWARD: (205, 88, 62),
    FELL_BACKWARD: (150, 108, 190),
}
UNKNOWN = (58, 62, 70)


def _shade(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


class Explorer:
    """Состояние приложения: сетка, кэш траекторий, воспроизведение."""

    def __init__(self, args):
        self.args = args
        self.system = WheeledPendulum(u_max=args.u_max)
        self.controller = LinearFeedbackController(K_LQR)
        self.integrator = RK4Integrator()
        self.spec = GridSpec(n=args.grid, theta_max=args.theta_max,
                             dtheta_max=args.dtheta_max)

        self.n_steps = int(round(args.horizon / args.dt))
        if self.n_steps % args.stride:
            self.n_steps += args.stride - self.n_steps % args.stride

        self.batch = None
        self.outcome = np.full(self.spec.n * self.spec.n, -1, dtype=int)
        self.t_fall = np.full(self.spec.n * self.spec.n, np.nan)
        self.cache: dict[int, object] = {}
        self.compute_seconds = 0.0

        if args.precompute:
            self._precompute()

        self.hover = None
        self.selected = None
        self.traj = None
        self.playing = True
        self.play_time = 0.0
        self._select_nearest(0.6 * args.theta_max, 0.0)

    # --- счёт --------------------------------------------------------------

    def _precompute(self):
        """Один векторный прогон всей сетки: M начальных условий одновременно."""
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)
        t0 = time.perf_counter()
        self.batch = rollout_many(self.system, self.controller, self.integrator,
                                  X0, self.args.dt, self.n_steps, self.args.stride)
        self.outcome, self.t_fall = classify(self.batch, I_THETA, self.args.theta_fall)
        self.compute_seconds = time.perf_counter() - t0
        print(f"сетка {self.spec.n}x{self.spec.n} = {len(self.batch)} НУ, "
              f"{self.n_steps} шагов: {self.compute_seconds:.2f} с, "
              f"{self.batch.nbytes / 2**20:.1f} МБ в памяти")

    def _trajectory(self, m: int):
        """Траектория клетки: из готовой пачки либо посчитанная по требованию."""
        if self.batch is not None:
            return self.batch[m]
        if m in self.cache:
            return self.cache[m]
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)[m:m + 1]
        t0 = time.perf_counter()
        batch = rollout_many(self.system, self.controller, self.integrator,
                             X0, self.args.dt, self.n_steps, self.args.stride)
        self.compute_seconds = time.perf_counter() - t0
        outcome, t_fall = classify(batch, I_THETA, self.args.theta_fall)
        self.outcome[m] = outcome[0]
        self.t_fall[m] = t_fall[0]
        self.cache[m] = batch[0]
        return self.cache[m]

    # --- выбор клетки ------------------------------------------------------

    def select(self, ix: int, iy: int):
        self.selected = (ix, iy)
        self.traj = self._trajectory(self.spec.index(ix, iy))
        self.play_time = 0.0
        self.playing = True

    def _select_nearest(self, theta, dtheta):
        ix = int(np.argmin(np.abs(self.spec.thetas - theta)))
        iy = int(np.argmin(np.abs(self.spec.dthetas - dtheta)))
        self.select(ix, iy)

    # --- воспроизведение ---------------------------------------------------

    @property
    def frame(self) -> int:
        if self.traj is None:
            return 0
        dt_frame = self.traj.t[1] - self.traj.t[0]
        return int(np.clip(self.play_time / dt_frame, 0, len(self.traj) - 1))

    def advance(self, dt_wall: float):
        if self.traj is None or not self.playing:
            return
        self.play_time += dt_wall * self.args.speed
        if self.play_time > self.traj.t[-1] - self.traj.t[0]:
            self.play_time = 0.0


# ---------------------------------------------------------------------------
#  Отрисовка.  Каждая функция рисует один прямоугольник и ничего не считает.
# ---------------------------------------------------------------------------

def _panel(screen, pg, rect, title, font):
    x, y, w, h = rect
    pg.draw.rect(screen, PANEL, (x, y, w, h), border_radius=6)
    screen.blit(font.render(title, True, DIM), (x + 10, y - 20))


def _map_surface(app, pg):
    n = app.spec.n
    surf = pg.Surface((n, n))
    for iy in range(n):
        for ix in range(n):
            m = app.spec.index(ix, iy)
            code = app.outcome[m]
            if code < 0:
                color = UNKNOWN
            else:
                base = OUTCOME_COLOR[code]
                if np.isnan(app.t_fall[m]):
                    color = base
                else:  # чем раньше упал, тем темнее клетка
                    frac = app.t_fall[m] / max(app.args.horizon, 1e-9)
                    color = _shade(base, 0.45 + 0.55 * frac)
            surf.set_at((ix, n - 1 - iy), color)
    return surf


def _to_map_px(app, theta, dtheta):
    x, y, w, h = MAP
    fx = (theta + app.spec.theta_max) / (2 * app.spec.theta_max)
    fy = (dtheta + app.spec.dtheta_max) / (2 * app.spec.dtheta_max)
    return x + fx * w, y + (1.0 - fy) * h


def draw_map(app, screen, pg, font):
    x, y, w, h = MAP
    _panel(screen, pg, MAP, "INITIAL CONDITIONS  theta0 (rad) x dtheta0 (rad/s)", font)
    screen.blit(pg.transform.scale(_map_surface(app, pg), (w, h)), (x, y))

    for frac in (0.5,):   # оси через ноль
        pg.draw.line(screen, GRID_LINE, (x + frac * w, y), (x + frac * w, y + h))
        pg.draw.line(screen, GRID_LINE, (x, y + frac * h), (x + w, y + frac * h))

    if app.traj is not None:   # фазовая кривая выбранной траектории
        pts = [_to_map_px(app, th, dth)
               for th, dth in zip(app.traj.x[:, I_THETA], app.traj.x[:, I_DTHETA])]
        pts = [(px, py) for px, py in pts
               if x - 400 < px < x + w + 400 and y - 400 < py < y + h + 400]
        if len(pts) > 1:
            clip = screen.get_clip()
            screen.set_clip((x, y, w, h))
            pg.draw.lines(screen, CURVE, False, pts, 2)
            head = pts[min(app.frame, len(pts) - 1)]
            pg.draw.circle(screen, ACCENT, (int(head[0]), int(head[1])), 5)
            screen.set_clip(clip)

    cell = w / app.spec.n
    for marker, color, width in ((app.hover, DIM, 1), (app.selected, ACCENT, 2)):
        if marker is None:
            continue
        ix, iy = marker
        pg.draw.rect(screen, color,
                     (x + ix * cell, y + (app.spec.n - 1 - iy) * cell, cell, cell), width)

    labels = [(f"-{app.spec.theta_max:.2f}", x, y + h + 6),
              (f"+{app.spec.theta_max:.2f}", x + w - 34, y + h + 6),
              (f"+{app.spec.dtheta_max:.1f}", x - 46, y),
              (f"-{app.spec.dtheta_max:.1f}", x - 46, y + h - 14)]
    for text, tx, ty in labels:
        screen.blit(font.render(text, True, DIM), (tx, ty))


def draw_robot(app, screen, pg, font):
    x, y, w, h = ANIM
    _panel(screen, pg, ANIM, "PLANT", font)
    if app.traj is None:
        return
    state = app.traj.x[app.frame]
    theta, phi = state[I_THETA], state[I_PHI]

    p = app.system.p
    wheel_px = 26.0
    m2px = wheel_px / p.r
    cx, cy = x + w / 2, y + h * 0.72

    pg.draw.line(screen, GRID_LINE, (x + 8, cy + wheel_px), (x + w - 8, cy + wheel_px), 2)
    shift = (p.r * phi * m2px) % (0.1 * m2px)     # бегущие метки дороги
    step = 0.1 * m2px
    k = -int(w / (2 * step)) - 1
    while x + w / 2 + k * step - shift < x + w:
        tx = x + w / 2 + k * step - shift
        if x + 8 < tx < x + w - 8:
            pg.draw.line(screen, GRID_LINE, (tx, cy + wheel_px), (tx, cy + wheel_px + 8), 2)
        k += 1

    pg.draw.circle(screen, (96, 104, 118), (int(cx), int(cy)), int(wheel_px), 3)
    spoke = (cx + wheel_px * np.sin(phi), cy - wheel_px * np.cos(phi))
    pg.draw.line(screen, (150, 158, 172), (cx, cy), spoke, 2)

    tip = (cx + p.l * m2px * np.sin(theta), cy - p.l * m2px * np.cos(theta))
    pg.draw.line(screen, (226, 230, 236), (cx, cy), tip, 6)
    pg.draw.circle(screen, ACCENT, (int(tip[0]), int(tip[1])), 9)

    torque = app.traj.u[min(app.frame, app.traj.n_steps - 1), 0]
    limit = app.args.u_max
    bar_w = 160
    bx, by = x + w - bar_w - 16, y + 16
    pg.draw.rect(screen, GRID_LINE, (bx, by, bar_w, 12), 1)
    pg.draw.rect(screen, ACCENT,
                 (bx + bar_w / 2, by + 1, np.clip(torque / limit, -1, 1) * bar_w / 2, 10))
    screen.blit(font.render(f"u = {torque:+6.2f} / {limit:.1f} N*m", True, DIM), (bx, by + 16))

    info = [f"t     = {app.traj.t[app.frame] - app.traj.t[0]:5.2f} s",
            f"theta = {theta:+.4f} rad",
            f"dtheta= {state[I_DTHETA]:+.4f} rad/s",
            f"phi   = {phi:+.3f} rad"]
    for i, line in enumerate(info):
        screen.blit(font.render(line, True, TEXT), (x + 14, y + 14 + 18 * i))


def _plot(screen, pg, rect, ts, ys, color, label, font, guides=()):
    x, y, w, h = rect
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1)
    lo, hi = float(np.min(ys)), float(np.max(ys))
    for g in guides:
        lo, hi = min(lo, -abs(g)), max(hi, abs(g))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def to_px(t, v):
        return (x + (t - ts[0]) / max(ts[-1] - ts[0], 1e-9) * w,
                y + h - (v - lo) / (hi - lo) * h)

    for g in guides:
        for sign in (+1, -1):
            gy = to_px(ts[0], sign * abs(g))[1]
            if y <= gy <= y + h:
                pg.draw.line(screen, (70, 60, 50), (x, gy), (x + w, gy), 1)
    if lo < 0 < hi:
        zy = to_px(ts[0], 0.0)[1]
        pg.draw.line(screen, GRID_LINE, (x, zy), (x + w, zy), 1)
    pg.draw.lines(screen, color, False, [to_px(t, v) for t, v in zip(ts, ys)], 2)
    screen.blit(font.render(label, True, DIM), (x + 6, y + 4))
    screen.blit(font.render(f"{hi:+.2f}", True, DIM), (x + w - 52, y + 4))
    screen.blit(font.render(f"{lo:+.2f}", True, DIM), (x + w - 52, y + h - 18))
    return to_px


def draw_plots(app, screen, pg, font):
    x, y, w, h = PLOT
    _panel(screen, pg, PLOT, "TRAJECTORY", font)
    if app.traj is None:
        return
    ts = app.traj.t - app.traj.t[0]
    half = (h - 30) / 2
    to_px1 = _plot(screen, pg, (x + 12, y + 10, w - 24, half),
                   ts, app.traj.x[:, I_THETA], (120, 180, 240), "theta (rad)", font,
                   guides=(app.args.theta_fall,))
    to_px2 = _plot(screen, pg, (x + 12, y + 20 + half, w - 24, half),
                   ts[:-1], app.traj.u[:, 0], (240, 170, 110), "u (N*m)", font,
                   guides=(app.args.u_max,))
    for to_px, rect_y in ((to_px1, y + 10), (to_px2, y + 20 + half)):
        cursor = to_px(ts[min(app.frame, len(ts) - 1)], 0.0)[0]
        pg.draw.line(screen, ACCENT, (cursor, rect_y), (cursor, rect_y + half), 1)


def draw_header(app, screen, pg, font, big):
    screen.blit(big.render("wpend explorer  --  wheeled pendulum, LQR feedback",
                           True, TEXT), (24, 14))
    mode = ("precomputed grid" if app.batch is not None
            else "on-demand (--no-precompute)")
    right = (f"u_max={app.args.u_max:g}  horizon={app.args.horizon:g}s  "
             f"dt={app.args.dt:g}  stride={app.args.stride}  {mode}  "
             f"[{app.compute_seconds:.2f}s]")
    screen.blit(font.render(right, True, DIM), (W - 24 - font.size(right)[0], 44))

    legend = [("held", HELD), ("fell forward", FELL_FORWARD), ("fell backward", FELL_BACKWARD)]
    lx = MAP[0] - 32
    for text, code in legend:
        pg.draw.rect(screen, OUTCOME_COLOR[code], (lx, H - 46, 14, 14))
        screen.blit(font.render(text, True, DIM), (lx + 20, H - 46))
        lx += 40 + font.size(text)[0]
    hint = "click a cell  |  SPACE pause  |  R restart  |  ESC quit"
    screen.blit(font.render(hint, True, GRID_LINE), (W - 24 - font.size(hint)[0], H - 46))


def run(app, max_frames=None, screenshot=None):
    import pygame as pg

    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("wpend explorer")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 15)
    big = pg.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
    clock = pg.time.Clock()

    x, y, w, h = MAP
    cell = w / app.spec.n
    running, frames = True, 0
    while running:
        dt_wall = clock.tick(60) / 1000.0
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.MOUSEMOTION:
                mx, my = event.pos
                inside = x <= mx < x + w and y <= my < y + h
                app.hover = ((int((mx - x) / cell),
                              app.spec.n - 1 - int((my - y) / cell)) if inside else None)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1 and app.hover:
                app.select(*app.hover)
            elif event.type == pg.KEYDOWN:
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif event.key == pg.K_SPACE:
                    app.playing = not app.playing
                elif event.key == pg.K_r:
                    app.play_time, app.playing = 0.0, True

        app.advance(dt_wall)

        screen.fill(BG)
        draw_header(app, screen, pg, font, big)
        draw_map(app, screen, pg, font)
        draw_robot(app, screen, pg, font)
        draw_plots(app, screen, pg, font)
        pg.display.flip()

        frames += 1
        if max_frames is not None and frames >= max_frames:
            running = False

    if screenshot:
        pg.image.save(screen, screenshot)
        print(f"скриншот: {screenshot}")
    pg.quit()


def build_parser():
    p = argparse.ArgumentParser(
        description="Карта начальных условий колёсного маятника + анимация.")
    p.add_argument("--grid", type=int, default=21,
                   help="сетка N x N начальных условий (по умолчанию 21)")
    p.add_argument("--precompute", dest="precompute", action="store_true",
                   help="посчитать всю сетку одним векторным прогоном (по умолчанию)")
    p.add_argument("--no-precompute", dest="precompute", action="store_false",
                   help="не считать заранее: клетка считается по клику")
    p.set_defaults(precompute=True)
    p.add_argument("--u-max", type=float, default=3.0, help="предел момента, Н*м")
    p.add_argument("--horizon", type=float, default=5.0, help="горизонт прогона, с")
    p.add_argument("--dt", type=float, default=1e-3, help="шаг интегрирования, с")
    p.add_argument("--stride", type=int, default=10,
                   help="сохранять каждый stride-й кадр (счёт идёт с шагом dt)")
    p.add_argument("--theta-max", type=float, default=0.8, help="границы карты по theta0")
    p.add_argument("--dtheta-max", type=float, default=5.0, help="границы карты по dtheta0")
    p.add_argument("--theta-fall", type=float, default=1.3,
                   help="угол, начиная с которого считаем, что корпус упал")
    p.add_argument("--speed", type=float, default=1.0, help="скорость воспроизведения")
    p.add_argument("--frames", type=int, default=None,
                   help="закрыть окно после N кадров (для headless-проверки)")
    p.add_argument("--screenshot", type=str, default=None,
                   help="сохранить кадр в PNG и выйти (включает headless-режим)")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        if args.frames is None:
            args.frames = 2
    run(Explorer(args), max_frames=args.frames, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
