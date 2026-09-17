"""Карты перебора цен в стиле окна исследователя: две цены рядом и их разница.

Читает то, что посчитал `examples/cost_sweep.py run`, и ничего не пересчитывает
по сетке. Слева карта цены A, в середине -- цены B, справа -- разница.
Внизу -- theta(t), u(t) и phi(t) выбранной клетки для обеих цен.

    uv run python examples/cost_map_view.py                       # base против R=100, ЛКР, 10 Н·м
    uv run python examples/cost_map_view.py --a base --b phi=1e-2 --u-max 3
    uv run python examples/cost_map_view.py --method mpc          # карты MPC 11x11

Цвета карт -- те же, что в `wpend.viz.explorer`: синий -- удержан, красный --
упал вперёд, фиолетовый -- упал назад; чем раньше упал, тем темнее. Зелёная
линия -- граница восстановимости (свойство системы и предела, не регулятора),
кружки -- сёдла.

Карта разницы:
    жёлтый   -- держит только A          бирюзовый -- держит только B
    тёмно-синий -- держат оба            тёмный    -- не держит никто

Управление: клик по клетке любой карты -- траектории обеих цен (для ЛКР
считаются на лету, ~0.5 с; для MPC берутся из файла). Кнопки сверху выбирают
метод, предел и цены A / B. Tab -- поменять A и B местами. Esc -- выход.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import LinearFeedbackController, RK4Integrator, rollout
from wpend.models import WheeledPendulum
from wpend.viz.explorer import (
    ACCENT, BG, BTN, BTN_ON, CURVE, DIM, GHOST, GRID_LINE, LIMIT, OUTCOME_COLOR,
    PANEL, TEXT, UNKNOWN,
)
from wpend.viz.grid import HELD

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cost_sweep as cs  # noqa: E402  -- пресеты и сетка берутся оттуда, а не копируются

W, H = 1280, 940
MAPS = {"A": (48, 176, 380, 400), "B": (468, 176, 380, 400), "diff": (888, 176, 360, 400)}
PLOTS = [(48, 668, 380, 220), (468, 668, 380, 220), (888, 668, 360, 220)]
ONLY_A, ONLY_B, BOTH, NONE = ACCENT, GHOST, (40, 80, 124), (40, 43, 50)
SURF_CACHE = {}


def shade(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


class Data:
    """Что лежит в .npz для выбранного метода и предела. Только чтение."""

    def __init__(self, lqr_path, mpc_path):
        self.files = {}
        for method, path in (("lqr", lqr_path), ("mpc", mpc_path)):
            if Path(path).exists():
                self.files[method] = np.load(path, allow_pickle=False)
        if "lqr" not in self.files:
            sys.exit(f"нет {lqr_path}: сначала `uv run python examples/cost_sweep.py run`")

    def u_maxes(self, method):
        return [float(u) for u in self.files[method]["u_maxes"]]

    def presets(self, method):
        return [str(p) for p in self.files[method]["presets"]]

    def has(self, method, u_max, name):
        d = self.files.get(method)
        return d is not None and f"{u_max}__{name}__outcome" in d

    def get(self, method, u_max, name, field):
        return self.files[method][f"{u_max}__{name}__{field}"]

    def axes(self, method, u_max):
        d = self.files[method]
        return d[f"{u_max}__thetas"], d[f"{u_max}__dthetas"], d[f"{u_max}__rec"]


class View:
    def __init__(self, args):
        self.data = Data(args.lqr, args.mpc_file)
        self.method = args.method if args.method in self.data.files else "lqr"
        self.u_max = args.u_max
        self.a, self.b = args.a, args.b
        self.selected = None           # (ix, iy)
        self.traj = {}                 # "A"/"B" -> (t, x, u) или None
        self.fix_state()

    # --- состояние ------------------------------------------------------------

    def fix_state(self):
        """Привести выбор к тому, что есть в файле, и пересчитать геометрию."""
        if self.u_max not in self.data.u_maxes(self.method):
            self.u_max = self.data.u_maxes(self.method)[0]
        avail = self.data.presets(self.method)
        if self.a not in avail:
            self.a = avail[0]
        if self.b not in avail:
            self.b = avail[min(1, len(avail) - 1)]
        self.thetas, self.dthetas, self.rec = self.data.axes(self.method, self.u_max)
        self.n = self.thetas.size
        world = WheeledPendulum(u_max=self.u_max)
        self.limit_theta = np.linspace(-1.5, 1.5, 1201)
        self.limit_floor, self.limit_ceiling = world.recoverable_bounds(self.u_max,
                                                                        self.limit_theta)
        self.theta_eq = world.saddle_angle(self.u_max)
        self.selected = None
        self.traj = {}

    def select(self, ix, iy):
        self.selected = (ix, iy)
        m = iy * self.n + ix
        x0 = np.array([self.thetas[ix], 0.0, self.dthetas[iy], 0.0])
        for slot, name in (("A", self.a), ("B", self.b)):
            if self.method == "mpc":
                X = self.data.get("mpc", self.u_max, name, "x")[m].astype(float)
                U = self.data.get("mpc", self.u_max, name, "u")[m].astype(float)
                t = self.data.files["mpc"]["t"]
                self.traj[slot] = (t, X, U)
            else:
                _, _, K, _ = cs.design(name)
                tr = rollout(WheeledPendulum(u_max=self.u_max), LinearFeedbackController(K),
                             RK4Integrator(), x0, cs.DT, int(round(float(
                                 self.data.files["lqr"]["horizon"]) / cs.DT)))
                self.traj[slot] = (tr.t[::10], tr.x[::10], tr.u[::10, 0])

    # --- геометрия карты -------------------------------------------------------

    def to_px(self, rect, theta, dtheta):
        """Как `_to_map_px` окна: узел -- центр клетки, края на полклетки дальше."""
        x, y, w, h = rect
        edge = self.n / max(self.n - 1, 1)
        half_th, half_dth = self.thetas[-1] * edge, self.dthetas[-1] * edge
        return (x + (theta + half_th) / (2 * half_th) * w,
                y + (1.0 - (dtheta + half_dth) / (2 * half_dth)) * h)

    def cell_at(self, pos):
        for rect in MAPS.values():
            x, y, w, h = rect
            if x <= pos[0] < x + w and y <= pos[1] < y + h:
                ix = int((pos[0] - x) / w * self.n)
                iy = self.n - 1 - int((pos[1] - y) / h * self.n)
                return ix, iy
        return None


# ---------------------------------------------------------------------------
#  Отрисовка
# ---------------------------------------------------------------------------

def outcome_surface(pg, view, name):
    n = view.n
    outcome = view.data.get(view.method, view.u_max, name, "outcome")
    t_fall = view.data.get(view.method, view.u_max, name, "tfall")
    horizon = float(view.data.files[view.method]["horizon"])
    surf = pg.Surface((n, n))
    for iy in range(n):
        for ix in range(n):
            m = iy * n + ix
            base = OUTCOME_COLOR.get(int(outcome[m]), UNKNOWN)
            color = base if np.isnan(t_fall[m]) else shade(base, 0.45 + 0.55 * t_fall[m] / horizon)
            surf.set_at((ix, n - 1 - iy), color)
    return surf


def diff_surface(pg, view):
    n = view.n
    ha = view.data.get(view.method, view.u_max, view.a, "outcome") == HELD
    hb = view.data.get(view.method, view.u_max, view.b, "outcome") == HELD
    surf = pg.Surface((n, n))
    for iy in range(n):
        for ix in range(n):
            m = iy * n + ix
            color = BOTH if ha[m] and hb[m] else ONLY_A if ha[m] else ONLY_B if hb[m] else NONE
            surf.set_at((ix, n - 1 - iy), color)
    return surf


def draw_limits(screen, pg, view, rect):
    x, y, w, h = rect
    clip = screen.get_clip()
    screen.set_clip(rect)
    for values in (view.limit_ceiling, view.limit_floor):
        run = []
        for th, dth in zip(view.limit_theta, values):
            if abs(dth) < 1e-12:                       # уровня нет -- линию рвём
                if len(run) > 1:
                    pg.draw.lines(screen, LIMIT, False, run, 2)
                run = []
                continue
            run.append(view.to_px(rect, th, dth))
        if len(run) > 1:
            pg.draw.lines(screen, LIMIT, False, run, 2)
    for sign in (+1, -1):
        px, py = view.to_px(rect, sign * view.theta_eq, 0.0)
        pg.draw.circle(screen, LIMIT, (int(px), int(py)), 4, 1)
    screen.set_clip(clip)


def draw_phase(screen, pg, view, rect, slot, color):
    if slot not in view.traj:
        return
    _, X, _ = view.traj[slot]
    pts = [view.to_px(rect, th, dth) for th, dth in zip(X[:, 0], X[:, 2])
           if np.isfinite(th) and np.isfinite(dth)]
    x, y, w, h = rect
    pts = [p for p in pts if x - 2000 < p[0] < x + w + 2000 and y - 2000 < p[1] < y + h + 2000]
    if len(pts) > 1:
        clip = screen.get_clip()
        screen.set_clip(rect)
        pg.draw.lines(screen, color, False, pts, 2)
        pg.draw.circle(screen, color, (int(pts[0][0]), int(pts[0][1])), 4)
        screen.set_clip(clip)


def draw_map(screen, pg, font, view, key, title, surf, curves):
    x, y, w, h = MAPS[key]
    pg.draw.rect(screen, PANEL, (x - 4, y - 4, w + 8, h + 8), border_radius=6)
    screen.blit(font.render(title, True, TEXT), (x, y - 44))
    screen.blit(pg.transform.scale(surf, (w, h)), (x, y))
    pg.draw.line(screen, GRID_LINE, (x + w / 2, y), (x + w / 2, y + h))
    pg.draw.line(screen, GRID_LINE, (x, y + h / 2), (x + w, y + h / 2))
    draw_limits(screen, pg, view, MAPS[key])
    for slot, color in curves:
        draw_phase(screen, pg, view, MAPS[key], slot, color)
    if view.selected is not None:
        ix, iy = view.selected
        cw, ch = w / view.n, h / view.n
        pg.draw.rect(screen, ACCENT, (x + ix * cw, y + (view.n - 1 - iy) * ch, cw, ch), 2)
    labels = [(f"{view.thetas[0]:+.2f}", x, y + h + 6),
              (f"{view.thetas[-1]:+.2f}", x + w - 40, y + h + 6)]
    if key == "A":       # оси у всех карт общие; слева от B и diff подпись легла бы на соседа
        labels += [(f"{view.dthetas[-1]:+.1f}", x - 44, y),
                   (f"{view.dthetas[0]:+.1f}", x - 44, y + h - 14)]
    for text, tx, ty in labels:
        screen.blit(font.render(text, True, DIM), (tx, ty))


def stats(view, name):
    held = view.data.get(view.method, view.u_max, name, "outcome") == HELD
    rec = view.rec
    return f"held {held.sum()}  lost {(rec & ~held).sum()} of {rec.sum()} recoverable"


def draw_plot(screen, pg, font, view, rect, title, index, limit=None):
    x, y, w, h = rect
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    screen.blit(font.render(title, True, DIM), (x + 8, y - 20))
    series = []
    for slot, color in (("B", GHOST), ("A", CURVE)):
        if slot in view.traj:
            t, X, U = view.traj[slot]
            ys = U if index is None else X[:, index]
            ts = t[:len(ys)]
            ok = np.isfinite(ys)
            series.append((ts[ok], ys[ok], color))
    if not series:
        screen.blit(font.render("click a cell", True, DIM), (x + w / 2 - 50, y + h / 2 - 8))
        return
    t_max = max(s[0][-1] for s in series if s[0].size)
    lo = min(s[1].min() for s in series if s[1].size)
    hi = max(s[1].max() for s in series if s[1].size)
    if limit is not None:
        lo, hi = min(lo, -limit), max(hi, limit)
    if hi - lo < 1e-9:
        lo, hi = lo - 1, hi + 1
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    def px(tv, yv):
        return x + 8 + tv / t_max * (w - 16), y + h - 8 - (yv - lo) / (hi - lo) * (h - 16)

    if lo < 0 < hi:
        pg.draw.line(screen, GRID_LINE, px(0, 0), px(t_max, 0))
    if limit is not None:
        for sgn in (+1, -1):
            a, b = px(0, sgn * limit), px(t_max, sgn * limit)
            for k in range(int(a[0]), int(b[0]), 10):
                pg.draw.line(screen, LIMIT, (k, a[1]), (min(k + 5, b[0]), a[1]))
    for ts, ys, color in series:
        pts = [px(a, b) for a, b in zip(ts, ys)]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 2)
    screen.blit(font.render(f"{hi:+.2f}", True, DIM), (x + w - 60, y + 4))
    screen.blit(font.render(f"{lo:+.2f}", True, DIM), (x + w - 60, y + h - 20))


class Buttons:
    """Прямоугольники кнопок пересобираются на каждом кадре: набор пресетов
    зависит от метода, и хранить устаревшие прямоугольники значило бы ловить
    клики по кнопкам, которых уже нет."""

    def __init__(self):
        self.items = []

    def add(self, screen, pg, font, rect, label, on, action, enabled=True):
        pg.draw.rect(screen, BTN_ON if on else BTN, rect, border_radius=4)
        color = TEXT if enabled else DIM
        screen.blit(font.render(label, True, color), (rect[0] + 6, rect[1] + 3))
        if not enabled:
            pg.draw.line(screen, DIM, (rect[0] + 3, rect[1] + rect[3] // 2),
                         (rect[0] + rect[2] - 3, rect[1] + rect[3] // 2))
        if enabled:
            self.items.append((pg.Rect(rect), action))

    def click(self, pos):
        for rect, action in self.items:
            if rect.collidepoint(pos):
                action()
                return True
        return False


def draw(screen, pg, font, view, buttons):
    screen.fill(BG)
    buttons.items = []

    # --- ряд 1: метод и предел ---
    x = 48
    screen.blit(font.render("method", True, DIM), (x, 16))
    x += 64
    for method, label in (("lqr", "LQR 41x41"), ("mpc", "MPC 11x11")):
        def pick(m=method):
            view.method = m
            view.fix_state()
        buttons.add(screen, pg, font, (x, 12, 104, 24), label, view.method == method, pick,
                    method in view.data.files)
        x += 112
    x += 24
    screen.blit(font.render("u_max", True, DIM), (x, 16))
    x += 56
    for u in view.data.u_maxes(view.method):
        def pick_u(u=u):
            view.u_max = u
            view.fix_state()
        buttons.add(screen, pg, font, (x, 12, 70, 24), f"{u:g} N*m", view.u_max == u, pick_u)
        x += 78

    # --- ряды 2-3: цены A и B ---
    all_presets = list(cs.PRESETS)
    for row, (slot, color) in enumerate((("A", CURVE), ("B", GHOST))):
        y = 48 + row * 30
        screen.blit(font.render(slot, True, color), (48, y + 4))
        x = 72
        for name in all_presets:
            ok = view.data.has(view.method, view.u_max, name)
            current = view.a if slot == "A" else view.b

            def pick_p(name=name, slot=slot):
                if slot == "A":
                    view.a = name
                else:
                    view.b = name
                sel = view.selected
                view.traj = {}
                if sel is not None:
                    view.select(*sel)
            width = 12 + 9 * len(name)
            buttons.add(screen, pg, font, (x, y, width, 24), name, current == name, pick_p, ok)
            x += width + 6

    # --- карты ---
    # Картинки клеток кэшируются по тому, от чего зависят: перерисовывать 41x41
    # клетку за клеткой 30 раз в секунду незачем, данные не меняются.
    def cached(key, make):
        if key not in SURF_CACHE:
            SURF_CACHE[key] = make()
        return SURF_CACHE[key]

    curves_ab = [("B", GHOST), ("A", CURVE)]
    draw_map(screen, pg, font, view, "A", f"A: {view.a}   {stats(view, view.a)}",
             cached((view.method, view.u_max, view.a),
                    lambda: outcome_surface(pg, view, view.a)), [("A", CURVE)])
    draw_map(screen, pg, font, view, "B", f"B: {view.b}   {stats(view, view.b)}",
             cached((view.method, view.u_max, view.b),
                    lambda: outcome_surface(pg, view, view.b)), [("B", GHOST)])
    ha = view.data.get(view.method, view.u_max, view.a, "outcome") == HELD
    hb = view.data.get(view.method, view.u_max, view.b, "outcome") == HELD
    draw_map(screen, pg, font, view, "diff",
             f"A vs B: only A {(ha & ~hb).sum()}, only B {(hb & ~ha).sum()}",
             cached((view.method, view.u_max, view.a, view.b, "diff"),
                    lambda: diff_surface(pg, view)), curves_ab)
    legend = [("held", OUTCOME_COLOR[HELD]), ("fell fwd", OUTCOME_COLOR[1]),
              ("fell back", OUTCOME_COLOR[2]), ("recoverable", LIMIT),
              ("only A", ONLY_A), ("only B", ONLY_B), ("both", BOTH)]
    lx = 48
    for label, color in legend:
        pg.draw.rect(screen, color, (lx, 598, 12, 12))
        screen.blit(font.render(label, True, DIM), (lx + 16, 595))
        lx += 26 + 9 * len(label) + 16

    # --- выбранная клетка ---
    if view.selected is not None:
        ix, iy = view.selected
        m = iy * view.n + ix
        parts = [f"theta0 = {view.thetas[ix]:+.3f}, dtheta0 = {view.dthetas[iy]:+.2f}, "
                 f"recoverable: {'yes' if view.rec[m] else 'no'}"]
        for slot, name in (("A", view.a), ("B", view.b)):
            code = int(view.data.get(view.method, view.u_max, name, "outcome")[m])
            tf = view.data.get(view.method, view.u_max, name, "tfall")[m]
            word = "held" if code == HELD else f"fell at {tf:.2f} s"
            if slot in view.traj:
                _, X, U = view.traj[slot]
                sat = np.mean(np.abs(U) >= view.u_max - 1e-6)
                word += f", saturated {sat:.0%}, phi(T) = {X[-1, 1]:+.1f}"
            parts.append(f"{slot}: {word}")
        screen.blit(font.render("   ".join(parts), True, TEXT), (48, 624))

    draw_plot(screen, pg, font, view, PLOTS[0], "theta(t), rad   A white, B teal", 0)
    draw_plot(screen, pg, font, view, PLOTS[1], "u(t), N*m   dashed: +-u_max", None, view.u_max)
    draw_plot(screen, pg, font, view, PLOTS[2], "phi(t), rad   wheel", 1)


def main():
    ap = argparse.ArgumentParser(description="карты перебора цен")
    ap.add_argument("--a", default="base")
    ap.add_argument("--b", default="R=100")
    ap.add_argument("--u-max", type=float, default=10.0)
    ap.add_argument("--method", choices=("lqr", "mpc"), default="lqr")
    ap.add_argument("--lqr", default=str(cs.DEFAULT_LQR))
    ap.add_argument("--mpc-file", default=str(cs.DEFAULT_MPC))
    ap.add_argument("--select", type=int, nargs=2, default=None, metavar=("IX", "IY"),
                    help="сразу выбрать клетку (для скриншотов)")
    ap.add_argument("--screenshot", default=None, help="снять кадр в файл и выйти")
    args = ap.parse_args()

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg

    view = View(args)
    if args.select is not None:
        view.select(*args.select)
    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("cost sweep maps")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 14)
    buttons = Buttons()

    if args.screenshot:
        draw(screen, pg, font, view, buttons)
        pg.image.save(screen, args.screenshot)
        return

    clock = pg.time.Clock()
    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT or (ev.type == pg.KEYDOWN and ev.key == pg.K_ESCAPE):
                return
            if ev.type == pg.KEYDOWN and ev.key == pg.K_TAB:
                view.a, view.b = view.b, view.a
                if view.selected is not None:
                    view.select(*view.selected)
            if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                if not buttons.click(ev.pos):
                    cell = view.cell_at(ev.pos)
                    if cell is not None:
                        view.select(*cell)
        draw(screen, pg, font, view, buttons)
        pg.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
