"""Окно для подбора цены ЛКР: две цены A и B рядом, их разница и предел момента.

    uv run python examples/lqr_cost_explorer.py
    uv run python examples/lqr_cost_explorer.py --dtheta-max 10 --grid 81   # карта по theta: от -pi/2 до pi/2
    uv run python examples/lqr_cost_explorer.py --theta-fall 1.571     # «упал» = лёг на землю

Что на экране
-------------
* Сверху слайдер u_max, как в окне исследователя: 0.25…10 Н·м шагом 0.25,
  клавиши [ и ], кнопка `no limit` (клавиша L). Карты пересчитываются на
  отпускании.
* Цена -- пять чисел: веса Q = diag(q_theta, q_phi, q_dtheta, q_dphi) и R,
  вводятся с клавиатуры. Клик по полю (или Tab, когда ничего не редактируется,
  переключает A / B) -- поле открыто на ввод, набранное заменяет старое число.
  Принимаются 100, 0.01, 1e-3. Enter -- применить и пересчитать карту,
  Tab / Shift+Tab -- применить и перейти к соседнему полю, Esc -- отменить,
  клик мимо поля -- применить. Кнопки A / B выбирают, какую цену правят поля;
  кнопки пресетов ставят в неё найденные ранее наборы (`examples/cost_sweep.py`).
* Три карты: цена A, цена B, разница. Клик по клетке -- траектории обеих цен
  на картах и графики theta(t), u(t), phi(t) снизу.

Нулевые веса колеса
-------------------
В полях q_phi и q_dphi разрешён 0 (в остальных -- только > 0). Уравнение Риккати с
нулевым весом phi не решается: phi -- чистый интегратор, которого цена не
видит, и стабилизирующего решения нет. Но phi не входит в правую часть вовсе
(Key_Formulas §1.4), поэтому подсистема без phi замкнута сама на себя, и ЛКР
для неё законен -- это обобщение `lqr_tilt` (A17):
    q_phi = 0              -> ЛКР по (theta, dtheta, dphi), K_phi = 0;
    q_phi = 0 и q_dphi = 0 -> ЛКР по (theta, dtheta), колесо не видим (= tilt).
Вес q_dphi = 0 при q_phi > 0 законен и так: phi наблюдаема через свой вес.

Исходы клетки
-------------
* упал вперёд / назад -- |theta| пересёк --theta-fall (по умолчанию pi, то есть
  корпус провернулся через низ). Земли в модели нет: корпус, лёгший за pi/2,
  может вернуться. Чтобы считать упавшим любое касание земли, --theta-fall 1.571.
* удержан -- не пересёк и к концу горизонта |theta| < 0.1;
* не успокоился (серый) -- не пересёк, но и не вернулся за горизонт.

Зелёная линия -- граница восстановимости `recoverable_bounds` при текущем
u_max (свойство системы и мотора, не регулятора). Проверена против реле на
карте до |theta| <= 0.5; на широкой карте это формула модели, а не замер.
"""

import argparse
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import LinearFeedbackController, RK4Integrator, rollout, rollout_many
from wpend.lqr import lqr
from wpend.models import WheeledPendulum
from wpend.viz.explorer import (
    ACCENT, BG, BTN, BTN_ON, CURVE, DIM, GHOST, GRID_LINE, LIMIT, OUTCOME_COLOR,
    PANEL, TEXT,
)
from wpend.viz.grid import FELL_BACKWARD, FELL_FORWARD, HELD, classify

NOT_SETTLED = 3
COLOR = dict(OUTCOME_COLOR)
COLOR[NOT_SETTLED] = (96, 100, 110)
ONLY_A, ONLY_B, BOTH, NONE = ACCENT, GHOST, (40, 80, 124), (40, 43, 50)

DT, STRIDE, SETTLE = 1e-3, 20, 0.1
U_MIN, U_MAX, U_STEP = 0.25, 10.0, 0.25          # как в окне исследователя

W, H = 1280, 940
U_SLIDER = (100, 16, 300, 14)
MAPS = {"A": (56, 186, 380, 390), "B": (472, 186, 380, 390), "diff": (888, 186, 360, 390)}
PLOTS = [(56, 676, 380, 220), (472, 676, 380, 220), (888, 676, 360, 220)]

#: Пять весов: ключ, подпись, можно ли ноль.
WEIGHTS = [
    ("q_theta", "q_theta", False),
    ("q_phi", "q_phi", True),
    ("q_dtheta", "q_dtheta", False),
    ("q_dphi", "q_dphi", True),
    ("R", "R", False),
]
W_X0, W_Y, W_GAP, W_LABEL, W_BOX = 24, 88, 200, 74, 104
FIELD_CHARS = set("0123456789.eE+-")

#: Пресеты -- наборы из `examples/cost_sweep.py` (16.09).
PRESETS = {
    "base":      dict(q_theta=100, q_phi=1, q_dtheta=10, q_dphi=1, R=1),
    "phi=1e-2":  dict(q_theta=100, q_phi=1e-2, q_dtheta=10, q_dphi=1, R=1),
    "phi=1e-3":  dict(q_theta=100, q_phi=1e-3, q_dtheta=10, q_dphi=1, R=1),
    "dphi=1e-2": dict(q_theta=100, q_phi=1, q_dtheta=10, q_dphi=1e-2, R=1),
    "R=0.1":     dict(q_theta=100, q_phi=1, q_dtheta=10, q_dphi=1, R=0.1),
    "R=10":      dict(q_theta=100, q_phi=1, q_dtheta=10, q_dphi=1, R=10),
    "R=100":     dict(q_theta=100, q_phi=1, q_dtheta=10, q_dphi=1, R=100),
    "theta x10": dict(q_theta=1000, q_phi=1, q_dtheta=100, q_dphi=1, R=1),
    "tilt":      dict(q_theta=100, q_phi=0, q_dtheta=10, q_dphi=0, R=1),
}


def shade(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


# ---------------------------------------------------------------------------
#  Синтез и расчёт карты
# ---------------------------------------------------------------------------

def design(model, cost):
    """K (1, 4) и полюса замкнутой ЛИНЕЙНОЙ системы для пяти весов.

    Координаты с нулевым весом, которые не входят в правую часть, выбрасываются
    из задачи (см. докстринг модуля), и K получает в их столбцах нули.
    """
    A, B = model.linearize_upright()
    keep = [0, 2]                                   # theta, dtheta -- всегда
    if cost["q_phi"] > 0:
        keep = [0, 1, 2, 3]
    elif cost["q_dphi"] > 0:
        keep = [0, 2, 3]
    q = np.array([cost["q_theta"], cost["q_phi"], cost["q_dtheta"], cost["q_dphi"]])
    idx = np.ix_(keep, keep)
    K_sub, _ = lqr(A[idx], B[keep, :], np.diag(q[keep]), np.array([[cost["R"]]]))
    K = np.zeros((1, 4))
    K[0, keep] = K_sub[0]
    poles = np.linalg.eigvals(A[idx] - B[keep, :] @ K_sub)
    return K, poles


def compute_map(model, K, u_max, X0, horizon, theta_fall):
    """Исход каждой клетки: classify (как окно) + проверка «успокоился»."""
    world = WheeledPendulum(u_max=u_max)
    batch = rollout_many(world, LinearFeedbackController(K), RK4Integrator(), X0, DT,
                         int(round(horizon / DT)), STRIDE)
    outcome, t_fall = classify(batch, 0, theta_fall)
    settled = np.abs(batch.x[:, -1, 0]) < SETTLE
    outcome = np.where((outcome == HELD) & ~settled, NOT_SETTLED, outcome)
    return outcome, t_fall


# ---------------------------------------------------------------------------
#  Состояние окна
# ---------------------------------------------------------------------------

class App:
    def __init__(self, args):
        self.args = args
        self.model = WheeledPendulum()
        n = args.grid
        self.thetas = np.linspace(-args.theta_max, args.theta_max, n)
        self.dthetas = np.linspace(-args.dtheta_max, args.dtheta_max, n)
        TH, DTH = np.meshgrid(self.thetas, self.dthetas, indexing="xy")
        self.X0 = np.zeros((n * n, 4))
        self.X0[:, 0], self.X0[:, 2] = TH.ravel(), DTH.ravel()
        self.n = n

        self.u_max = args.u_max if np.isfinite(args.u_max) else None
        self.u_finite = args.u_max if np.isfinite(args.u_max) else 3.0
        self.u_preview = None           # значение под ручкой во время перетаскивания
        self.cost = {"A": dict(PRESETS[args.a]), "B": dict(PRESETS[args.b])}
        self.editing = "A"
        self.drag = None                # "u_max" | None
        self.field = None               # индекс редактируемого веса | None
        self.buffer = ""                # набранный текст
        self.fresh = False              # первый символ заменяет старое число
        self.field_error = None
        self.stale = {"A": True, "B": True}
        self.K = {}
        self.poles = {}
        self.error = {"A": None, "B": None}
        self.result = {}                # слот -> (outcome, t_fall)
        self.cache = {}                 # (u_max, веса) -> (outcome, t_fall)
        self.selected = None
        self.traj = {}
        self.busy = None
        for slot in "AB":
            self.synthesize(slot)

    # --- синтез и пересчёт ------------------------------------------------------

    def synthesize(self, slot):
        """K дёшев (миллисекунды), поэтому считается сразу, даже при перетаскивании."""
        try:
            self.K[slot], self.poles[slot] = design(self.model, self.cost[slot])
            self.error[slot] = None
        except Exception as exc:        # scipy: нет стабилизирующего решения и т. п.
            self.error[slot] = str(exc).split("\n")[0][:60]

    def recompute(self, slot):
        if self.error[slot] is not None:
            return
        key = (self.u_max, tuple(sorted(self.cost[slot].items())))
        if key not in self.cache:
            self.cache[key] = compute_map(self.model, self.K[slot], self.u_max, self.X0,
                                          self.args.horizon, self.args.theta_fall)
        self.result[slot] = self.cache[key]
        self.stale[slot] = False
        if self.selected is not None:
            self.select(*self.selected)

    def world(self):
        return WheeledPendulum(u_max=self.u_max)

    def select(self, ix, iy):
        self.selected = (ix, iy)
        x0 = np.array([self.thetas[ix], 0.0, self.dthetas[iy], 0.0])
        for slot in "AB":
            if self.error[slot] is None:
                tr = rollout(self.world(), LinearFeedbackController(self.K[slot]),
                             RK4Integrator(), x0, DT, int(round(self.args.horizon / DT)))
                self.traj[slot] = (tr.t[::10], tr.x[::10], tr.u[::10, 0])

    def apply_preset(self, name):
        self.cost[self.editing] = dict(PRESETS[name])
        self.synthesize(self.editing)
        self.stale[self.editing] = True

    # --- ввод весов с клавиатуры -----------------------------------------------

    def open_field(self, i):
        self.field = i
        self.buffer = f"{self.cost[self.editing][WEIGHTS[i][0]]:g}"
        self.fresh = True
        self.field_error = None

    def type_text(self, text):
        chars = "".join(c for c in text if c in FIELD_CHARS)
        if not chars:
            return
        self.buffer = chars if self.fresh else self.buffer + chars
        self.fresh = False
        self.field_error = None

    def backspace(self):
        self.buffer = "" if self.fresh else self.buffer[:-1]
        self.fresh = False
        self.field_error = None

    def commit_field(self):
        """Применить набранное. False -- ввод неверен, поле остаётся открытым."""
        key, _, zero_ok = WEIGHTS[self.field]
        value, err = parse_weight(self.buffer, zero_ok)
        if err is not None:
            self.field_error = f"{key}: {err}"
            return False
        if value != self.cost[self.editing][key]:
            self.cost[self.editing][key] = value
            self.synthesize(self.editing)
            self.stale[self.editing] = True
        self.field = None
        self.field_error = None
        return True

    def cancel_field(self):
        self.field = None
        self.field_error = None

    def set_u_max(self, u):
        self.u_max = u
        if u is not None:
            self.u_finite = u
        self.stale = {"A": True, "B": True}

    # --- геометрия -------------------------------------------------------------

    def to_px(self, rect, theta, dtheta):
        """Как `_to_map_px` окна: узел -- центр клетки."""
        x, y, w, h = rect
        edge = self.n / max(self.n - 1, 1)
        ht, hd = self.thetas[-1] * edge, self.dthetas[-1] * edge
        return x + (theta + ht) / (2 * ht) * w, y + (1 - (dtheta + hd) / (2 * hd)) * h

    def cell_at(self, pos):
        for rect in MAPS.values():
            x, y, w, h = rect
            if x <= pos[0] < x + w and y <= pos[1] < y + h:
                return int((pos[0] - x) / w * self.n), self.n - 1 - int((pos[1] - y) / h * self.n)
        return None


# ---------------------------------------------------------------------------
#  Слайдеры: пиксель <-> значение
# ---------------------------------------------------------------------------

def u_to_px(u):
    x, _, w, _ = U_SLIDER
    return x + (np.clip(u, U_MIN, U_MAX) - U_MIN) / (U_MAX - U_MIN) * w


def u_from_px(px):
    x, _, w, _ = U_SLIDER
    f = float(np.clip((px - x) / w, 0, 1))
    return U_MIN + round(f * (U_MAX - U_MIN) / U_STEP) * U_STEP


def weight_box(i):
    """Прямоугольник поля i-го веса. Одна функция на отрисовку и на клик."""
    return W_X0 + i * W_GAP + W_LABEL, W_Y - 4, W_BOX, 22


def parse_weight(text, zero_ok):
    """Строка из поля -> (число, None) или (None, что не так).

    Проверка здесь, а не в синтезе: отрицательный вес или nan дали бы
    невнятную ошибку scipy, а человеку нужно сказать, что не так с вводом.
    """
    try:
        v = float(text)
    except ValueError:
        return None, f"'{text}' is not a number"
    if not np.isfinite(v):
        return None, "must be finite"
    if v < 0 or (v == 0 and not zero_ok):
        return None, ">= 0 required" if zero_ok else "> 0 required"
    return v, None


# ---------------------------------------------------------------------------
#  Отрисовка
# ---------------------------------------------------------------------------

class Buttons:
    """Прямоугольники кнопок пересобираются каждый кадр: одна раскладка и на
    рисование, и на клик."""

    def __init__(self):
        self.items = []

    def add(self, screen, pg, font, rect, label, on, action):
        pg.draw.rect(screen, BTN_ON if on else BTN, rect, border_radius=4)
        screen.blit(font.render(label, True, TEXT), (rect[0] + 7, rect[1] + 3))
        self.items.append((pg.Rect(rect), action))

    def click(self, pos):
        for rect, action in self.items:
            if rect.collidepoint(pos):
                action()
                return True
        return False


def map_surface(pg, app, slot):
    n = app.n
    img = np.zeros((n, n, 3), dtype=np.uint8)
    if slot in app.result:
        outcome, t_fall = app.result[slot]
        for code, base in COLOR.items():
            mask = outcome == code
            frac = np.where(np.isnan(t_fall), 1.0, t_fall / app.args.horizon)
            factor = 0.45 + 0.55 * frac
            for c in range(3):
                img[..., c] = np.where(mask.reshape(n, n),
                                       np.clip(base[c] * factor, 0, 255).reshape(n, n),
                                       img[..., c])
    return pg.surfarray.make_surface(np.flipud(img).transpose(1, 0, 2))


def diff_surface(pg, app):
    n = app.n
    img = np.zeros((n, n, 3), dtype=np.uint8)
    if "A" in app.result and "B" in app.result:
        ha = (app.result["A"][0] == HELD).reshape(n, n)
        hb = (app.result["B"][0] == HELD).reshape(n, n)
        for mask, color in ((ha & hb, BOTH), (ha & ~hb, ONLY_A), (~ha & hb, ONLY_B),
                            (~ha & ~hb, NONE)):
            img[mask] = color
    return pg.surfarray.make_surface(np.flipud(img).transpose(1, 0, 2))


def draw_limits(screen, pg, app, rect):
    if app.u_max is None:
        return
    world = app.world()
    th = np.linspace(-app.args.theta_max * 1.05, app.args.theta_max * 1.05, 1601)
    floor, ceiling = world.recoverable_bounds(app.u_max, th)
    clip = screen.get_clip()
    screen.set_clip(rect)
    for values in (ceiling, floor):
        run = []
        for t, d in zip(th, values):
            if abs(d) < 1e-12:
                if len(run) > 1:
                    pg.draw.lines(screen, LIMIT, False, run, 2)
                run = []
                continue
            run.append(app.to_px(rect, t, d))
        if len(run) > 1:
            pg.draw.lines(screen, LIMIT, False, run, 2)
    eq = world.saddle_angle(app.u_max)
    for s in (+1, -1):
        px, py = app.to_px(rect, s * eq, 0.0)
        pg.draw.circle(screen, LIMIT, (int(px), int(py)), 4, 1)
    screen.set_clip(clip)


def draw_map(screen, pg, font, app, key, title, surf, curves):
    x, y, w, h = MAPS[key]
    pg.draw.rect(screen, PANEL, (x - 4, y - 4, w + 8, h + 8), border_radius=6)
    screen.blit(font.render(title, True, TEXT), (x, y - 24))
    screen.blit(pg.transform.scale(surf, (w, h)), (x, y))
    stale = key == "diff" and (app.stale["A"] or app.stale["B"]) or app.stale.get(key, False)
    if stale:
        veil = pg.Surface((w, h), pg.SRCALPHA)
        veil.fill(BG + (170,))
        screen.blit(veil, (x, y))
    # Оси через ноль и отметки pi/2: на широкой карте глаз ищет их первыми.
    zx, zy = app.to_px(MAPS[key], 0.0, 0.0)
    pg.draw.line(screen, GRID_LINE, (zx, y), (zx, y + h))
    pg.draw.line(screen, GRID_LINE, (x, zy), (x + w, zy))
    for s in (+1, -1):
        px, _ = app.to_px(MAPS[key], s * np.pi / 2, 0.0)
        if x <= px <= x + w:
            for top in range(int(y), int(y + h), 11):
                pg.draw.line(screen, DIM, (px, top), (px, min(top + 6, y + h)))
    draw_limits(screen, pg, app, MAPS[key])
    clip = screen.get_clip()
    screen.set_clip(MAPS[key])
    for slot, color in curves:
        if slot in app.traj:
            _, X, _ = app.traj[slot]
            pts = [app.to_px(MAPS[key], a, b) for a, b in zip(X[:, 0], X[:, 2])
                   if np.isfinite(a) and np.isfinite(b)]
            pts = [p for p in pts if abs(p[0]) < 1e5 and abs(p[1]) < 1e5]
            if len(pts) > 1:
                pg.draw.lines(screen, color, False, pts, 2)
                pg.draw.circle(screen, color, (int(pts[0][0]), int(pts[0][1])), 4)
    screen.set_clip(clip)
    if app.selected is not None:
        ix, iy = app.selected
        cw, ch = w / app.n, h / app.n
        pg.draw.rect(screen, ACCENT, (x + ix * cw, y + (app.n - 1 - iy) * ch,
                                      max(cw, 3), max(ch, 3)), 2)
    labels = [(f"{app.thetas[0]:+.2f}", x, y + h + 6),
              (f"{app.thetas[-1]:+.2f}", x + w - 40, y + h + 6)]
    if key == "A":
        labels += [(f"{app.dthetas[-1]:+.1f}", x - 48, y),
                   (f"{app.dthetas[0]:+.1f}", x - 48, y + h - 14)]
    for text, tx, ty in labels:
        screen.blit(font.render(text, True, DIM), (tx, ty))


def stats(app, slot):
    if slot not in app.result:
        return app.error[slot] or "computing..."
    held = app.result[slot][0] == HELD
    if app.u_max is None:
        return f"held {held.sum()} of {held.size}"
    rec = app.world().is_recoverable(app.u_max, app.X0[:, 0], app.X0[:, 2])
    return f"held {held.sum()}, lost {(rec & ~held).sum()} of {rec.sum()} rec."


def draw_plot(screen, pg, font, app, rect, title, index, limit=None):
    x, y, w, h = rect
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    screen.blit(font.render(title, True, DIM), (x + 8, y - 20))
    series = []
    for slot, color in (("B", GHOST), ("A", CURVE)):
        if slot in app.traj:
            t, X, U = app.traj[slot]
            ys = U if index is None else X[:, index]
            ok = np.isfinite(ys)
            series.append((t[:len(ys)][ok], ys[ok], color))
    if not series:
        screen.blit(font.render("click a cell", True, DIM), (x + w / 2 - 50, y + h / 2 - 8))
        return
    t_max = max(s[0][-1] for s in series)
    lo = min(s[1].min() for s in series)
    hi = max(s[1].max() for s in series)
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
        for s in (+1, -1):
            a, b = px(0, s * limit), px(t_max, s * limit)
            for k in range(int(a[0]), int(b[0]), 10):
                pg.draw.line(screen, LIMIT, (k, a[1]), (min(k + 5, b[0]), a[1]))
    for ts, ys, color in series:
        pts = [px(a, b) for a, b in zip(ts, ys)]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 2)
    screen.blit(font.render(f"{hi:+.2f}", True, DIM), (x + w - 64, y + 4))
    screen.blit(font.render(f"{lo:+.2f}", True, DIM), (x + w - 64, y + h - 20))


def draw(screen, pg, font, app, buttons):
    screen.fill(BG)
    buttons.items = []

    # --- u_max ---
    x, y, w, h = U_SLIDER
    screen.blit(font.render("u_max", True, DIM), (24, y - 1))
    shown = app.u_preview if app.u_preview is not None else app.u_finite
    off = app.u_max is None and app.u_preview is None
    pg.draw.rect(screen, BTN, (x, y, w, h), border_radius=4)
    pg.draw.rect(screen, shade(BTN_ON, 0.45 if off else 1.0),
                 (x, y, max(u_to_px(shown) - x, 1), h), border_radius=4)
    pg.draw.circle(screen, DIM if off else TEXT, (int(u_to_px(shown)), int(y + h / 2)), 7)
    screen.blit(font.render("  off" if off else f"{shown:5.2f} N*m", True, TEXT),
                (x + w + 14, y - 1))
    buttons.add(screen, pg, font, (x + w + 110, y - 4, 84, 22), "no limit",
                app.u_max is None, lambda: toggle_limit(app))
    note = ("release to recompute" if app.u_preview is not None else
            app.busy or f"grid {app.n}x{app.n}, horizon {app.args.horizon:g} s, "
                        f"fall at |theta| >= {app.args.theta_fall:.3g}")
    screen.blit(font.render(note, True, ACCENT if app.busy or app.u_preview else DIM),
                (x + w + 210, y - 1))

    # --- A / B и пресеты ---
    y2 = 52
    screen.blit(font.render("edit", True, DIM), (24, y2 + 3))
    for i, slot in enumerate("AB"):
        buttons.add(screen, pg, font, (64 + i * 40, y2, 32, 22), slot,
                    app.editing == slot, lambda s=slot: setattr(app, "editing", s))
    px_ = 160
    screen.blit(font.render("presets", True, DIM), (px_, y2 + 3))
    px_ += 70
    for name in PRESETS:
        bw = 16 + 8 * len(name)
        buttons.add(screen, pg, font, (px_, y2, bw, 22), name,
                    app.cost[app.editing] == PRESETS[name],
                    lambda nm=name: apply_preset(app, nm))
        px_ += bw + 6

    # --- пять весов ---
    color = CURVE if app.editing == "A" else GHOST
    for i, (key, label, _) in enumerate(WEIGHTS):
        bx, by, bw, bh = weight_box(i)
        screen.blit(font.render(label, True, color), (W_X0 + i * W_GAP, by + 3))
        active = app.field == i
        pg.draw.rect(screen, PANEL if active else BTN, (bx, by, bw, bh), border_radius=4)
        border = ((205, 88, 62) if active and app.field_error else
                  ACCENT if active else GRID_LINE)
        pg.draw.rect(screen, border, (bx, by, bw, bh), 2 if active else 1, border_radius=4)
        text = app.buffer if active else f"{app.cost[app.editing][key]:g}"
        screen.blit(font.render(text, True, TEXT), (bx + 6, by + 3))
        if active and (pg.time.get_ticks() // 500) % 2 == 0:      # мигающий курсор
            cx = bx + 6 + font.size(text)[0] + 1
            pg.draw.line(screen, TEXT, (cx, by + 4), (cx, by + bh - 5))
    if app.field is not None:
        hint = app.field_error or "Enter apply   Tab next   Esc cancel"
        screen.blit(font.render(hint, True, (205, 88, 62) if app.field_error else DIM),
                    (W_X0 + 5 * W_GAP, W_Y - 1))

    # --- K и полюса ---
    for row, slot in enumerate("AB"):
        c = CURVE if slot == "A" else GHOST
        if app.error[slot]:
            line = f"{slot}: design failed: {app.error[slot]}"
        else:
            K = app.K[slot][0]
            slow = min(abs(p.real) for p in app.poles[slot])
            line = (f"{slot}: K = [{K[0]:.2f}, {K[1]:.3g}, {K[2]:.2f}, {K[3]:.3g}]   "
                    f"slowest pole Re = -{slow:.2f}  (tau = {1 / slow:.1f} s)   "
                    f"weights {', '.join(f'{v:g}' for v in app.cost[slot].values())}")
        screen.blit(font.render(line, True, c), (24, 116 + row * 18))

    # --- карты ---
    draw_map(screen, pg, font, app, "A", f"A   {stats(app, 'A')}",
             map_surface(pg, app, "A"), [("A", CURVE)])
    draw_map(screen, pg, font, app, "B", f"B   {stats(app, 'B')}",
             map_surface(pg, app, "B"), [("B", GHOST)])
    if "A" in app.result and "B" in app.result:
        ha, hb = app.result["A"][0] == HELD, app.result["B"][0] == HELD
        dtitle = f"A vs B: only A {(ha & ~hb).sum()}, only B {(hb & ~ha).sum()}"
    else:
        dtitle = "A vs B"
    draw_map(screen, pg, font, app, "diff", dtitle, diff_surface(pg, app),
             [("B", GHOST), ("A", CURVE)])

    lx = 56
    for label, c in (("held", COLOR[HELD]), ("fell fwd", COLOR[FELL_FORWARD]),
                     ("fell back", COLOR[FELL_BACKWARD]), ("not settled", COLOR[NOT_SETTLED]),
                     ("recoverable", LIMIT), ("only A", ONLY_A), ("only B", ONLY_B),
                     ("both", BOTH)):
        pg.draw.rect(screen, c, (lx, 600, 12, 12))
        screen.blit(font.render(label, True, DIM), (lx + 16, 597))
        lx += 34 + 8 * len(label)

    if app.selected is not None:
        ix, iy = app.selected
        m = iy * app.n + ix
        parts = [f"theta0 = {app.thetas[ix]:+.3f}, dtheta0 = {app.dthetas[iy]:+.2f}"]
        for slot in "AB":
            if slot in app.traj and slot in app.result:
                code = int(app.result[slot][0][m])
                tf = app.result[slot][1][m]
                word = {HELD: "held", NOT_SETTLED: "not settled"}.get(code, f"fell at {tf:.2f} s")
                _, X, U = app.traj[slot]
                if app.u_max is not None:
                    word += f", sat {np.mean(np.abs(U) >= app.u_max - 1e-6):.0%}"
                word += f", phi(T) {X[-1, 1]:+.1f}"
                parts.append(f"{slot}: {word}")
        screen.blit(font.render("   ".join(parts), True, TEXT), (56, 624))

    draw_plot(screen, pg, font, app, PLOTS[0], "theta(t), rad   A white, B teal", 0)
    draw_plot(screen, pg, font, app, PLOTS[1], "u(t), N*m   dashed: +-u_max", None, app.u_max)
    draw_plot(screen, pg, font, app, PLOTS[2], "phi(t), rad   wheel", 1)


def toggle_limit(app):
    app.set_u_max(None if app.u_max is not None else app.u_finite)


def apply_preset(app, name):
    app.apply_preset(name)


def refresh(app, screen, pg, font, buttons):
    """Пересчитать устаревшие карты, показав перед этим, что идёт расчёт:
    на сетке 61x61 это секунды, и замерший без объяснений экран выглядел бы
    как зависание."""
    for slot in "AB":
        if app.stale[slot] and app.error[slot] is None:
            app.busy = f"computing map {slot}..."
            draw(screen, pg, font, app, buttons)
            pg.display.flip()
            t0 = time.perf_counter()
            app.recompute(slot)
            app.busy = None
            app.last_time = time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--theta-max", type=float, default=float(np.pi / 2),
                    help="край карты по theta; по умолчанию pi/2 -- соглашение для визуализаций")
    ap.add_argument("--dtheta-max", type=float, default=8.0)
    ap.add_argument("--grid", type=int, default=61)
    ap.add_argument("--horizon", type=float, default=5.0)
    ap.add_argument("--theta-fall", type=float, default=float(np.pi))
    ap.add_argument("--u-max", type=float, default=10.0, help="inf -- без предела")
    ap.add_argument("--a", default="base", choices=list(PRESETS))
    ap.add_argument("--b", default="R=100", choices=list(PRESETS))
    ap.add_argument("--select", type=float, nargs=2, default=None, metavar=("THETA0", "DTHETA0"))
    ap.add_argument("--screenshot", default=None, help="снять кадр и выйти")
    args = ap.parse_args()

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg

    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("LQR cost explorer")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 14)
    app = App(args)
    buttons = Buttons()
    refresh(app, screen, pg, font, buttons)
    if args.select is not None:
        ix = int(np.argmin(np.abs(app.thetas - args.select[0])))
        iy = int(np.argmin(np.abs(app.dthetas - args.select[1])))
        app.select(ix, iy)

    if args.screenshot:
        draw(screen, pg, font, app, buttons)
        pg.image.save(screen, args.screenshot)
        return

    clock = pg.time.Clock()
    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return
            if app.field is not None:
                # Пока поле открыто, клавиатура принадлежит ему: иначе «1e-3»
                # по дороге нажало бы L и выключило предел момента.
                if ev.type == pg.TEXTINPUT:
                    app.type_text(ev.text)
                    continue
                if ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        app.commit_field()
                    elif ev.key == pg.K_ESCAPE:
                        app.cancel_field()
                    elif ev.key == pg.K_BACKSPACE:
                        app.backspace()
                    elif ev.key == pg.K_TAB:
                        i = app.field
                        if app.commit_field():
                            step = -1 if ev.mod & pg.KMOD_SHIFT else 1
                            app.open_field((i + step) % len(WEIGHTS))
                    continue
                if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    hit = next((i for i in range(len(WEIGHTS))
                                if pg.Rect(weight_box(i)).collidepoint(ev.pos)), None)
                    if hit == app.field:
                        continue
                    if not app.commit_field():
                        continue            # неверный ввод: не уходим из поля
                    if hit is not None:
                        app.open_field(hit)
                        continue
                    # клик по остальному окну обрабатывается дальше как обычно
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    return
                if ev.key == pg.K_l:
                    toggle_limit(app)
                if ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    step = U_STEP if ev.key == pg.K_RIGHTBRACKET else -U_STEP
                    app.set_u_max(float(np.clip(app.u_finite + step, U_MIN, U_MAX)))
                if ev.key == pg.K_TAB:
                    app.editing = "B" if app.editing == "A" else "A"
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                x, y, w, h = U_SLIDER
                if x - 8 <= ev.pos[0] <= x + w + 8 and y - 8 <= ev.pos[1] <= y + h + 8:
                    app.drag = "u_max"
                    app.u_preview = u_from_px(ev.pos[0])
                    continue
                hit = next((i for i in range(len(WEIGHTS))
                            if pg.Rect(weight_box(i)).collidepoint(ev.pos)), None)
                if hit is not None:
                    app.open_field(hit)
                    continue
                if buttons.click(ev.pos):
                    app.cancel_field()
                    continue
                cell = app.cell_at(ev.pos)
                if cell is not None:
                    app.select(*cell)
            elif ev.type == pg.MOUSEMOTION and app.drag == "u_max":
                app.u_preview = u_from_px(ev.pos[0])
            elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1 and app.drag == "u_max":
                # Пересчёт карты -- на отпускании, как у слайдера в окне исследователя.
                app.set_u_max(app.u_preview)
                app.u_preview = None
                app.drag = None
        if app.drag is None and app.field is None:
            refresh(app, screen, pg, font, buttons)
        draw(screen, pg, font, app, buttons)
        pg.display.flip()
        clock.tick(30)


if __name__ == "__main__":
    main()
