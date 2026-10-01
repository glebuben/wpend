"""Окно поля экстремалей: та же диаграмма, но с дорожкой времени (ER-023).

    uv run --extra viz python examples/extremal_explorer.py
    uv run --extra viz python examples/extremal_explorer.py --u-max 10 --n-psi 25
    uv run --extra viz python examples/extremal_explorer.py --screenshot /tmp/f.png

Зачем окно, если есть `examples/extremal_field.py`
--------------------------------------------------
Статическая диаграмма отвечает на вопрос «как устроено поле», но не на вопрос
«что при этом делает робот».  Здесь к полю приделана дорожка времени: t = 0 --
выбранная точка, t < 0 -- как в неё попали (обратная ветвь), t > 0 -- что будет
дальше.  Курсор едет по кривой в плоскости (psi, dpsi), корпус и колесо слева
показывают тот же момент времени, а графики снизу -- тот же t.  Так видно, что
«ветвь назад» -- это не математическая условность, а реальное прошлое
траектории.

Что на экране
-------------
* Слева -- поле экстремалей при текущем пределе момента: синие кривые
  u = +u_max, оранжевые u = -u_max (обе ветви одной точки -- ОДНА кривая,
  линия уровня первого интеграла), поверх -- sigma = 0 из
  `BangBangLQRController` и граница восстановимости из `recoverable_bounds`,
  кружки -- сёдла при u = -+u_max.  Серая заливка -- восстановимое множество.
* Клик по полю выбирает точку: через неё проходят ДВЕ кривые (по одной на знак
  момента).  Яркая -- выбранная (кнопки `u = +u_max` / `u = -u_max` или
  клавиша T), бледная -- вторая.
* Справа сверху -- корпус и колесо в момент t; справа снизу -- psi(t), dpsi(t)
  и угол колеса theta(t) с курсором на том же t.
* Внизу -- дорожка времени по всей ширине.  Ноль отмечен: слева от него
  прошлое, справа будущее.  ПРОБЕЛ -- пуск/пауза, стрелки -- на кадр,
  перетаскивание ручки останавливает воспроизведение и держит кадр (правило
  дорожки времени из ER-015).
* Слайдер u_max сверху (клавиши [ и ]) -- поле пересчитывается на отпускании,
  как в окне исследователя.

Кривые через сёдла (клавиша M, интерес Глеба 24.09)
---------------------------------------------------
Отдельный интерес -- что делает робот НА многообразиях седла: не на прямой
вдоль собственного вектора, а на самой кривой, которой эта прямая касается.
Клавиша M листает восемь таких кривых: два седла (+-psi_eq) x устойчивое и
неустойчивое многообразие x две ветви (выше и ниже седла); Shift+M -- назад.
Касательная прямая вдоль собственного вектора не рисуется (решение Глеба
24.09: отвлекает от кривой) -- её наклон равен -lam и напечатан числом.
Знак момента при этом выставляется сам: седло при +psi_eq существует только
для u = -u_max, при -psi_eq -- только для +u_max.

Выбранная точка кладётся не В седло (там робот стоит вечно и смотреть не на
что), а на eps вдоль собственного вектора, и дальше работает та же машинерия,
что для любой точки поля: ветвь назад и ветвь вперёд.

Что происходит с роботом. Половина кривых -- вход в седло, половина -- выход, и
путать их не надо. На НЕУСТОЙЧИВОМ многообразии корпус из седла уходит, и это
не ошибка счёта: неустойчивое многообразие и есть множество траекторий,
покидающих равновесие (в седле они были при t -> -бесконечность). Скорость
ухода -- та же lam, от eps зависит только начало отсчёта времени.

На устойчивом многообразии корпус приходит к седлу и
ОСТАЁТСЯ под седловым углом: равновесие достигается за бесконечное время, и
уйти оттуда решение не может -- это и есть определение устойчивого
многообразия. Колесо при этом не стоит: седло -- равновесие только приведённой
динамики наклона, и под постоянным моментом колесо всё это время разгоняется.

Почему кривая обрезана. Точка, сдвинутая на eps, лежит на многообразии лишь с
точностью до первого порядка (теорема о устойчивом многообразии даёт касание,
а не совпадение), поэтому неустойчивая составляющая порядка eps² растёт как
e^(lam t), и численная кривая после подхода к седлу сваливается. Этот хвост --
ошибка метода, а не поведение робота, поэтому встречная ветвь режется в момент
наибольшего приближения (`_trim_to_manifold`), и в панели написано, что дальше
корпус остаётся под седловым углом. Достоверна та ветвь, которая многообразие
ВЫМЕТАЕТ: назад по времени для устойчивого, вперёд для неустойчивого -- ровно
так его и рисует `extremal_field.stable_manifold`.

Легенда -- обязательна и нарисована прямо в поле (ER-015).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from wpend.models import WheeledPendulum
from wpend.viz.explorer import (
    ACCENT, BG, BTN, BTN_ON, CURVE, DIM, GHOST, GRID_LINE, LIMIT, PANEL, TEXT,
)

import extremal_field as ef  # noqa: E402  -- ветви, аналитика и числа оттуда

W, H = 1280, 880
FIELD = (20, 84, 612, 612)
PLANT = (652, 84, 608, 286)
PLOTS = (652, 382, 608, 314)
TRACK = (20, 726, 1240, 30)
U_SLIDER = (150, 34, 360, 14)

U_MIN, U_MAX, U_STEP = 0.25, 10.0, 0.25

#: Цвета знаков момента. К цвету всегда добавлен второй признак -- подпись в
#: легенде и толщина выбранной кривой, поэтому картинка читается и без цвета.
C_PLUS, C_MINUS = (86, 156, 240), (240, 140, 80)
SIGN_COLOR = {+1: C_PLUS, -1: C_MINUS}

#: Кривые через сёдла: (какое седло, какое многообразие, с какой стороны).
#: `s = +1` -- седло при +psi_eq (живёт при u = -u_max), `s = -1` -- зеркальное.
#: `side = -1` -- ветвь СО СТОРОНЫ ВЕРТИКАЛИ (её и видно как границу
#: восстановимости), `+1` -- дальше седла, откуда возврата уже нет.
#: Порядок листания: сначала оба многообразия одного седла, потом зеркало.
SADDLE_PICKS = [
    (+1, "stable", -1), (+1, "stable", +1),
    (+1, "unstable", -1), (+1, "unstable", +1),
    (-1, "stable", -1), (-1, "stable", +1),
    (-1, "unstable", -1), (-1, "unstable", +1),
]
SADDLE_KIND_LABEL = {"stable": "устойчивое", "unstable": "неустойчивое"}
SADDLE_SIDE_LABEL = {-1: "ветвь от вертикали", +1: "ветвь за седлом"}
#: Цвет выбранного седла и касательной к многообразию. Отличается и от цветов
#: знака момента, и от ACCENT (кадр): это третья по смыслу вещь на картинке.
SADDLE = (200, 150, 240)


# ---------------------------------------------------------------------------
#  Состояние окна
# ---------------------------------------------------------------------------


class App:
    """Поле при текущем u_max плюс выбранная точка и время на её кривой.

    Считается ровно две вещи: фон (пучок ветвей, дорого, кэшируется) и две
    кривые через выбранную точку (дёшево, пересчитывается на каждый клик).
    Ничего из этого не является прогоном замкнутого контура -- регулятора тут
    нет вовсе, момент постоянен по построению.
    """

    def __init__(self, args):
        self.args = args
        self.system = WheeledPendulum()
        self.u_max = float(args.u_max)
        self.u_preview = None
        self.drag = None
        self.sign = +1
        self.playing = not args.screenshot
        self.t = 0.0
        self.field = None
        self.surface = None          # кэш фона; None -- перерисовать
        self.curves = {}
        self.point = None
        #: Выбрана ли кривая через седло, и какая (см. select_saddle).
        self.saddle = None
        self.rebuild()
        if args.saddle is not None:
            self.select_saddle(int(args.saddle))
        else:
            self.select(*(args.select if args.select else (-0.6, 0.0)))

    # --- фон -------------------------------------------------------------

    def rebuild(self):
        """Пересчитать пучок под текущий u_max.

        Шаг здесь крупнее, чем в оракулах: фон -- это картинка, а не проверка.
        Точность проверяется в `tests/test_extremal_field.py`, и мельчить тут
        значит платить секундами за пиксели, которых не видно.
        """
        self.field = ef.extremal_field(self.system, self.u_max,
                                       n_psi=self.args.n_psi, dt=self.args.field_dt)
        self.psi_lim = self.field["psi_lim"]
        self.dpsi_lim = self.field["dpsi_lim"]
        self.psi_eq = self.system.saddle_angle(self.u_max)
        self.surface = None

    def set_u_max(self, value):
        value = float(np.clip(value, U_MIN, U_MAX))
        if value == self.u_max:
            return
        self.u_max = value
        self.rebuild()
        # Кривая через седло переносится НЕ по запомненной точке: седло уезжает
        # вместе с пределом момента (psi_eq растёт с u_max), и старая точка при
        # новом пределе равновесием уже не является -- окно показывало бы
        # кривую «через седло», проходящую мимо него.
        if self.saddle is not None:
            self.select_saddle(self.saddle["index"])
        elif self.point is not None:
            self.select(*self.point)

    # --- выбранная точка --------------------------------------------------

    def select(self, psi, dpsi, *, keep_saddle=False):
        """Две кривые через точку: по одной на знак момента.

        Точка не обязана лежать на оси dpsi = 0 -- в отличие от сетки, которой
        рисуется фон (решение Глеба 21.09: перебираем только угол).  Поле от
        этого не меняется: через любую точку плоскости проходит ровно по одной
        линии уровня каждого семейства.

        `keep_saddle` ставит только `select_saddle`: обычный клик по полю
        снимает отметку «мы на многообразии», потому что после него мы на ней
        уже не находимся, а подпись осталась бы и врала.
        """
        psi = float(np.clip(psi, -self.psi_lim, self.psi_lim))
        dpsi = float(np.clip(dpsi, -self.dpsi_lim, self.dpsi_lim))
        if not keep_saddle:
            self.saddle = None
        self.point = (psi, dpsi)
        x0 = np.array([psi, 0.0, dpsi, 0.0])
        lim = dict(psi_lim=self.psi_lim, dpsi_lim=self.dpsi_lim,
                   max_arc=2.0 * (self.psi_lim + self.dpsi_lim),
                   max_steps=int(self.args.max_steps))
        self.curves = {}
        for sign in (+1, -1):
            u = sign * self.u_max
            t_f, X_f = ef.branch(self.system, x0, u, +self.args.dt, **lim)
            t_b, X_b = ef.branch(self.system, x0, u, -self.args.dt, **lim)
            self.curves[sign] = ef.whole_branch(
                {"t_fwd": t_f, "X_fwd": X_f, "t_bwd": t_b, "X_bwd": X_b})
        self.t = 0.0

    # --- кривые через сёдла ----------------------------------------------

    def select_saddle(self, index: int):
        """Встать на многообразие седла и смотреть, что делает робот.

        Точка -- седло плюс eps вдоль собственного вектора; знак момента
        выставляется под это седло, иначе кривая была бы чужой: при +psi_eq
        равновесие существует только для u = -u_max.

        Собственные числа и вектор берутся у `extremal_field`, а не выводятся
        здесь заново: там они посчитаны конечной разностью и проверены
        оракулом против якобиана полной четырёхмерной f. Неустойчивый вектор
        отличается от устойчивого только знаком lam -- якобиан седла
        [[0, 1], [g', 0]] симметричен по времени.

        Зеркальное седло получается заменой (psi, dpsi, u) -> (-psi, -dpsi, -u):
        правая часть при этом меняет знак вся целиком (sin нечётен, cos чётен),
        поэтому вторая половина списка -- не новый расчёт, а отражение.
        """
        s, kind, side = SADDLE_PICKS[index % len(SADDLE_PICKS)]
        psi_eq, lam, v_stable = ef.saddle_stable_dir(self.system, self.u_max)
        v = v_stable if kind == "stable" else np.array([1.0, +lam])
        eps = float(self.args.saddle_eps)
        psi0 = s * (psi_eq + side * eps * v[0])
        dpsi0 = s * (side * eps * v[1])
        # Седло при +psi_eq -- равновесие при u = -u_max, и наоборот.
        self.sign = -s
        self.select(psi0, dpsi0, keep_saddle=True)
        self.saddle = dict(index=index % len(SADDLE_PICKS), s=s, kind=kind,
                           side=side, psi_eq=psi_eq, lam=float(lam), eps=eps)
        self._trim_to_manifold()
        self.saddle.update(self.saddle_stats())

    def _trim_to_manifold(self):
        """Обрезать кривую там, где она перестаёт быть многообразием.

        На устойчивом многообразии корпус обязан ОСТАТЬСЯ в седле: решение
        приходит в равновесие за бесконечное время и никуда оттуда не уходит.
        Численно же точка лежит на многообразии лишь с точностью до первого
        порядка (теорема о устойчивом многообразии даёт касание, а не
        совпадение), поэтому неустойчивая составляющая -- порядка eps² плюс
        шум арифметики -- растёт как e^(lam t), и кривая рано или поздно
        сваливается. Этот хвост -- не поведение робота, а ошибка метода, и
        рисовать его рядом с честной половиной нельзя.

        Поэтому оставляем ту ветвь, которая многообразие ВЫМЕТАЕТ, и режем
        встречную в момент наибольшего приближения к седлу:

          * устойчивое многообразие -- достоверна ветвь НАЗАД (назад по
            времени устойчивое направление становится неустойчивым и уводит
            траекторию прочь от седла, то есть рисует многообразие); вперёд
            оставляем только подход, дальше корпус стоит под седловым углом;
          * неустойчивое -- зеркально: достоверна ветвь ВПЕРЁД, а прошлое
            обрезается на том же признаке.

        Обрезанное не подменяется ничем: дорисовывать «стояние» в седле
        выдуманными отсчётами значило бы показывать то, чего не считали.
        """
        sd = self.saddle
        t, X = self.curves[self.sign]
        d = np.abs(X[:, 0] - sd["s"] * sd["psi_eq"])
        k = int(np.argmin(d))
        keep = slice(None, k + 1) if sd["kind"] == "stable" else slice(k, None)
        self.curves[self.sign] = (t[keep], X[keep])
        sd["t_cut"] = float(t[k])
        # Время не переносим: ноль по-прежнему в выбранной точке, и дорожка
        # так же делит прошлое и будущее. У устойчивой кривой будущего просто
        # немного -- ровно до седла.
        self.t = float(np.clip(self.t, t[keep][0], t[keep][-1]))

    def saddle_stats(self) -> dict:
        """Сколько корпус держится у седла и когда подходит ближе всего.

        Это и есть ответ на «что с роботом»: на устойчивом многообразии он
        асимптотически замирает под седловым углом, но точка сдвинута на eps,
        неустойчивая составляющая растёт как e^(lam t), и корпус уходит.
        `t_near` -- момент наибольшего приближения, `linger` -- сколько всего
        времени |psi - psi_eq| держится в пределах одного градуса.
        """
        if self.saddle is None:
            return {}
        t, X = self.curve
        target = self.saddle["s"] * self.saddle["psi_eq"]
        d = np.abs(X[:, 0] - target)
        near = d <= np.radians(1.0)
        dt = float(self.args.dt)
        return {"t_near": float(t[int(np.argmin(d))]),
                "d_min": float(np.min(d)),
                "linger": float(near.sum() * dt),
                "settle": float(abs(t[-1] - t[0]))}

    @property
    def curve(self):
        return self.curves[self.sign]

    def span(self):
        t, _ = self.curve
        return float(t[0]), float(t[-1])

    def frame(self) -> int:
        """Индекс кадра на выбранной кривой по текущему t."""
        t, _ = self.curve
        return int(np.clip(np.searchsorted(t, self.t), 0, len(t) - 1))

    def state(self):
        _, X = self.curve
        return X[self.frame()]

    def scrub(self, t: float):
        """Поставить время руками.  Останавливает воспроизведение и ДЕРЖИТ
        кадр до ПРОБЕЛА -- то же правило, что у дорожки в окне исследователя
        (решение Глеба 21.09, ER-015)."""
        lo, hi = self.span()
        self.t = float(np.clip(t, lo, hi))
        self.playing = False

    def step_frame(self, delta: int):
        t, _ = self.curve
        k = int(np.clip(self.frame() + delta, 0, len(t) - 1))
        self.t = float(t[k])
        self.playing = False

    def advance(self, dt_wall: float):
        """Проигрывание в реальном времени, с зацикливанием на конце."""
        if not self.playing:
            return
        lo, hi = self.span()
        self.t += dt_wall * self.args.speed
        if self.t > hi:
            self.t = lo


# ---------------------------------------------------------------------------
#  Геометрия: величины <-> пиксели
# ---------------------------------------------------------------------------


def to_px(app, psi, dpsi):
    x, y, w, h = FIELD
    fx = (psi + app.psi_lim) / (2.0 * app.psi_lim)
    fy = (dpsi + app.dpsi_lim) / (2.0 * app.dpsi_lim)
    return x + fx * w, y + (1.0 - fy) * h


def from_px(app, px, py):
    x, y, w, h = FIELD
    psi = (px - x) / w * 2.0 * app.psi_lim - app.psi_lim
    dpsi = (1.0 - (py - y) / h) * 2.0 * app.dpsi_lim - app.dpsi_lim
    return psi, dpsi


def time_to_px(app, t: float) -> float:
    x, _, w, _ = TRACK
    lo, hi = app.span()
    return x + (t - lo) / max(hi - lo, 1e-9) * w


def time_from_px(app, px: float) -> float:
    x, _, w, _ = TRACK
    lo, hi = app.span()
    return lo + (px - x) / w * (hi - lo)


def u_to_px(value):
    x, _, w, _ = U_SLIDER
    return x + (value - U_MIN) / (U_MAX - U_MIN) * w


def u_from_px(px):
    x, _, w, _ = U_SLIDER
    raw = U_MIN + (px - x) / w * (U_MAX - U_MIN)
    return float(np.clip(round(raw / U_STEP) * U_STEP, U_MIN, U_MAX))


# ---------------------------------------------------------------------------
#  Рисование
# ---------------------------------------------------------------------------


def _panel(screen, pg, rect, title, font):
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    screen.blit(font.render(title, True, DIM), (rect[0] + 10, rect[1] + 6))


def _polyline(target, pg, pts, color, width):
    """Ломаная с отсечением: pygame не любит координаты в миллионы пикселей."""
    if len(pts) >= 2:
        pg.draw.lines(target, color, False, pts, width)


def _runs(mask):
    """Непрерывные куски True.

    Нужно ровно затем, чтобы не соединить прямой линией две РАЗНЫЕ ветви одной
    кривой: у sigma = 0 при большом u_max есть вторая компонента за седлом, и
    без разрезания ломаная перечеркнула бы всю картинку.
    """
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    return np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1)


def _curve_px(app, X, stride=1):
    pts = []
    for psi, dpsi in zip(X[::stride, 0], X[::stride, 2]):
        px, py = to_px(app, psi, dpsi)
        if -4000 < px < 4000 and -4000 < py < 4000:
            pts.append((px, py))
    return pts


def field_surface(app, pg):
    """Фон: пучок + аналитика.  Кэшируется -- на каждом кадре это дорого."""
    if app.surface is not None:
        return app.surface
    x, y, w, h = FIELD
    surf = pg.Surface((w, h))
    surf.fill(PANEL)

    off = lambda p: (p[0] - x, p[1] - y)

    # Восстановимое множество -- заливкой: белое поле вокруг и есть «оттуда не
    # спасает никакое управление» (ER-017).
    grid = np.linspace(-app.psi_lim, app.psi_lim, 241)
    floor, ceiling = app.system.recoverable_bounds(app.u_max, grid)
    # Только там, где потолок выше пола: иначе многоугольник самопересекается
    # и заливает то, что как раз НЕ восстановимо.
    idx = np.flatnonzero(ceiling > floor)
    if idx.size:
        sl = slice(idx[0], idx[-1] + 1)
        band = [off(to_px(app, p, c)) for p, c in zip(grid[sl], ceiling[sl])]
        band += [off(to_px(app, p, f))
                 for p, f in zip(grid[sl][::-1], floor[sl][::-1])]
        pg.draw.polygon(surf, (40, 44, 52), band)

    for gx in np.arange(-2.0, 2.01, 0.5):
        px = off(to_px(app, gx, 0.0))[0]
        pg.draw.line(surf, GRID_LINE, (px, 0), (px, h), 1)
    for gy in np.arange(-20.0, 20.01, 2.0):
        py = off(to_px(app, 0.0, gy))[1]
        if 0 <= py <= h:
            pg.draw.line(surf, GRID_LINE, (0, py), (w, py), 1)

    for b in app.field["branches"]:
        color = SIGN_COLOR[+1 if b["u"] > 0 else -1]
        pts = [off(p) for p in _curve_px(app, ef.whole_curve(b), stride=4)]
        _polyline(surf, pg, pts, _shade(color, 0.55), 1)

    sw = ef.sigma_zero_dpsi(app.system, app.u_max, grid)
    for xs, ys in ((grid, sw), (-grid, -sw)):
        for run in _runs(np.isfinite(ys)):
            pts = [off(to_px(app, p, d)) for p, d in zip(xs[run], ys[run])]
            _polyline(surf, pg, pts, CURVE, 2)

    for edge in (ceiling, floor):
        pts = [off(to_px(app, p, d)) for p, d in zip(grid, edge)]
        _polyline(surf, pg, pts, LIMIT, 2)

    for s in (+1, -1):
        px, py = off(to_px(app, s * app.psi_eq, 0.0))
        pg.draw.circle(surf, TEXT, (int(px), int(py)), 6, 2)
    for s in (+1, -1):
        px = off(to_px(app, s * np.pi / 2, 0.0))[0]
        pg.draw.line(surf, GRID_LINE, (px, 0), (px, h), 2)

    app.surface = surf
    return surf


def _shade(color, factor):
    return tuple(int(c * factor) for c in color)


def _legend(screen, pg, font, x, y, items):
    """Легенда строкой (ER-015): цвет, подпись, следующий."""
    for color, label, width in items:
        pg.draw.line(screen, color, (x, y + 7), (x + 22, y + 7), width)
        screen.blit(font.render(label, True, TEXT), (x + 28, y))
        x += 34 + font.size(label)[0] + 14
    return x


def draw_field(app, screen, pg, font):
    x, y, w, h = FIELD
    screen.blit(field_surface(app, pg), (x, y))
    pg.draw.rect(screen, GRID_LINE, FIELD, 1, border_radius=6)

    # Вторая кривая -- бледная: видно, что через точку их ровно две. На кривой
    # через седло её не рисуем: она проходит через ту же точку, но
    # многообразием не является и уходит от седла -- глаз читает это как
    # «робот всё-таки выбрался», хотя это просто другая линия уровня.
    if app.saddle is None:
        other = -app.sign
        _polyline(screen, pg, _curve_px(app, app.curves[other][1], stride=2),
                  _shade(SIGN_COLOR[other], 0.8), 2)
    _polyline(screen, pg, _curve_px(app, app.curve[1], stride=1),
              SIGN_COLOR[app.sign], 3)

    # Прошлое выбранной кривой -- пунктиром по точкам: t < 0 это «как попали».
    t, X = app.curve
    past = X[t <= 0.0]
    for p in _curve_px(app, past, stride=14):
        pg.draw.circle(screen, ACCENT, (int(p[0]), int(p[1])), 2)

    # Выбранное седло -- залитый кружок. Касательная прямая вдоль собственного
    # вектора здесь РИСОВАЛАСЬ и убрана (решение Глеба 24.09: отвлекает от
    # кривой). Сама прямая никуда не делась из смысла: наклон многообразия в
    # седле равен -lam, и это число напечатано в панели.
    if app.saddle is not None:
        cx0, cy0 = to_px(app, app.saddle["s"] * app.saddle["psi_eq"], 0.0)
        pg.draw.circle(screen, SADDLE, (int(cx0), int(cy0)), 7)

    px, py = to_px(app, *app.point)
    pg.draw.circle(screen, TEXT, (int(px), int(py)), 5, 1)
    cx, cy = to_px(app, app.state()[0], app.state()[2])
    pg.draw.circle(screen, ACCENT, (int(cx), int(cy)), 6)

    screen.blit(font.render(f"поле экстремалей, u_max = {app.u_max:g} Н·м, "
                            f"сёдла ±{np.degrees(app.psi_eq):.1f}°",
                            True, DIM), (x + 10, y + 6))
    screen.blit(font.render("ψ, рад →", True, DIM), (x + w - 70, y + h - 20))
    screen.blit(font.render("↑ dψ/dt, рад/с", True, DIM), (x + 10, y + 26))
    _legend(screen, pg, font, x + 10, y + h - 44,
            [(C_PLUS, "u=+u_max", 2), (C_MINUS, "u=−u_max", 2),
             (CURVE, "σ=0", 2), (LIMIT, "восстановимость", 2)])
    _legend(screen, pg, font, x + 10, y + h - 26,
            [(ACCENT, "прошлое (t<0) и кадр", 2), (TEXT, "седло / выбор", 1),
             (SADDLE, "выбранное седло", 1)])


def draw_plant(app, screen, pg, font):
    x, y, w, h = PLANT
    _panel(screen, pg, PLANT, "PLANT", font)
    state = app.state()
    psi, theta = float(state[0]), float(state[1])
    p = app.system.p
    wheel_px = 34.0
    m2px = wheel_px / p.r
    # Корпус смещён вправо: слева под числами четыре строки про выбранную
    # кривую через седло, и по центру они оказывались бы под корпусом.
    cx, cy = x + w * 0.78, y + h * 0.78

    pg.draw.line(screen, GRID_LINE, (x + 20, cy + wheel_px), (x + w - 20, cy + wheel_px), 1)
    pg.draw.circle(screen, GHOST, (int(cx), int(cy)), int(wheel_px), 3)
    spoke = (cx + wheel_px * np.sin(theta), cy - wheel_px * np.cos(theta))
    pg.draw.line(screen, GHOST, (cx, cy), spoke, 2)
    tip = (cx + p.l * m2px * np.sin(psi), cy - p.l * m2px * np.cos(psi))
    pg.draw.line(screen, CURVE, (cx, cy), tip, 6)
    pg.draw.circle(screen, SIGN_COLOR[app.sign], (int(tip[0]), int(tip[1])), 10)

    lines = [f"t = {app.t:+.3f} с   (t<0 -- прошлое ветви)",
             f"ψ = {psi:+.3f} рад   dψ/dt = {float(state[2]):+.3f} рад/с",
             f"θ = {theta:+.3f} рад   dθ/dt = {float(state[3]):+.3f} рад/с",
             f"u = {app.sign * app.u_max:+g} Н·м (постоянный)",
             f"H = {float(app.system.first_integral(state, app.sign * app.u_max)):.6f}"]
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, TEXT), (x + 12, y + 30 + 18 * i))

    if app.saddle is not None:
        sd = app.saddle
        way = "вход в седло" if sd["kind"] == "stable" else "ВЫХОД из седла"
        title = (f"седло {sd['s'] * np.degrees(sd['psi_eq']):+.2f}°: "
                 f"{SADDLE_KIND_LABEL[sd['kind']]} многообразие — {way}")
        # Вторая строка -- то, чего картинка сама не скажет: момент ухода с
        # седла задан eps и арифметикой, а не физикой.
        # Три строки, а не одна: в ширину панели помещается около 78 знаков
        # моноширинного шрифта, и склеенная строка молча обрезалась бы ровно
        # на числах, ради которых она и написана.
        # Что происходит ПОСЛЕ конца кривой -- главное, чего сама кривая не
        # говорит: на устойчивом многообразии корпус остаётся под седловым
        # углом, на неустойчивом он в седле БЫЛ до начала кривой.
        tail = ("дальше корпус остаётся под седловым углом"
                if sd["kind"] == "stable"
                else "корпус уходит из седла — это путь наружу")
        rows = (title,
                f"{SADDLE_SIDE_LABEL[sd['side']]}, eps = {sd['eps']:.0e}",
                f"λ = {sd['lam']:.3f} 1/с, e-фолд 1/λ = {1.0 / sd['lam']:.3f} с",
                f"подход {sd['linger']:.2f} с, |Δψ| на конце {sd['d_min']:.1e} рад",
                tail)
        for i, line in enumerate(rows):
            screen.blit(font.render(line, True, SADDLE),
                        (x + 12, y + 30 + 18 * (len(lines) + i) + 6))


def _plot(screen, pg, rect, ts, ys, color, label, font, t_now):
    x, y, w, h = rect
    pg.draw.rect(screen, (28, 31, 37), rect, border_radius=4)
    lo, hi = float(np.min(ys)), float(np.max(ys))
    if hi - lo < 1e-9:
        lo, hi = lo - 1.0, hi + 1.0
    t0, t1 = float(ts[0]), float(ts[-1])
    to_x = lambda t: x + (t - t0) / max(t1 - t0, 1e-9) * w
    to_y = lambda v: y + h - (v - lo) / (hi - lo) * h

    if lo < 0.0 < hi:
        pg.draw.line(screen, GRID_LINE, (x, to_y(0.0)), (x + w, to_y(0.0)), 1)
    if t0 < 0.0 < t1:
        pg.draw.line(screen, GRID_LINE, (to_x(0.0), y), (to_x(0.0), y + h), 1)
    step = max(1, len(ts) // w)
    pts = [(to_x(t), to_y(v)) for t, v in zip(ts[::step], ys[::step])]
    _polyline(screen, pg, pts, color, 2)
    pg.draw.line(screen, ACCENT, (to_x(t_now), y), (to_x(t_now), y + h), 1)
    screen.blit(font.render(f"{label}  [{lo:+.2f}, {hi:+.2f}]", True, DIM),
                (x + 6, y + 2))


def draw_plots(app, screen, pg, font):
    x, y, w, h = PLOTS
    _panel(screen, pg, PLOTS, "вдоль выбранной кривой", font)
    t, X = app.curve
    rows = [(X[:, 0], "ψ(t), рад", SIGN_COLOR[app.sign]),
            (X[:, 2], "dψ/dt(t), рад/с", SIGN_COLOR[app.sign]),
            (X[:, 1], "θ(t), рад -- колесо ведомое", GHOST)]
    top, gap = y + 28, 6
    ph = (h - 34 - gap * (len(rows) - 1)) // len(rows)
    for i, (ys, label, color) in enumerate(rows):
        _plot(screen, pg, (x + 10, top + i * (ph + gap), w - 20, ph),
              t, ys, color, label, font, app.t)


def draw_track(app, screen, pg, font):
    x, y, w, h = TRACK
    lo, hi = app.span()
    pg.draw.rect(screen, PANEL, TRACK, border_radius=6)
    if lo < 0.0 < hi:
        zx = time_to_px(app, 0.0)
        pg.draw.line(screen, TEXT, (zx, y + 4), (zx, y + h - 4), 2)
        screen.blit(font.render("t = 0 (выбранная точка)", True, DIM), (zx + 6, y + h + 2))
    px = time_to_px(app, app.t)
    pg.draw.rect(screen, SIGN_COLOR[app.sign], (x + 2, y + h / 2 - 2, px - x - 2, 4))
    pg.draw.circle(screen, ACCENT, (int(px), int(y + h / 2)), 8)
    screen.blit(font.render(f"{lo:+.2f} с", True, DIM), (x + 6, y + h + 2))
    right = f"{hi:+.2f} с"
    screen.blit(font.render(right, True, DIM), (x + w - font.size(right)[0] - 6, y + h + 2))


def draw_top(app, screen, pg, font, buttons):
    x, y, w, h = U_SLIDER
    value = app.u_preview if app.u_preview is not None else app.u_max
    pg.draw.rect(screen, BTN, (x, y, w, h), border_radius=7)
    pg.draw.rect(screen, BTN_ON, (x, y, u_to_px(value) - x, h), border_radius=7)
    pg.draw.circle(screen, ACCENT, (int(u_to_px(value)), int(y + h / 2)), 9)
    screen.blit(font.render(f"u_max = {value:g} Н·м  [ ]", True, TEXT), (16, y - 2))
    for rect, label, on in buttons:
        pg.draw.rect(screen, BTN_ON if on else BTN, rect, border_radius=5)
        screen.blit(font.render(label, True, TEXT), (rect[0] + 10, rect[1] + 5))


def buttons_of(app, font):
    """Кнопки выбора знака момента: та же пара, что цвета в поле."""
    out = []
    for i, sign in enumerate((+1, -1)):
        label = f"u = {'+' if sign > 0 else '−'}u_max"
        out.append(((560 + i * 150, 28, 140, 26), label, app.sign == sign))
    return out


def draw(app, screen, pg, font):
    screen.fill(BG)
    draw_top(app, screen, pg, font, buttons_of(app, font))
    draw_field(app, screen, pg, font)
    draw_plant(app, screen, pg, font)
    draw_plots(app, screen, pg, font)
    draw_track(app, screen, pg, font)
    hint = ("клик по полю -- точка;  M -- кривая через седло (Shift+M назад);  "
            "T -- другой знак момента;  ПРОБЕЛ -- пуск/пауза;  "
            "стрелки -- кадр;  [ ] -- предел момента;  ESC -- выход")
    screen.blit(font.render(hint, True, DIM), (20, H - 28))


# ---------------------------------------------------------------------------
#  Цикл
# ---------------------------------------------------------------------------


def run(app, *, max_frames=None, screenshot=None):
    import pygame as pg

    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("extremal field explorer")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 14)
    clock = pg.time.Clock()
    frames = 0
    while True:
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                return
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    return
                if ev.key == pg.K_SPACE:
                    app.playing = not app.playing
                if ev.key == pg.K_t:
                    # На кривой через седло смена знака момента означает не
                    # «вторую кривую через ту же точку» (она многообразием не
                    # является, а подпись осталась бы и врала), а ЗЕРКАЛЬНОЕ
                    # седло: оно как раз живёт при другом знаке.
                    if app.saddle is not None:
                        app.select_saddle(app.saddle["index"] + len(SADDLE_PICKS) // 2)
                    else:
                        app.sign = -app.sign
                if ev.key == pg.K_m:
                    # Листаем кривые через сёдла; Shift -- назад по списку.
                    step = -1 if ev.mod & pg.KMOD_SHIFT else +1
                    base = 0 if app.saddle is None else app.saddle["index"] + step
                    app.select_saddle(base)
                if ev.key in (pg.K_LEFT, pg.K_RIGHT):
                    app.step_frame(-1 if ev.key == pg.K_LEFT else +1)
                if ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    step = U_STEP if ev.key == pg.K_RIGHTBRACKET else -U_STEP
                    app.set_u_max(app.u_max + step)
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                ux, uy, uw, uh = U_SLIDER
                tx, ty, tw, th = TRACK
                if ux - 10 <= ev.pos[0] <= ux + uw + 10 and uy - 10 <= ev.pos[1] <= uy + uh + 10:
                    app.drag, app.u_preview = "u_max", u_from_px(ev.pos[0])
                elif tx <= ev.pos[0] <= tx + tw and ty - 6 <= ev.pos[1] <= ty + th + 6:
                    app.drag = "time"
                    app.scrub(time_from_px(app, ev.pos[0]))
                elif any(pg.Rect(r).collidepoint(ev.pos) for r, _, _ in buttons_of(app, font)):
                    hit = [r for r, _, _ in buttons_of(app, font)
                           if pg.Rect(r).collidepoint(ev.pos)][0]
                    want = +1 if hit[0] < 700 else -1
                    if app.saddle is not None and want != app.sign:
                        # То же, что клавиша T: остаёмся на многообразии,
                        # переходя к зеркальному седлу.
                        app.select_saddle(app.saddle["index"] + len(SADDLE_PICKS) // 2)
                    else:
                        app.sign = want
                elif pg.Rect(FIELD).collidepoint(ev.pos):
                    app.select(*from_px(app, *ev.pos))
            elif ev.type == pg.MOUSEMOTION and app.drag is not None:
                if app.drag == "u_max":
                    app.u_preview = u_from_px(ev.pos[0])
                else:
                    app.scrub(time_from_px(app, ev.pos[0]))
            elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1:
                if app.drag == "u_max":
                    # Пересчёт поля -- на отпускании: тянуть ручку и считать
                    # пучок на каждом пикселе нельзя, это секунды.
                    app.set_u_max(app.u_preview)
                    app.u_preview = None
                app.drag = None

        app.advance(clock.get_time() / 1000.0)
        draw(app, screen, pg, font)
        pg.display.flip()
        frames += 1
        if screenshot and frames >= 1:
            pg.image.save(screen, screenshot)
            return
        if max_frames is not None and frames >= max_frames:
            return
        clock.tick(30)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--u-max", type=float, default=3.0)
    ap.add_argument("--n-psi", type=int, default=17, help="ветвей в фоне")
    ap.add_argument("--dt", type=float, default=1e-3, help="шаг выбранной кривой")
    ap.add_argument("--field-dt", type=float, default=2e-3, help="шаг фона")
    ap.add_argument("--speed", type=float, default=1.0, help="скорость воспроизведения")
    ap.add_argument("--select", type=float, nargs=2, default=None,
                    metavar=("PSI", "DPSI"))
    ap.add_argument("--saddle", type=int, default=None,
                    help=f"стартовать с кривой через седло, 0..{len(SADDLE_PICKS) - 1} "
                         "(клавиша M листает их же)")
    ap.add_argument("--saddle-eps", type=float, default=1e-6,
                    help="сдвиг от седла вдоль собственного вектора, рад")
    ap.add_argument("--max-steps", type=int, default=30000,
                    help="потолок шагов одной ветви: у седла ветвь ползёт "
                         "экспоненциально медленно и по дуге не останавливается")
    ap.add_argument("--screenshot", default=None, help="снять кадр и выйти")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    app = App(args)
    run(app, screenshot=args.screenshot)


if __name__ == "__main__":
    main()
