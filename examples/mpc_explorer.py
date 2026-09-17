"""Окно MPC: как box-DDP MPC ведёт себя из одного начального состояния.

    uv run python examples/mpc_explorer.py
    uv run python examples/mpc_explorer.py --u-max 10 --select -0.495 0.46
    uv run python examples/mpc_explorer.py --dtheta-max 8        # своя граница по скорости
    uv run python examples/mpc_explorer.py --theta-fall 1.571    # «упал» = лёг на землю

Устроено как окно исследователя, но с поправкой на цену расчёта: карту MPC
целиком не посчитать (8–15 с на клетку), поэтому

* ФОН карты -- ЛКР + clip с той же ценой и тем же пределом (секунда-две);
* клик по клетке запускает MPC для этого старта в фоновом потоке. Окно не
  замирает: пока идёт расчёт, видна полоса прогресса, время на такт и
  итерации iLQR; уже посчитанные старты остаются на карте кружками цвета
  исхода MPC (для текущих параметров);
* после расчёта траектория проигрывается: робот MPC и полупрозрачный
  робот-ЛКР, фазовые кривые, графики theta(t), u(t), phi(t);
* главное -- ПЛАН. На каждом кадре поверх карты и графиков пунктиром рисуется
  то, что MPC предсказывал на последнем такте: куда собирался привести
  корпус и каким моментом. Расхождение плана с тем, что случилось, --
  ровно то, что хочется видеть при смене горизонта или цены.

Параметры (любая смена отменяет текущий расчёт и пересчитывает фон):
* u_max -- слайдер как в explorer, [ ] и L (no limit);
* цена -- пять весов с клавиатуры и пресеты, как в `lqr_cost_explorer.py`;
* MPC -- такт dt_plan (кратен 1 мс), горизонт в секундах, итераций iLQR на
  такт и на первом такте;
* `limit` / `blind` -- знает ли планировщик предел (box-DDP) или нет (мир
  всё равно обрезает);
* `LQR ghost` -- рисовать ли ЛКР рядом.

Сравнение с ЛКР на карте
------------------------
* `map: LQR` -- фон ЛКР + clip, кружки -- исходы MPC там, где он посчитан.
* `map: MPC vs LQR` -- там, где есть ответ MPC, клетка окрашена по паре
  исходов: синий -- держат оба, жёлтый -- только MPC, бирюзовый -- только ЛКР,
  тёмный -- никто. В заголовке -- счётчики.
* `fill MPC` -- посчитать MPC на прореженной сетке (каждая --fill-step-я клетка,
  по умолчанию 11x11 = 121 старт) в пуле процессов, карта заполняется по мере
  готовности. Повторное нажатие -- остановить. Каждая клетка заливки
  закрашивает квадрат step x step вокруг себя. В этом режиме клик по квадрату
  выбирает именно старт, которым он закрашен (а не соседнюю мелкую клетку).
* Всё посчитанное -- и кликом, и заливкой -- хранится вместе с траекторией и
  планами по параметрам: клик по уже посчитанному старту показывает его сразу,
  без пересчёта (в строке статуса «cached»). Вернувшись к прежним параметрам,
  увидишь прежнюю карту.

Клавиши: пробел -- пауза, R -- проиграть сначала, , и . -- скорость
проигрывания, Esc -- выход (при открытом поле -- отмена ввода).

Карта и исходы (соглашение для визуализаций, Глеб 16.09)
-------------------------------------------------------
* По theta карта всегда от -pi/2 до pi/2 (пунктир на краях -- горизонт корпуса).
  По dtheta -- в 1.3 раза шире множества восстановимости на этом отрезке
  (зелёная линия), чтобы полоса целиком помещалась при любом u_max.
* Упал -- |theta| пересёк --theta-fall (по умолчанию pi: провернулся через
  низ; земли в модели нет). Не успокоился (серый) -- не пересёк, но к концу
  прогона |theta| >= 0.1. Иначе удержан.
"""

import argparse
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0]))
sys.path.insert(0, str(HERE))

import numpy as np

from wpend import (
    Controller, LinearFeedbackController, MPCController, RK4Integrator, rollout,
)
from wpend.models import WheeledPendulum
from wpend.viz.explorer import (
    ACCENT, BG, BTN, BTN_ON, CURVE, DIM, GHOST, GRID_LINE, LIMIT, PANEL, TEXT,
)
from wpend.viz.grid import HELD

import lqr_cost_explorer as lce  # noqa: E402  -- цена, пресеты, поля и фоновая карта оттуда

DT = 1e-3
MAP_FIT = 1.3
THETA_MAP = np.pi / 2          # край карты по theta: соглашение для визуализаций (Глеб 16.09)
W, H = 1280, 940
U_SLIDER = (100, 14, 300, 14)
MAP = (56, 206, 500, 500)
ANIM = (600, 206, 650, 214)
PLOTS = [(600, 450, 650, 140), (600, 616, 650, 130), (600, 772, 650, 120)]
PLAN = (240, 150, 70)
FALL_COLOR = lce.COLOR

MPC_FIELDS = [
    # ключ, подпись, тип, проверка
    ("dt_plan", "dt_plan, s", float, lambda v: v >= DT),
    ("horizon_s", "horizon, s", float, lambda v: v > 0),
    ("n_iter", "iters/tact", int, lambda v: v >= 1),
    ("first_iter", "iters first", int, lambda v: v >= 1),
]


class Cancelled(Exception):
    pass


class Watch(Controller):
    """Обёртка вокруг MPC: отдаёт окну прогресс и позволяет отменить расчёт.

    Живёт в примере, а не в ядре: это забота окна. Правило 2 цело -- память
    (текущее t) у регулятора."""

    def __init__(self, inner, cancel):
        self.inner, self.cancel, self.t = inner, cancel, 0.0

    def reset(self):
        self.inner.reset()

    def act(self, t, x_hat):
        if self.cancel.is_set():
            raise Cancelled
        self.t = t
        return self.inner.act(t, x_hat)


# ---------------------------------------------------------------------------
#  MPC одного старта -- функции верхнего уровня, чтобы их можно было отдать
#  отдельному процессу (заливка карты MPC идёт в пуле процессов)
# ---------------------------------------------------------------------------

def build_mpc(cost, u_max, mpc_params, box, record_plans=False):
    """(MPCController, K) для параметров окна.

    Терминальная цена -- P полной задачи Риккати; при нулевых весах колеса
    полная задача не решается, тогда P берётся у подсистемы (как K в lce.design).
    """
    from wpend.lqr import lqr
    K, _ = lce.design(WheeledPendulum(), cost)
    A, B = WheeledPendulum().linearize_upright()
    q = np.array([cost["q_theta"], cost["q_phi"], cost["q_dtheta"], cost["q_dphi"]])
    Q, R = np.diag(q), np.array([[cost["R"]]])
    try:
        _, P = lqr(A, B, Q, R)
    except Exception:
        keep = [0, 2] if q[3] == 0 else [0, 2, 3]
        _, P_sub = lqr(A[np.ix_(keep, keep)], B[keep, :], np.diag(q[keep]), R)
        P = np.zeros((4, 4))
        P[np.ix_(keep, keep)] = P_sub
    steps_plan = max(1, int(round(mpc_params["dt_plan"] / DT)))
    dt_plan = steps_plan * DT
    horizon = max(1, int(round(mpc_params["horizon_s"] / dt_plan)))
    planner = WheeledPendulum(u_max=u_max) if box else WheeledPendulum()
    ctrl = MPCController(planner, RK4Integrator(), dt_plan, horizon, Q, R, P,
                         n_iter=mpc_params["n_iter"], first_iter=mpc_params["first_iter"],
                         K_init=K, record_plans=record_plans)
    return ctrl, K


def outcome_of(x, theta_fall):
    """Исход прогона -- те же правила, что у фоновой карты (lce.compute_map)."""
    fell = np.flatnonzero(np.abs(x[1:, 0]) >= theta_fall)
    if fell.size:
        return 1 if x[fell[0] + 1, 0] > 0 else 2
    return HELD if abs(x[-1, 0]) < lce.SETTLE else lce.NOT_SETTLED


STORE_STRIDE = 10      # траектория в кэше -- каждый 10-й шаг (10 мс): 121 старт ~ 30 МБ


def pack_run(tr, plans, theta_fall, wall):
    """Всё, что окну нужно, чтобы показать старт без пересчёта: исход,
    прореженная траектория, планы тактов (float32) и сводка по скорости."""
    walls = [p["wall"] for p in plans]
    iters = [p["iters"] for p in plans]
    return dict(
        outcome=outcome_of(tr.x, theta_fall),
        t=tr.t[::STORE_STRIDE],
        x=tr.x[::STORE_STRIDE].astype(np.float32),
        u=tr.u[::STORE_STRIDE, 0].astype(np.float32),
        plans=[dict(t=p["t"], x=p["x"].astype(np.float32), u=p["u"].astype(np.float32))
               for p in plans],
        status=(f"MPC {wall:.1f} s: {len(walls)} tacts, {1e3 * np.mean(walls):.0f} ms/tact "
                f"(max {1e3 * np.max(walls):.0f}), accepted iLQR steps per tact "
                f"{np.mean(iters[1:] or iters):.1f}"),
    )


def mpc_cell(job):
    """Один старт для пула. Возвращает то же, что считает клик по клетке, --
    чтобы клик по уже залитой клетке показывал траекторию сразу."""
    x0, cost, u_max, mpc_params, box, horizon, theta_fall = job
    ctrl, _ = build_mpc(cost, u_max, mpc_params, box, record_plans=True)
    t0 = time.perf_counter()
    tr = rollout(WheeledPendulum(u_max=u_max), ctrl, RK4Integrator(), x0, DT,
                 int(round(horizon / DT)))
    return pack_run(tr, ctrl.plans, theta_fall, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
#  Состояние
# ---------------------------------------------------------------------------

class App:
    def __init__(self, args):
        self.args = args
        self.u_max = args.u_max if np.isfinite(args.u_max) else None
        self.u_finite = self.u_max if self.u_max is not None else 10.0
        self.u_preview = None
        self.cost = dict(lce.PRESETS[args.preset])
        self.mpc = dict(dt_plan=0.02, horizon_s=1.0, n_iter=2, first_iter=50)
        self.box = True
        self.ghost = True
        self.drag = None
        self.field = None               # ("w", i) | ("m", i)
        self.buffer, self.fresh, self.field_error = "", False, None

        self.selected = None
        self.traj = None                # MPC: (t, x, u)
        self.plans = []
        self.traj_lqr = None
        self.marks = {}                 # ключ параметров -> {(ix, iy): исход}
        self.runs = {}                  # ключ параметров -> {(ix, iy): pack_run(...)}
        self.mode = "lqr"               # "lqr" | "compare"
        self.pool, self.fill_futures, self.fill_total = None, {}, 0
        self.fill_step = args.fill_step
        self.frame_t, self.playing, self.speed = 0.0, True, 1.0
        self.worker, self.cancel, self.watch = None, threading.Event(), None
        self.status = ""
        self.K = None
        self.rebuild()

    # --- параметры --------------------------------------------------------------

    def params_key(self):
        return (self.u_max, tuple(sorted(self.cost.items())), tuple(sorted(self.mpc.items())),
                self.box, self.args.horizon)

    def rebuild(self):
        """Всё, что зависит от параметров: K, сетка, фон ЛКР, выбранный старт."""
        self.cancel_worker()
        if hasattr(self, "fill_futures"):
            self.stop_fill()
        u = self.u_max if self.u_max is not None else self.u_finite
        world = WheeledPendulum(u_max=u)
        th = THETA_MAP
        # Полоса восстановимости наклонена: у края карты она уходит к большим
        # |dtheta|. Граница по скорости берётся по всему отрезку theta, иначе
        # полоса вылезала бы за верх и низ карты.
        grid_th = np.linspace(-th, th, 401)
        floor, ceil = world.recoverable_bounds(u, grid_th)
        reach = np.abs(np.concatenate([floor, ceil]))
        dth = self.args.dtheta_max or MAP_FIT * float(reach[np.isfinite(reach)].max())
        n = self.args.grid
        self.thetas = np.linspace(-th, th, n)
        self.dthetas = np.linspace(-dth, dth, n)
        TH, DTH = np.meshgrid(self.thetas, self.dthetas, indexing="xy")
        self.X0 = np.zeros((n * n, 4))
        self.X0[:, 0], self.X0[:, 2] = TH.ravel(), DTH.ravel()
        self.n = n
        self.bg = None                  # посчитается в ensure_background (с сообщением на экране)
        # Сетка строится до синтеза: если Риккати не решится, окну всё равно
        # есть что рисовать, а ошибка пишется строкой.
        try:
            self.K, _ = lce.design(WheeledPendulum(), self.cost)
            self.design_error = None
        except Exception as exc:
            self.design_error = str(exc).split("\n")[0][:70]
            self.traj = self.traj_lqr = None
            return
        if self.selected is not None:
            self.start(*self.selected)

    def compute_background(self):
        self.bg = lce.compute_map(WheeledPendulum(), self.K, self.u_max, self.X0,
                                  self.args.horizon, self.args.theta_fall)

    # --- расчёт выбранного старта ---------------------------------------------------

    def cancel_worker(self):
        if self.worker is not None and self.worker.is_alive():
            self.cancel.set()
            self.worker.join()
        self.worker = None
        self.cancel = threading.Event()

    def load_run(self, run):
        # В кэше float32 (память), для рисования -- обычные float: pygame не
        # принимает numpy.float32 в координатах.
        self.traj = (run["t"], run["x"].astype(float), run["u"].astype(float))
        self.plans = [dict(t=p["t"], x=p["x"].astype(float), u=p["u"].astype(float))
                      for p in run["plans"]]
        self.status = run["status"]
        self.frame_t, self.playing = 0.0, True

    def start(self, ix, iy):
        self.cancel_worker()
        self.selected = (ix, iy)
        x0 = np.array([self.thetas[ix], 0.0, self.dthetas[iy], 0.0])
        world = WheeledPendulum(u_max=self.u_max)
        n_steps = int(round(self.args.horizon / DT))
        lq = rollout(world, LinearFeedbackController(self.K), RK4Integrator(), x0, DT, n_steps)
        self.traj_lqr = (lq.t, lq.x, lq.u[:, 0])
        self.traj, self.plans, self.frame_t = None, [], 0.0
        key, cell = self.params_key(), (ix, iy)
        cached = self.runs.get(key, {}).get(cell)
        if cached is not None:
            # Уже посчитано (кликом или заливкой) при этих же параметрах --
            # показываем сразу, без пересчёта.
            self.load_run(cached)
            self.status += "   (cached)"
            return
        self.status = "MPC: computing..."

        mpc, _ = build_mpc(self.cost, self.u_max, self.mpc, self.box, record_plans=True)
        self.watch = Watch(mpc, self.cancel)

        def job(watch=self.watch):
            t0 = time.perf_counter()
            try:
                tr = rollout(world, watch, RK4Integrator(), x0, DT, n_steps)
            except Cancelled:
                return
            except Exception as exc:        # показать, а не уронить окно
                self.status = f"MPC failed: {str(exc)[:80]}"
                return
            run = pack_run(tr, mpc.plans, self.args.theta_fall, time.perf_counter() - t0)
            self.runs.setdefault(key, {})[cell] = run
            self.marks.setdefault(key, {})[cell] = run["outcome"]
            if self.selected == cell and self.params_key() == key:
                self.load_run(run)

        self.worker = threading.Thread(target=job, daemon=True)
        self.worker.start()

    # --- заливка карты MPC ------------------------------------------------------

    def fill_cells(self):
        """Прореженная сетка: каждая step-я клетка. 41x41 целиком -- часы."""
        step = self.fill_step
        idx = list(range(0, self.n, step))
        if idx[-1] != self.n - 1:
            idx.append(self.n - 1)
        return [(ix, iy) for iy in idx for ix in idx]

    def start_fill(self):
        from concurrent.futures import ProcessPoolExecutor
        if self.pool is None:
            self.pool = ProcessPoolExecutor(max(1, (os.cpu_count() or 2) - 1))
        key = self.params_key()
        done = self.runs.get(key, {})
        todo = [c for c in self.fill_cells() if c not in done]
        for ix, iy in todo:
            x0 = np.array([self.thetas[ix], 0.0, self.dthetas[iy], 0.0])
            job = (x0, dict(self.cost), self.u_max, dict(self.mpc), self.box,
                   self.args.horizon, self.args.theta_fall)
            self.fill_futures[self.pool.submit(mpc_cell, job)] = (key, (ix, iy))
        self.fill_total = len(self.fill_cells())

    def stop_fill(self):
        """Отменить то, что ещё не началось. Уже идущие клетки досчитаются и
        лягут в журнал под своими параметрами -- пригодятся, если вернуться."""
        for f in self.fill_futures:
            f.cancel()

    def poll_fill(self):
        for f in [f for f in self.fill_futures if f.done()]:
            key, cell = self.fill_futures.pop(f)
            if not f.cancelled() and f.exception() is None:
                run = f.result()
                self.runs.setdefault(key, {})[cell] = run
                self.marks.setdefault(key, {})[cell] = run["outcome"]
                # Если этот старт выбран и ещё считается кликом -- отдать готовое.
                if self.selected == cell and key == self.params_key() and self.traj is None:
                    self.cancel_worker()
                    self.load_run(run)

    def fill_running(self):
        key = self.params_key()
        return any(k == key for k, _ in self.fill_futures.values())

    # --- поля ввода -------------------------------------------------------------

    def field_value(self, field):
        kind, i = field
        return (self.cost[lce.WEIGHTS[i][0]] if kind == "w" else self.mpc[MPC_FIELDS[i][0]])

    def open_field(self, field):
        self.field, self.buffer, self.fresh, self.field_error = field, f"{self.field_value(field):g}", True, None

    def commit_field(self):
        kind, i = self.field
        if kind == "w":
            key, _, zero_ok = lce.WEIGHTS[i]
            value, err = lce.parse_weight(self.buffer, zero_ok)
            target = self.cost
        else:
            key, _, typ, ok = MPC_FIELDS[i]
            try:
                value = typ(float(self.buffer))
                err = None if ok(value) else "out of range"
            except ValueError:
                value, err = None, f"'{self.buffer}' is not a number"
            target = self.mpc
        if err:
            self.field_error = f"{key}: {err}"
            return False
        self.field, self.field_error = None, None
        if value != target[key]:
            target[key] = value
            self.rebuild()
        return True

    # --- геометрия -----------------------------------------------------------------

    def to_px(self, theta, dtheta):
        x, y, w, h = MAP
        edge = self.n / max(self.n - 1, 1)
        ht, hd = self.thetas[-1] * edge, self.dthetas[-1] * edge
        return x + (theta + ht) / (2 * ht) * w, y + (1 - (dtheta + hd) / (2 * hd)) * h

    def cell_at(self, pos):
        """Клетка под курсором. В режиме сравнения -- ближайшая клетка заливки:
        там карта нарисована квадратами step x step, и клик внутри квадрата
        должен выбирать тот старт, которым квадрат закрашен, а не соседнюю
        мелкую клетку, для которой MPC ещё не считался."""
        x, y, w, h = MAP
        if not (x <= pos[0] < x + w and y <= pos[1] < y + h):
            return None
        ix = int((pos[0] - x) / w * self.n)
        iy = self.n - 1 - int((pos[1] - y) / h * self.n)
        if self.mode == "compare":
            idx = sorted({c[0] for c in self.fill_cells()})
            ix = min(idx, key=lambda i: abs(i - ix))
            iy = min(idx, key=lambda i: abs(i - iy))
        return ix, iy

    def field_rects(self):
        rects = []
        for i in range(len(lce.WEIGHTS)):
            rects.append((("w", i), (24 + i * 200 + 74, 44, 100, 22)))
        for i in range(len(MPC_FIELDS)):
            rects.append((("m", i), (24 + i * 200 + 94, 76, 80, 22)))
        return rects


# ---------------------------------------------------------------------------
#  Отрисовка
# ---------------------------------------------------------------------------

def u_to_px(u):
    x, _, w, _ = U_SLIDER
    return x + (np.clip(u, lce.U_MIN, lce.U_MAX) - lce.U_MIN) / (lce.U_MAX - lce.U_MIN) * w


def u_from_px(px):
    x, _, w, _ = U_SLIDER
    f = float(np.clip((px - x) / w, 0, 1))
    return lce.U_MIN + round(f * (lce.U_MAX - lce.U_MIN) / lce.U_STEP) * lce.U_STEP


def stride_for(ts, seconds):
    """Шаг прореживания, чтобы рисовать точку раз в `seconds` при любом шаге записи."""
    return max(1, int(round(seconds / max(ts[1] - ts[0], 1e-12))))


def at_time(traj, t):
    ts = traj[0]
    return min(int(np.searchsorted(ts, t)), len(ts) - 1)


def current_plan(app):
    """Последний план, начатый не позже текущего кадра."""
    best = None
    for p in app.plans:
        if p["t"] <= app.frame_t + 1e-9:
            best = p
        else:
            break
    return best


def draw_map(screen, pg, font, app):
    x, y, w, h = MAP
    pg.draw.rect(screen, PANEL, (x - 4, y - 4, w + 8, h + 8), border_radius=6)
    marks = app.marks.get(app.params_key(), {})
    if app.mode == "compare":
        title = compare_title(app, marks)
    else:
        title = "INITIAL CONDITIONS  background: LQR + clip,  dots: MPC runs"
    screen.blit(font.render(title, True, DIM), (x, y - 22))
    if app.bg is not None:
        n = app.n
        outcome, t_fall = app.bg
        img = np.zeros((n, n, 3), dtype=np.uint8)
        frac = np.where(np.isnan(t_fall), 1.0, t_fall / app.args.horizon).reshape(n, n)
        for code, base in FALL_COLOR.items():
            mask = (outcome == code).reshape(n, n)
            for c in range(3):
                img[..., c] = np.where(mask, np.clip(base[c] * (0.45 + 0.55 * frac), 0, 255),
                                       img[..., c])
        surf = pg.surfarray.make_surface(np.flipud(img).transpose(1, 0, 2))
        screen.blit(pg.transform.scale(surf, (w, h)), (x, y))
        veil = pg.Surface((w, h), pg.SRCALPHA)
        # В режиме сравнения фон ЛКР почти гаснет: на первом плане -- клетки,
        # где есть оба ответа.
        veil.fill(BG + ((215,) if app.mode == "compare" else (90,)))
        screen.blit(veil, (x, y))
    zx, zy = app.to_px(0.0, 0.0)
    pg.draw.line(screen, GRID_LINE, (zx, y), (zx, y + h))
    pg.draw.line(screen, GRID_LINE, (x, zy), (x + w, zy))
    for sgn in (+1, -1):                                 # горизонт корпуса, pi/2
        hx, _ = app.to_px(sgn * np.pi / 2, 0.0)
        for top in range(int(y), int(y + h), 11):
            pg.draw.line(screen, DIM, (hx, top), (hx, min(top + 6, y + h)))

    clip = screen.get_clip()
    screen.set_clip(MAP)
    cw, ch = w / app.n, h / app.n
    if app.mode == "compare" and app.bg is not None:
        # Клетка заливки представляет квадрат step x step вокруг себя: так
        # прореженная сетка читается как грубая карта, а не как россыпь точек.
        # Одиночные клики -- квадрат в одну клетку.
        fill = set(app.fill_cells())
        ordered = sorted(marks.items(), key=lambda kv: kv[0] not in fill)   # мелкие -- поверх
        for (ix, iy), code in ordered:
            span = app.fill_step if (ix, iy) in fill else 1
            color = diff_color(code, app.bg[0][iy * app.n + ix])
            px0 = x + (ix + 0.5 - span / 2) * cw
            py0 = y + (app.n - 1 - iy + 0.5 - span / 2) * ch
            pg.draw.rect(screen, color, (px0, py0, span * cw + 1, span * ch + 1))
    else:
        # кружки -- уже посчитанные MPC-старты при текущих параметрах
        for (ix, iy), code in marks.items():
            cx, cy = x + (ix + 0.5) * cw, y + (app.n - 1 - iy + 0.5) * ch
            r = max(3, int(min(cw, ch) / 3))
            pg.draw.circle(screen, FALL_COLOR[code], (int(cx), int(cy)), r)
            pg.draw.circle(screen, TEXT, (int(cx), int(cy)), r, 1)

    if app.u_max is not None:
        world = WheeledPendulum(u_max=app.u_max)
        th = np.linspace(app.thetas[0] * 1.05, app.thetas[-1] * 1.05, 800)
        floor, ceil = world.recoverable_bounds(app.u_max, th)
        for vals in (ceil, floor):
            pts = [app.to_px(a, b) for a, b in zip(th, vals) if abs(b) > 1e-12]
            if len(pts) > 1:
                pg.draw.lines(screen, LIMIT, False, pts, 2)

    def phase(traj, color, width, upto):
        k = at_time(traj, upto)
        st = stride_for(traj[0], 0.01)      # кэш хранит шаг 10 мс, ЛКР -- 1 мс
        pts = [app.to_px(a, b) for a, b in zip(traj[1][:k + 1:st, 0], traj[1][:k + 1:st, 2])]
        pts = [p for p in pts if abs(p[0]) < 1e5 and abs(p[1]) < 1e5]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, width)
        if pts:
            pg.draw.circle(screen, color, (int(pts[-1][0]), int(pts[-1][1])), 5)

    if app.ghost and app.traj_lqr is not None:
        phase(app.traj_lqr, GHOST, 2, app.frame_t)
    if app.traj is not None:
        phase(app.traj, CURVE, 2, app.frame_t)
        plan = current_plan(app)
        if plan is not None:
            pts = [app.to_px(a, b) for a, b in zip(plan["x"][:, 0], plan["x"][:, 2])]
            for a, b in zip(pts[:-1:2], pts[1::2]):          # пунктир
                pg.draw.line(screen, PLAN, a, b, 2)
    screen.set_clip(clip)

    if app.selected is not None:
        ix, iy = app.selected
        span = app.fill_step if app.mode == "compare" and (ix, iy) in set(app.fill_cells()) else 1
        pg.draw.rect(screen, ACCENT, (x + (ix + 0.5 - span / 2) * cw,
                                      y + (app.n - 1 - iy + 0.5 - span / 2) * ch,
                                      max(span * cw, 3), max(span * ch, 3)), 2)
    for text, tx, ty in ((f"{app.thetas[0]:+.2f}", x, y + h + 6),
                         (f"{app.thetas[-1]:+.2f}", x + w - 40, y + h + 6),
                         (f"{app.dthetas[-1]:+.1f}", x - 48, y),
                         (f"{app.dthetas[0]:+.1f}", x - 48, y + h - 14)):
        screen.blit(font.render(text, True, DIM), (tx, ty))


ONLY_MPC, ONLY_LQR, BOTH, NEITHER = ACCENT, GHOST, (58, 132, 196), (48, 51, 58)


def diff_color(mpc_code, lqr_code):
    m, l = mpc_code == HELD, lqr_code == HELD
    return BOTH if m and l else ONLY_MPC if m else ONLY_LQR if l else NEITHER


def compare_counts(app, marks):
    out = dict(both=0, mpc=0, lqr=0, neither=0)
    if app.bg is None:
        return out
    for (ix, iy), code in marks.items():
        m, l = code == HELD, app.bg[0][iy * app.n + ix] == HELD
        out["both" if m and l else "mpc" if m else "lqr" if l else "neither"] += 1
    return out


def compare_title(app, marks):
    c = compare_counts(app, marks)
    return (f"MPC vs LQR, {len(marks)} starts: both {c['both']} | only MPC {c['mpc']} | "
            f"only LQR {c['lqr']} | none {c['neither']}")


def draw_robot(screen, pg, font, app):
    x, y, w, h = ANIM
    pg.draw.rect(screen, PANEL, ANIM, border_radius=6)
    screen.blit(font.render("PLANT  white: MPC,  teal: LQR", True, DIM), (x, y - 22))
    p = WheeledPendulum().p
    wheel_px = 22.0
    m2px = wheel_px / p.r
    cy = y + h * 0.78
    pg.draw.line(screen, GRID_LINE, (x + 8, cy + wheel_px), (x + w - 8, cy + wheel_px), 2)

    def body(state, color, cx):
        th, ph = state[0], state[1]
        if not np.isfinite(state).all():
            return
        pg.draw.circle(screen, color, (int(cx), int(cy)), int(wheel_px), 2)
        pg.draw.line(screen, color, (cx, cy),
                     (cx + wheel_px * np.sin(ph), cy - wheel_px * np.cos(ph)), 2)
        tip = (cx + p.l * m2px * 0.8 * np.sin(th), cy - p.l * m2px * 0.8 * np.cos(th))
        pg.draw.line(screen, color, (cx, cy), tip, 5)
        pg.draw.circle(screen, color, (int(tip[0]), int(tip[1])), 7)

    if app.ghost and app.traj_lqr is not None:
        body(app.traj_lqr[1][at_time(app.traj_lqr, app.frame_t)], GHOST, x + w * 0.3)
    if app.traj is not None:
        body(app.traj[1][at_time(app.traj, app.frame_t)], CURVE, x + w * 0.7)
    elif app.selected is not None:
        prog = app.watch.t / app.args.horizon if app.watch else 0.0
        bx, by = x + w * 0.5 - 120, y + h * 0.4
        pg.draw.rect(screen, BTN, (bx, by, 240, 14), border_radius=4)
        pg.draw.rect(screen, ACCENT, (bx, by, 240 * prog, 14), border_radius=4)
        screen.blit(font.render(f"MPC t = {app.watch.t:.2f} / {app.args.horizon:g} s",
                                True, TEXT), (bx, by - 20))
    screen.blit(font.render(f"t = {app.frame_t:5.2f} s   x{app.speed:g}"
                            f"{'' if app.playing else '  paused'}", True, TEXT),
                (x + 10, y + 8))


def draw_plot(screen, pg, font, app, rect, title, index, limit=None):
    x, y, w, h = rect
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    screen.blit(font.render(title, True, DIM), (x + 8, y - 18))
    T = app.args.horizon
    series = []
    if app.ghost and app.traj_lqr is not None:
        series.append((app.traj_lqr, GHOST))
    if app.traj is not None:
        series.append((app.traj, CURVE))
    if not series:
        return
    vals = []
    for (t, X, U), _ in series:
        ys = U if index is None else X[:, index]
        vals.append(ys[np.isfinite(ys)])
    lo = min(v.min() for v in vals if v.size)
    hi = max(v.max() for v in vals if v.size)
    if limit is not None:
        lo, hi = min(lo, -limit), max(hi, limit)
    if index == 0:
        lo, hi = max(lo, -1.1 * app.args.theta_fall), min(hi, 1.1 * app.args.theta_fall)
    pad = 0.08 * max(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad

    def px(tv, yv):
        return (x + 8 + tv / T * (w - 16),
                y + h - 6 - (np.clip(yv, lo, hi) - lo) / (hi - lo) * (h - 12))

    if lo < 0 < hi:
        pg.draw.line(screen, GRID_LINE, px(0, 0), px(T, 0))
    if limit is not None:
        for s in (+1, -1):
            a, b = px(0, s * limit), px(T, s * limit)
            for k in range(int(a[0]), int(b[0]), 10):
                pg.draw.line(screen, LIMIT, (k, a[1]), (min(k + 5, b[0]), a[1]))
    for (t, X, U), color in series:
        ys = U if index is None else X[:, index]
        n = len(ys)
        st = stride_for(t, 0.02)
        pts = [px(a, b) for a, b in zip(t[:n:st], ys[::st]) if np.isfinite(b)]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 2)
    # план текущего такта
    plan = current_plan(app) if app.traj is not None else None
    if plan is not None and index != 1:
        ys = plan["u"] if index is None else plan["x"][:, index]
        dtp = app.plans[1]["t"] - app.plans[0]["t"] if len(app.plans) > 1 else app.mpc["dt_plan"]
        ts = plan["t"] + dtp * np.arange(len(ys))
        pts = [px(a, b) for a, b in zip(ts, ys) if a <= T]
        for a, b in zip(pts[:-1:2], pts[1::2]):
            pg.draw.line(screen, PLAN, a, b, 2)
    cx, _ = px(app.frame_t, 0)
    pg.draw.line(screen, ACCENT, (cx, y + 2), (cx, y + h - 2), 1)
    screen.blit(font.render(f"{hi:+.2f}", True, DIM), (x + w - 64, y + 2))
    screen.blit(font.render(f"{lo:+.2f}", True, DIM), (x + w - 64, y + h - 18))


def draw(screen, pg, font, app, buttons):
    screen.fill(BG)
    buttons.items = []

    # u_max
    x, y, w, h = U_SLIDER
    screen.blit(font.render("u_max", True, DIM), (24, y - 1))
    shown = app.u_preview if app.u_preview is not None else app.u_finite
    off = app.u_max is None and app.u_preview is None
    pg.draw.rect(screen, BTN, (x, y, w, h), border_radius=4)
    pg.draw.rect(screen, lce.shade(BTN_ON, 0.45 if off else 1.0),
                 (x, y, max(u_to_px(shown) - x, 1), h), border_radius=4)
    pg.draw.circle(screen, DIM if off else TEXT, (int(u_to_px(shown)), int(y + h / 2)), 7)
    screen.blit(font.render("  off" if off else f"{shown:5.2f} N*m", True, TEXT), (x + w + 14, y - 1))
    buttons.add(screen, pg, font, (x + w + 110, y - 4, 84, 22), "no limit", app.u_max is None,
                lambda: set_u(app, None if app.u_max is not None else app.u_finite))
    buttons.add(screen, pg, font, (x + w + 210, y - 4, 60, 22), "limit", app.box,
                lambda: toggle(app, "box", True))
    buttons.add(screen, pg, font, (x + w + 276, y - 4, 60, 22), "blind", not app.box,
                lambda: toggle(app, "box", False))
    buttons.add(screen, pg, font, (x + w + 350, y - 4, 94, 22), "LQR ghost", app.ghost,
                lambda: setattr(app, "ghost", not app.ghost))
    buttons.add(screen, pg, font, (x + w + 460, y - 4, 90, 22), "map: LQR", app.mode == "lqr",
                lambda: setattr(app, "mode", "lqr"))
    buttons.add(screen, pg, font, (x + w + 556, y - 4, 140, 22), "map: MPC vs LQR",
                app.mode == "compare", lambda: setattr(app, "mode", "compare"))
    running = app.fill_running()
    done = sum(1 for c in app.fill_cells() if c in app.marks.get(app.params_key(), {}))
    label = (f"fill MPC {done}/{len(app.fill_cells())}" if running or done
             else f"fill MPC ({len(app.fill_cells())} starts)")
    buttons.add(screen, pg, font, (x + w + 702, y - 4, 170, 22), label, running,
                lambda: (app.stop_fill() if app.fill_running() else
                         (app.start_fill(), setattr(app, "mode", "compare"))))
    px_ = 24
    screen.blit(font.render("presets", True, DIM), (px_, 111))
    px_ += 70
    for name in lce.PRESETS:
        bw = 14 + 8 * len(name)
        buttons.add(screen, pg, font, (px_, 108, bw, 22), name, app.cost == lce.PRESETS[name],
                    lambda nm=name: apply_preset(app, nm))
        px_ += bw + 4

    # поля
    for field, rect in app.field_rects():
        kind, i = field
        label = lce.WEIGHTS[i][1] if kind == "w" else MPC_FIELDS[i][1]
        lx = rect[0] - (74 if kind == "w" else 94)
        screen.blit(font.render(label, True, TEXT if kind == "w" else ACCENT), (lx, rect[1] + 3))
        active = app.field == field
        pg.draw.rect(screen, PANEL if active else BTN, rect, border_radius=4)
        border = (205, 88, 62) if active and app.field_error else ACCENT if active else GRID_LINE
        pg.draw.rect(screen, border, rect, 2 if active else 1, border_radius=4)
        text = app.buffer if active else f"{app.field_value(field):g}"
        screen.blit(font.render(text, True, TEXT), (rect[0] + 6, rect[1] + 3))
    hint = (app.field_error or "Enter apply   Tab next   Esc cancel") if app.field else ""
    if app.design_error:
        hint = f"LQR design failed: {app.design_error}"
    screen.blit(font.render(hint, True, (205, 88, 62) if app.field_error or app.design_error
                            else DIM), (24 + 4 * 200 + 20, 79))

    line = app.status
    if app.selected is not None:
        ix, iy = app.selected
        line = f"theta0 = {app.thetas[ix]:+.3f}, dtheta0 = {app.dthetas[iy]:+.2f}   {line}"
    screen.blit(font.render(line, True, TEXT), (24, 138))
    legend = "dashed orange: MPC plan made at the last tact (where it expected to go)"
    screen.blit(font.render(legend, True, PLAN), (24, 158))
    if app.mode == "compare":
        lx = 640
        for text, color in (("both held", BOTH), ("only MPC", ONLY_MPC),
                            ("only LQR", ONLY_LQR), ("neither", NEITHER)):
            pg.draw.rect(screen, color, (lx, 160, 12, 12))
            screen.blit(font.render(text, True, DIM), (lx + 16, 158))
            lx += 34 + 8 * len(text)

    draw_map(screen, pg, font, app)
    draw_robot(screen, pg, font, app)
    draw_plot(screen, pg, font, app, PLOTS[0], "theta(t), rad", 0)
    draw_plot(screen, pg, font, app, PLOTS[1], "u(t), N*m  dashed green: +-u_max", None, app.u_max)
    draw_plot(screen, pg, font, app, PLOTS[2], "phi(t), rad  (wheel)", 1)
    if app.bg is None:
        screen.blit(font.render("computing LQR background...", True, ACCENT), (MAP[0] + 10, MAP[1] + 10))


def set_u(app, u):
    app.u_max = u
    if u is not None:
        app.u_finite = u
    app.rebuild()


def toggle(app, attr, value):
    if getattr(app, attr) != value:
        setattr(app, attr, value)
        app.rebuild()


def apply_preset(app, name):
    app.cost = dict(lce.PRESETS[name])
    app.rebuild()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--u-max", type=float, default=10.0)
    ap.add_argument("--preset", default="base", choices=list(lce.PRESETS))
    ap.add_argument("--grid", type=int, default=41)
    ap.add_argument("--horizon", type=float, default=5.0, help="длина прогона, с")
    ap.add_argument("--dtheta-max", type=float, default=None,
                    help="граница по dtheta; по умолчанию -- по множеству восстановимости")
    ap.add_argument("--theta-fall", type=float, default=float(np.pi))
    ap.add_argument("--fill-step", type=int, default=4,
                    help="заливка MPC: каждая k-я клетка сетки (41 и 4 -> 11x11 = 121 старт)")
    ap.add_argument("--fill", action="store_true", help="сразу начать заливку MPC")
    ap.add_argument("--select", type=float, nargs=2, default=None, metavar=("THETA0", "DTHETA0"))
    ap.add_argument("--screenshot", default=None, help="дождаться MPC, снять кадр на --shot-t и выйти")
    ap.add_argument("--shot-t", type=float, default=1.0)
    args = ap.parse_args()

    if args.screenshot:
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    import pygame as pg

    pg.init()
    screen = pg.display.set_mode((W, H))
    pg.display.set_caption("MPC explorer")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 14)
    app = App(args)
    buttons = lce.Buttons()

    def ensure_background():
        if app.bg is None and app.design_error is None:
            draw(screen, pg, font, app, buttons)
            pg.display.flip()
            app.compute_background()

    ensure_background()
    if args.select is not None:
        app.start(int(np.argmin(np.abs(app.thetas - args.select[0]))),
                  int(np.argmin(np.abs(app.dthetas - args.select[1]))))

    if args.fill:
        app.start_fill()
        app.mode = "compare"

    if args.screenshot:
        if app.worker is not None:
            app.worker.join()
        while app.fill_futures:
            time.sleep(0.5)
            app.poll_fill()
        app.frame_t, app.playing = args.shot_t, False
        draw(screen, pg, font, app, buttons)
        pg.image.save(screen, args.screenshot)
        return

    clock = pg.time.Clock()
    while True:
        dt_wall = clock.tick(30) / 1000.0
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                app.cancel_worker()
                app.stop_fill()
                return
            if app.field is not None:
                if ev.type == pg.TEXTINPUT:
                    chars = "".join(c for c in ev.text if c in lce.FIELD_CHARS)
                    if chars:
                        app.buffer = chars if app.fresh else app.buffer + chars
                        app.fresh, app.field_error = False, None
                    continue
                if ev.type == pg.KEYDOWN:
                    if ev.key in (pg.K_RETURN, pg.K_KP_ENTER):
                        app.commit_field()
                    elif ev.key == pg.K_ESCAPE:
                        app.field = app.field_error = None
                    elif ev.key == pg.K_BACKSPACE:
                        app.buffer = "" if app.fresh else app.buffer[:-1]
                        app.fresh, app.field_error = False, None
                    elif ev.key == pg.K_TAB:
                        order = [f for f, _ in app.field_rects()]
                        i = order.index(app.field)
                        if app.commit_field():
                            step = -1 if ev.mod & pg.KMOD_SHIFT else 1
                            app.open_field(order[(i + step) % len(order)])
                    continue
                if ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                    hit = next((f for f, r in app.field_rects() if pg.Rect(r).collidepoint(ev.pos)), None)
                    if hit == app.field or not app.commit_field():
                        continue
                    if hit is not None:
                        app.open_field(hit)
                        continue
            if ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    app.cancel_worker()
                    app.stop_fill()
                    return
                if ev.key == pg.K_SPACE:
                    app.playing = not app.playing
                if ev.key == pg.K_r:
                    app.frame_t, app.playing = 0.0, True
                if ev.key == pg.K_COMMA:
                    app.speed = max(0.125, app.speed / 2)
                if ev.key == pg.K_PERIOD:
                    app.speed = min(8.0, app.speed * 2)
                if ev.key == pg.K_l:
                    set_u(app, None if app.u_max is not None else app.u_finite)
                if ev.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    step = lce.U_STEP if ev.key == pg.K_RIGHTBRACKET else -lce.U_STEP
                    set_u(app, float(np.clip(app.u_finite + step, lce.U_MIN, lce.U_MAX)))
            elif ev.type == pg.MOUSEBUTTONDOWN and ev.button == 1:
                x, y, w, h = U_SLIDER
                if x - 8 <= ev.pos[0] <= x + w + 8 and y - 8 <= ev.pos[1] <= y + h + 8:
                    app.drag, app.u_preview = "u_max", u_from_px(ev.pos[0])
                    continue
                hit = next((f for f, r in app.field_rects() if pg.Rect(r).collidepoint(ev.pos)), None)
                if hit is not None:
                    app.open_field(hit)
                    continue
                if buttons.click(ev.pos):
                    continue
                cell = app.cell_at(ev.pos)
                if cell is not None:
                    app.start(*cell)
            elif ev.type == pg.MOUSEMOTION and app.drag == "u_max":
                app.u_preview = u_from_px(ev.pos[0])
            elif ev.type == pg.MOUSEBUTTONUP and ev.button == 1 and app.drag == "u_max":
                app.drag = None
                u, app.u_preview = app.u_preview, None
                set_u(app, u)
        ensure_background()
        app.poll_fill()
        if app.traj is not None and app.playing:
            app.frame_t += dt_wall * app.speed
            if app.frame_t >= app.args.horizon:
                app.frame_t, app.playing = app.args.horizon, False
        elif app.traj is None and app.traj_lqr is not None and app.watch is not None:
            app.frame_t = app.watch.t          # пока MPC считается, ЛКР-призрак идёт вровень
        draw(screen, pg, font, app, buttons)
        pg.display.flip()


if __name__ == "__main__":
    main()
