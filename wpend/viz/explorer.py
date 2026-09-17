"""Окно исследователя: карта начальных условий слева, робот справа.

    python -m wpend.viz.explorer                 # сетка 21x21 считается заранее
    python -m wpend.viz.explorer --grid 41
    python -m wpend.viz.explorer --no-precompute # считать по одной по клику
    python -m wpend.viz.explorer --main bang-ell --compare lqr

Два регулятора одновременно: ОСНОВНОЙ красит карту и рисуется как раньше,
СРАВНЕНИЕ считается только для выбранной клетки и рисуется полупрозрачным
поверх -- робот-призрак, вторая фазовая кривая, вторые графики. Так видно не
«какой лучше в среднем», а что именно эти двое делают из ОДНОГО состояния.

Управление: клик по клетке -- проиграть её траекторию; кнопки сверху либо
1..8 (основной) и Shift+1..8 (сравнение, Shift+0 -- выключить); слайдер u_max
или клавиши [ и ]; кнопка settings или S -- панель цен ЛКР (Q, R), eps и
горизонта; ПРОБЕЛ -- пауза; R -- сначала; Esc -- выход.

Раскладка окна -- фиксированные 1280x940 (см. W, H и прямоугольники ниже).
Если экран меньше, окно не перевёрстывается, а ужимается целиком: рисунок
идёт на холст своего размера, а в окно кладётся масштабированная копия. Окно
можно тянуть мышью, пропорция при этом сохраняется. Скриншоты (--screenshot)
снимаются с холста и всегда 1280x940, иначе их было бы не сравнить.

Предел момента можно и выключить (кнопка `no limit`, клавиша L, `--u-max inf`):
мотор становится безграничным, и видно, чего стоит «может всё» -- ЛКР удерживает
всю карту, запрашивая около 175 Н*м при весе корпуса D = 29.4 Н*м. Реле в этом
режиме не определено (u = +-u_max), и его варианты окно отключает.

Предел момента -- не декорация, а параметр, от которого зависят СРАЗУ ДВЕ
разные вещи: множество восстановимых состояний (свойство системы, замкнутая
форма) и сертифицированный уровень c* (свойство проекта, считается численно).
Поэтому слайдер во время перетаскивания двигает только первое -- это даром, --
а второе и сетку пересчитывает на отпускании. Границы карты -- по стандарту
проекта: theta до ±1.3·pi/2, dtheta -- 1.3 от восстановимого множества на
[-pi/2, pi/2], см. Explorer._map_bounds.

Модуль читает только Trajectory / TrajectoryBatch: ни одной формулы динамики
здесь нет. Регуляторы он не изобретает, а собирает из готовых -- см.
CONTROLLERS ниже.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

from ..controller import (
    BangBangLQRController,
    LinearFeedbackController,
    ellipsoid_region,
    grid_region,
    theta_band_region,
)
from ..estimator import ComplementaryEstimator, optimal_tau
from ..integrator import RK4Integrator
from ..models import WheeledPendulum
from ..rollout import rollout_many
from ..sensor import EncoderSensor, IMUSensor, StackedSensor
from .grid import FELL_BACKWARD, FELL_FORWARD, HELD, GridSpec, classify

# ЛКР для линеаризации в верхнем положении, Q = diag(100, 1, 10, 1), R = 1.
# Синтез (wpend.lqr.lqr) требует scipy и вызывается только при старте окна;
# число здесь -- страховка и одновременно проверка, что параметры модели не
# уехали: если синтез даст другое K, окно скажет об этом вслух.
K_LQR = np.array([[95.122221, 1.0, 19.807871, 1.449624]])

I_THETA, I_PHI, I_DTHETA, I_DPHI = 0, 1, 2, 3

W, H = 1280, 940
PICK = (24, 44)                     # x, y первой строки кнопок
PICK_ROW = 27
SLIDER = (132, 129, 300, 14)        # дорожка u_max: x, y, w, h
NOISE_Y = 156                       # ряд слайдеров датчика и фильтра
MAP = (56, 214, 568, 632)           # x, y, w, h
ANIM = (656, 214, 600, 316)
PLOT = (656, 546, 600, 300)

#: Слайдеры датчика и фильтра: ключ, подпись, границы, логарифмическая ли
#: шкала, формат.  Шум задаётся ПЛОТНОСТЬЮ и меняется на порядки, поэтому
#: линейная шкала для него бессмысленна: почти весь ход ручки пришёлся бы на
#: значения, при которых уже ничего не видно.
NOISE_SPECS = [
    ("sigma_g", "sig_g", 1e-6, 1e-2, True, "{:.1e}"),
    ("sigma_a", "sig_a", 1e-5, 1e-1, True, "{:.1e}"),
    ("b0_g", "b0_g", 1e-4, 1e-1, True, "{:.1e}"),
    ("tau", "tau", 1e-2, 3.0, True, "{:.3f}"),
]
NOISE_TRACK = 104                   # длина дорожки одного слайдера
NOISE_STEP = 96                     # x-шаг между блоками сверх дорожки

# Диапазон слайдера. Сверху 10 Н*м: там седло уже за theta_fall, и карта
# перестаёт следовать за восстановимым множеством -- окно говорит об этом
# вслух. Снизу 0.25 -- режим, где восстановимо почти ничто. Шаг общий для мыши
# и клавиатуры, иначе один и тот же замер не повторить.
U_MIN, U_MAX, U_STEP = 0.25, 10.0, 0.25

#: Ключи вариантов, которым нужен КОНЕЧНЫЙ предел: реле подаёт u = +-u_max и
#: переключается по сепаратрисе при том же пределе. Без предела не определено
#: ни то, ни другое, поэтому окно их отключает -- как отключает эллипсоид без
#: scipy: кнопка перечёркнута, и рядом написано, чего не хватает.
RELAY_KEYS = ("bang", "bang-ell", "bang-ell-phi", "bang-tilt", "bang-map",
              "bang-eps", "bang-eps-ell")

#: Цена ЛКР второй фазы у `bang-eps`: веса колеса в сто раз меньше базовых.
#: Замер (PROPOSALS.md A26, карта 31x31, theta +-pi/2): с этой ценой и ЛКР, и
#: реле -> |theta| < eps -> ЛКР удерживают всё восстановимое при u_max = 1.5, 3, 10;
#: с базовой ценой реле -> eps теряет 4 и 10 клеток при 3 и 10 Н*м.
Q_SOFT_WHEEL = np.diag([100.0, 1e-2, 10.0, 1e-2])

#: Цены ЛКР, с которыми окно стартует: (q_theta, q_phi, q_dtheta, q_dphi, R).
#: Панель настроек (кнопка `settings`, клавиша S) меняет КОПИЮ -- app.cost, --
#: а эти числа остаются опорой: кнопка `defaults` возвращает ровно их.
#: base -- K для `lqr`, `bang-ell*`, `bang-map` и третьей фазы `bang-eps-ell`;
#: soft -- K второй фазы `bang-eps*`. ЛКР наклона (`bang-tilt`) панель не трогает.
DEFAULT_COST = {
    "base": (100.0, 1.0, 10.0, 1.0, 1.0),
    "soft": (100.0, 1e-2, 10.0, 1e-2, 1.0),
}
COST_LABELS = ("q_theta", "q_phi", "q_dtheta", "q_dphi", "R")

#: Во сколько раз карта шире восстановимого множества. 1.0 -- граница ровно по
#: краю, и не видно, что снаружи неё карта обязана быть красной у любого
#: регулятора; 1.3 оставляет поля, в которых это видно.
MAP_FIT = 1.3
#: Горизонт корпуса. Отмечается на карте пунктиром, когда попадает в границы.
HORIZON_ANGLE = float(np.pi / 2)
#: Стандарт карт (CLAUDE.md, Глеб 16.09 и 17.09): интересен theta в
#: [-pi/2, pi/2], сетка берётся с запасом MAP_FIT по обеим осям.
THETA_MAP = HORIZON_ANGLE

BG = (22, 24, 28)
PANEL = (33, 36, 42)
GRID_LINE = (52, 56, 64)
TEXT = (216, 219, 224)
DIM = (140, 146, 156)
ACCENT = (240, 198, 92)
CURVE = (245, 247, 250)

# Сравнение всюду рисуется этим цветом и этой прозрачностью -- один цвет на
# все три панели, чтобы глаз связывал призрака робота, вторую фазовую кривую и
# вторые графики в одного и того же «второго».
GHOST = (104, 214, 200)
GHOST_A = 132
LIMIT = (170, 255, 120)     # граница восстановимости -- свойство системы
#: Оценка состояния. Свой цвет, не GHOST: призрак -- это ДРУГОЙ регулятор на
#: той же истине, а оценка -- то же самое состояние, увиденное фильтром.
#: Путать эти две вещи одним цветом было бы хуже, чем не рисовать вовсе.
ESTIMATE = (235, 130, 180)
#: Оцениватель СРАВНЕНИЯ: его истинная траектория. Отдельный цвет от GHOST --
#: тот занят вторым РЕГУЛЯТОРОМ, и путать «другой закон» с «другой оценкой»
#: одним цветом значило бы сделать обе кнопки бесполезными.
GHOST_EST = (190, 150, 245)
#: Метки оптимального tau. Их две, и это принципиально: аналитическая
#: считает ошибку акселерометра БЕЛОЙ, а измеренная берёт её как есть, вместе
#: с кажущейся вертикалью. Расстояние между метками -- и есть вклад того, чего
#: в формуле нет.
TAU_MODEL = (240, 198, 92)
TAU_MEASURED = (120, 220, 160)
BTN = (44, 48, 56)
BTN_ON = (58, 132, 196)

OUTCOME_COLOR = {
    HELD: (58, 132, 196),
    FELL_FORWARD: (205, 88, 62),
    FELL_BACKWARD: (150, 108, 190),
}
UNKNOWN = (58, 62, 70)


def _cost_matrices(cost):
    """(q_theta, q_phi, q_dtheta, q_dphi, R) -> (Q, R) для `wpend.lqr.lqr`.
    Порядок весов -- порядок состояния (I_THETA, I_PHI, I_DTHETA, I_DPHI)."""
    q = np.zeros(4)
    q[[I_THETA, I_PHI, I_DTHETA, I_DPHI]] = cost[:4]
    return np.diag(q), np.array([[float(cost[4])]])


#: Поля панели настроек в порядке обхода по Tab: сначала столбец base, потом
#: soft, потом eps и горизонт.
SETTINGS_KEYS = ([f"base.{i}" for i in range(5)] + [f"soft.{i}" for i in range(5)]
                 + ["eps", "horizon"])
SETTINGS = (300, 190, 680, 520)     # панель настроек: x, y, w, h
#: Символы, которые принимает поле ввода: число в любой записи Python.
SETTINGS_CHARS = set("0123456789.eE-+")


def _finite_or_none(value):
    """Число -> число, inf/nan/None -> None («предела нет»).

    Одно место перевода: командная строка знает про `--u-max inf`, модель --
    про `u_max=None`, и стыкуются они здесь, а не в пяти местах по вкусу.
    """
    if value is None:
        return None
    value = float(value)
    return None if not np.isfinite(value) else value


def _finite(ts, ys):
    """Оставить пары (t, y) с конечным y.

    Без предела момента ЛКР -- это ЛИНЕЙНЫЙ закон, продолженный на все углы:
    перевалившись за горизонт, корпус получает момент, который его же и
    раскручивает, и угловые клетки карты уходят в переполнение за секунды
    (замер: |u| до 2e6 против 175 в остальных). Классификация это ловит --
    theta пересекает theta_fall задолго до inf, и клетка честно красится
    «упал», -- но рисовать по inf нельзя: масштаб оси станет nan, а int(nan)
    в pygame -- исключение. Поэтому кривые строятся по конечным точкам, а сам
    факт расхождения окно пишет словами.
    """
    ts, ys = np.asarray(ts, dtype=float), np.asarray(ys, dtype=float)
    m = np.isfinite(ts) & np.isfinite(ys)
    return ts[m], ys[m]


def _fmt(v):
    """Число для подписи оси. Без предела момента ЛКР успевает запросить
    8e9 Н*м, и `%+.2f` выезжает за панель на пол-экрана: крупные величины
    печатаем мантиссой с порядком."""
    v = float(v)
    if not np.isfinite(v):                           # pragma: no cover
        return "  n/a"
    return f"{v:+.2f}" if abs(v) < 1e3 else f"{v:+.3g}"


def _span(*arrays):
    """(min, max) по конечным значениям всех массивов; (0, 1), если их нет."""
    vals = np.concatenate([np.asarray(a, dtype=float).ravel() for a in arrays])
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:                               # pragma: no cover
        return 0.0, 1.0
    return float(vals.min()), float(vals.max())


def _shade(color, factor):
    return tuple(int(np.clip(c * factor, 0, 255)) for c in color)


# ---------------------------------------------------------------------------
#  Какие регуляторы предлагает окно
# ---------------------------------------------------------------------------
#  Каждая запись -- ключ для командной строки, подпись на кнопке и фабрика,
#  собирающая готовый Controller. Предел момента фабрики берут у окна
#  (app.u_max), а не у командной строки: слайдер меняет именно его, и реле
#  обязано просить ровно столько, сколько даёт мотор. Ничего нового здесь не изобретается: реле по
#  сепаратрисе и оба предиката региона живут в wpend/controller.py, синтез K, P
#  и уровня c* -- в wpend/lqr.py. Окно только выбирает.

def _never(x):
    """Регион, в который нельзя войти: изолирует чистую релейную фазу."""
    return np.zeros(np.shape(x)[:-1], dtype=bool)


CONTROLLERS = [
    ("lqr", "LQR",
     lambda app: LinearFeedbackController(app.K)),
    ("bang", "bang-bang",
     lambda app: BangBangLQRController(app.K, app.u_max, _never, app.system)),
    ("bang-ell", "bang -> ell",
     lambda app: BangBangLQRController(app.K, app.u_max,
                                       ellipsoid_region(app.P, app.c_star),
                                       app.system)),
    ("bang-ell-phi", "bang -> ell+phi",
     lambda app: BangBangLQRController(app.K, app.u_max,
                                       ellipsoid_region(app.P, app.c_star),
                                       app.system, wheel_ref=True)),
    ("bang-tilt", "bang -> tilt",
     lambda app: BangBangLQRController(app.K_tilt, app.u_max,
                                       ellipsoid_region(app.P_tilt, app.c_tilt),
                                       app.system)),
    ("bang-map", "bang -> map",
     lambda app: BangBangLQRController(app.K, app.u_max,
                                       grid_region(app.lqr_mask(), app.spec.thetas,
                                                   app.spec.dthetas),
                                       app.system)),
    # Реле сразу, передача по |theta| < eps (eps -- --eps), дальше ЛКР с
    # пониженными гейнами по phi, dphi. Критерий без сертификата, поэтому
    # K_soft, а не K: базовый ЛКР после такой передачи роняет корпус чаще.
    ("bang-eps", "bang -> eps",
     lambda app: BangBangLQRController(app.K_soft, app.u_max,
                                       theta_band_region(app.args.eps),
                                       app.system)),
    # То же, плюс третья фаза (A27): мягкий ЛКР отдаёт управление базовому K,
    # когда состояние входит в ЕГО сертифицированный эллипсоид x^T P x <= c*.
    # c* окно и так пересчитывает под текущий u_max (_certify), поэтому пара
    # «критерий + регулятор» здесь честная.
    ("bang-eps-ell", "bang -> eps -> ell",
     lambda app: BangBangLQRController(app.K_soft, app.u_max,
                                       theta_band_region(app.args.eps),
                                       app.system, K_final=app.K,
                                       final_region=ellipsoid_region(app.P, app.c_star))),
]
CONTROLLER_KEYS = [key for key, _, _ in CONTROLLERS]
CONTROLLER_LABEL = {key: label for key, label, _ in CONTROLLERS}
CONTROLLER_BUILD = {key: build for key, _, build in CONTROLLERS}


# ---------------------------------------------------------------------------
#  Какие оцениватели предлагает окно
# ---------------------------------------------------------------------------
#  Запись: ключ, подпись, фабрика ДАТЧИКА, фабрика ОЦЕНИВАТЕЛЯ.  None у обеих
#  фабрик -- идеальный вариант: rollout сам подставит FullStateSensor и
#  PassthroughEstimator, то есть регулятор увидит истину побитово.  Это опора
#  сравнения: разница карт с ней и есть цена оценки.
#
#  Колесо при одном ИДУ НЕНАБЛЮДАЕМО (docs/imu_noise.md, ER-008), поэтому все
#  неидеальные варианты идут с wheel="zero": выдуманное число в оценку колеса
#  не кладётся вовсе.  Законно это ровно с регулятором, у которого нули в
#  столбцах phi и dphi (bang-tilt); про остальные окно предупреждает строкой.


def _imu(app):
    return IMUSensor(app.system, dt=app.args.dt, mode="imu", seed=app.args.seed,
                     sigma_g=app.noise["sigma_g"], sigma_a=app.noise["sigma_a"],
                     b0_g=app.noise["b0_g"], b0_a=app.noise["b0_g"] * 9.8)


def _imu_enc(app):
    """ИДУ плюс моторный энкодер. Порядок -- часть контракта оценивателя."""
    return StackedSensor(_imu(app),
                         EncoderSensor(app.system, dt=app.args.dt,
                                       counts_per_rev=app.args.counts))


def _comp(tau=None, wheel="zero"):
    """tau=None -- взять со слайдера; число -- вырожденный режим (0 или inf)."""
    return lambda app: ComplementaryEstimator(
        app.system, dt=app.args.dt,
        tau=app.noise["tau"] if tau is None else tau, wheel=wheel)


ESTIMATORS = [
    ("ideal", "ideal", None, None),
    ("comp", "complementary", _imu, _comp()),
    # Вырожденные концы шкалы tau держим отдельными кнопками: слайдер до 0 и
    # до бесконечности не доезжает, а именно они показывают, что каждый
    # датчик умеет в одиночку.
    ("gyro", "gyro only", _imu, _comp(np.inf)),
    ("acc", "accel only", _imu, _comp(0.0)),
    # С энкодером колесо перестаёт быть ненаблюдаемым (rank 4 против 2), и
    # ошибка колеса из растущей становится ограниченной -- её задаёт уже
    # разрешение энкодера, а не фильтр.
    ("enc", "IMU+encoder", _imu_enc, _comp(wheel="encoder")),
]
ESTIMATOR_KEYS = [key for key, _, _, _ in ESTIMATORS]
ESTIMATOR_LABEL = {key: label for key, label, _, _ in ESTIMATORS}
ESTIMATOR_SENSOR = {key: mk for key, _, mk, _ in ESTIMATORS}
ESTIMATOR_BUILD = {key: mk for key, _, _, mk in ESTIMATORS}


class Explorer:
    """Состояние приложения: сетка, кэш траекторий, воспроизведение."""

    def __init__(self, args):
        self.args = args
        # Предел момента: число либо None -- «мотор без предела». None -- не
        # выдумка окна, а задокументированный режим модели: при u_bounds is
        # None System.clip_action возвращает управление как есть.
        # u_finite помнит последнее конечное значение, поэтому выключить и
        # включить обратно -- два нажатия, и карта возвращается ровно та же.
        self.u_finite = float(np.clip(_finite_or_none(args.u_max) or 3.0,
                                      U_MIN, U_MAX))
        self.u_max = _finite_or_none(args.u_max)
        if self.u_max is not None:
            self.u_max = self.u_finite
        self.system = WheeledPendulum(u_max=self.u_max)
        self.integrator = RK4Integrator()
        self.pinned = args.theta_max is not None or args.dtheta_max is not None
        self.spec = GridSpec(args.grid, *self._map_bounds())
        self.u_clip = self._clip_u_max()

        self.n_steps = int(round(args.horizon / args.dt))
        if self.n_steps % args.stride:
            self.n_steps += args.stride - self.n_steps % args.stride

        # Граница восстановимости не зависит ни от регулятора, ни от клетки, но
        # зависит от предела мотора -- слайдер её двигает. Формула живёт в
        # модели (WheeledPendulum), здесь только её значения: в wpend/viz/
        # формул динамики нет. Форма замкнутая, поэтому пересчёт бесплатен и
        # идёт прямо во время перетаскивания.
        self._refresh_limits()

        # Цены ЛКР -- копия DEFAULT_COST, которую двигает панель настроек.
        self.cost = {k: tuple(v) for k, v in DEFAULT_COST.items()}
        self.settings_open = False
        self.settings_text = {}
        self.settings_focus = None
        self.settings_error = ""
        self.K, self.P, self.K_tilt, self.P_tilt, self._drift = self._synthesize()
        self.K_soft = self._synthesize_soft()
        self.design_note = "no scipy"
        self.c_star, self.c_tilt = self._certify()
        self._refresh_hint()
        self._lqr_mask_cache = None
        self.stale = False          # предел сдвинут, а сетка ещё от старого
        self.dragging = False
        # Предел, при котором посчитаны ТРАЕКТОРИИ. Пока слайдер тащат, он
        # отстаёт от self.u_max, и панели, описывающие уже посчитанный прогон
        # (полоса момента, коридор ±u_max на графике), обязаны жить по нему:
        # мерить старую траекторию новой линейкой -- врать тише, но так же.
        self.run_u_max = self.u_max

        # `--main bang --u-max inf` -- законная просьба о невозможном: реле
        # без конечного предела не определено. Отказываем НЕ молча: вариант
        # заменяется на чистый ЛКР, а причина написана рядом с кнопками.
        self.main_key = args.main if self.available(args.main) else "lqr"
        self.cmp_key = (args.compare if args.compare is None
                        or self.available(args.compare) else None)
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        # Оцениватель -- третья ось выбора рядом с основным и сравнением.
        # Он красит карту наравне с регулятором: мир развивается по истинному
        # состоянию, но управление считается по оценке (правило 4), значит
        # исход клетки зависит и от фильтра.
        # Параметры датчика и фильтра -- в одном месте: их читают фабрики
        # ESTIMATORS, их же двигают слайдеры.  Дублировать значения в
        # аргументах командной строки И в объекте значило бы завести два
        # источника правды.
        self.noise = {"sigma_g": args.sigma_g, "sigma_a": args.sigma_a,
                      "b0_g": args.b0_g, "tau": args.tau}
        self.noise_drag = None          # какой слайдер тащат
        self.tau_star_measured = None   # argmin замера; None -- ещё не мерили
        self.est_key = args.estimator
        # Оцениватель СРАВНЕНИЯ считается только для выбранной клетки -- ровно
        # как второй регулятор. Карту он не красит: карта принадлежит
        # основному, иначе непонятно, чей это бассейн.
        self.est_cmp_key = args.estimator_compare
        self.traj_est_cmp = None
        self._refresh_hint()

        self.batch = None
        self.outcome = np.full(self.spec.n * self.spec.n, -1, dtype=int)
        self.t_fall = np.full(self.spec.n * self.spec.n, np.nan)
        self.cache: dict[int, object] = {}
        self.compute_seconds = 0.0

        if args.precompute:
            self._precompute()

        self.hover = None
        self.selected = None
        self.selected_state = (0.0, 0.0)
        self.traj = None
        self.traj_cmp = None
        self.traj_est_cmp = None
        self.playing = True
        self.play_time = 0.0
        # Стартовая клетка -- внутри множества восстановимости (_start_state):
        # на широкой карте «0.6 края» -- это уже лёгший корпус, и первое, что
        # видит пользователь, была бы падающая траектория.
        self._select_nearest(*self._start_state())

    # --- проект регулятора -------------------------------------------------

    def _synthesize(self):
        """Решения Риккати: (K, P, K_наклон, P_наклон, пояснение).

        Считается ОДИН раз за жизнь окна. Предел мотора сюда не входит вовсе:
        линеаризация (A, B) обрезки не видит, задача ЛКР линейно-квадратичная и
        про |u| <= u_max не знает. Поэтому слайдер пересчитывает не K, а только
        сертификат -- см. _certify. Если бы K зависел от предела, слайдер стоил
        бы ещё 0.4 с на каждое движение.

        K синтезируется, а не берётся из константы: так параметры модели и
        матрица обратной связи не могут разъехаться молча. K_LQR сверху --
        контрольное число, расхождение с ним печатается. Если scipy нет (окно
        ставили без extra `design`), берём K_LQR, а варианты, которым нужен
        эллипсоид, окно отключит.

        Второй проект -- ЛКР по ОДНОМУ наклону, уже вложенный в полное
        пространство: P выходит вырожденной (ранг 2), её множество уровня --
        цилиндр, а не эллипсоид.
        """
        A, B = self.system.linearize_upright()
        Q, R = _cost_matrices(self.cost["base"])
        try:
            from ..lqr import lqr, lqr_tilt
            K, P = lqr(A, B, Q, R)
            K_t, P_t = lqr_tilt(self.system, np.diag([100.0, 10.0]),
                                np.array([[1.0]]))
        except ImportError:                              # pragma: no cover
            return K_LQR, None, None, None, ""
        drift = float(np.abs(K - K_LQR).max())
        note = ""
        # Сверка с константой имеет смысл только для той цены, под которую
        # константа записана: при цене из панели K обязано отличаться.
        if drift > 1e-4 and self.cost["base"] == DEFAULT_COST["base"]:  # pragma: no cover
            note = f"  (!) K ушло от константы на {drift:.2g}"
        return K, P, K_t, P_t, note

    def _synthesize_soft(self):
        """ЛКР второй фазы `bang-eps`: та же задача, что в _synthesize, но с
        ценой self.cost["soft"]. По умолчанию это Q_SOFT_WHEEL, R = 1 -- гейны
        по phi и dphi ниже базовых в 10 и 4.4 раза.

        Без scipy -- None, и `bang-eps` окно отключает, как эллипсоид.
        """
        A, B = self.system.linearize_upright()
        try:
            from ..lqr import lqr
        except ImportError:                              # pragma: no cover
            return None
        K_soft, _ = lqr(A, B, *_cost_matrices(self.cost["soft"]))
        return K_soft

    def _start_state(self):
        """(theta, dtheta) стартовой клетки: узел сетки с |theta| <= pi/2 внутри
        множества восстановимости и, если сетка уже посчитана, удержанный
        основным регулятором; из таких -- с наибольшим |theta| (при равенстве
        ближе к середине отрезка скоростей). На широкой карте наугад
        выбранная клетка почти всегда -- лёгший корпус, а центр -- неподвижный
        робот; ни то, ни другое не показывает, что делает регулятор."""
        u = self.u_finite if self.u_max is None else self.u_max
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)
        th, dth = X0[:, I_THETA], X0[:, I_DTHETA]
        ok = self.system.is_recoverable(u, th, dth) & (np.abs(th) <= THETA_MAP)
        if (self.outcome >= 0).all() and (ok & (self.outcome == HELD)).any():
            ok &= self.outcome == HELD
        if not ok.any():
            return 0.0, 0.0
        floor, ceiling = self.system.recoverable_bounds(u, th)
        score = np.where(ok, np.abs(th) - 1e-6 * np.abs(dth - 0.5 * (floor + ceiling)),
                         -np.inf)
        m = int(np.argmax(score))
        return float(th[m]), float(dth[m])

    # --- панель настроек ---------------------------------------------------

    def settings_values(self) -> dict:
        """Текущие настраиваемые числа под ключами полей панели."""
        values = {}
        for group in ("base", "soft"):
            for i, v in enumerate(self.cost[group]):
                values[f"{group}.{i}"] = float(v)
        values["eps"] = float(self.args.eps)
        values["horizon"] = float(self.args.horizon)
        return values

    def open_settings(self):
        """Поля заполняются текущими числами: панель показывает то, чем сейчас
        посчитана карта, а не то, что вводили в прошлый раз."""
        self.settings_text = {k: f"{v:g}" for k, v in self.settings_values().items()}
        self.settings_focus = SETTINGS_KEYS[0]
        self.settings_error = ""
        self.settings_open = True

    def apply_settings(self, values: dict):
        """Принять новые цены ЛКР, eps и горизонт; пересчитать всё зависимое.

        Значения -- числа или строки из полей панели. Возвращает None при
        успехе либо строку с причиной отказа; при отказе окно не меняется ни в
        чём -- ни K, ни карта. Проверка до пересчёта, а синтез -- в пробном
        прогоне: Риккати с нулевым весом, при котором пара (A, Q) теряет
        обнаружимость, может не решиться, и узнать это можно только решив.

        Что поедет. K и P базы, K второй фазы -- синтез; c* -- сертификат под
        новую P при текущем u_max (иначе эллипсоид от старой цены при новом K --
        пара «критерий + регулятор» разошлась бы, CLAUDE.md); маска чистого ЛКР
        и сетка -- заново; горизонт меняет число шагов. ЛКР наклона не зависит
        от этих цен и не пересчитывается.
        """
        merged = self.settings_values()
        try:
            for key, value in values.items():
                if key not in merged:
                    return f"unknown field {key}"
                merged[key] = float(value)
        except (TypeError, ValueError):
            return f"not a number: {key} = {value!r}"
        if not all(np.isfinite(v) for v in merged.values()):
            return "all values must be finite"
        cost = {g: tuple(merged[f"{g}.{i}"] for i in range(5)) for g in ("base", "soft")}
        for g, c in cost.items():
            if min(c[:4]) < 0:
                return f"{g}: Q weights must be >= 0"
            if c[4] <= 0:
                return f"{g}: R must be > 0"
        if merged["eps"] <= 0:
            return "eps must be > 0"
        if merged["horizon"] < 5 * self.args.dt * self.args.stride:
            return "horizon too short"

        A, B = self.system.linearize_upright()
        try:
            from ..lqr import lqr
            with np.errstate(all="raise"):
                trial = {g: lqr(A, B, *_cost_matrices(c)) for g, c in cost.items()}
        except ImportError:                              # pragma: no cover
            return "needs scipy:  uv sync --extra dev"
        except Exception as exc:                         # noqa: BLE001
            return f"Riccati failed: {type(exc).__name__}"
        for g, (K, P) in trial.items():
            if not (np.isfinite(K).all() and np.isfinite(P).all()):
                return f"{g}: Riccati gave non-finite K"
            closed = np.linalg.eigvals(A - B @ K)
            if closed.real.max() >= 0:
                return f"{g}: closed loop not stable (Q too small?)"

        self.cost = cost
        self.args.eps = merged["eps"]
        self.args.horizon = merged["horizon"]
        self.n_steps = int(round(self.args.horizon / self.args.dt))
        if self.n_steps % self.args.stride:
            self.n_steps += self.args.stride - self.n_steps % self.args.stride
        self.K, self.P, self.K_tilt, self.P_tilt, self._drift = self._synthesize()
        self.K_soft = self._synthesize_soft()
        self.c_star, self.c_tilt = self._certify()
        self._refresh_hint()
        if not self.available(self.main_key):
            self.main_key = "lqr"
        if self.cmp_key is not None and not self.available(self.cmp_key):
            self.cmp_key = None
        self._invalidate(mask=True)
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        if self.args.precompute:
            self._precompute()
        self._select_nearest(*self.selected_state)
        self.settings_error = ""
        return None

    def _certify(self):
        """Сертифицированные уровни (c*, c*_наклон) для ТЕКУЩЕГО предела.

        Единственное место проекта, куда предел мотора входит по существу:
        уровень строится по vdot ПРИ ОБРЕЗАННОМ управлении, и забыть здесь
        u_max -- получить ответ для другого мотора (при u_max = 1.5 он выходит
        в 16 раз оптимистичнее). Цилиндрической форме обязательно передаются
        coords: без них луч вдоль оси цилиндра даёт c* = 0 молча.

        Цена -- 0.14 с на полную форму и 0.63 с на наклонную; ради этих 0.8 с
        слайдер и пересчитывает на отпускании, а не на каждом пикселе.
        """
        if self.P is None:                               # pragma: no cover
            self.design_note = "no scipy"
            return None, None
        from ..lqr import certified_level

        c_star, _ = certified_level(self.system, self.K, self.P, u_max=self.u_max,
                                    n_dirs=1500, ds=4e-3, s_max=8.0,
                                    theta_max=self.args.theta_fall)
        c_tilt, _ = certified_level(self.system, self.K_tilt, self.P_tilt,
                                    u_max=self.u_max, coords=(I_THETA, I_DTHETA),
                                    n_dirs=4000, ds=2e-3, s_max=8.0,
                                    theta_max=self.args.theta_fall)
        self.design_note = (f"c*={c_star:.3g}"
                            + ("" if self.u_max is not None else " unclipped")
                            + self._drift)
        return c_star, c_tilt

    # --- карта под предел момента ------------------------------------------

    def _map_bounds(self):
        """Границы карты (theta_max, dtheta_max) по стандарту проекта.

        Стандарт (CLAUDE.md, решения Глеба 16.09 и 17.09): интересен наклон
        theta в [-pi/2, pi/2], а сетка берётся с запасом MAP_FIT по обеим осям.
        По theta граница поэтому постоянная, MAP_FIT * pi/2 ≈ 2.04 рад: карта
        одна и та же при любом моторе, и пунктир pi/2 на ней виден всегда.

        По dtheta граница по-прежнему идёт за мотором: восстановимое множество
        растёт с u_max почти линейно, и одни и те же границы годятся ровно для
        одного мотора. Берётся MAP_FIT от наибольшего |dtheta| на границе
        восстановимости при theta в [-pi/2, pi/2] -- не «горло» при theta = 0:
        множество наклонное, и у края отрезка оно шире, чем в центре.

        theta_fall по умолчанию -- pi («упал», земли в модели нет), так что
        карта до него не доходит. Явные --theta-max / --dtheta-max
        замораживают границы: тогда карты при разных u_max сравнимы по
        пикселям, а не только по подписям.
        """
        # При выключенном пределе восстановимо ВСЁ, и подгонять карту не подо
        # что: границы замораживаются на последнем конечном пределе. Это и есть
        # смысл опыта -- те же клетки, другой мотор: видно, какие поменяли
        # цвет, а не как поехали оси.
        u = self.u_finite if self.u_max is None else self.u_max
        theta_max = min(MAP_FIT * THETA_MAP, 0.98 * self.args.theta_fall)
        th = np.linspace(-THETA_MAP, THETA_MAP, 401)
        floor, ceiling = self.system.recoverable_bounds(u, th)
        reach = np.abs(np.concatenate([floor, ceiling]))
        dtheta_max = MAP_FIT * float(reach[np.isfinite(reach)].max())
        if self.args.theta_max is not None:
            theta_max = self.args.theta_max
        if self.args.dtheta_max is not None:
            dtheta_max = self.args.dtheta_max
        return theta_max, dtheta_max

    def _clip_u_max(self):
        """Предел момента, с которого карта перестаёт содержать множество
        восстановимости по theta. При стандартной карте (±1.3·pi/2 до
        theta_fall = pi) такого нет -- None. Остаётся для --theta-fall меньше
        MAP_FIT·pi/2, где карта обрезается и окно обязано сказать об этом."""
        if MAP_FIT * THETA_MAP <= 0.98 * self.args.theta_fall:
            return None
        return U_MIN

    def _refresh_limits(self):
        """Границу восстановимости и седло -- под текущий предел и текущие
        границы карты. Обе величины замкнутые, поэтому вызывается и на каждом
        движении слайдера."""
        if self.u_max is None:
            # Без предела восстановимо любое состояние. Нарисовать здесь линию
            # значило бы дорисовать границу, которой нет, поэтому границы
            # просто нет -- и легенда говорит об этом словами.
            self.limit_theta = self.limit_floor = self.limit_ceiling = None
            self.theta_eq = None
            return
        self.limit_theta = np.linspace(-self.spec.theta_max * 1.05,
                                       self.spec.theta_max * 1.05, 601)
        self.limit_floor, self.limit_ceiling = self.system.recoverable_bounds(
            self.u_max, self.limit_theta)
        self.theta_eq = self.system.saddle_angle(self.u_max)

    def toggle_limit(self):
        """Выключить предел или вернуть последнее конечное значение.

        Идёт тем же путём, что и слайдер: preview меняет систему и границы,
        commit -- сертификат, регулятор и сетку. Отдельной ветки пересчёта
        нет, поэтому и рассогласоваться нечему.
        """
        self.preview_u_max(self.u_finite if self.u_max is None else None)
        self.commit_u_max()

    def preview_u_max(self, value):
        """Слайдер тащат: меняем систему, границы карты и границу
        восстановимости -- всё, что считается замкнутой формой и потому даром.

        Сетка и сертификат остаются от прежнего предела и честно помечаются
        устаревшими (stale): рисовать чужую карту как свою -- врать. Видно
        при этом ровно правильное -- сначала едет физический предел, и только
        потом, на отпускании, догоняет то, что регулятор из него сумел.
        """
        value = None if value is None else float(np.clip(value, U_MIN, U_MAX))
        if value == self.u_max:
            return
        if value is not None:
            self.u_finite = value
        self.u_max = value
        self.system = WheeledPendulum(u_max=value)
        self.spec = GridSpec(self.spec.n, *self._map_bounds())
        self._refresh_limits()
        self.stale = True

    def commit_u_max(self):
        """Слайдер отпустили: сертификат, регулятор и сетка -- под новый предел.

        Предел момента входит в ТРИ разных места, и все три обязаны поехать
        вместе: обрезка в System.u_bounds, модуль релейного момента в
        BangBangLQRController и u_max в certified_level. Оставить хоть одно
        старым -- получить реле, просящее больше, чем даёт мотор, или
        сертификат от другой машины. Маска чистого ЛКР посчитана при старом
        пределе, поэтому её кэш тоже сбрасывается.

        Клетка выбирается заново по СОСТОЯНИЮ, а не по индексу: сетка
        перестроилась, и тот же (ix, iy) означал бы другой наклон.
        """
        if not self.stale:
            return
        # Реле без конечного предела не определено. Отказать молча нельзя --
        # окно падает на чистый ЛКР и пишет, почему кнопки перечёркнуты.
        self._refresh_hint()
        if not self.available(self.main_key):
            self.main_key = "lqr"
        if self.cmp_key is not None and not self.available(self.cmp_key):
            self.cmp_key = None
        self.c_star, self.c_tilt = self._certify()
        self._invalidate(mask=True)
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        if self.args.precompute:
            self._precompute()
        self.stale = False
        self.run_u_max = self.u_max
        self._select_nearest(*self.selected_state)

    # --- слой оценки -------------------------------------------------------

    def _sensor(self, key=None):
        """Датчик оценивателя key (None -- идеальный, y = x)."""
        make = ESTIMATOR_SENSOR[key or self.est_key]
        return None if make is None else make(self)

    def _estimator(self, key=None):
        """Оцениватель (None -- тождественный, x_hat = y)."""
        make = ESTIMATOR_BUILD[key or self.est_key]
        return None if make is None else make(self)

    @property
    def records_estimate(self) -> bool:
        """Писать ли x_hat в прогон.

        У идеального оценивателя x_hat совпадает с x побитово -- писать нечего,
        и незачем платить +80 % памяти пачки. У остальных оценка и есть предмет
        интереса.
        """
        return self.est_key != "ideal"

    # --- оптимальное tau ---------------------------------------------------

    @property
    def tau_star_model(self):
        """Аналитический оптимум: замкнутая форма, ошибка акселерометра БЕЛАЯ.

        Считается мгновенно, поэтому метка двигается прямо во время
        перетаскивания -- как зелёная граница восстановимости за слайдером
        момента.  Что она НЕ учитывает, написано в докстринге optimal_tau и
        видно по расстоянию до второй метки.
        """
        return optimal_tau(sigma_g=self.noise["sigma_g"],
                           sigma_a=self.noise["sigma_a"],
                           b=self.noise["b0_g"], g=float(self.system.p.g))

    def measure_tau_star(self, n_points: int = 7, n_reps: int = 3):
        """Измеренный оптимум: argmin RMSE оценки по сетке tau, одна клетка.

        Считается ОДНОЙ пачкой из n_points * n_reps строк: у ComplementaryEstimator
        tau может быть массивом по строкам, и семь прогонов превращаются в один.
        Замер: 9.1 с семью отдельными прогонами против ~1.3 с пачкой -- цена
        сидит в питоновском цикле по шагам, а не в числе строк, ровно как в
        rollout_many.

        n_reps реализаций шума на каждое tau: датчик разыгрывает своё смещение
        включения на КАЖДУЮ строку, поэтому усреднение по ним убирает разброс
        от одного неудачного розыгрыша. Отдельными прогонами это стоило бы
        втрое дороже, пачкой -- бесплатно.

        Мерится на ВЫБРАННОЙ клетке: оптимум зависит от того, насколько резкий
        манёвр, и общего числа не существует (замер: 15x аналитического у
        мягкого старта, 39x у крупного наклона).
        """
        if self.selected is None or self.est_key == "ideal":
            self.tau_star_measured = None
            return
        lo, hi = NOISE_SPECS[3][2], NOISE_SPECS[3][3]
        taus = np.exp(np.linspace(np.log(lo), np.log(hi), n_points))
        m = self.spec.index(*self.selected)
        x0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)[m]
        X0 = np.tile(x0, (n_points * n_reps, 1))

        keep = self.noise["tau"]
        try:
            self.noise["tau"] = np.repeat(taus, n_reps)
            with self._errstate():
                batch = rollout_many(
                    self.system, self.controller, self.integrator, X0,
                    self.args.dt, self.n_steps, self.args.stride,
                    sensor=self._sensor(), estimator=self._estimator(),
                    record_estimate=True)
        finally:
            self.noise["tau"] = keep
        if batch.x_hat is None:
            self.tau_star_measured = None
            return

        err = batch.x_hat[:, :, I_THETA] - batch.x[:, :-1, I_THETA]
        err = np.where(np.isfinite(err), err, np.inf)
        rmse = np.sqrt(np.mean(err ** 2, axis=1)).reshape(n_points, n_reps)
        score = np.mean(rmse, axis=1)
        if not np.any(np.isfinite(score)):
            self.tau_star_measured = None
            return
        self.tau_star_measured = float(taus[int(np.nanargmin(
            np.where(np.isfinite(score), score, np.nan)))])

    # --- слайдеры датчика --------------------------------------------------

    def preview_noise(self, key: str, value: float):
        """Пока тащат -- меняем только число: аналитическая метка бесплатна,
        а сетка нет."""
        spec = next(sp for sp in NOISE_SPECS if sp[0] == key)
        self.noise[key] = float(np.clip(value, spec[2], spec[3]))
        self.stale = True

    def commit_noise(self):
        """На отпускании: пересчитать сетку. Развёртку по tau -- НЕ трогать.

        Замер: сетка 21x21 -- 2.1 с, развёртка по семи tau -- 8.9 с.  То есть
        измерение оптимума стоило вчетверо дороже всего остального, и слайдер
        от этого провисал на одиннадцать секунд.  Причина поучительная: прогон
        ОДНОЙ клетки стоит почти столько же, сколько вся сетка из 441, потому
        что цена сидит в питоновском цикле по шагам, а не в числе клеток --
        ровно то, ради чего в проекте есть rollout_many.

        Развёртка векторизована (одна пачка вместо семи прогонов) и стоит
        теперь малую долю от сетки, поэтому снова считается здесь же.
        Клавиша T пересчитывает её отдельно, не трогая сетку.
        """
        self.stale = False
        self._invalidate(mask=True)
        if self.args.precompute:
            self._precompute()
        if self.selected is not None:
            self.select(*self.selected)
        self.measure_tau_star()

    def set_estimator_compare(self, key):
        """Второй оцениватель: один прогон выбранной клетки, как у регулятора."""
        if key is not None and key not in ESTIMATOR_LABEL:
            return
        self.est_cmp_key = None if key == self.est_cmp_key else key
        self._update_estimator_compare()

    def _update_estimator_compare(self):
        """Пересчитать траекторию второго оценивателя для выбранной клетки.

        Тот же регулятор и то же начальное условие -- отличается ровно слой
        оценки, поэтому расхождение истинных траекторий и есть его цена.
        """
        if self.est_cmp_key is None or self.selected is None:
            self.traj_est_cmp = None
            return
        m = self.spec.index(*self.selected)
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)[m:m + 1]
        with self._errstate():
            batch = rollout_many(self.system, self.controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(self.est_cmp_key),
                                 estimator=self._estimator(self.est_cmp_key),
                                 record_estimate=self.est_cmp_key != "ideal")
        self.traj_est_cmp = batch[0]

    def set_estimator(self, key: str):
        """Смена оценивателя красит карту заново -- как и смена основного.

        Плюс маска чистого ЛКР: она не зависит от того, кто красит карту, но
        зависит от того, что регулятор видит, а значит от фильтра.
        """
        if key == self.est_key or key not in ESTIMATOR_LABEL:
            return
        self.est_key = key
        self._refresh_hint()
        self._invalidate(mask=True)
        if self.args.precompute:
            self._precompute()
        if self.selected is not None:
            self.select(*self.selected)
        self.tau_star_measured = None

    def available(self, key: str) -> bool:
        """Вариант с эллипсоидом требует P (то есть scipy), любое реле --
        конечного предела момента."""
        if key in RELAY_KEYS and self.u_max is None:
            return False
        if key in ("bang-ell", "bang-ell-phi"):
            return self.P is not None
        if key == "bang-tilt":
            return self.P_tilt is not None
        if key == "bang-eps":
            return self.K_soft is not None
        if key == "bang-eps-ell":
            return self.K_soft is not None and self.P is not None
        return True

    def _refresh_hint(self):
        """Строка о том, почему часть кнопок перечёркнута. Молчащая мёртвая
        кнопка -- баг интерфейса, поэтому причина всегда написана рядом."""
        # Про реле пишет строка слайдера: там есть место, а рядом с кнопками
        # длинная фраза не помещается и обрезается краем окна.
        if self.P is None:
            self.design_hint = "ellipsoid needs scipy:  uv sync --extra dev"
        elif getattr(self, "est_key", "ideal") != "ideal" and self.K is not None \
                and not np.allclose(self.K[0, [I_PHI, I_DPHI]], 0.0):
            # Колесо при одном ИДУ ненаблюдаемо, а K его всё равно спрашивает.
            # Не отключаем -- увидеть, как это ломается, полезно, -- но молчать
            # нельзя: иначе карта врёт, а причина не написана.
            self.design_hint = "IMU cannot see the wheel, yet K[phi], K[dphi] != 0"
        else:
            self.design_hint = ""

    def lqr_mask(self):
        """Маска клеток, из которых чистый ЛКР удержал корпус, -- та самая
        «карта», по которой один из вариантов решает, когда отдавать
        управление. Считается лениво и один раз: если основной регулятор и
        так ЛКР и сетка посчитана, берём готовое."""
        if self._lqr_mask_cache is None:
            if self.main_key == "lqr" and self.batch is not None:
                outcome = self.outcome
            else:
                X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)
                with self._errstate():
                    batch = rollout_many(self.system, LinearFeedbackController(self.K),
                                         self.integrator, X0, self.args.dt,
                                         self.n_steps, self.args.stride,
                                         sensor=self._sensor(),
                                         estimator=self._estimator())
                outcome, _ = classify(batch, I_THETA, self.args.theta_fall)
            self._lqr_mask_cache = (outcome == HELD).reshape(self.spec.n, self.spec.n)
        return self._lqr_mask_cache

    # --- смена регулятора --------------------------------------------------

    def _invalidate(self, *, mask: bool = False):
        """Забыть посчитанное. mask=True -- ещё и маску чистого ЛКР: она не
        зависит от того, кто красит карту, но зависит от предела момента."""
        self.batch = None
        self.outcome[:] = -1
        self.t_fall[:] = np.nan
        self.cache.clear()
        if mask:
            self._lqr_mask_cache = None

    def set_main(self, key: str):
        """Основной красит карту, поэтому сетку приходится пересчитать."""
        if key == self.main_key or not self.available(key):
            return
        self.main_key = key
        self._invalidate()
        self.controller = CONTROLLER_BUILD[key](self)
        self.traj_est_cmp = None
        if self.args.precompute:
            self._precompute()
        if self.selected is not None:
            self.select(*self.selected)

    def set_compare(self, key):
        """Сравнение считается только для выбранной клетки -- один прогон."""
        if key is not None and not self.available(key):
            return
        self.cmp_key = key
        self._update_compare()

    # --- счёт --------------------------------------------------------------

    def _errstate(self):
        """Переполнение -- ожидаемый исход БЕЗ предела момента и подозрительное
        событие с ним, поэтому предупреждения numpy глушим только в первом
        случае. Факт расхождения при этом не теряется: число разошедшихся
        клеток окно печатает само."""
        return (np.errstate(over="ignore", invalid="ignore") if self.u_max is None
                else np.errstate())

    def _precompute(self):
        """Один векторный прогон всей сетки: M начальных условий одновременно."""
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)
        t0 = time.perf_counter()
        with self._errstate():
            self.batch = rollout_many(self.system, self.controller, self.integrator,
                                      X0, self.args.dt, self.n_steps,
                                      self.args.stride,
                                      sensor=self._sensor(),
                                      estimator=self._estimator(),
                                      record_estimate=self.records_estimate)
        self.outcome, self.t_fall = classify(self.batch, I_THETA, self.args.theta_fall)
        self.compute_seconds = time.perf_counter() - t0
        # Без предела ЛКР -- линейный закон, продолженный на все углы: за
        # горизонтом он раскручивает корпус вместо того, чтобы ловить, и
        # угловые клетки уходят в переполнение. Классификация их всё равно
        # ловит (theta пересекает theta_fall задолго до inf), но молчать об
        # этом нельзя -- иначе выглядит как «numpy что-то ругался».
        gone = int((~np.isfinite(self.batch.x).all(axis=(1, 2))).sum())
        print(f"[{CONTROLLER_LABEL[self.main_key]} / {ESTIMATOR_LABEL[self.est_key]}] "
              f"сетка {self.spec.n}x{self.spec.n} "
              f"= {len(self.batch)} НУ, {self.n_steps} шагов: "
              f"{self.compute_seconds:.2f} с, "
              f"{self.batch.nbytes / 2**20:.1f} МБ в памяти"
              + (f", разошлось {gone}" if gone else ""))

    def _trajectory(self, m: int):
        """Траектория клетки: из готовой пачки либо посчитанная по требованию."""
        if self.batch is not None:
            return self.batch[m]
        if m in self.cache:
            return self.cache[m]
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)[m:m + 1]
        t0 = time.perf_counter()
        with self._errstate():
            batch = rollout_many(self.system, self.controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(), estimator=self._estimator(),
                                 record_estimate=self.records_estimate)
        self.compute_seconds = time.perf_counter() - t0
        outcome, t_fall = classify(batch, I_THETA, self.args.theta_fall)
        self.outcome[m] = outcome[0]
        self.t_fall[m] = t_fall[0]
        self.cache[m] = batch[0]
        return self.cache[m]

    def frame_of(self, traj) -> int:
        """Кадр траектории на текущем времени воспроизведения. У сравнения та
        же сетка времени, но длина может отличаться, если прогон оборвался."""
        if traj is None:
            return 0
        dt_frame = traj.t[1] - traj.t[0]
        return int(np.clip(self.play_time / dt_frame, 0, len(traj) - 1))

    # --- выбор клетки ------------------------------------------------------

    def _run_one(self, controller, m: int):
        """Один прогон одной клетки выбранным регулятором."""
        X0 = self.spec.initial_states(self.system.n_state, I_THETA, I_DTHETA)[m:m + 1]
        with self._errstate():
            batch = rollout_many(self.system, controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(), estimator=self._estimator(),
                                 record_estimate=self.records_estimate)
        return batch[0]

    def _update_compare(self):
        """Пересчитать траекторию сравнения для выбранной клетки."""
        if self.cmp_key is None or self.selected is None:
            self.traj_cmp = None
            return
        m = self.spec.index(*self.selected)
        self.traj_cmp = self._run_one(CONTROLLER_BUILD[self.cmp_key](self), m)

    def select(self, ix: int, iy: int):
        self.selected = (ix, iy)
        # Запоминаем не номер клетки, а её СОСТОЯНИЕ: при смене предела сетка
        # перестраивается, и вопрос «что этот же старт делает при другом
        # моторе» требует именно состояния.
        self.selected_state = (float(self.spec.thetas[ix]),
                               float(self.spec.dthetas[iy]))
        self.traj = self._trajectory(self.spec.index(ix, iy))
        self._update_compare()
        self._update_estimator_compare()
        self.play_time = 0.0
        self.playing = True

    def _select_nearest(self, theta, dtheta):
        ix = int(np.argmin(np.abs(self.spec.thetas - theta)))
        iy = int(np.argmin(np.abs(self.spec.dthetas - dtheta)))
        self.select(ix, iy)

    # --- воспроизведение ---------------------------------------------------

    @property
    def frame(self) -> int:
        return self.frame_of(self.traj)

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
    """Точка (theta, dtheta) -> пиксель карты.

    Карта рисуется как n x n клеток: узел сетки ix -- это не точка, а целый
    пиксель surface, растянутый в клетку [ix, ix+1).  Поэтому узел ix лежит в
    (ix + 0.5) / n ширины, а не в ix / (n - 1), и пересчёт ведётся по границам
    картинки, отступающим от крайних узлов на полклетки.  Наивная нормировка на
    theta_max совпадает с подсветкой клетки только в центре карты, а к краям
    расходится линейно, до полклетки в углах.
    """
    x, y, w, h = MAP
    n = app.spec.n
    edge = n / max(n - 1, 1)          # (полширины картинки) / theta_max
    half_th = app.spec.theta_max * edge
    half_dth = app.spec.dtheta_max * edge
    fx = (theta + half_th) / (2 * half_th)
    fy = (dtheta + half_dth) / (2 * half_dth)
    return x + fx * w, y + (1.0 - fy) * h


def _draw_phase(target, pg, app, traj, color, head_color, frame, ox, oy, width=2):
    """Фазовая кривая (theta, dtheta) в координатах target со сдвигом (ox, oy).

    Сдвиг нужен, чтобы одна и та же функция рисовала и прямо на экране, и на
    полупрозрачной подложке, у которой своё начало координат.
    """
    x, y, w, h = MAP
    pts = [_to_map_px(app, th, dth)
           for th, dth in zip(traj.x[:, I_THETA], traj.x[:, I_DTHETA])]
    # Сравнение отсеивает и nan: любое сравнение с nan ложно, поэтому
    # разошедшийся хвост просто не попадает в список точек.
    pts = [(px + ox, py + oy) for px, py in pts
           if x - 400 < px < x + w + 400 and y - 400 < py < y + h + 400]
    if len(pts) < 2:
        return
    pg.draw.lines(target, color, False, pts, width)
    hx, hy = pts[min(frame, len(pts) - 1)]
    pg.draw.circle(target, head_color, (int(hx), int(hy)), 5)


def _draw_limits(app, screen, pg):
    """Граница восстановимости поверх карты.

    Две кривые: выше верхней корпус уже не остановить вперёд, ниже нижней --
    назад, и это НЕ свойство регулятора, а свойство системы с данным u_max.
    Внутри карта может быть красной (этот регулятор не справился), но снаружи
    она обязана быть красной у любого.

    Отрезки, где уровень не существует (подкоренное выражение отрицательно и
    обрезано в ноль), рвём: рисовать там прямую по нулю значило бы дорисовать
    границу, которой нет.
    """
    if app.limit_ceiling is None:
        return                      # предела нет -- восстановимо всё
    x, y, w, h = MAP
    clip = screen.get_clip()
    screen.set_clip((x, y, w, h))
    for values in (app.limit_ceiling, app.limit_floor):
        run = []
        for th, dth in zip(app.limit_theta, values):
            if abs(dth) < 1e-12:                  # уровень отсутствует
                if len(run) > 1:
                    pg.draw.lines(screen, LIMIT, False, run, 2)
                run = []
                continue
            px, py = _to_map_px(app, th, dth)
            if x - 4000 < px < x + w + 4000 and y - 4000 < py < y + h + 4000:
                run.append((px, py))
            elif len(run) > 1:
                pg.draw.lines(screen, LIMIT, False, run, 2)
                run = []
        if len(run) > 1:
            pg.draw.lines(screen, LIMIT, False, run, 2)
    # Сёдла: отсюда из состояния покоя уже не вернуться.
    for sign in (+1, -1):
        px, py = _to_map_px(app, sign * app.theta_eq, 0.0)
        if x <= px <= x + w and y <= py <= y + h:
            pg.draw.circle(screen, LIMIT, (int(px), int(py)), 4, 1)
    screen.set_clip(clip)


def _draw_horizon(app, screen, pg, font):
    """Пунктир на theta = +-pi/2: корпус лёг горизонтально.

    Отметка не про регулятор, а про задачу. Момент тяжести идёт как D*sin(theta)
    и ровно на pi/2 проходит максимум: правее он снова УБЫВАЕТ, и «дальше --
    всегда тяжелее» перестаёт быть правдой. Без линии на широкой карте это
    место ничем не отмечено, а глаз ищет его первым.

    Рисуется только когда pi/2 попал в границы: на приколоченной узкой
    карте (--theta-max 0.15) линия ушла бы за край, и pygame нарисовал бы её по
    самой кромке панели -- отметка не там, где написано, хуже, чем её
    отсутствие.
    """
    x, y, w, h = MAP
    if app.spec.theta_max < HORIZON_ANGLE:
        return
    for sign in (+1, -1):
        px, _ = _to_map_px(app, sign * HORIZON_ANGLE, 0.0)
        if not x <= px <= x + w:
            continue
        for top in range(int(y), int(y + h), 11):       # штрих 6, пропуск 5
            pg.draw.line(screen, DIM, (px, top), (px, min(top + 6, y + h)))
        label = font.render("pi/2", True, DIM)
        # Подпись уводится внутрь карты, чтобы не залезть на рамку панели.
        lx = px + 4 if sign > 0 else px - 4 - label.get_width()
        screen.blit(label, (min(max(lx, x + 2), x + w - label.get_width() - 2), y + 4))


def draw_map(app, screen, pg, font):
    x, y, w, h = MAP
    _panel(screen, pg, MAP, "INITIAL CONDITIONS  theta0 (rad) x dtheta0 (rad/s)", font)
    screen.blit(pg.transform.scale(_map_surface(app, pg), (w, h)), (x, y))
    if app.stale:
        # Карта посчитана при другом пределе момента. Не пометить её -- значит
        # выдавать чужой результат за текущий; стереть -- потерять то, с чем
        # сравнивать глазом. Гасим, а граница восстановимости поверх остаётся
        # яркой: она-то уже от нового предела.
        veil = pg.Surface((w, h), pg.SRCALPHA)
        veil.fill(BG + (170,))
        screen.blit(veil, (x, y))

    for frac in (0.5,):   # оси через ноль
        pg.draw.line(screen, GRID_LINE, (x + frac * w, y), (x + frac * w, y + h))
        pg.draw.line(screen, GRID_LINE, (x, y + frac * h), (x + w, y + frac * h))

    _draw_horizon(app, screen, pg, font)
    _draw_limits(app, screen, pg)

    # Сравнение рисуется ПОД основным: основная кривая должна оставаться
    # читаемой там, где они совпадают.
    if app.traj_cmp is not None:
        ghost = pg.Surface((w, h), pg.SRCALPHA)
        # Шире основной: если регуляторы дали одну и ту же траекторию (а это
        # осмысленный результат, а не поломка), призрак виден как ореол.
        _draw_phase(ghost, pg, app, app.traj_cmp, GHOST + (GHOST_A,),
                    GHOST + (255,), app.frame_of(app.traj_cmp), -x, -y, width=5)
        screen.blit(ghost, (x, y))
    if app.traj_est_cmp is not None:
        ghost = pg.Surface((w, h), pg.SRCALPHA)
        _draw_phase(ghost, pg, app, app.traj_est_cmp, GHOST_EST + (GHOST_A,),
                    GHOST_EST + (255,), app.frame_of(app.traj_est_cmp),
                    -x, -y, width=5)
        screen.blit(ghost, (x, y))
    if app.traj is not None:
        clip = screen.get_clip()
        screen.set_clip((x, y, w, h))
        _draw_phase(screen, pg, app, app.traj, CURVE, ACCENT, app.frame, 0, 0)
        screen.set_clip(clip)

    # Панель не квадратная, а surface растягивается в (w, h): клетка тоже не
    # квадратная, и высоту нельзя брать из ширины -- иначе рамка уезжает вниз
    # тем сильнее, чем ниже строка.
    cell_w, cell_h = w / app.spec.n, h / app.spec.n
    for marker, color, width in ((app.hover, DIM, 1), (app.selected, ACCENT, 2)):
        if marker is None:
            continue
        ix, iy = marker
        pg.draw.rect(screen, color,
                     (x + ix * cell_w, y + (app.spec.n - 1 - iy) * cell_h,
                      cell_w, cell_h), width)

    labels = [(f"-{app.spec.theta_max:.2f}", x, y + h + 6),
              (f"+{app.spec.theta_max:.2f}", x + w - 34, y + h + 6),
              (f"+{app.spec.dtheta_max:.1f}", x - 46, y),
              (f"-{app.spec.dtheta_max:.1f}", x - 46, y + h - 14)]
    for text, tx, ty in labels:
        screen.blit(font.render(text, True, DIM), (tx, ty))


def _draw_body(target, pg, app, state, ox, oy, body_color, wheel_color, tip_color):
    """Корпус и колесо в координатах target со сдвигом (ox, oy).

    Одна функция и для основного робота (прямо на экране), и для призрака (на
    полупрозрачной подложке): различаются только цвета и куда рисуем.
    """
    _, _, w, h = ANIM
    p = app.system.p
    wheel_px = 26.0
    m2px = wheel_px / p.r
    cx, cy = ox + w / 2, oy + h * 0.72
    theta, phi = state[I_THETA], state[I_PHI]

    pg.draw.circle(target, wheel_color, (int(cx), int(cy)), int(wheel_px), 3)
    spoke = (cx + wheel_px * np.sin(phi), cy - wheel_px * np.cos(phi))
    pg.draw.line(target, wheel_color, (cx, cy), spoke, 2)

    tip = (cx + p.l * m2px * np.sin(theta), cy - p.l * m2px * np.cos(theta))
    pg.draw.line(target, body_color, (cx, cy), tip, 6)
    pg.draw.circle(target, tip_color, (int(tip[0]), int(tip[1])), 9)


def draw_robot(app, screen, pg, font):
    x, y, w, h = ANIM
    _panel(screen, pg, ANIM, "PLANT", font)
    if app.traj is None:
        return
    state = app.traj.x[app.frame]
    phi = state[I_PHI]
    # Разошедшаяся траектория: рисовать корпус по inf нельзя (int(nan) --
    # исключение), а сказать об этом надо -- это и есть ответ на вопрос
    # «что будет без предела» для угловых клеток.
    alive = bool(np.isfinite(state).all())
    finite_t = app.traj.t[np.isfinite(app.traj.x).all(axis=1)]
    diverged = len(finite_t) < len(app.traj)

    p = app.system.p
    wheel_px = 26.0
    m2px = wheel_px / p.r
    cy = y + h * 0.72

    # Дорога и её бегущие метки -- декорация сцены, а не робота: рисуются один
    # раз и по phi ОСНОВНОГО прогона, иначе два робота ехали бы по двум дорогам.
    pg.draw.line(screen, GRID_LINE, (x + 8, cy + wheel_px), (x + w - 8, cy + wheel_px), 2)
    shift = (p.r * phi * m2px) % (0.1 * m2px)
    step = 0.1 * m2px
    k = -int(w / (2 * step)) - 1
    while x + w / 2 + k * step - shift < x + w:
        tx = x + w / 2 + k * step - shift
        if x + 8 < tx < x + w - 8:
            pg.draw.line(screen, GRID_LINE, (tx, cy + wheel_px), (tx, cy + wheel_px + 8), 2)
        k += 1

    if app.traj_cmp is not None:
        cmp_state = app.traj_cmp.x[app.frame_of(app.traj_cmp)]
        if np.isfinite(cmp_state).all():
            ghost = pg.Surface((w, h), pg.SRCALPHA)
            _draw_body(ghost, pg, app, cmp_state, 0, 0, GHOST + (GHOST_A,),
                       GHOST + (GHOST_A,), GHOST + (GHOST_A,))
            screen.blit(ghost, (x, y))

    if alive:
        _draw_body(screen, pg, app, state, x, y, (226, 230, 236), (96, 104, 118), ACCENT)

    torque = app.traj.u[min(app.frame, app.traj.n_steps - 1), 0]
    # Без предела делить не на что, а вопрос «сколько же он попросил» -- ровно
    # то, ради чего предел и выключают. Масштаб берём по пику самой
    # траектории и пик подписываем числом.
    peak = _span(np.abs(app.traj.u))[1]
    limit = app.run_u_max if app.run_u_max is not None else max(peak, 1e-9)
    bar_w = 160
    bx, by = x + w - bar_w - 16, y + 16
    pg.draw.rect(screen, GRID_LINE, (bx, by, bar_w, 12), 1)
    # pg.draw.rect с отрицательной шириной не рисует НИЧЕГО, поэтому полосу
    # приходится нормализовать: иначе отрицательный момент просто не виден.
    if np.isfinite(torque):
        span = np.clip(torque / limit, -1, 1) * bar_w / 2
        pg.draw.rect(screen, ACCENT,
                     (bx + bar_w / 2 + min(span, 0.0), by + 1, abs(span), 10))
    label = (f"u = {_fmt(torque)} / {limit:.1f} N*m" if app.run_u_max is not None
             else f"u = {_fmt(torque)} / peak {_fmt(peak)} N*m  (no limit)")
    screen.blit(font.render(label, True, DIM),
                (x + w - 16 - font.size(label)[0], by + 16))
    if app.traj_cmp is not None:
        f2 = min(app.frame_of(app.traj_cmp), app.traj_cmp.n_steps - 1)
        t2 = app.traj_cmp.u[f2, 0]
        px = bx + bar_w / 2 + np.clip(t2 / limit, -1, 1) * bar_w / 2
        pg.draw.line(screen, GHOST, (px, by - 2), (px, by + 14), 2)

    info = [f"t     = {app.traj.t[app.frame] - app.traj.t[0]:5.2f} s",
            f"theta = {state[I_THETA]:+.4f} rad",
            f"dtheta= {state[I_DTHETA]:+.4f} rad/s",
            f"phi   = {phi:+.3f} rad"]
    for i, line in enumerate(info):
        screen.blit(font.render(line, True, TEXT), (x + 14, y + 14 + 18 * i))
    if diverged:
        gone = float(finite_t[-1] - app.traj.t[0]) if len(finite_t) else 0.0
        screen.blit(font.render(f"diverged at t = {gone:.2f} s", True, ACCENT),
                    (x + 14, y + 14 + 18 * len(info)))
    if app.traj_cmp is not None:
        s2 = app.traj_cmp.x[app.frame_of(app.traj_cmp)]
        cmp_info = [f"theta = {s2[I_THETA]:+.4f}", f"phi   = {s2[I_PHI]:+.3f}"]
        for i, line in enumerate(cmp_info):
            screen.blit(font.render(line, True, GHOST), (x + 14, y + 92 + 18 * i))


def _plot(screen, pg, rect, ts, ys, color, label, font, guides=(), ghosts=(),
          limit=None, estimate=None):
    """ghosts -- список (ts, ys, цвет): вторые кривые полупрозрачным. Их может
    быть две -- другой РЕГУЛЯТОР и другой ОЦЕНИВАТЕЛЬ, -- и цвета у них разные,
    потому что вопросы разные. Масштаб оси берётся по ВСЕМ кривым, иначе
    сравнивать нечего: несколько картинок с разными осями.

    estimate = (ts3, ys3) -- ОЦЕНКА той же величины, тонкой линией цвета
    ESTIMATE. Это не третий регулятор, а то же самое состояние, увиденное
    фильтром; расстояние до основной кривой и есть ошибка оценки."""
    x, y, w, h = rect
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1)
    lo, hi = _span(ys)
    for g in ghosts:
        g_lo, g_hi = _span(g[1])
        lo, hi = min(lo, g_lo), max(hi, g_hi)
    if estimate is not None:
        e_lo, e_hi = _span(estimate[1])
        lo, hi = min(lo, e_lo), max(hi, e_hi)
    if limit is not None:
        lo, hi = min(lo, -abs(limit)), max(hi, abs(limit))
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
    if limit is not None:
        # ±theta_eq: за этим наклоном из состояния покоя не вернуться никаким
        # управлением. Не путать с theta_fall -- тот просто «считаем упавшим».
        for sign in (+1, -1):
            ly = to_px(ts[0], sign * abs(limit))[1]
            if y <= ly <= y + h:
                pg.draw.line(screen, LIMIT, (x, ly), (x + w, ly), 1)
    if lo < 0 < hi:
        zy = to_px(ts[0], 0.0)[1]
        pg.draw.line(screen, GRID_LINE, (x, zy), (x + w, zy), 1)
    for g_ts, g_ys, g_color in ghosts:
        layer = pg.Surface((w, h), pg.SRCALPHA)
        pts = [(px - x, py - y) for px, py in
               (to_px(t, v) for t, v in zip(*_finite(g_ts, g_ys)))]
        if len(pts) > 1:
            pg.draw.lines(layer, g_color + (GHOST_A,), False, pts, 4)
        screen.blit(layer, (x, y))
    if estimate is not None:
        pts = [to_px(t, v) for t, v in zip(*_finite(*estimate))]
        if len(pts) > 1:
            pg.draw.lines(screen, ESTIMATE, False, pts, 1)
    pts = [to_px(t, v) for t, v in zip(*_finite(ts, ys))]
    if len(pts) > 1:
        pg.draw.lines(screen, color, False, pts, 2)
    screen.blit(font.render(label, True, DIM), (x + 6, y + 4))
    for text, ty in ((_fmt(hi), y + 4), (_fmt(lo), y + h - 18)):
        screen.blit(font.render(text, True, DIM),
                    (x + w - font.size(text)[0] - 6, ty))
    return to_px


#: Три разложения функции Ляпунова V = x^T P x. Вопрос «почему реле не отдаёт
#: управление» имеет численный ответ, и ответ этот -- в том, КАКАЯ координата
#: держит V наверху. Поэтому кривых три, а не одна.
V_COLORS = ((230, 120, 140), (196, 160, 240), (120, 220, 160))
V_LABELS = ("V full", "V wheel-anchored", "V tilt (P_t)")


def _lyapunov_curves(app, traj):
    """(V полное, V с привязкой phi, V только наклон) вдоль траектории.

    Читается только Trajectory плюс константы проекта P и c*, которые окно и
    так держит, чтобы собрать регулятор. Закон управления здесь не
    повторяется: это диагностика сертификата, а не вторая копия регулятора.
    """
    X = traj.x
    anchored = X.copy()
    anchored[:, I_PHI] = 0.0                       # колесо объявлено стоящим
    # Третья кривая считается ДРУГОЙ формой -- P от lqr_tilt, -- и сравнивать
    # её надо с c*_t, а не с c*. Подставлять сюда P от четырёхмерного проекта
    # и класть рядом уровень c* значило бы сравнивать разные квадратичные
    # формы: числа получились бы, а смысла в них -- нет.
    return (np.einsum("ti,ij,tj->t", X, app.P, X),
            np.einsum("ti,ij,tj->t", anchored, app.P, anchored),
            np.einsum("ti,ij,tj->t", X, app.P_tilt, X))


def draw_lyapunov(app, screen, pg, rect, font):
    """log10 V против log10 c*. В линейной шкале эта картинка бесполезна:
    полное V выше уровня в сотни раз и прижимает остальные две к нулю."""
    x, y, w, h = rect
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1)
    if app.P is None or app.P_tilt is None or app.traj is None:
        screen.blit(font.render("V = xTPx  --  needs scipy", True, GRID_LINE),
                    (x + 6, y + 4))
        return
    curves = [np.log10(np.maximum(v, 1e-12)) for v in _lyapunov_curves(app, app.traj)]
    # Два сертификата -- две пары «кривая + свой уровень». Красная и
    # фиолетовая живут с c* (форма P), зелёная -- с c*_t (форма P_t).
    levels = [(float(np.log10(max(app.c_star, 1e-12))), ACCENT, f"c* = {app.c_star:.3g}"),
              (float(np.log10(max(app.c_tilt, 1e-12))), V_COLORS[2],
               f"c*t = {app.c_tilt:.3g}")]
    ts = app.traj.t - app.traj.t[0]

    c_lo, c_hi = _span(*curves)
    lo = min(min(l for l, _, _ in levels), c_lo)
    hi = max(max(l for l, _, _ in levels), c_hi)
    lo = max(lo, hi - 12.0)                        # ноль в логарифме -- -inf
    pad = 0.08 * max(hi - lo, 1e-9)
    lo, hi = lo - pad, hi + pad

    def to_px(t, v):
        return (x + (t - ts[0]) / max(ts[-1] - ts[0], 1e-9) * w,
                y + h - (np.clip(v, lo, hi) - lo) / (hi - lo) * h)

    # Уровни бывают близки (c* = 2.53 против c*t = 3.27 -- в логарифме почти
    # одно и то же), и подписи налезали друг на друга. Вторую в таком случае
    # кладём под линию, а не над.
    taken = []
    for value, color, text in levels:
        ly = to_px(ts[0], value)[1]
        pg.draw.line(screen, color, (x, ly), (x + w, ly), 1)
        ty = ly - 17
        if any(abs(ty - t) < 15 for t in taken):
            ty = ly + 2
        taken.append(ty)
        screen.blit(font.render(text, True, color),
                    (x + w - font.size(text)[0] - 6, ty))

    for curve, color in zip(curves, V_COLORS):
        pts = [to_px(t, v) for t, v in zip(*_finite(ts, curve))]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 2)
    if app.traj_cmp is not None:
        layer = pg.Surface((w, h), pg.SRCALPHA)
        c2 = np.log10(np.maximum(_lyapunov_curves(app, app.traj_cmp)[0], 1e-12))
        ts2 = app.traj_cmp.t - app.traj_cmp.t[0]
        pts = [(px - x, py - y) for px, py in
               (to_px(t, v) for t, v in zip(*_finite(ts2, c2)))]
        if len(pts) > 1:
            pg.draw.lines(layer, GHOST + (GHOST_A,), False, pts, 4)
        screen.blit(layer, (x, y))

    lx = x + 6
    for label, color in zip(V_LABELS, V_COLORS):
        screen.blit(font.render(label, True, color), (lx, y + 4))
        lx += font.size(label)[0] + 14
    rng = f"log10  {hi:+.1f}..{lo:+.1f}"
    screen.blit(font.render(rng, True, DIM), (x + w - font.size(rng)[0] - 6, y + 4))
    cursor = to_px(ts[min(app.frame, len(ts) - 1)], lo)[0]
    pg.draw.line(screen, ACCENT, (cursor, y), (cursor, y + h), 1)


def draw_plots(app, screen, pg, font):
    x, y, w, h = PLOT
    _panel(screen, pg, PLOT, "TRAJECTORY", font)
    if app.traj is None:
        return
    ts = app.traj.t - app.traj.t[0]
    half = (h - 40) / 3
    g_theta, g_u = [], []
    for traj2, color in ((app.traj_cmp, GHOST), (app.traj_est_cmp, GHOST_EST)):
        if traj2 is None:
            continue
        ts2 = traj2.t - traj2.t[0]
        g_theta.append((ts2, traj2.x[:, I_THETA], color))
        g_u.append((ts2[:-1], traj2.u[:, 0], color))
    # Оценка приходит из самой Trajectory (правило 6): окно не пересчитывает
    # фильтр и вообще не знает, какой датчик её породил. x_hat выровнена по u,
    # то есть на один кадр короче x -- отсюда ts[:-1].
    e_theta = None
    if app.traj.x_hat is not None:
        e_theta = (ts[:-1], app.traj.x_hat[:, I_THETA])
    to_px1 = _plot(screen, pg, (x + 12, y + 10, w - 24, half),
                   ts, app.traj.x[:, I_THETA], (120, 180, 240), "theta (rad)", font,
                   guides=(app.args.theta_fall,), ghosts=g_theta,
                   limit=app.theta_eq, estimate=e_theta)
    if e_theta is not None:
        err = app.traj.x_hat[:, I_THETA] - app.traj.x[:-1, I_THETA]
        finite = err[np.isfinite(err)]
        rmse = float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")
        # Слева под подписью панели: справа стоят числа шкалы, и надпись
        # наезжала бы на верхнюю границу диапазона.
        text = f"est RMSE {rmse:.4f}"
        screen.blit(font.render(text, True, ESTIMATE), (x + 18, y + 10 + 22))
        if app.traj_est_cmp is not None and app.traj_est_cmp.x_hat is not None:
            e2 = (app.traj_est_cmp.x_hat[:, I_THETA]
                  - app.traj_est_cmp.x[:-1, I_THETA])
            f2 = e2[np.isfinite(e2)]
            r2 = float(np.sqrt(np.mean(f2 ** 2))) if f2.size else float("nan")
            screen.blit(font.render(f"vs {r2:.4f}", True, GHOST_EST),
                        (x + 18 + font.size(text)[0] + 10, y + 10 + 22))
    to_px2 = _plot(screen, pg, (x + 12, y + 20 + half, w - 24, half),
                   ts[:-1], app.traj.u[:, 0], (240, 170, 110), "u (N*m)", font,
                   guides=() if app.run_u_max is None else (app.run_u_max,),
                   ghosts=g_u)
    for to_px, rect_y in ((to_px1, y + 10), (to_px2, y + 20 + half)):
        cursor = to_px(ts[min(app.frame, len(ts) - 1)], 0.0)[0]
        pg.draw.line(screen, ACCENT, (cursor, rect_y), (cursor, rect_y + half), 1)
    draw_lyapunov(app, screen, pg, (x + 12, y + 30 + 2 * half, w - 24, half), font)


#: Подписи рядов кнопок. row 0 -- регулятор, красящий карту; row 1 -- второй
#: регулятор, рисуемый призраком; row 2 -- ОЦЕНИВАТЕЛЬ, который тоже красит
#: карту: исход клетки зависит и от того, что регулятор видит.
PICK_ROWS = ("MAIN", "COMPARE", "ESTIMATOR")


def pick_label(row: int, key) -> str:
    """Подпись кнопки. У рядов разные словари, и путать их нельзя."""
    if key is None:
        return "off"
    return ESTIMATOR_LABEL[key] if row == 2 else CONTROLLER_LABEL[key]


def picker_layout(app, font):
    """Прямоугольники кнопок: [(rect, row, key), ...]. row 0 -- основной,
    row 1 -- сравнение, row 2 -- оцениватель. Одна и та же раскладка нужна и
    отрисовке, и обработке клика, поэтому она -- чистая функция, а не побочный
    эффект рисования."""
    out = []
    rows = (CONTROLLER_KEYS, CONTROLLER_KEYS + [None], ESTIMATOR_KEYS)
    for row, keys in enumerate(rows):
        bx = PICK[0] + 108
        by = PICK[1] + row * PICK_ROW
        for key in keys:
            bw = font.size(pick_label(row, key))[0] + 18
            out.append(((bx, by, bw, 22), row, key))
            bx += bw + 6
    return out


def noise_layout(font):
    """Раскладка ряда слайдеров: [(spec, (label_x, y), track_rect, value_x)].

    Чистая функция, как picker_layout: одна раскладка и на отрисовку, и на
    обработку клика.  Разъехавшись, они дали бы слайдер, который рисуется в
    одном месте, а ловит мышь в другом -- баг, который ищут глазами.
    """
    out, x = [], 24
    for spec in NOISE_SPECS:
        lw = font.size(spec[1])[0]
        track = (x + lw + 8, NOISE_Y + 5, NOISE_TRACK, 8)
        value_x = track[0] + NOISE_TRACK + 10
        out.append((spec, (x, NOISE_Y), track, value_x))
        x = value_x + font.size(spec[5].format(0.000123))[0] + 20
    return out


def noise_to_px(spec, value, track):
    """Значение -> пиксель. Шкала логарифмическая: шум живёт на порядках."""
    _, _, lo, hi, log, _ = spec
    v = float(np.clip(value, lo, hi))
    f = (np.log(v / lo) / np.log(hi / lo)) if log else (v - lo) / (hi - lo)
    return track[0] + f * track[2]


def noise_from_px(spec, px, track):
    """Пиксель -> значение."""
    _, _, lo, hi, log, _ = spec
    f = float(np.clip((px - track[0]) / track[2], 0.0, 1.0))
    return lo * (hi / lo) ** f if log else lo + f * (hi - lo)


def noise_hit(track, pos) -> bool:
    x, y, w, h = track
    return x - 8 <= pos[0] <= x + w + 8 and y - 9 <= pos[1] <= y + h + 9


def draw_noise(app, screen, pg, font):
    """Ряд слайдеров датчика и фильтра плюс две метки оптимального tau."""
    live = app.est_key != "ideal"
    for spec, (lx, ly), track, vx in noise_layout(font):
        key, label, lo, hi, _, fmt = spec
        screen.blit(font.render(label, True, DIM if live else GRID_LINE), (lx, ly))
        pg.draw.rect(screen, PANEL if live else BG, track, border_radius=4)
        pg.draw.rect(screen, GRID_LINE, track, 1, border_radius=4)
        value = app.noise[key]
        if key == "tau":
            # Метки рисуются ПОД дорожкой, чтобы не спорить с ручкой.
            for tau_star, color in ((app.tau_star_model, TAU_MODEL),
                                    (app.tau_star_measured, TAU_MEASURED)):
                if tau_star is None or not (lo <= tau_star <= hi):
                    continue
                mx = noise_to_px(spec, tau_star, track)
                pg.draw.line(screen, color, (mx, track[1] - 5),
                             (mx, track[1] + track[3] + 5), 2)
        px = noise_to_px(spec, value, track)
        fill = BTN_ON if live else GRID_LINE
        pg.draw.rect(screen, fill, (track[0], track[1], px - track[0], track[3]),
                     border_radius=4)
        pg.draw.circle(screen, TEXT if live else GRID_LINE,
                       (int(px), track[1] + track[3] // 2), 6)
        screen.blit(font.render(fmt.format(value), True, TEXT if live else GRID_LINE),
                    (vx, ly))

    last = noise_layout(font)[-1]
    hx = last[3] + font.size(last[0][5].format(0.000123))[0] + 22
    if not live:
        screen.blit(font.render("ideal: sensor knobs do nothing", True, GRID_LINE),
                    (hx, NOISE_Y))
        return
    # Две метки подписаны рядом с дорожкой, чтобы не гадать, какая из них
    # какая. "white" -- напоминание, что модель считает ошибку акселерометра
    # белой; всё расхождение с замером живёт именно в этом слове.
    for value, color, name in ((app.tau_star_model, TAU_MODEL, "model"),
                               (app.tau_star_measured, TAU_MEASURED, "meas")):
        if value is None and name == "meas":
            text = "meas: press T"
        else:
            text = f"{name} {'--' if value is None else format(value, '.3f')}"
        if hx + 12 + font.size(text)[0] > W - 24:
            break
        pg.draw.rect(screen, color, (hx, NOISE_Y + 4, 8, 8))
        screen.blit(font.render(text, True, color), (hx + 12, NOISE_Y))
        hx += 12 + font.size(text)[0] + 14
    tail = "tau*, white-accel model vs measured"
    if hx + font.size(tail)[0] <= W - 24:
        screen.blit(font.render(tail, True, GRID_LINE), (hx, NOISE_Y))


def u_max_to_px(u: float) -> float:
    """Предел момента -> пиксель на дорожке слайдера."""
    x, _, w, _ = SLIDER
    return x + (float(np.clip(u, U_MIN, U_MAX)) - U_MIN) / (U_MAX - U_MIN) * w


def u_max_from_px(px: float) -> float:
    """Пиксель -> предел момента, округлённый до шага слайдера.

    Округление живёт здесь, а не в обработчике: мышь и клавиши обязаны давать
    одни и те же значения, иначе замер «при 2.75 Н*м было так» не повторить.
    """
    x, _, w, _ = SLIDER
    f = float(np.clip((px - x) / w, 0.0, 1.0))
    return U_MIN + round(f * (U_MAX - U_MIN) / U_STEP) * U_STEP


def slider_hit(pos) -> bool:
    """Попали ли мышью в слайдер. Область на 8 px шире дорожки: ручка
    выступает за неё, и промах по собственной ручке -- это баг интерфейса."""
    x, y, w, h = SLIDER
    return x - 8 <= pos[0] <= x + w + 8 and y - 8 <= pos[1] <= y + h + 8


def limit_button_rect(font):
    """Прямоугольник кнопки «no limit». Чистая функция, как picker_layout:
    одна раскладка и на отрисовку, и на обработку клика."""
    x, y, w, h = SLIDER
    return (x + w + 122, y - 4, font.size("no limit")[0] + 18, 22)


def draw_slider(app, screen, pg, font):
    """Дорожка, ручка, число, кнопка отключения и строка о состоянии карты."""
    x, y, w, h = SLIDER
    off = app.u_max is None
    screen.blit(font.render("u_max", True, DIM), (PICK[0], y - 1))
    pg.draw.rect(screen, BTN, (x, y, w, h), border_radius=4)
    # При выключенном пределе дорожка гаснет, но ручка остаётся на своём
    # числе: слайдер помнит, куда вернётся, и это видно.
    shown = app.u_finite if off else app.u_max
    fill = max(u_max_to_px(shown) - x, 1.0)
    tone = BTN_ON
    if app.stale:
        tone = _shade(tone, 0.6)
    if off:
        tone = _shade(tone, 0.45)
    pg.draw.rect(screen, tone, (x, y, fill, h), border_radius=4)
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1, border_radius=4)

    # Отметка предела, за которым карта упирается в theta_fall: цвет тот же,
    # что у границы восстановимости, потому что это про неё и есть.
    if app.u_clip is not None:
        cx = u_max_to_px(app.u_clip)
        pg.draw.line(screen, LIMIT, (cx, y - 4), (cx, y + h + 4), 1)

    knob_color = DIM if off else (ACCENT if app.stale else TEXT)
    pg.draw.circle(screen, knob_color,
                   (int(u_max_to_px(shown)), int(y + h / 2)), 7)
    value = "  off    " if off else f"{app.u_max:5.2f} N*m"
    screen.blit(font.render(value, True, knob_color), (x + w + 14, y - 1))

    rect = limit_button_rect(font)
    pg.draw.rect(screen, BTN_ON if off else BTN, rect, border_radius=4)
    if not off:
        pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=4)
    screen.blit(font.render("no limit", True, BG if off else TEXT),
                (rect[0] + 9, rect[1] + 3))

    if app.stale:
        note, color = "release to recompute grid and c*", ACCENT
    elif off:
        note, color = "no limit: relay disabled, map frozen", LIMIT
    elif app.pinned:
        note, color = "map pinned by --theta-max / --dtheta-max", DIM
    elif app.u_clip is not None and app.u_max > app.u_clip:
        note = f"map clipped by theta_fall = {app.args.theta_fall:g}"
        color = LIMIT
    else:
        note = f"map: theta +-{MAP_FIT:g}*pi/2, dtheta {MAP_FIT:g}x recoverable"
        color = DIM
    screen.blit(font.render(note, True, color), (rect[0] + rect[2] + 16, y - 1))

    srect = settings_button_rect(font)
    pg.draw.rect(screen, BTN_ON if app.settings_open else BTN, srect, border_radius=4)
    pg.draw.rect(screen, GRID_LINE, srect, 1, border_radius=4)
    screen.blit(font.render("settings", True, BG if app.settings_open else TEXT),
                (srect[0] + 9, srect[1] + 3))


def settings_button_rect(font):
    """Кнопка «settings» в правом конце ряда u_max. Чистая функция, как
    limit_button_rect: одна раскладка и на отрисовку, и на обработку клика."""
    w = font.size("settings")[0] + 18
    return (W - 24 - w, SLIDER[1] - 4, w, 22)


def settings_layout(font):
    """Раскладка панели настроек: [(kind, key, rect)], kind -- "field" или
    "button". Чистая функция: отрисовка и клик читают одну и ту же раскладку."""
    x, y, w, h = SETTINGS
    out = []
    col_x = {"base": x + 160, "soft": x + 340}
    for group in ("base", "soft"):
        for i in range(5):
            out.append(("field", f"{group}.{i}", (col_x[group], y + 90 + i * 30, 150, 24)))
    out.append(("field", "eps", (col_x["base"], y + 250, 150, 24)))
    out.append(("field", "horizon", (col_x["base"], y + 280, 150, 24)))
    bx = x + 24
    for key in ("apply", "defaults", "close"):
        bw = font.size(key)[0] + 24
        out.append(("button", key, (bx, y + h - 44, bw, 26)))
        bx += bw + 10
    return out


def draw_settings(app, screen, pg, font):
    """Панель настроек поверх окна: веса Q и R двух ЛКР, eps и горизонт.

    Рисует только текст полей и числа K, посчитанные окном; сама проверка и
    синтез -- в Explorer.apply_settings.
    """
    if not app.settings_open:
        return
    veil = pg.Surface((W, H), pg.SRCALPHA)
    veil.fill(BG + (150,))
    screen.blit(veil, (0, 0))
    x, y, w, h = SETTINGS
    pg.draw.rect(screen, PANEL, SETTINGS, border_radius=6)
    pg.draw.rect(screen, GRID_LINE, SETTINGS, 1, border_radius=6)
    screen.blit(font.render("SETTINGS  --  LQR cost  Q = diag(q_theta, q_phi, q_dtheta, q_dphi)",
                            True, TEXT), (x + 24, y + 16))
    screen.blit(font.render("base: lqr, bang-ell*, bang-map, phase 3;  soft: phase 2 of bang-eps*",
                            True, DIM), (x + 24, y + 38))
    screen.blit(font.render("base", True, TEXT), (x + 160, y + 66))
    screen.blit(font.render("soft", True, TEXT), (x + 340, y + 66))
    for i, label in enumerate(COST_LABELS):
        screen.blit(font.render(label, True, DIM), (x + 24, y + 94 + i * 30))
    screen.blit(font.render("eps, rad", True, DIM), (x + 24, y + 254))
    screen.blit(font.render("horizon, s", True, DIM), (x + 24, y + 284))

    for kind, key, rect in settings_layout(font):
        if kind == "field":
            focused = key == app.settings_focus
            pg.draw.rect(screen, BG, rect, border_radius=3)
            pg.draw.rect(screen, ACCENT if focused else GRID_LINE, rect, 1, border_radius=3)
            text = app.settings_text.get(key, "")
            screen.blit(font.render(text + ("_" if focused else ""), True, TEXT),
                        (rect[0] + 6, rect[1] + 4))
        else:
            fill = BTN_ON if key == "apply" else BTN
            pg.draw.rect(screen, fill, rect, border_radius=4)
            screen.blit(font.render(key, True, BG if key == "apply" else TEXT),
                        (rect[0] + 12, rect[1] + 5))

    # Числа, которыми посчитана ТЕКУЩАЯ карта: видно, что именно сделала цена.
    lines = []
    for name, K in (("K base", app.K), ("K soft", app.K_soft)):
        if K is not None:
            lines.append(f"{name} = [" + ", ".join(f"{v:.3g}" for v in K[0]) + "]")
    if app.P is not None and app.K is not None:
        A, B = app.system.linearize_upright()
        for name, K in (("base", app.K), ("soft", app.K_soft)):
            if K is None:
                continue
            slow = float(np.linalg.eigvals(A - B @ K).real.max())
            lines.append(f"slowest pole {name}: {slow:.3g}  (~{-4.0 / slow:.0f} s to 2%)")
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, DIM), (x + 24, y + 330 + i * 20))
    if app.settings_error:
        screen.blit(font.render(app.settings_error, True, (235, 110, 90)),
                    (x + 24, y + h - 78))
    hint = "Enter apply   Tab next   Esc close"
    screen.blit(font.render(hint, True, DIM), (x + w - 24 - font.size(hint)[0], y + h - 39))


def settings_click(app, pos, font):
    """Клик при открытой панели: фокус поля или кнопка. Вне панели -- ничего:
    панель модальная, иначе клик по карте молча пересчитал бы клетку под ней."""
    for kind, key, (rx, ry, rw, rh) in settings_layout(font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            if kind == "field":
                app.settings_focus = key
            elif key == "apply":
                settings_submit(app)
            elif key == "defaults":
                for group in ("base", "soft"):
                    for i, v in enumerate(DEFAULT_COST[group]):
                        app.settings_text[f"{group}.{i}"] = f"{v:g}"
            elif key == "close":
                app.settings_open = False
            return


def settings_submit(app):
    """Применить введённое. Ошибка остаётся на панели, окно -- прежним."""
    error = app.apply_settings(dict(app.settings_text))
    app.settings_error = error or ""
    if error is None:
        app.settings_open = False


def settings_key(app, event, pg):
    """Клавиша при открытой панели. Возвращает True, если её съела панель."""
    if event.key == pg.K_ESCAPE:
        app.settings_open = False
    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
        settings_submit(app)
    elif event.key in (pg.K_TAB, pg.K_DOWN, pg.K_UP):
        step = -1 if (event.key == pg.K_UP or event.mod & pg.KMOD_SHIFT) else 1
        i = SETTINGS_KEYS.index(app.settings_focus) if app.settings_focus else -1
        app.settings_focus = SETTINGS_KEYS[(i + step) % len(SETTINGS_KEYS)]
    elif app.settings_focus is not None:
        text = app.settings_text.get(app.settings_focus, "")
        if event.key == pg.K_BACKSPACE:
            app.settings_text[app.settings_focus] = text[:-1]
        elif event.unicode and event.unicode in SETTINGS_CHARS:
            app.settings_text[app.settings_focus] = text + event.unicode
    return True


def draw_picker(app, screen, pg, font):
    for row, name in enumerate(PICK_ROWS):
        screen.blit(font.render(name, True, DIM),
                    (PICK[0], PICK[1] + row * PICK_ROW + 3))
    for rect, row, key in picker_layout(app, font):
        active = key == (app.main_key, app.cmp_key, app.est_key)[row]
        # На ряду оценивателей кнопка может быть выбрана ВТОРЫМ образом:
        # Shift-клик назначает её сравнением. Цвет тогда другой -- тот же,
        # которым нарисована её траектория.
        as_cmp = row == 2 and key == app.est_cmp_key and not active
        enabled = row == 2 or key is None or app.available(key)
        if active:
            fill = (BTN_ON, GHOST, ESTIMATE)[row]
        elif as_cmp:
            fill = GHOST_EST
        else:
            fill = BTN
        pg.draw.rect(screen, fill, rect, border_radius=4)
        if not (active or as_cmp):
            pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=4)
        label = pick_label(row, key)
        color = BG if (active or as_cmp) else (TEXT if enabled else GRID_LINE)
        screen.blit(font.render(label, True, color), (rect[0] + 9, rect[1] + 3))
        if not enabled:
            # Перечёркнута -- значит выключена не «просто так»: рядом написано,
            # чего не хватает. Молчащая мёртвая кнопка -- это баг интерфейса.
            x, y, w, h = rect
            pg.draw.line(screen, DIM, (x + 6, y + h // 2), (x + w - 6, y + h // 2), 1)

    if app.design_hint:
        # Подсказка встаёт справа от ряда ОЦЕНИВАТЕЛЬ: он самый короткий, и
        # только там есть место. У ряда MAIN кнопки доходят почти до края, и
        # подсказка наезжала бы на них.
        last = max((it for it in picker_layout(app, font) if it[1] == 2),
                   key=lambda item: item[0][0] + item[0][2])
        # Не даём подсказке уехать за край окна: обрезанная подсказка хуже,
        # чем подсказка, наехавшая на кнопку.
        hx = min(last[0][0] + last[0][2] + 16,
                 W - 24 - font.size(app.design_hint)[0])
        screen.blit(font.render(app.design_hint, True, ACCENT),
                    (hx, PICK[1] + 2 * PICK_ROW + 3))

    draw_slider(app, screen, pg, font)
    draw_noise(app, screen, pg, font)


def draw_header(app, screen, pg, font, big):
    screen.blit(big.render("wpend explorer  --  wheeled pendulum", True, TEXT), (24, 12))
    mode = "grid" if app.batch is not None else "on-demand"
    right = (f"horizon={app.args.horizon:g}s  "
             f"dt={app.args.dt:g}  stride={app.args.stride}  {mode}  "
             f"[{app.compute_seconds:.2f}s]  {app.design_note}"
             + ("" if app.c_tilt is None else f"  c*t={app.c_tilt:.3g}"))
    screen.blit(font.render(right, True, ACCENT if app.design_hint else DIM),
                (W - 24 - font.size(right)[0], 16))
    draw_picker(app, screen, pg, font)

    legend = [("held", HELD), ("fell forward", FELL_FORWARD), ("fell backward", FELL_BACKWARD)]
    lx = MAP[0] - 32
    for text, code in legend:
        pg.draw.rect(screen, OUTCOME_COLOR[code], (lx, H - 52, 14, 14))
        screen.blit(font.render(text, True, DIM), (lx + 20, H - 52))
        lx += 40 + font.size(text)[0]
    if app.theta_eq is None:
        # Свотча нет сознательно: на карте нет цвета, который он объяснял бы.
        lim_text = "no torque limit -- every state is recoverable"
        screen.blit(font.render(lim_text, True, DIM), (lx, H - 52))
        lx += 20 + font.size(lim_text)[0]
    else:
        pg.draw.rect(screen, LIMIT, (lx, H - 52, 14, 14))
        lim_text = f"recoverable limit (theta_eq = {app.theta_eq:.3f})"
        screen.blit(font.render(lim_text, True, LIMIT), (lx + 20, H - 52))
        lx += 40 + font.size(lim_text)[0]
    if app.cmp_key is not None:
        pg.draw.rect(screen, GHOST, (lx, H - 52, 14, 14))
        text = f"compare: {CONTROLLER_LABEL[app.cmp_key]}"
        screen.blit(font.render(text, True, GHOST), (lx + 20, H - 52))
        lx += 40 + font.size(text)[0]
    if app.est_key != "ideal":
        pg.draw.rect(screen, ESTIMATE, (lx, H - 52, 14, 14))
        text = f"estimate: {ESTIMATOR_LABEL[app.est_key]}"
        screen.blit(font.render(text, True, ESTIMATE), (lx + 20, H - 52))
        lx += 40 + font.size(text)[0]
    if app.est_cmp_key is not None:
        pg.draw.rect(screen, GHOST_EST, (lx, H - 52, 14, 14))
        screen.blit(font.render(f"vs est: {ESTIMATOR_LABEL[app.est_cmp_key]}",
                                True, GHOST_EST), (lx + 20, H - 52))
    hint = ("click a cell  |  1-8 main, Shift+1-8 compare  |  "
            "E / Shift+E estimator, T measure tau*  |  [ ] u_max, L no limit  |  "
            "SPACE pause  R restart  ESC quit")
    screen.blit(font.render(hint, True, GRID_LINE), (W - 24 - font.size(hint)[0], H - 26))


# Раскладка выше (W, H, MAP, ANIM, PLOT, SLIDER) задана в пикселях и подогнана
# вручную: панели стоят впритык, а отступы внутри draw_* записаны литералами.
# Поэтому под маленький экран окно НЕ перевёрстывается. Оно рисуется в свой
# размер на холсте W x H, а на экран кладётся уменьшенной копией: геометрия
# остаётся одна на всех машинах, скриншоты сравнимы между запусками, и ни одна
# из отрисовок не знает, что её ужали.
#
# Флаг pg.SCALED сделал бы то же силами SDL, но он не ужимает: при экране
# меньше холста окно всё равно создаётся 1280x940 и уезжает за край, а менять
# его размер SCALED не даёт (проверено на pygame 2.6.1 / SDL 2.28.4). Отсюда
# масштаб руками.

TASKBAR = 96                        # запас под заголовок окна и панель задач


def fit_scale(pg):
    """Во сколько раз ужать холст, чтобы он поместился на рабочий стол.

    Больше единицы не возвращает: растягивать пятнадцатый шрифт незачем, а на
    большом экране окно и так на своём месте.

    Отдельная функция, а не пара строк в run: в headless её ответ подменяется,
    и тогда путь «клик в ужатом окне» становится проверяемым.
    """
    if os.environ.get("SDL_VIDEODRIVER") == "dummy":
        # Экрана нет, dummy-драйвер сообщает выдуманные 1024x768.
        return 1.0
    try:
        dw, dh = pg.display.get_desktop_sizes()[0]
    except (AttributeError, IndexError, pg.error):
        return 1.0                  # старый pygame или экрана нет -- как было
    return max(min(dw / W, (dh - TASKBAR) / H, 1.0), 0.2)


def view_rect(pg, size):
    """Куда лечь холсту в окне: масштаб общий по осям, остаток -- поля.

    Общий, а не по каждой оси отдельно, потому что карта -- квадратная сетка
    начальных условий: разное сжатие по x и y превратило бы клетки в
    прямоугольники, а робота -- в эллипс, и картинка начала бы врать о том,
    что нарисовано.
    """
    ww, wh = size
    k = min(ww / W, wh / H)
    w, h = max(round(W * k), 1), max(round(H * k), 1)
    return pg.Rect((ww - w) // 2, (wh - h) // 2, w, h)


def canvas_pos(pos, view):
    """Координаты мыши из окна в холст.

    Обработчики сравнивают позицию с прямоугольниками раскладки (MAP, SLIDER,
    кнопки), а те заданы в координатах холста. Забыть перевод -- получить окно,
    где всё нарисовано верно, но клик попадает мимо, и промах тем больше, чем
    сильнее ужато окно. Ошибка тихая: программа не падает.
    """
    return (round((pos[0] - view.x) * W / view.w),
            round((pos[1] - view.y) * H / view.h))


def run(app, max_frames=None, screenshot=None):
    import pygame as pg

    pg.init()
    k = fit_scale(pg)
    window = pg.display.set_mode((round(W * k), round(H * k)), pg.RESIZABLE)
    # Холст заводится ПОСЛЕ set_mode: так он берёт формат дисплея, и
    # smoothscale ниже не упирается в неподходящую глубину цвета.
    screen = pg.Surface((W, H))
    pg.display.set_caption("wpend explorer")
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 15)
    big = pg.font.SysFont("consolas,dejavusansmono,monospace", 20, bold=True)
    clock = pg.time.Clock()

    x, y, w, h = MAP
    cell_w, cell_h = w / app.spec.n, h / app.spec.n
    running, frames = True, 0
    while running:
        dt_wall = clock.tick(60) / 1000.0
        view = view_rect(pg, window.get_size())
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
            elif event.type == pg.VIDEORESIZE:
                window = pg.display.set_mode(event.size, pg.RESIZABLE)
                view = view_rect(pg, window.get_size())
            elif app.settings_open and event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    settings_click(app, canvas_pos(event.pos, view), font)
            elif app.settings_open and event.type == pg.KEYDOWN:
                settings_key(app, event, pg)
            elif app.settings_open and event.type in (pg.MOUSEMOTION, pg.MOUSEBUTTONUP):
                continue
            elif event.type == pg.MOUSEMOTION:
                mx, my = canvas_pos(event.pos, view)
                if app.dragging:
                    app.preview_u_max(u_max_from_px(mx))
                    continue
                if app.noise_drag is not None:
                    spec, track = next((sp, tr) for sp, _, tr, _ in noise_layout(font)
                                       if sp[0] == app.noise_drag)
                    app.preview_noise(app.noise_drag, noise_from_px(spec, mx, track))
                    continue
                inside = x <= mx < x + w and y <= my < y + h
                app.hover = ((int((mx - x) / cell_w),
                              app.spec.n - 1 - int((my - y) / cell_h)) if inside else None)
            elif event.type == pg.MOUSEBUTTONDOWN and event.button == 1:
                pos = canvas_pos(event.pos, view)
                shift = bool(pg.key.get_mods() & pg.KMOD_SHIFT)
                bx, by, bw, bh = limit_button_rect(font)
                if bx <= pos[0] < bx + bw and by <= pos[1] < by + bh:
                    app.toggle_limit()
                    continue
                sx, sy, sw, sh = settings_button_rect(font)
                if sx <= pos[0] < sx + sw and sy <= pos[1] < sy + sh:
                    app.open_settings()
                    continue
                if slider_hit(pos):
                    app.dragging = True
                    app.preview_u_max(u_max_from_px(pos[0]))
                    continue
                if app.est_key != "ideal":
                    hit = next((sp for sp, _, tr, _ in noise_layout(font)
                                if noise_hit(tr, pos)), None)
                    if hit is not None:
                        track = next(tr for sp, _, tr, _ in noise_layout(font)
                                     if sp[0] == hit[0])
                        app.noise_drag = hit[0]
                        app.preview_noise(hit[0], noise_from_px(hit, pos[0], track))
                        continue
                for rect, row, key in picker_layout(app, font):
                    rx, ry, rw, rh = rect
                    if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
                        if row == 2 and shift:
                            # Повторный Shift-клик по той же кнопке выключает
                            # сравнение: отдельной кнопки "off" ряду не нужно.
                            app.set_estimator_compare(key)
                        else:
                            (app.set_main, app.set_compare,
                             app.set_estimator)[row](key)
                        break
                else:
                    if app.hover:
                        app.select(*app.hover)
            elif event.type == pg.MOUSEBUTTONUP and event.button == 1:
                # Пересчёт ровно здесь: на отпускании, а не на каждом пикселе.
                # 0.8 с на сертификат плюс секунда-две на сетку -- это не
                # частота кадров, и тащить слайдер стало бы нельзя.
                if app.dragging:
                    app.dragging = False
                    app.commit_u_max()
                elif app.noise_drag is not None:
                    # Пересчёт ровно здесь, как и у слайдера момента: сетка
                    # плюс развёртка по tau -- это секунды, а не кадр.
                    app.noise_drag = None
                    app.commit_noise()
            elif event.type == pg.KEYDOWN:
                shift = bool(event.mod & pg.KMOD_SHIFT)
                digit = event.key - pg.K_0
                if event.key in (pg.K_ESCAPE, pg.K_q):
                    running = False
                elif event.key == pg.K_SPACE:
                    app.playing = not app.playing
                elif event.key == pg.K_r:
                    app.play_time, app.playing = 0.0, True
                elif event.key == pg.K_l:
                    app.toggle_limit()
                elif event.key == pg.K_s:
                    app.open_settings()
                elif event.key == pg.K_t:
                    # Дорого (секунды), поэтому по явной просьбе, а не на
                    # каждом движении слайдера.
                    app.measure_tau_star()
                elif event.key == pg.K_e:
                    # Перебор по кругу: отдельных цифр под оценивателей нет,
                    # а Shift+цифра уже занят сравнением регуляторов.
                    # Shift+E двигает ВТОРОЙ оцениватель -- так же, как Shift
                    # всюду в окне означает "сравнение".
                    if shift:
                        keys = [None] + ESTIMATOR_KEYS
                        i = keys.index(app.est_cmp_key)
                        nxt = keys[(i + 1) % len(keys)]
                        app.est_cmp_key = nxt
                        app._update_estimator_compare()
                    else:
                        i = ESTIMATOR_KEYS.index(app.est_key)
                        app.set_estimator(ESTIMATOR_KEYS[(i + 1) % len(ESTIMATOR_KEYS)])
                elif event.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    # Шаг при выключенном пределе возвращает его: иначе
                    # клавиша молча ничего не делает.
                    step = U_STEP if event.key == pg.K_RIGHTBRACKET else -U_STEP
                    app.preview_u_max(app.u_finite if app.u_max is None
                                      else app.u_max + step)
                    app.commit_u_max()
                elif shift and digit == 0:
                    app.set_compare(None)
                elif 1 <= digit <= len(CONTROLLER_KEYS):
                    key = CONTROLLER_KEYS[digit - 1]
                    (app.set_compare if shift else app.set_main)(key)

        app.advance(dt_wall)

        screen.fill(BG)
        draw_header(app, screen, pg, font, big)
        draw_map(app, screen, pg, font)
        draw_robot(app, screen, pg, font)
        draw_plots(app, screen, pg, font)
        draw_settings(app, screen, pg, font)
        if view.size == (W, H):
            # Масштаб 1 -- кладём как есть: пересемплировать нечего, и текст
            # остаётся ровно тем, что нарисовал шрифт.
            window.blit(screen, view)
        else:
            window.fill(BG)         # поля, если окно другой пропорции
            window.blit(pg.transform.smoothscale(screen, view.size), view)
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
    p.add_argument("--main", choices=CONTROLLER_KEYS, default="lqr",
                   help="основной регулятор: он красит карту")
    p.add_argument("--compare", choices=CONTROLLER_KEYS, default=None,
                   help="регулятор сравнения: рисуется полупрозрачным поверх")
    p.add_argument("--estimator", choices=ESTIMATOR_KEYS, default="ideal",
                   help="что видит регулятор: ideal -- истину, остальные -- "
                        "показания ИДУ через комплементарный фильтр")
    p.add_argument("--estimator-compare", choices=ESTIMATOR_KEYS, default=None,
                   help="второй оцениватель: считается для выбранной клетки и "
                        "накладывается полупрозрачным")
    p.add_argument("--sigma-g", type=float, default=1e-4,
                   help="плотность шума гироскопа, рад/с/sqrt(Гц)")
    p.add_argument("--sigma-a", type=float, default=1e-3,
                   help="плотность шума акселерометра, м/с^2/sqrt(Гц)")
    p.add_argument("--b0-g", type=float, default=1e-2,
                   help="СКО смещения включения гироскопа, рад/с")
    p.add_argument("--tau", type=float, default=0.30,
                   help="постоянная времени комплементарного фильтра, с")
    p.add_argument("--counts", type=int, default=2048,
                   help="меток на оборот у энкодера (вариант IMU+encoder)")
    p.add_argument("--seed", type=int, default=0,
                   help="зерно шума датчика (при --estimator, отличном от ideal)")
    p.add_argument("--eps", type=float, default=0.05,
                   help="bang-eps: реле отдаёт управление ЛКР, когда |theta| < eps, рад")
    p.add_argument("--u-max", type=float, default=3.0,
                   help="предел момента, Н*м -- начальное положение слайдера "
                        f"(диапазон {U_MIN:g}..{U_MAX:g}); inf -- запустить "
                        "без ограничения")
    p.add_argument("--horizon", type=float, default=5.0, help="горизонт прогона, с")
    p.add_argument("--dt", type=float, default=1e-3, help="шаг интегрирования, с")
    p.add_argument("--stride", type=int, default=10,
                   help="сохранять каждый stride-й кадр (счёт идёт с шагом dt)")
    p.add_argument("--theta-max", type=float, default=None,
                   help="границы карты по theta0; по умолчанию подгоняются под "
                        "u_max, явное значение их замораживает")
    p.add_argument("--dtheta-max", type=float, default=None,
                   help="границы карты по dtheta0; по умолчанию подгоняются под "
                        "u_max, явное значение их замораживает")
    p.add_argument("--theta-fall", type=float, default=float(np.pi),
                   help="угол, начиная с которого считаем, что корпус упал "
                        "(по умолчанию pi: земли в модели нет)")
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
