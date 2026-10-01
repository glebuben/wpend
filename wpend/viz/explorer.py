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
горизонта; кнопка noise или N -- панель корреляций каналов ИДУ; кнопка switch
или C -- показать / спрятать кривую переключения реле sigma = 0 на карте
(линия уровня первого интеграла через (0, 0)); ПРОБЕЛ -- пауза; R -- сначала;
Esc -- выход.

Мир, модель и фильтр Калмана (ER-025, docs/explorer_kalman.md): W или кнопка
`world` -- чистый мир / ветер (DisturbedWheeledPendulum); M или `model` --
физика мира и номинала (что знают регулятор и фильтр) и параметры ветра;
K или `kalman` -- чип (калиброванный / сырой MPU-6050), априор смещений
(code / калибровка на подставке), модель толчка в фильтре; P или `[P] u | w`
-- нижний график: момент или толчок (истина против оценки kalman+enc).
Виды фильтра (docs/kalman_views.md): V или `[V] P ellipse` -- ошибка
(psi, dpsi) и эллипс 1 sigma по P; H или `[H] y vs h(x,u)` -- показания и
предсказание фильтра против чистого h(x, u).  Вид закрывает правую половину,
карта и дорожка времени остаются.
В окне две модели: app.world -- что происходит (прогон, датчики, граница
восстановимости, карта), app.system -- что знает проект (K, P, c*, реле,
фильтры).  По умолчанию это один объект.

Время выбирается дорожкой под графиками (ER-015). Она не перематывает прогон,
а выбирает кадр УЖЕ посчитанной траектории, и на этот кадр смотрят сразу все
панели: карта, робот, оба графика и обе сравниваемые системы. Схваченная ручка
останавливает воспроизведение и держит кадр, пока не нажмут ПРОБЕЛ; стрелки
<- и -> двигают ровно на кадр, с Shift -- на десять. Каждая панель подписана
легендой теми же цветами, какими нарисованы кривые: без неё картинку читает
только тот, кто писал код. Панель V = x^T P x убрана (решение Глеба 21.09):
в логарифмической шкале три кривые и два уровня не читались.

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
проекта: psi до ±1.3·pi/2, dpsi -- 1.3 от восстановимого множества на
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
    EnergyBangBangController,
    LinearFeedbackController,
    all_regions,
    ellipsoid_region,
    energy_band_region,
    psi_band_region,
)
from ..estimator import (ComplementaryEstimator, KalmanEstimator, calibrate_on_stand,
                         optimal_tau)
from ..integrator import RK4Integrator
from ..models import DisturbedWheeledPendulum, WheeledPendulum, WheeledPendulumParams
from ..rollout import rollout_many
from ..sensor import (MPU6050_RAW_B0, EncoderSensor, IMUSensor, StackedSensor,
                      draw_turn_on_bias)
from .grid import FELL_BACKWARD, FELL_FORWARD, HELD, GridSpec, classify
from .phases import (PHASE_BANG, PHASE_FINAL, PHASE_LABEL, PHASE_LQR,
                     track_line, track_summary)

# ЛКР для линеаризации в верхнем положении, Q = diag(100, 1, 10, 1), R = 1.
# Синтез (wpend.lqr.lqr) требует scipy и вызывается только при старте окна;
# число здесь -- страховка и одновременно проверка, что параметры модели не
# уехали: если синтез даст другое K, окно скажет об этом вслух.
K_LQR = np.array([[95.122221, 1.0, 19.807871, 1.449624]])

I_PSI, I_THETA, I_DPSI, I_DTHETA = 0, 1, 2, 3

W, H = 1280, 940
PICK = (24, 44)                     # x, y первой строки кнопок
PICK_ROW = 27
SLIDER = (132, 129, 300, 14)        # дорожка u_max: x, y, w, h
NOISE_Y = 156                       # ряд слайдеров датчика и фильтра
MAP = (56, 214, 568, 632)           # x, y, w, h
ANIM = (656, 214, 600, 316)
PLOT = (656, 546, 600, 300)
#: Дорожка времени и подписи под ней живут ВНУТРИ PLOT, см. plot_layout.
TIME_H = 14
TIME_LABEL_H = 18
#: Дорожка фаз (ER-016) -- между графиком u и дорожкой времени. Высота на три
#: строки: основной регулятор и до двух призраков; одна строка тоньше 5 px
#: уже не читается, отсюда 18.
PHASE_H = 18
#: Легенда карты -- две строки под ней. Верхняя принадлежит задаче (цвета
#: исходов, граница восстановимости), нижняя -- нажатым кнопкам.
LEGEND_Y = 870
LEGEND_ROW = 22

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

# Диапазон слайдера. Сверху 10 Н*м: там седло уже за psi_fall, и карта
# перестаёт следовать за восстановимым множеством -- окно говорит об этом
# вслух. Снизу 0.25 -- режим, где восстановимо почти ничто. Шаг общий для мыши
# и клавиатуры, иначе один и тот же замер не повторить.
U_MIN, U_MAX, U_STEP = 0.25, 10.0, 0.25

#: Ключи вариантов, которым нужен КОНЕЧНЫЙ предел: реле подаёт u = +-u_max и
#: переключается по сепаратрисе при том же пределе. Без предела не определено
#: ни то, ни другое, поэтому окно их отключает -- как отключает эллипсоид без
#: scipy: кнопка перечёркнута, и рядом написано, чего не хватает.
RELAY_KEYS = ("bang-eps-ell", "bang-energy-ell")

#: Цена ЛКР второй фазы у `bang-eps`, R = 250 (ER-017, A38; прежняя --
#: diag(100, 1e-2, 10, 1e-2) при R = 1, A26). Прежняя держала всё
#: восстановимое только на карте +-pi/2; на стандартной карте с запасом 1.3
#: при слабом моторе мягкий ЛКР после передачи упирался в предел и выносил
#: состояние за сепаратрису: 9 из 27 клеток при 1.5 Н*м, 53 из 57 при 3.
#: Почему R: при R >> q_psi гейны по наклону почти не меняются (58 -> 52,
#: полюса наклона -- зеркало неустойчивых), а гейн по скорости колеса
#: падает 0.53 -> 0.14; именно его доля момента и не помещалась в предел.
#: Остаётся одна ручка -- отношение q_theta / R: оно задаёт медленную пару
#: полюсов колеса. q_theta = 2e-2 -- наибольшее из проверенных, при котором
#: реле -> |psi| < 0.05 -> soft держит 100% восстановимого при u_max =
#: 1, 1.5, 2, 3, 5, 10 (карта 31x31, 30 с; при 1 и 1.5 ещё и 61x61);
#: 3e-2 уже теряет 2 из 91 при 1 Н*м. Цена -- колесо: медленный полюс -0.065
#: против -0.216, колесо возвращается за ~45 с вместо ~15.
Q_SOFT_WHEEL = np.diag([100.0, 2e-2, 10.0, 1e-2])
R_SOFT = 250.0

#: Цены ЛКР, с которыми окно стартует: (q_psi, q_theta, q_dpsi, q_dtheta, R).
#: Панель настроек (кнопка `settings`, клавиша S) меняет КОПИЮ -- app.cost, --
#: а эти числа остаются опорой: кнопка `defaults` возвращает ровно их.
#: base -- K для `lqr` и третьей фазы обеих релейных схем;
#: soft -- K второй фазы обеих релейных схем.
DEFAULT_COST = {
    "base": (100.0, 1.0, 10.0, 1.0, 1.0),
    "soft": tuple(float(q) for q in np.diag(Q_SOFT_WHEEL)) + (R_SOFT,),
}
COST_LABELS = ("q_psi", "q_theta", "q_dpsi", "q_dtheta", "R")

#: Во сколько раз карта шире восстановимого множества. 1.0 -- граница ровно по
#: краю, и не видно, что снаружи неё карта обязана быть красной у любого
#: регулятора; 1.3 оставляет поля, в которых это видно.
MAP_FIT = 1.3
#: Горизонт корпуса. Отмечается на карте пунктиром, когда попадает в границы.
HORIZON_ANGLE = float(np.pi / 2)
#: Стандарт карт (CLAUDE.md, Глеб 16.09 и 17.09): интересен psi в
#: [-pi/2, pi/2], сетка берётся с запасом MAP_FIT по обеим осям.
PSI_MAP = HORIZON_ANGLE

BG = (22, 24, 28)
PANEL = (33, 36, 42)
GRID_LINE = (52, 56, 64)
TEXT = (216, 219, 224)
DIM = (140, 146, 156)
ACCENT = (240, 198, 92)
CURVE = (245, 247, 250)
#: Направляющие на графиках (psi_fall, +-u_max) и их подписи в легенде. Были
#: почти чёрными (70, 60, 50): саму линию на панели видно, а подпись её
#: цветом -- уже нет, и легенда получилась бы нечитаемой ровно там, где она
#: и нужна.
GUIDE = (152, 134, 102)
#: Цвета кривых графиков. Вынесены из draw_plots: их называет легенда, и
#: разойтись рисунку с расшифровкой нельзя.
PSI_CURVE = (120, 180, 240)
U_CURVE = (240, 170, 110)
#: Цвета дорожки фаз (ER-016). Красный конец -- реле: мотор выжат в упор;
#: синий -- ЛКР после передачи; зелёный -- третья фаза, сертифицированный
#: эллипсоид. Цвета нарочно не совпадают с цветами кривых: полоса отвечает на
#: другой вопрос -- не «сколько», а «кто сейчас рулит».
PHASE_COLOR = {
    PHASE_BANG: (198, 112, 72),
    PHASE_LQR: (72, 140, 200),
    PHASE_FINAL: (110, 200, 160),
}

# Сравнение всюду рисуется этим цветом и этой прозрачностью -- один цвет на
# все три панели, чтобы глаз связывал призрака робота, вторую фазовую кривую и
# вторые графики в одного и того же «второго».
GHOST = (104, 214, 200)
GHOST_A = 132
LIMIT = (170, 255, 120)     # граница восстановимости -- свойство системы
#: Кривая переключения реле sigma = 0 (кнопка `switch`, клавиша C). Цвет --
#: родственник красного конца дорожки фаз (реле), но ярче: кривая лежит
#: поверх карты и должна читаться и на синем, и на красном фоне.
SWITCH = (255, 150, 70)
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
    """(q_psi, q_theta, q_dpsi, q_dtheta, R) -> (Q, R) для `wpend.lqr.lqr`.
    Порядок весов -- порядок состояния (I_PSI, I_THETA, I_DPSI, I_DTHETA)."""
    q = np.zeros(4)
    q[[I_PSI, I_THETA, I_DPSI, I_DTHETA]] = cost[:4]
    return np.diag(q), np.array([[float(cost[4])]])


#: Поля панели настроек в порядке обхода по Tab: сначала столбец base, потом
#: soft, потом eps и горизонт.
SETTINGS_KEYS = ([f"base.{i}" for i in range(5)] + [f"soft.{i}" for i in range(5)]
                 + ["eps", "horizon"])
SETTINGS = (300, 190, 680, 520)     # панель настроек: x, y, w, h
#: Символы, которые принимает поле ввода: число в любой записи Python.
SETTINGS_CHARS = set("0123456789.eE-+")

#: Панель шума (кнопка `noise`, клавиша N): корреляции каналов ИДУ
#: [a_x, a_z, gyro] -- по матрице на компоненту, как в IMUSensor
#: (PROPOSALS.md A29).  Матрица 3x3 с единицами на диагонали задаётся тремя
#: числами над диагональю; остальное -- симметрия, поэтому несимметричную
#: матрицу из панели ввести нельзя в принципе.
CORR_MATRICES = (("w", "white n", "corr_w"),
                 ("b", "drift b", "corr_b"),
                 ("b0", "turn-on b0", "corr_b0"))
CORR_PAIRS = ((0, 1, "a_x - a_z"), (0, 2, "a_x - gyro"), (1, 2, "a_z - gyro"))
#: Старт -- каналы независимы, ровно прежняя модель.
DEFAULT_CORR = {m: (0.0, 0.0, 0.0) for m, _, _ in CORR_MATRICES}
#: Порядок обхода по Tab: матрица за матрицей (столбец за столбцом панели).
NOISE_KEYS = [f"{m}.{i}" for m, _, _ in CORR_MATRICES for i in range(len(CORR_PAIRS))]
NOISE_PANEL = (300, 190, 680, 400)  # панель шума: x, y, w, h


# ---------------------------------------------------------------------------
#  Мир, номинал и фильтр Калмана (ER-025, docs/explorer_kalman.md)
# ---------------------------------------------------------------------------
#  В окне ДВЕ модели: app.world -- что происходит (по ней идёт прогон, её
#  чувствуют датчики, от неё граница восстановимости и карта), app.system --
#  что знает проект (K, P, c*, реле, прогноз фильтров).  По умолчанию это
#  один и тот же объект, и окно бит в бит прежнее.

#: Физика машины по умолчанию -- поля WheeledPendulumParams.
PHYS_KEYS = ("mb", "mw", "l", "r")
DEFAULT_PHYS = {k: float(getattr(WheeledPendulumParams(), k)) for k in PHYS_KEYS}
#: Ветер (DisturbedWheeledPendulum): установившееся СКО по каналам, время
#: корреляции, зерно.  Решение Глеба 30.09: всё настраиваемое.
DEFAULT_WIND = {"sigma_psi": 0.3, "sigma_theta": 0.3, "tau_w": 0.2, "seed": 7.0}
#: Фильтр Калмана и процедура запуска: чип (свойство мира), априор смещений,
#: стоянка, уход, модель возмущения в фильтре, q_acc, итерации.
#: q_acc = 1e-4 рад/с^2/sqrt(Гц), а не 0 (решение Глеба 01.10, kalman_views §17.1):
#: при Q = 0 блок P по (psi, dpsi) вырождается в отрезок и фильтр самоуверен.
DEFAULT_KF = {"chip": "calibrated", "prior": "code", "T_c": 1.0, "sigma_jig": 1e-2,
              "drift": "off", "w_model": "matched", "sigma_psi": 0.3,
              "sigma_theta": 0.3, "tau_w": 0.2, "q_acc": 1e-4, "iterations": 1.0}
KF_CHOICES = {"chip": ("calibrated", "raw"), "prior": ("code", "calibrate"),
              "drift": ("off", "on"), "w_model": ("off", "matched", "own")}
#: Цвет истинного толчка на графике `w`.
WIND = (200, 200, 120)
#: Панель model (клавиша M): поля в порядке обхода по Tab.
MODEL_KEYS = ([f"world.{k}" for k in PHYS_KEYS] + [f"nominal.{k}" for k in PHYS_KEYS]
              + [f"wind.{k}" for k in DEFAULT_WIND])
MODEL_PANEL = (300, 170, 680, 540)
#: Панель kalman (клавиша K): числовые поля; выборы -- кнопками.
KF_FIELDS = ("T_c", "sigma_jig", "sigma_psi", "sigma_theta", "tau_w", "q_acc",
             "iterations")
KF_LABELS = {"T_c": "stand T_c, s", "sigma_jig": "stand sigma_jig, rad",
             "sigma_psi": "own w: sigma_psi", "sigma_theta": "own w: sigma_theta",
             "tau_w": "own w: tau_w, s", "q_acc": "q_acc", "iterations": "iterations"}
KF_PANEL = (300, 170, 680, 460)


def corr_matrix(rho):
    """Три числа над диагональью (порядок CORR_PAIRS) -> матрица корреляции 3x3."""
    R = np.eye(3)
    for (i, j, _), r in zip(CORR_PAIRS, rho):
        R[i, j] = R[j, i] = float(r)
    return R


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
    psi пересекает psi_fall задолго до inf, и клетка честно красится
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

def _energy_region(app):
    """Критерий передачи энергетической схемы: энергия на уровне И корпус в
    полосе. Собирается здесь, а не в лямбде ниже: два экземпляра одного
    условия разъехались бы при первой же правке."""
    return all_regions(
        energy_band_region(app.system, app.args.eps_e, energy=app.args.energy),
        psi_band_region(app.args.eps))


#: Три варианта, а не десять (решение Глеба 24.09). Раньше окно предлагало
#: восемь релейных схем -- следы поиска: реле с эллипсоидом, с привязкой
#: колеса, с ЛКР наклона, с картой, двухфазные версии. Поиск кончился, и
#: держать их на кнопках значит держать на экране вопросы, на которые уже
#: ответили (числа -- PROPOSALS.md A26, A27, §B8). Осталось ровно то, что
#: сравнивают сейчас: две трёхфазные схемы, различающиеся ПЕРВОЙ фазой и
#: критерием передачи, и чистый ЛКР как база.
#:
#: ЛКР держим не из осторожности, а потому что он нужен по существу: это
#: ответ на вопрос «что было бы без первой фазы вообще», и это единственный
#: вариант, который работает и без scipy, и при снятом пределе момента
#: (кнопка `no limit`) -- туда окно откатывает выбор, когда реле недоступно.
CONTROLLERS = [
    ("lqr", "LQR",
     lambda app: LinearFeedbackController(app.K)),
    # Реле по сепаратрисе -> |psi| < eps -> мягкий ЛКР -> базовый ЛКР по его
    # сертифицированному эллипсоиду (A26, A27).
    ("bang-eps-ell", "bang -> eps -> ell",
     lambda app: BangBangLQRController(app.K_soft, app.u_max,
                                       psi_band_region(app.args.eps),
                                       app.system, K_final=app.K,
                                       final_region=ellipsoid_region(app.P, app.c_star))),
    # То же, но первая фаза качает энергию к уровню верхнего положения, а
    # передача идёт по КОНЪЮНКЦИИ «энергия на уровне И корпус в полосе»
    # (ER-018, решение Глеба 22.09). Какую энергию -- задаёт --energy:
    # "tilt" (по умолчанию) -- энергию приведённой динамики наклона,
    # "total" -- полную энергию машины. Замер 22.09: tilt держит 9 / 55 / 153
    # клетки при 1.5 / 3 / 10 Н*м, total -- 1 / 3 / 11, сепаратриса -- 9 / 53 / 173.
    ("bang-energy-ell", "bang-energy -> ell",
     lambda app: EnergyBangBangController(
         app.K_soft, app.u_max, _energy_region(app), app.system,
         energy=app.args.energy, K_final=app.K,
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
#  Все неидеальные варианты -- ИДУ ПЛЮС МОТОРНЫЙ ЭНКОДЕР (решение Глеба и
#  куратора 17.09): при одном ИДУ колесо ненаблюдаемо (rank 2 из 4), и любой
#  регулятор с K[theta], K[dtheta] != 0 получал выдуманное колесо -- смотреть на
#  такую карту нет смысла.  С энкодером rank 4, ошибка колеса ограничена.
#
#  Все с компенсацией кажущейся вертикали (accel_offset = IMU_D, PROPOSALS.md
#  A31): после смены l, r модели 17.09 без неё ЛКР с фильтром раскачивается
#  при любом tau.  Вынос у датчика и у оценивателя -- одно число, IMU_D:
#  разъехавшись, они молча вычитали бы не ту инерционную часть.

#: Вынос ИДУ от оси колеса вдоль корпуса [м] (значение IMUSensor по умолчанию).
IMU_D = 0.20


class _BiasTap:
    """Обёртка датчика для видов фильтра: запоминает настоящее смещение ИДУ
    на каждом отсчёте (IMUSensor.bias -- величина, которая есть только в
    симуляции).  Только для диагностического пересчёта клетки."""

    def __init__(self, sensor):
        self.sensor = sensor
        self.imu = sensor.sensors[0]
        self.biases = []

    def reset(self):
        self.sensor.reset()
        self.biases = []

    def measure(self, t, x, u_prev=None):
        y = self.sensor.measure(t, x, u_prev)
        self.biases.append(self.imu.bias)
        return y


class _NeesTap:
    """Обёртка оценивателя ПАЧКИ: после шага считает NEES смещений по каждой
    клетке и хранит среднее по клеткам (kalman_speed §6.2, решение Глеба 01.10).

        eps_m = e_m^T P_bb,m^-1 e_m,   e_m = b^_m - b_m,

    у честного фильтра eps ~ chi^2_3, среднее по M клеткам ~ 3.  Настоящее
    смещение берётся у датчика пачки (IMUSensor.bias -- только в симуляции),
    сразу после estimate: датчик только что отдал отсчёт этого шага.
    P всех клеток не хранится -- одно число на шаг.  Считается каждый
    every-й шаг (как кадры пачки): решение 3x3 на клетку не бесплатно."""

    def __init__(self, est, imu, every=1):
        self.est, self.imu, self.every = est, imu, max(1, int(every))
        self.t, self.mean, self.count, self.wild = [], [], [], []
        self._k = 0

    def __getattr__(self, name):                 # aux, state, ... -- от фильтра
        return getattr(self.est, name)

    def reset(self, y0=None):
        self.est.reset(y0)
        self.t, self.mean, self.count, self.wild = [], [], [], []
        self._k = 0

    def estimate(self, t, y, u_prev):
        x = self.est.estimate(t, y, u_prev)
        if self._k % self.every == 0:
            self._record(t)
        self._k += 1
        return x

    def _record(self, t):
        est = self.est
        ib = est._ib
        P = est.covariance.reshape(-1, est.n, est.n)[:, ib][:, :, ib]
        e = est.aux["b_hat"].reshape(-1, 3) - np.asarray(self.imu.bias).reshape(-1, 3)
        with np.errstate(all="ignore"):
            try:
                eps = np.einsum("mi,mi->m", e, np.linalg.solve(P, e[..., None])[..., 0])
            except np.linalg.LinAlgError:
                eps = np.full(len(e), np.nan)
        ok = np.isfinite(eps)
        self.t.append(float(t))
        self.mean.append(float(eps[ok].mean()) if ok.any() else np.nan)
        self.count.append(int(ok.sum()))
        # Клетки, где eps выше квантиля 0.999 chi^2_3: у честного фильтра их
        # ~0.1 %.  Среднее одно их не покажет -- оно просто улетит.
        self.wild.append(int(np.sum(eps[ok] > NEES_WILD)))

    def result(self) -> dict:
        return {"t": np.array(self.t), "mean": np.array(self.mean),
                "count": np.array(self.count), "wild": np.array(self.wild)}


#: Квантиль 0.999 распределения chi^2_3: выше -- клетка «дикая».
NEES_WILD = 16.27


def _chi2_mean_band(dof, M, z=1.96):
    """Коридор 95 % для среднего M независимых chi^2_dof: квантили chi^2_{dof M},
    делённые на M.  Квантиль -- приближение Уилсона--Хилферти
    nu (1 - 2/(9 nu) + z sqrt(2/(9 nu)))^3: при nu ~ 10^3 ошибка ~1e-4,
    и scipy окну не нужен."""
    nu = np.maximum(dof * np.asarray(M, float), 1.0)
    a = 2.0 / (9.0 * nu)
    return (nu * (1 - a - z * np.sqrt(a)) ** 3 / np.maximum(M, 1),
            nu * (1 - a + z * np.sqrt(a)) ** 3 / np.maximum(M, 1))


def _imu_params(app) -> dict:
    """Параметры ИДУ окна -- ОДНИ для датчика прогона и для подставки
    калибровки (calibrate_on_stand): разойдясь, калибровка мерила бы другой
    датчик."""
    return dict(dt=app.args.dt, d=IMU_D,
                sigma_g=app.noise["sigma_g"], sigma_a=app.noise["sigma_a"],
                b0_g=app.noise["b0_g"], b0_a=app.noise["b0_g"] * 9.8,
                **{name: corr_matrix(app.corr[m]) for m, _, name in CORR_MATRICES})


#  Фабрики получают (app, rows): rows -- номера клеток карты, которые идут
#  строками пачки (None -- вся карта).  Ветер и чип у клетки m ОДНИ И ТЕ ЖЕ и
#  в пачке, и в отдельном пересчёте (решение Глеба 30.09: у сравниваемых
#  регуляторов и оценивателей один ветер) -- см. Explorer._world_for и
#  Explorer._chip_for.

def _imu(app, rows=None):
    return IMUSensor(app._world_for(rows), mode="imu", seed=app.args.seed,
                     b_init=app._chip_for(rows), **_imu_params(app))


def _imu_enc(app, rows=None):
    """ИДУ плюс моторный энкодер. Порядок -- часть контракта оценивателя."""
    return StackedSensor(_imu(app, rows),
                         EncoderSensor(app._world_for(rows), dt=app.args.dt,
                                       counts_per_rev=app.args.counts))


def _comp(tau=None):
    """tau=None -- взять со слайдера; число -- вырожденный режим (0 или inf).
    Фильтр знает НОМИНАЛ (app.system), а не мир."""
    return lambda app, rows=None: ComplementaryEstimator(
        app.system, dt=app.args.dt,
        tau=app.noise["tau"] if tau is None else tau,
        wheel="encoder", accel_offset=IMU_D)


def _kalman(app, rows=None, diagnostics=False):
    """Фильтр Калмана с уравнениями акселерометра (ER-025).

    Модель -- НОМИНАЛ app.system (никогда не мир).  Шум ИДУ -- с тех же
    слайдеров и из той же панели noise, что у датчика: по умолчанию фильтр
    знает датчик точно.  Остальное -- из панели kalman (app.kf).
    """
    kf = app.kf
    prior = None
    if kf["prior"] == "calibrate":
        b_hat, P_bb = app._calibration()
        prior = (b_hat if rows is None else b_hat[rows], P_bb)
    drift = None
    if kf["drift"] == "on":
        s = IMUSensor(app.system, dt=app.args.dt, mode="imu")   # плотности ухода датчика
        drift = dict(sigma_ba=float(s.sigma_b[0]), sigma_bg=float(s.sigma_b[2]))
    dist = None
    if kf["w_model"] == "matched" and app.phys["wind"]:
        dist = dict(sigma_w=(app.phys["sigma_psi"], app.phys["sigma_theta"]),
                    tau_w=app.phys["tau_w"])
    elif kf["w_model"] == "own":
        dist = dict(sigma_w=(kf["sigma_psi"], kf["sigma_theta"]), tau_w=kf["tau_w"])
    return KalmanEstimator(
        app.system, dt=app.args.dt, d=IMU_D,
        sigma_a=app.noise["sigma_a"], sigma_g=app.noise["sigma_g"],
        corr_w=corr_matrix(app.corr["w"]), counts_per_rev=app.args.counts,
        b0_a=app.noise["b0_g"] * 9.8, b0_g=app.noise["b0_g"], bias_prior=prior,
        bias_drift=drift, disturbance=dist, q_acc=kf["q_acc"],
        iterations=int(kf["iterations"]), diagnostics=diagnostics)


ESTIMATORS = [
    ("ideal", "ideal", None, None),
    ("comp", "comp+enc", _imu_enc, _comp()),
    # Вырожденные концы шкалы tau держим отдельными кнопками: слайдер до 0 и
    # до бесконечности не доезжает, а именно они показывают, что каждый
    # датчик ИДУ умеет в одиночку (колесо -- от энкодера в обоих).
    ("gyro", "gyro+enc", _imu_enc, _comp(np.inf)),
    ("acc", "accel+enc", _imu_enc, _comp(0.0)),
    # ER-025: оба канала акселерометра как есть, смещения и толчок в состоянии.
    ("kalman", "kalman+enc", _imu_enc, _kalman),
]
ESTIMATOR_KEYS = [key for key, _, _, _ in ESTIMATORS]
ESTIMATOR_LABEL = {key: label for key, label, _, _ in ESTIMATORS}
ESTIMATOR_SENSOR = {key: mk for key, _, mk, _ in ESTIMATORS}
ESTIMATOR_BUILD = {key: mk for key, _, _, mk in ESTIMATORS}


def _parse_params(text, base):
    """'mb=10,l=0.8' -> копия base с этими полями."""
    out = dict(base)
    if not text:
        return out
    for item in text.split(","):
        key, _, value = item.partition("=")
        key = key.strip()
        if key not in PHYS_KEYS:
            raise ValueError(f"неизвестный параметр {key!r}: нужны {PHYS_KEYS}")
        out[key] = float(value)
    return out


def _phys_from_args(args) -> dict:
    """Мир, номинал и ветер из командной строки (docs/explorer_kalman.md §7)."""
    world = _parse_params(getattr(args, "world_params", None), DEFAULT_PHYS)
    nominal_text = getattr(args, "nominal_params", None)
    nominal = _parse_params(nominal_text, world) if nominal_text else dict(world)
    sigma = getattr(args, "sigma_w", None) or (DEFAULT_WIND["sigma_psi"],
                                               DEFAULT_WIND["sigma_theta"])
    return {"world": world, "nominal": nominal, "same": nominal == world,
            "wind": getattr(args, "world", "clean") == "wind",
            "sigma_psi": float(sigma[0]), "sigma_theta": float(sigma[1]),
            "tau_w": float(getattr(args, "tau_w", None) or DEFAULT_WIND["tau_w"]),
            "seed": float(getattr(args, "wind_seed", None) or DEFAULT_WIND["seed"])}


def _kf_from_args(args) -> dict:
    kf = dict(DEFAULT_KF)
    for key, attr in (("chip", "chip"), ("prior", "bias_prior"), ("T_c", "calib_time"),
                      ("sigma_jig", "sigma_jig"), ("w_model", "kf_w"),
                      ("q_acc", "kf_q_acc"), ("iterations", "kf_iter")):
        value = getattr(args, attr, None)
        if value is not None:
            kf[key] = value if key in KF_CHOICES else float(value)
    if getattr(args, "kf_drift", False):
        kf["drift"] = "on"
    return kf


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
        # Мир и номинал (docs/explorer_kalman.md §1).  phys -- единственный
        # источник правды о физике; обе модели собирает _build_models, и её же
        # зовут слайдер u_max, кнопка no limit и панель model.
        self.phys = _phys_from_args(args)
        self.kf = _kf_from_args(args)
        self._bias_key = None           # кэш чипа и калибровки
        self._bias_val = (None, None)
        self._build_models()
        self.integrator = RK4Integrator()
        self.pinned = args.psi_max is not None or args.dpsi_max is not None
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
        # Кривая переключения реле на карте (кнопка `switch`, клавиша C).
        self.show_switch = True

        # Цены ЛКР -- копия DEFAULT_COST, которую двигает панель настроек.
        self.cost = {k: tuple(v) for k, v in DEFAULT_COST.items()}
        self.settings_open = False
        self.settings_text = {}
        self.settings_focus = None
        self.settings_error = ""
        self.K, self.P, self._drift = self._synthesize()
        self.K_soft = self._synthesize_soft()
        self.design_note = "no scipy"
        self.c_star = self._certify()
        self._refresh_hint()
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
        # Панели model и kalman (ER-025) -- устроены как settings и noise.
        self.model_open = False
        self.model_text = {}
        self.model_focus = None
        self.model_error = ""
        self.kf_open = False
        self.kf_text = {}
        self.kf_focus = None
        self.kf_error = ""
        # Нижний график: момент u или толчок w (клавиша P).
        self.plot_w = False
        self.w_true = None              # истинный толчок выбранной клетки
        # Виды фильтра (docs/kalman_views.md): None, "A" -- эллипс 1σ по P на
        # плоскости (psi, dpsi), "B" -- показания y против h(x, u) и h(x^-, u).
        # Открываются кнопками, закрывают правую половину окна; карта и
        # дорожка времени остаются.
        self.view = None
        self.diag = None
        self.diag_note = ""
        # Шрифт кнопок считается один раз на окно и здесь же и живёт: объект
        # шрифта привязан к инициализированному pygame, и модульный кэш
        # пережил бы pg.quit() мёртвой ссылкой.
        self._pick_font = None
        # Корреляции каналов ИДУ -- тоже параметры датчика, но не слайдеры:
        # у матрицы есть условие допустимости (положительная определённость),
        # которое у отдельного числа не проверить, поэтому они вводятся
        # панелью и принимаются целиком или никак -- как цены ЛКР.
        self.corr = {k: tuple(v) for k, v in DEFAULT_CORR.items()}
        self.noise_open = False
        self.noise_text = {}
        self.noise_focus = None
        self.noise_error = ""
        self.tau_star_measured = None   # argmin замера; None -- ещё не мерили
        self.est_key = args.estimator
        # Оцениватель СРАВНЕНИЯ считается только для выбранной клетки -- ровно
        # как второй регулятор. Карту он не красит: карта принадлежит
        # основному, иначе непонятно, чей это бассейн.
        self.est_cmp_key = args.estimator_compare
        self.traj_est_cmp = None
        self._refresh_hint()

        self.batch = None
        self.nees = None          # NEES смещений по ансамблю клеток (вид C)
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
        self.controller_cmp = None
        # Дорожки фаз выбранной клетки (ER-016): кто рулил на каждом шаге и
        # числа к этому -- время передачи, число релейных переключений.
        # Считаются в _update_tracks вместе с траекториями, а не при отрисовке.
        self.track = None
        self.track_cmp = None
        self.track_est_cmp = None
        self.playing = True
        self.play_time = 0.0
        self.time_drag = False      # ручку дорожки времени держат мышью
        # Стартовая клетка -- внутри множества восстановимости (_start_state):
        # на широкой карте «0.6 края» -- это уже лёгший корпус, и первое, что
        # видит пользователь, была бы падающая траектория.
        self._select_nearest(*self._start_state())

    # --- мир и номинал (ER-025) ---------------------------------------------

    def _build_models(self):
        """Собрать app.system (номинал) и app.world (мир) из self.phys и
        текущего предела момента.  Предел ставится ОБЕИМ: мотор один.

        Пока номинал равен миру и ветра нет, это один и тот же объект -- окно
        бит в бит прежнее (оракул tests/test_explorer_kalman.py)."""
        ph = self.phys
        pw = WheeledPendulumParams(**ph["world"], u_max=self.u_max)
        pn = pw if ph["same"] else WheeledPendulumParams(**ph["nominal"], u_max=self.u_max)
        self.system = WheeledPendulum(pn)
        if ph["wind"]:
            self.world = DisturbedWheeledPendulum(
                pw, sigma_w=(ph["sigma_psi"], ph["sigma_theta"]), tau_w=ph["tau_w"],
                dt=self.args.dt, seed=int(ph["seed"]))
        elif ph["same"]:
            self.world = self.system
        else:
            self.world = WheeledPendulum(pw)

    def _world_for(self, rows):
        """Мир для строк пачки rows (None -- вся карта).  С ветром клетка m
        получает путь СВОЕЙ строки (row_offset): у второго регулятора и
        второго оценивателя тот же ветер, что у основного."""
        if rows is None or not isinstance(self.world, DisturbedWheeledPendulum):
            return self.world
        return self.world.with_row_offset(int(np.asarray(rows).ravel()[0]))

    def _bias_state(self):
        """(чип, калибровка) карты -- по строке на клетку.  Кэш по всему, от
        чего они зависят.

        Чип разыгрывается явно, если он сырой или нужна калибровка (ей надо
        знать, что именно калибровать); иначе -- None, и смещение, как
        раньше, разыгрывает сам IMUSensor (бит в бит прежнее окно)."""
        kf, n = self.kf, self.spec.n * self.spec.n
        explicit = kf["chip"] == "raw" or kf["prior"] == "calibrate"
        key = (n, explicit, kf["chip"], kf["prior"], kf["T_c"], kf["sigma_jig"],
               self.args.seed, tuple(sorted((k, float(np.sum(v))) for k, v in
                                            _imu_params(self).items())))
        if key == self._bias_key:
            return self._bias_val
        chip = calib = None
        if explicit:
            if kf["chip"] == "raw":
                chip = draw_turn_on_bias(**MPU6050_RAW_B0, shape=n, seed=self.args.seed + 1)
            else:
                b0 = self.noise["b0_g"]
                chip = draw_turn_on_bias(b0_a=9.8 * b0, b0_g=b0, shape=n,
                                         seed=self.args.seed + 1,
                                         corr_b0=corr_matrix(self.corr["b0"]))
            if kf["prior"] == "calibrate":
                params = _imu_params(self)
                params.pop("b0_g"), params.pop("b0_a"), params.pop("corr_b0")
                calib = calibrate_on_stand(self.world, params, b_init=chip,
                                           T_c=kf["T_c"], sigma_jig=kf["sigma_jig"],
                                           seed=self.args.seed + 2)
        self._bias_key, self._bias_val = key, (chip, calib)
        return self._bias_val

    def _chip_for(self, rows):
        chip = self._bias_state()[0]
        if chip is None:
            return None
        return chip if rows is None else chip[rows]

    def _calibration(self):
        return self._bias_state()[1]

    def mismatch_growth(self):
        """max Re eig(A_w - B_w K_n): номинальный ЛКР в МИРЕ (§1.2).  None --
        номинал равен миру."""
        if self.phys["same"]:
            return None
        A, B = self.world.linearize_upright()
        return float(np.linalg.eigvals(A - B @ self.K).real.max())

    def _rebuild_all(self):
        """Физика или ветер поменялись: всё, что от них зависит, -- заново.
        Мир: границы карты, граница восстановимости, сетка.  Номинал: K, P,
        K_soft, c*, кривая реле.  Клетка -- по состоянию, как у слайдера."""
        self._build_models()
        self.spec = GridSpec(self.spec.n, *self._map_bounds())
        self._refresh_limits()
        self.K, self.P, self._drift = self._synthesize()
        self.K_soft = self._synthesize_soft()
        self.c_star = self._certify()
        self._refresh_hint()
        if not self.available(self.main_key):
            self.main_key = "lqr"
        if self.cmp_key is not None and not self.available(self.cmp_key):
            self.cmp_key = None
        self._invalidate()
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        if self.args.precompute:
            self._precompute()
        self._select_nearest(*self.selected_state)

    def toggle_world(self):
        """Клавиша W: чистый мир <-> ветер."""
        self.phys["wind"] = not self.phys["wind"]
        if not self.w_available:
            self.plot_w = False
        self._rebuild_all()

    @property
    def w_available(self) -> bool:
        """График толчка осмыслен, только если толчок есть."""
        return bool(self.phys["wind"])

    def toggle_plot_w(self):
        if self.w_available:
            self.plot_w = not self.plot_w

    # --- виды фильтра (docs/kalman_views.md) ---------------------------------

    def toggle_view(self, key):
        """V -- вид A, H -- вид B, B -- вид C (смещения); повторное нажатие
        закрывает."""
        self.view = None if self.view == key else key
        self._update_diag()

    def _update_diag(self):
        """Диагностический пересчёт выбранной клетки: фильтр отдаёт P, y,
        h(x^-, u) и S на КАЖДОМ шаге (record_aux), а окно -- чистые показания
        мира h(x, u): настоящее состояние, без шума и смещения.

        Пересчёт отдельный (около 4 с на клетку): записывать P для всей карты
        -- это гигабайты (§5.2).  Ветер и чип у клетки те же, что в пачке
        (row_offset, _chip_for); белый шум датчика -- свой розыгрыш.
        """
        self.diag = None
        if self.view is None or self.selected is None:
            return
        if self.est_key != "kalman":
            self.diag_note = "views show the Kalman filter: choose kalman+enc"
            return
        from ..rollout import rollout
        m = self.spec.index(*self.selected)
        x0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)[m]
        world = self._world_for(m)
        tap = _BiasTap(_imu_enc(self, m))
        with self._errstate():
            traj = rollout(world, CONTROLLER_BUILD[self.main_key](self), self.integrator,
                           x0, self.args.dt, self.n_steps,
                           sensor=tap, estimator=_kalman(self, m, True),
                           record_aux=True)
        t = traj.t[:-1]
        X = traj.x[:-1]
        U_prev = np.vstack([np.zeros((1, 1)), traj.u[:-1]])
        clean = WheeledPendulum(world.p)
        W = (world.disturbance(t) if isinstance(world, DisturbedWheeledPendulum) else None)
        a_x, a_z = clean.specific_force(0.0, X, U_prev, IMU_D, w=W)
        y_clean = np.column_stack([a_x, a_z, X[:, I_DPSI], X[:, I_THETA] - X[:, I_PSI]])
        # Масштаб увеличенного эллипса на фазовой плоскости (§14): степень
        # десяти, ОДНА на прогон -- эллипс первого кадра ~10 % ширины.
        P0 = traj.aux["P"][0]
        span_psi = max(np.ptp(X[:, I_PSI]), 1e-6)
        span_dpsi = max(np.ptp(X[:, I_DPSI]), 1e-6)
        rel = max(np.sqrt(P0[I_PSI, I_PSI]) / span_psi, np.sqrt(P0[I_DPSI, I_DPSI]) / span_dpsi)
        scale_k = 10.0 ** np.round(np.log10(0.05 / max(rel, 1e-15)))
        # Смещение до коррекции: при Q_b = 0 априорное b^- равно оценке
        # прошлого шага; на первом шаге -- самой первой оценке.
        b_hat = traj.aux["b_hat"]
        b_prior = np.vstack([b_hat[:1], b_hat[:-1]])
        self.diag = {"t": t, "x": X, "x_hat": traj.x_hat, "P": traj.aux["P"],
                     "y": traj.aux["y"], "y_pred": traj.aux["y_pred"],
                     "S": traj.aux["S_diag"], "y_clean": y_clean[:, :traj.aux["y"].shape[1]],
                     "b_prior": b_prior, "b_hat": b_hat,
                     "b_true": np.asarray(tap.biases[1:1 + len(t)]),
                     "scale_k": float(scale_k), "cell": m,
                     "lens_half": (1.5 * float(np.sqrt(P0[I_PSI, I_PSI])),
                                   1.5 * float(np.sqrt(P0[I_DPSI, I_DPSI])))}
        self.diag_note = ""

    def diag_frame(self) -> int:
        if self.diag is None:
            return 0
        return int(np.clip(round(self.play_time / self.args.dt), 0, len(self.diag["t"]) - 1))

    # --- панель model --------------------------------------------------------

    def model_values(self) -> dict:
        ph = self.phys
        v = {f"world.{k}": ph["world"][k] for k in PHYS_KEYS}
        v.update({f"nominal.{k}": ph["nominal"][k] for k in PHYS_KEYS})
        v.update({f"wind.{k}": ph[k] for k in DEFAULT_WIND})
        return v

    def open_model(self):
        self.model_text = {k: f"{v:g}" for k, v in self.model_values().items()}
        self.model_text["same"] = "yes" if self.phys["same"] else "no"
        self.model_text["wind"] = "on" if self.phys["wind"] else "off"
        self.model_focus = MODEL_KEYS[0]
        self.model_error = ""
        self.model_open = True
        self.settings_open = self.noise_open = self.kf_open = False

    def apply_model(self, values: dict):
        """Принять физику мира, номинала и ветра.  None -- успех, строка --
        отказ; при отказе окно не меняется ни в чём."""
        merged = self.model_values()
        same = self.phys["same"]
        wind = self.phys["wind"]
        try:
            for key, value in values.items():
                if key == "same":
                    same = value in ("yes", True)
                elif key == "wind":
                    wind = value in ("on", True)
                elif key in merged:
                    merged[key] = float(value)
                else:
                    return f"unknown field {key}"
        except (TypeError, ValueError):
            return f"not a number: {key} = {value!r}"
        if not all(np.isfinite(v) for v in merged.values()):
            return "all values must be finite"
        for group in ("world", "nominal"):
            for k in PHYS_KEYS:
                if merged[f"{group}.{k}"] <= 0:
                    return f"{group}.{k} must be > 0"
        if merged["wind.sigma_psi"] < 0 or merged["wind.sigma_theta"] < 0:
            return "wind sigma must be >= 0"
        if merged["wind.tau_w"] <= 0:
            return "wind tau_w must be > 0"
        if merged["wind.seed"] < 0 or merged["wind.seed"] != int(merged["wind.seed"]):
            return "wind seed must be a non-negative integer"
        world = {k: merged[f"world.{k}"] for k in PHYS_KEYS}
        nominal = dict(world) if same else {k: merged[f"nominal.{k}"] for k in PHYS_KEYS}
        # Пробный синтез по номиналу: Риккати может не решиться -- узнать
        # можно, только решив (как в apply_settings).
        try:
            trial = WheeledPendulum(WheeledPendulumParams(**nominal))
            A, B = trial.linearize_upright()
            from ..lqr import lqr
            with np.errstate(all="raise"):
                lqr(A, B, *_cost_matrices(self.cost["base"]))
        except ImportError:                              # pragma: no cover
            pass
        except Exception as exc:                         # noqa: BLE001
            return f"nominal: Riccati failed ({type(exc).__name__})"
        self.phys = {"world": world, "nominal": nominal, "same": same, "wind": wind,
                     **{k: merged[f"wind.{k}"] for k in DEFAULT_WIND}}
        if not self.w_available:
            self.plot_w = False
        self.model_error = ""
        self._rebuild_all()
        return None

    # --- панель kalman -------------------------------------------------------

    def kf_values(self) -> dict:
        return dict(self.kf)

    def open_kf(self):
        self.kf_text = {k: (v if isinstance(v, str) else f"{v:g}")
                        for k, v in self.kf_values().items()}
        self.kf_focus = "q_acc"
        self.kf_error = ""
        self.kf_open = True
        self.settings_open = self.noise_open = self.model_open = False

    def apply_kf(self, values: dict):
        """Принять настройки фильтра Калмана и чипа.  None -- успех."""
        merged = self.kf_values()
        try:
            for key, value in values.items():
                if key not in merged:
                    return f"unknown field {key}"
                if key in KF_CHOICES:
                    if value not in KF_CHOICES[key]:
                        return f"{key}: one of {'/'.join(KF_CHOICES[key])}"
                    merged[key] = value
                else:
                    merged[key] = float(value)
        except (TypeError, ValueError):
            return f"not a number: {key} = {value!r}"
        nums = [v for k, v in merged.items() if k not in KF_CHOICES]
        if not all(np.isfinite(v) for v in nums):
            return "all values must be finite"
        if merged["T_c"] <= 0:
            return "T_c must be > 0"
        if merged["sigma_jig"] < 0:
            return "sigma_jig must be >= 0"
        if merged["sigma_psi"] < 0 or merged["sigma_theta"] < 0:
            return "filter sigma_w must be >= 0"
        if merged["tau_w"] <= 0:
            return "filter tau_w must be > 0"
        if merged["q_acc"] < 0:
            return "q_acc must be >= 0"
        it = merged["iterations"]
        if it != int(it) or not 1 <= it <= 5:
            return "iterations: integer 1..5"
        changed = merged != self.kf
        self.kf = merged
        self.kf_error = ""
        # Чип -- свойство мира: он меняет датчик у ВСЕХ оценивателей.
        uses_imu = self.est_key != "ideal" or self.est_cmp_key is not None
        if changed and uses_imu:
            self.commit_noise()
        return None

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

        ЛКР по ОДНОМУ наклону (`lqr_tilt`) окно больше не синтезирует: его
        читал только вариант `bang -> tilt`, убранный 24.09. Сама функция
        осталась в `wpend/lqr.py` -- её читают примеры и тесты.
        """
        A, B = self.system.linearize_upright()
        Q, R = _cost_matrices(self.cost["base"])
        try:
            from ..lqr import lqr
            K, P = lqr(A, B, Q, R)
        except ImportError:                              # pragma: no cover
            return K_LQR, None, ""
        drift = float(np.abs(K - K_LQR).max())
        note = ""
        # Сверка с константой имеет смысл только для той цены, под которую
        # константа записана: при цене из панели K обязано отличаться.
        if drift > 1e-4 and self.cost["base"] == DEFAULT_COST["base"]:  # pragma: no cover
            note = f"  (!) K ушло от константы на {drift:.2g}"
        return K, P, note

    def _synthesize_soft(self):
        """ЛКР второй фазы `bang-eps`: та же задача, что в _synthesize, но с
        ценой self.cost["soft"]. По умолчанию это Q_SOFT_WHEEL, R_SOFT = 250
        (ER-017): гейн по скорости колеса ниже базового в 17 раз, по наклону --
        в 1.5, почему так -- комментарий у Q_SOFT_WHEEL.

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
        """(psi, dpsi) стартовой клетки: узел сетки с |psi| <= pi/2 внутри
        множества восстановимости и, если сетка уже посчитана, удержанный
        основным регулятором; из таких -- с наибольшим |psi| (при равенстве
        ближе к середине отрезка скоростей). На широкой карте наугад
        выбранная клетка почти всегда -- лёгший корпус, а центр -- неподвижный
        робот; ни то, ни другое не показывает, что делает регулятор."""
        u = self.u_finite if self.u_max is None else self.u_max
        X0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)
        th, dth = X0[:, I_PSI], X0[:, I_DPSI]
        ok = self.world.is_recoverable(u, th, dth) & (np.abs(th) <= PSI_MAP)
        if (self.outcome >= 0).all() and (ok & (self.outcome == HELD)).any():
            ok &= self.outcome == HELD
        if not ok.any():
            return 0.0, 0.0
        floor, ceiling = self.world.recoverable_bounds(u, th)
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
        self.noise_open = False         # модальная панель -- одна за раз
        self.model_open = self.kf_open = False

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
        self.K, self.P, self._drift = self._synthesize()
        self.K_soft = self._synthesize_soft()
        self.c_star = self._certify()
        self._refresh_hint()
        if not self.available(self.main_key):
            self.main_key = "lqr"
        if self.cmp_key is not None and not self.available(self.cmp_key):
            self.cmp_key = None
        self._invalidate()
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        if self.args.precompute:
            self._precompute()
        self._select_nearest(*self.selected_state)
        self.settings_error = ""
        return None

    # --- панель шума -------------------------------------------------------

    def noise_values(self) -> dict:
        """Текущие корреляции под ключами полей панели шума."""
        return {f"{m}.{i}": float(r)
                for m, _, _ in CORR_MATRICES for i, r in enumerate(self.corr[m])}

    def open_noise(self):
        """Как open_settings: поля -- то, чем посчитана текущая карта."""
        self.noise_text = {k: f"{v:g}" for k, v in self.noise_values().items()}
        self.noise_focus = NOISE_KEYS[0]
        self.noise_error = ""
        self.noise_open = True
        self.settings_open = False
        self.model_open = self.kf_open = False

    def apply_noise(self, values: dict):
        """Принять корреляции каналов ИДУ; None -- успех, строка -- отказ.

        Допустимость матрицы проверяет сам IMUSensor (пробная сборка), а не
        окно: правило одно и живёт в одном месте.  Каждая матрица пробуется
        отдельно, чтобы в отказе назвать, какая именно невозможна.  При отказе
        окно не меняется ни в чём.

        Пересчёт -- как у слайдеров шума (commit_noise), но только если датчик
        сейчас кому-то нужен: при ideal у основного и без второго оценивателя
        карта от корреляций не зависит, и платить за неё секунды незачем.
        Фабрики датчиков читают app.corr при каждой сборке, поэтому
        переключиться на comp позже -- уже с новыми числами.
        """
        merged = self.noise_values()
        try:
            for key, value in values.items():
                if key not in merged:
                    return f"unknown field {key}"
                merged[key] = float(value)
        except (TypeError, ValueError):
            return f"not a number: {key} = {value!r}"
        if not all(np.isfinite(v) for v in merged.values()):
            return "all values must be finite"
        corr = {m: tuple(merged[f"{m}.{i}"] for i in range(len(CORR_PAIRS)))
                for m, _, _ in CORR_MATRICES}
        for m, label, name in CORR_MATRICES:
            if max(abs(r) for r in corr[m]) >= 1.0:
                return f"{label}: need |rho| < 1"
            try:
                IMUSensor(self.system, dt=self.args.dt, mode="imu",
                          **{name: corr_matrix(corr[m])})
            except ValueError:
                return f"{label}: impossible correlations (not positive definite)"

        changed = corr != self.corr
        self.corr = corr
        self.noise_error = ""
        if changed and (self.est_key != "ideal" or self.est_cmp_key is not None):
            self.commit_noise()
        return None

    def _certify(self):
        """Сертифицированный уровень c* для ТЕКУЩЕГО предела.

        Единственное место проекта, куда предел мотора входит по существу:
        уровень строится по vdot ПРИ ОБРЕЗАННОМ управлении, и забыть здесь
        u_max -- получить ответ для другого мотора (при u_max = 1.5 он выходит
        в 16 раз оптимистичнее). Цена -- 0.14 с, ради неё
        слайдер и пересчитывает уровень на отпускании, а не на каждом пикселе.
        """
        if self.P is None:                               # pragma: no cover
            self.design_note = "no scipy"
            return None
        from ..lqr import certified_level

        c_star, _ = certified_level(self.system, self.K, self.P, u_max=self.u_max,
                                    n_dirs=1500, ds=4e-3, s_max=8.0,
                                    psi_max=self.args.psi_fall)
        self.design_note = (f"c*={c_star:.3g}"
                            + ("" if self.u_max is not None else " unclipped")
                            + self._drift)
        return c_star

    # --- карта под предел момента ------------------------------------------

    def _map_bounds(self):
        """Границы карты (psi_max, dpsi_max) по стандарту проекта.

        Стандарт (CLAUDE.md, решения Глеба 16.09 и 17.09): интересен наклон
        psi в [-pi/2, pi/2], а сетка берётся с запасом MAP_FIT по обеим осям.
        По psi граница поэтому постоянная, MAP_FIT * pi/2 ≈ 2.04 рад: карта
        одна и та же при любом моторе, и пунктир pi/2 на ней виден всегда.

        По dpsi граница по-прежнему идёт за мотором: восстановимое множество
        растёт с u_max почти линейно, и одни и те же границы годятся ровно для
        одного мотора. Берётся MAP_FIT от наибольшего |dpsi| на границе
        восстановимости при psi в [-pi/2, pi/2] -- не «горло» при psi = 0:
        множество наклонное, и у края отрезка оно шире, чем в центре.

        psi_fall по умолчанию -- pi («упал», земли в модели нет), так что
        карта до него не доходит. Явные --psi-max / --dpsi-max
        замораживают границы: тогда карты при разных u_max сравнимы по
        пикселям, а не только по подписям.
        """
        # При выключенном пределе восстановимо ВСЁ, и подгонять карту не подо
        # что: границы замораживаются на последнем конечном пределе. Это и есть
        # смысл опыта -- те же клетки, другой мотор: видно, какие поменяли
        # цвет, а не как поехали оси.
        u = self.u_finite if self.u_max is None else self.u_max
        psi_max = min(MAP_FIT * PSI_MAP, 0.98 * self.args.psi_fall)
        th = np.linspace(-PSI_MAP, PSI_MAP, 401)
        floor, ceiling = self.world.recoverable_bounds(u, th)
        reach = np.abs(np.concatenate([floor, ceiling]))
        dpsi_max = MAP_FIT * float(reach[np.isfinite(reach)].max())
        if self.args.psi_max is not None:
            psi_max = self.args.psi_max
        if self.args.dpsi_max is not None:
            dpsi_max = self.args.dpsi_max
        return psi_max, dpsi_max

    def _clip_u_max(self):
        """Предел момента, с которого карта перестаёт содержать множество
        восстановимости по psi. При стандартной карте (±1.3·pi/2 до
        psi_fall = pi) такого нет -- None. Остаётся для --psi-fall меньше
        MAP_FIT·pi/2, где карта обрезается и окно обязано сказать об этом."""
        if MAP_FIT * PSI_MAP <= 0.98 * self.args.psi_fall:
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
            self.limit_psi = self.limit_floor = self.limit_ceiling = None
            self.psi_eq = None
            self.switch_up = self.switch_down = None
            return
        self.limit_psi = np.linspace(-self.spec.psi_max * 1.05,
                                       self.spec.psi_max * 1.05, 601)
        # Граница и седло -- свойства МИРА; кривая реле -- того, что знает
        # реле, то есть номинала (docs/explorer_kalman.md §1.1).
        self.limit_floor, self.limit_ceiling = self.world.recoverable_bounds(
            self.u_max, self.limit_psi)
        self.psi_eq = self.world.saddle_angle(self.u_max)
        self.switch_up, self.switch_down = switching_curve(
            self.system, self.u_max, self.limit_psi)

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
        # Обе модели -- из self.phys: собрать здесь WheeledPendulum(u_max=...)
        # значило бы молча сбросить физику из панели model на заводскую.
        self._build_models()
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
        self.c_star = self._certify()
        self._invalidate()
        self.controller = CONTROLLER_BUILD[self.main_key](self)
        if self.args.precompute:
            self._precompute()
        self.stale = False
        self.run_u_max = self.u_max
        self._select_nearest(*self.selected_state)

    # --- слой оценки -------------------------------------------------------

    def _sensor(self, key=None, rows=None):
        """Датчик оценивателя key (None -- идеальный, y = x).  rows -- клетки
        карты, идущие строками пачки (None -- вся карта)."""
        make = ESTIMATOR_SENSOR[key or self.est_key]
        return None if make is None else make(self, rows)

    def _estimator(self, key=None, rows=None):
        """Оцениватель (None -- тождественный, x_hat = y)."""
        make = ESTIMATOR_BUILD[key or self.est_key]
        return None if make is None else make(self, rows)

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
        x0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)[m]
        X0 = np.tile(x0, (n_points * n_reps, 1))

        keep = self.noise["tau"]
        try:
            self.noise["tau"] = np.repeat(taus, n_reps)
            with self._errstate():
                rows = np.full(len(X0), m)
            batch = rollout_many(
                    self._world_for(rows), self.controller, self.integrator, X0,
                    self.args.dt, self.n_steps, self.args.stride,
                    sensor=self._sensor(rows=rows), estimator=self._estimator(rows=rows),
                    record_estimate=True)
        finally:
            self.noise["tau"] = keep
        if batch.x_hat is None:
            self.tau_star_measured = None
            return

        err = batch.x_hat[:, :, I_PSI] - batch.x[:, :-1, I_PSI]
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
        self._invalidate()
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
        self._update_tracks()

    def _update_estimator_compare(self):
        """Пересчитать траекторию второго оценивателя для выбранной клетки.

        Тот же регулятор и то же начальное условие -- отличается ровно слой
        оценки, поэтому расхождение истинных траекторий и есть его цена.
        """
        if self.est_cmp_key is None or self.selected is None:
            self.traj_est_cmp = None
            return
        m = self.spec.index(*self.selected)
        X0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)[m:m + 1]
        rows = np.array([m])
        with self._errstate():
            batch = rollout_many(self._world_for(rows), self.controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(self.est_cmp_key, rows),
                                 estimator=self._estimator(self.est_cmp_key, rows),
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
        self._invalidate()
        if self.args.precompute:
            self._precompute()
        if self.selected is not None:
            self.select(*self.selected)
        self.tau_star_measured = None

    def pick_font(self, pg):
        """Шрифт кнопок, посчитанный один раз на окно (см. `picker_font_size`)."""
        if self._pick_font is None:
            self._pick_font = picker_font(pg)
        return self._pick_font

    def available(self, key: str) -> bool:
        """Обе трёхфазные схемы требуют конечного предела момента (реле и
        накачка подают ровно +-u_max) и scipy (третья фаза -- эллипсоид по P,
        вторая -- мягкий ЛКР). ЛКР доступен всегда: он и есть то, куда окно
        откатывается, когда остальное недоступно."""
        if key in RELAY_KEYS:
            return (self.u_max is not None and self.K_soft is not None
                    and self.P is not None)
        return True

    def _refresh_hint(self):
        """Строка о том, почему часть кнопок перечёркнута. Молчащая мёртвая
        кнопка -- баг интерфейса, поэтому причина всегда написана рядом."""
        # Про реле пишет строка слайдера: там есть место, а рядом с кнопками
        # длинная фраза не помещается и обрезается краем окна.
        if self.P is None:
            self.design_hint = "ellipsoid needs scipy:  uv sync --extra dev"
        else:
            self.design_hint = ""

    # --- смена регулятора --------------------------------------------------

    def _invalidate(self):
        """Забыть посчитанное: сетку, исходы, моменты падения и кэш клеток."""
        self.batch = None
        self.nees = None
        self.outcome[:] = -1
        self.t_fall[:] = np.nan
        self.cache.clear()

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
        X0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)
        t0 = time.perf_counter()
        sensor, estimator = self._sensor(), self._estimator()
        self.nees = None
        if self.est_key == "kalman":
            # Честность фильтра по ансамблю клеток -- для вида C (§6.2).
            estimator = _NeesTap(estimator, sensor.sensors[0], every=self.args.stride)
        with self._errstate():
            self.batch = rollout_many(self.world, self.controller, self.integrator,
                                      X0, self.args.dt, self.n_steps,
                                      self.args.stride,
                                      sensor=sensor,
                                      estimator=estimator,
                                      record_estimate=self.records_estimate)
        if isinstance(estimator, _NeesTap):
            self.nees = estimator.result()
        self.outcome, self.t_fall = classify(self.batch, I_PSI, self.args.psi_fall)
        self.compute_seconds = time.perf_counter() - t0
        # Без предела ЛКР -- линейный закон, продолженный на все углы: за
        # горизонтом он раскручивает корпус вместо того, чтобы ловить, и
        # угловые клетки уходят в переполнение. Классификация их всё равно
        # ловит (psi пересекает psi_fall задолго до inf), но молчать об
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
        X0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)[m:m + 1]
        rows = np.array([m])
        t0 = time.perf_counter()
        with self._errstate():
            batch = rollout_many(self._world_for(rows), self.controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(rows=rows),
                                 estimator=self._estimator(rows=rows),
                                 record_estimate=self.records_estimate)
        self.compute_seconds = time.perf_counter() - t0
        outcome, t_fall = classify(batch, I_PSI, self.args.psi_fall)
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
        X0 = self.spec.initial_states(self.system.n_state, I_PSI, I_DPSI)[m:m + 1]
        rows = np.array([m])
        with self._errstate():
            batch = rollout_many(self._world_for(rows), controller, self.integrator,
                                 X0, self.args.dt, self.n_steps, self.args.stride,
                                 sensor=self._sensor(rows=rows),
                                 estimator=self._estimator(rows=rows),
                                 record_estimate=self.records_estimate)
        return batch[0]

    def _update_compare(self):
        """Пересчитать траекторию сравнения для выбранной клетки."""
        if self.cmp_key is None or self.selected is None:
            self.traj_cmp = None
            self.controller_cmp = None
            return
        m = self.spec.index(*self.selected)
        # Регулятор сравнения запоминается, а не выбрасывается: дорожке фаз он
        # нужен целиком -- с его регионом, его K и его защёлкой. Собрать его
        # заново значило бы получить ВТОРОЙ объект с той же ценой, и при
        # изменившихся настройках это были бы разные регуляторы.
        self.controller_cmp = CONTROLLER_BUILD[self.cmp_key](self)
        self.traj_cmp = self._run_one(self.controller_cmp, m)

    def _update_tracks(self):
        """Дорожки фаз для всех показанных траекторий (ER-016).

        Считаются ОДИН раз на выбранную клетку, а не каждый кадр: повтор
        прогоняет регулятор по всем сохранённым отсчётам, и на 60 кадрах в
        секунду это была бы самая дорогая вещь в окне при неизменном ответе.

        Оцениватель сравнения идёт под ОСНОВНЫМ регулятором (так его и
        считает `_update_estimator_compare`), поэтому и фаза у него считается
        основным -- другим он и не управлялся.
        """
        def track(traj, controller):
            if traj is None or controller is None:
                return None
            return track_summary(traj, controller, self.system)

        self.track = track(self.traj, self.controller)
        self.track_cmp = track(self.traj_cmp, getattr(self, "controller_cmp", None))
        self.track_est_cmp = track(self.traj_est_cmp, self.controller)

    def select(self, ix: int, iy: int):
        self.selected = (ix, iy)
        # Запоминаем не номер клетки, а её СОСТОЯНИЕ: при смене предела сетка
        # перестраивается, и вопрос «что этот же старт делает при другом
        # моторе» требует именно состояния.
        self.selected_state = (float(self.spec.psis[ix]),
                               float(self.spec.dpsis[iy]))
        m = self.spec.index(ix, iy)
        self.traj = self._trajectory(m)
        # Истинный толчок клетки: окно строило мир, окно его и спрашивает;
        # рисование получает готовый массив (правило 6).
        self.w_true = (self._world_for([m]).disturbance(self.traj.t[:-1])
                       if isinstance(self.world, DisturbedWheeledPendulum) else None)
        self._update_compare()
        self._update_estimator_compare()
        self._update_tracks()
        self._update_diag()
        self.play_time = 0.0
        self.playing = True

    def _select_nearest(self, psi, dpsi):
        ix = int(np.argmin(np.abs(self.spec.psis - psi)))
        iy = int(np.argmin(np.abs(self.spec.dpsis - dpsi)))
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

    @property
    def play_span(self) -> float:
        """Длина прогона в секундах: правый край дорожки времени. 0, пока
        клетка не выбрана."""
        if self.traj is None:
            return 0.0
        return float(self.traj.t[-1] - self.traj.t[0])

    def scrub(self, t: float):
        """Встать на момент t и ОСТАНОВИТЬ воспроизведение.

        Решение Глеба 21.09: отпущенная ручка держит кадр, а не едет дальше.
        Слайдер существует ради «остановиться на передаче управления и
        посмотреть, что в этот момент делают обе системы»; автоматический пуск
        отнимал бы ровно это. Вернуть ход -- ПРОБЕЛ.

        Траектория не пересчитывается: кадр уже лежит в Trajectory, меняется
        одно число.
        """
        if self.traj is None:
            return
        self.play_time = float(np.clip(t, 0.0, self.play_span))
        self.playing = False

    def step_frame(self, delta: int):
        """Шаг ровно на кадр траектории, а не на «немного времени».

        Встаём В СЕРЕДИНУ кадра: frame_of переводит время в номер отбрасыванием
        дробной части, и время ровно на границе кадра при накопленной ошибке
        округления попадало бы то в k, то в k-1 -- стрелка через раз стояла бы
        на месте.
        """
        if self.traj is None:
            return
        dt_frame = float(self.traj.t[1] - self.traj.t[0])
        k = int(np.clip(self.frame + delta, 0, len(self.traj) - 1))
        self.scrub((k + 0.5) * dt_frame)


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


def _to_map_px(app, psi, dpsi):
    """Точка (psi, dpsi) -> пиксель карты.

    Карта рисуется как n x n клеток: узел сетки ix -- это не точка, а целый
    пиксель surface, растянутый в клетку [ix, ix+1).  Поэтому узел ix лежит в
    (ix + 0.5) / n ширины, а не в ix / (n - 1), и пересчёт ведётся по границам
    картинки, отступающим от крайних узлов на полклетки.  Наивная нормировка на
    psi_max совпадает с подсветкой клетки только в центре карты, а к краям
    расходится линейно, до полклетки в углах.
    """
    x, y, w, h = MAP
    n = app.spec.n
    edge = n / max(n - 1, 1)          # (полширины картинки) / psi_max
    half_th = app.spec.psi_max * edge
    half_dth = app.spec.dpsi_max * edge
    fx = (psi + half_th) / (2 * half_th)
    fy = (dpsi + half_dth) / (2 * half_dth)
    return x + fx * w, y + (1.0 - fy) * h


def _draw_phase(target, pg, app, traj, color, head_color, frame, ox, oy, width=2):
    """Фазовая кривая (psi, dpsi) в координатах target со сдвигом (ox, oy).

    Сдвиг нужен, чтобы одна и та же функция рисовала и прямо на экране, и на
    полупрозрачной подложке, у которой своё начало координат.
    """
    x, y, w, h = MAP
    pts = [_to_map_px(app, th, dth)
           for th, dth in zip(traj.x[:, I_PSI], traj.x[:, I_DPSI])]
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
        for th, dth in zip(app.limit_psi, values):
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
        px, py = _to_map_px(app, sign * app.psi_eq, 0.0)
        if x <= px <= x + w and y <= py <= y + h:
            pg.draw.circle(screen, LIMIT, (int(px), int(py)), 4, 1)
    screen.set_clip(clip)


def switching_curve(system, u_max, psi):
    """Кривая переключения реле sigma = 0 -- ветвь линии уровня первого
    интеграла, проходящая через (0, 0). Возвращает (up, down): dpsi кривой при
    dpsi > 0 и при dpsi < 0; где ветви нет -- nan.

    Реле (`BangBangLQRController`, ARCHITECTURE.md «Двухфазный регулятор»)
    переключается на уровне H(psi, dpsi; -u_max d) = H(0, 0) = gamma D,
    d = sign dpsi (Key_Formulas §3). Отсюда при d = +-1

        1/2 Delta(psi) dpsi^2 = H(0, 0) - H(psi, 0; -u_max d),

    и dpsi = d sqrt(...). H берётся у самой модели (`first_integral`), а не
    переписывается здесь: формула живёт в одном месте, и кривая на карте не
    может разойтись с тем, по чему переключается регулятор.

    Берётся только компонента через ноль: ветвь d = +1 -- при psi <= 0
    (корпус летит к вертикали слева и тормозит), d = -1 -- при psi >= 0. У
    того же уровня есть вторая компонента за psi* ~ 2 psi_eq (A34), целиком
    вне множества восстановимости; она про вырождение закона, а не про
    переключение, и здесь не рисуется.
    """
    psi = np.asarray(psi, dtype=float)
    H0 = system.first_integral(np.zeros(4), 0.0)
    x = np.zeros(psi.shape + (4,))
    x[..., 0] = psi
    out = []
    for d in (+1.0, -1.0):
        rhs = 2.0 * (H0 - system.first_integral(x, -u_max * d)) / system.Delta(psi)
        own = (d * psi <= 0.0) & (rhs >= 0.0)
        out.append(np.where(own, d * np.sqrt(np.clip(rhs, 0.0, None)), np.nan))
    return out[0], out[1]


def _draw_switch(app, screen, pg):
    """Кривая переключения реле поверх карты. Выше неё (sigma > 0) реле
    подаёт -u_max, ниже -- +u_max; по ней самой корпус приезжает в вертикаль
    с нулевой скоростью. Траектория с одним переключением доезжает до кривой
    по одной линии уровня и дальше идёт вдоль неё."""
    if not app.show_switch or app.switch_up is None:
        return
    x, y, w, h = MAP
    clip = screen.get_clip()
    screen.set_clip((x, y, w, h))
    for values in (app.switch_up, app.switch_down):
        run = []
        for th, dth in zip(app.limit_psi, values):
            if not np.isfinite(dth):
                if len(run) > 1:
                    pg.draw.lines(screen, SWITCH, False, run, 2)
                run = []
                continue
            run.append(_to_map_px(app, th, dth))
        if len(run) > 1:
            pg.draw.lines(screen, SWITCH, False, run, 2)
    screen.set_clip(clip)


def switch_button_rect(font):
    """Кнопка «switch» в строке заголовка карты, у правого края панели.
    Чистая функция, как limit_button_rect: одна раскладка и на отрисовку, и на
    обработку клика."""
    x, y, w, _ = MAP
    bw = font.size("switch")[0] + 18
    return (x + w - bw, y - 25, bw, 22)


def draw_switch_button(app, screen, pg, font):
    rect = switch_button_rect(font)
    on = app.show_switch
    # Без предела реле не определено и рисовать нечего: кнопка гаснет, но
    # помнит своё положение -- как ручка слайдера при `no limit`.
    live = app.u_max is not None
    fill = BTN_ON if on else BTN
    if not live:
        fill = _shade(fill, 0.5)
    pg.draw.rect(screen, fill, rect, border_radius=4)
    pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=4)
    color = (BG if on else TEXT) if live else DIM
    screen.blit(font.render("switch", True, color), (rect[0] + 9, rect[1] + 3))


def _draw_horizon(app, screen, pg, font):
    """Пунктир на psi = +-pi/2: корпус лёг горизонтально.

    Отметка не про регулятор, а про задачу. Момент тяжести идёт как D*sin(psi)
    и ровно на pi/2 проходит максимум: правее он снова УБЫВАЕТ, и «дальше --
    всегда тяжелее» перестаёт быть правдой. Без линии на широкой карте это
    место ничем не отмечено, а глаз ищет его первым.

    Рисуется только когда pi/2 попал в границы: на приколоченной узкой
    карте (--psi-max 0.15) линия ушла бы за край, и pygame нарисовал бы её по
    самой кромке панели -- отметка не там, где написано, хуже, чем её
    отсутствие.
    """
    x, y, w, h = MAP
    if app.spec.psi_max < HORIZON_ANGLE:
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
    _panel(screen, pg, MAP, "INITIAL CONDITIONS  psi0 (rad) x dpsi0 (rad/s)", font)
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
    _draw_switch(app, screen, pg)
    draw_switch_button(app, screen, pg, font)

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

    labels = [(f"-{app.spec.psi_max:.2f}", x, y + h + 6),
              (f"+{app.spec.psi_max:.2f}", x + w - 34, y + h + 6),
              (f"+{app.spec.dpsi_max:.1f}", x - 46, y),
              (f"-{app.spec.dpsi_max:.1f}", x - 46, y + h - 14)]
    for text, tx, ty in labels:
        screen.blit(font.render(text, True, DIM), (tx, ty))


def _draw_body(target, pg, app, state, ox, oy, body_color, wheel_color, tip_color):
    """Корпус и колесо в координатах target со сдвигом (ox, oy).

    Одна функция и для основного робота (прямо на экране), и для призрака (на
    полупрозрачной подложке): различаются только цвета и куда рисуем.
    """
    _, _, w, h = ANIM
    p = app.world.p
    wheel_px = 26.0
    m2px = wheel_px / p.r
    cx, cy = ox + w / 2, oy + h * 0.72
    psi, theta = state[I_PSI], state[I_THETA]

    pg.draw.circle(target, wheel_color, (int(cx), int(cy)), int(wheel_px), 3)
    spoke = (cx + wheel_px * np.sin(theta), cy - wheel_px * np.cos(theta))
    pg.draw.line(target, wheel_color, (cx, cy), spoke, 2)

    tip = (cx + p.l * m2px * np.sin(psi), cy - p.l * m2px * np.cos(psi))
    pg.draw.line(target, body_color, (cx, cy), tip, 6)
    pg.draw.circle(target, tip_color, (int(tip[0]), int(tip[1])), 9)


def draw_robot(app, screen, pg, font):
    x, y, w, h = ANIM
    _panel(screen, pg, ANIM, "PLANT", font)
    if app.traj is None:
        return
    state = app.traj.x[app.frame]
    theta = state[I_THETA]
    # Разошедшаяся траектория: рисовать корпус по inf нельзя (int(nan) --
    # исключение), а сказать об этом надо -- это и есть ответ на вопрос
    # «что будет без предела» для угловых клеток.
    alive = bool(np.isfinite(state).all())
    finite_t = app.traj.t[np.isfinite(app.traj.x).all(axis=1)]
    diverged = len(finite_t) < len(app.traj)

    p = app.world.p
    wheel_px = 26.0
    m2px = wheel_px / p.r
    cy = y + h * 0.72

    # Дорога и её бегущие метки -- декорация сцены, а не робота: рисуются один
    # раз и по theta ОСНОВНОГО прогона, иначе два робота ехали бы по двум дорогам.
    road_x = x + 190            # левее стоит колонка чисел, см. ниже
    pg.draw.line(screen, GRID_LINE, (road_x, cy + wheel_px),
                 (x + w - 8, cy + wheel_px), 2)
    shift = (p.r * theta * m2px) % (0.1 * m2px)
    step = 0.1 * m2px
    k = -int(w / (2 * step)) - 1
    while x + w / 2 + k * step - shift < x + w:
        tx = x + w / 2 + k * step - shift
        if road_x < tx < x + w - 8:
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

    # Состояние на ВЫБРАННОМ кадре -- блоком на каждую систему, одними и теми
    # же строками и цветом своей кривой. Ради этого сравнения дорожка времени
    # и заведена: «здесь основной уже отдал управление, а сравнение ещё
    # упирается в предел» читается по числам, а не восстанавливается в голове.
    # Дорога начинается правее (road_x): линия шла бы сквозь цифры.
    # Колонка "=" выровнена вручную: имена после смены нотации (ER-022) стали
    # другой длины, автоподбора тут нет -- шрифт моноширинный.
    line_h = 16
    ty = y + 14
    screen.blit(font.render("t = %5.2f s" % (app.traj.t[app.frame] - app.traj.t[0]),
                            True, ACCENT), (x + 14, ty))
    ty += line_h + 6
    blocks = [("main   ", CONTROLLER_LABEL[app.main_key], app.traj, TEXT, app.frame)]
    if app.traj_cmp is not None:
        blocks.append(("compare", CONTROLLER_LABEL[app.cmp_key], app.traj_cmp,
                       GHOST, app.frame_of(app.traj_cmp)))
    if app.traj_est_cmp is not None:
        blocks.append(("vs est ", ESTIMATOR_LABEL[app.est_cmp_key],
                       app.traj_est_cmp, GHOST_EST,
                       app.frame_of(app.traj_est_cmp)))
    for name, label, traj, color, frame in blocks:
        s = traj.x[frame]
        torque_k = traj.u[min(frame, traj.n_steps - 1), 0]
        screen.blit(font.render("%s %s" % (name, label), True, color), (x + 14, ty))
        rows = [" psi   = %+.4f" % s[I_PSI], " dpsi  = %+.4f" % s[I_DPSI],
                " theta = %+.3f" % s[I_THETA], " u     = %s" % _fmt(torque_k)]
        for i, line in enumerate(rows):
            screen.blit(font.render(line, True, color),
                        (x + 14, ty + line_h * (i + 1)))
        ty += line_h * (len(rows) + 1) + 6
    if diverged:
        gone = float(finite_t[-1] - app.traj.t[0]) if len(finite_t) else 0.0
        screen.blit(font.render("diverged at t = %.2f s" % gone, True, ACCENT),
                    (x + 14, ty))


def _legend_inline(screen, pg, font, x, y, stop, items):
    """Легенда одной строкой: отрезок цвета кривой плюс подпись рядом.

    Правый верхний угол панели занят числами шкалы, поэтому строка обязана
    оборваться, не доходя до них. Обрыв не молчаливый: если поместилось не
    всё, в конце стоит `+n`. Молча пропавшая подпись читается как «такой
    кривой нет» -- это хуже, чем признаться, что не влезло.
    """
    left = list(items)
    while left:
        text, color = left[0]
        need = 16 + font.size(text)[0] + 14
        if x + need > stop:
            break
        pg.draw.line(screen, color, (x, y + 7), (x + 12, y + 7), 3)
        screen.blit(font.render(text, True, color), (x + 16, y))
        x += need
        left.pop(0)
    if left:
        tail = "+%d" % len(left)
        screen.blit(font.render(tail, True, DIM),
                    (min(x, stop - font.size(tail)[0]), y))


def _plot(screen, pg, rect, ts, ys, color, label, font, guides=(), ghosts=(),
          limit=None, estimate=None, legend=()):
    """ghosts -- список (ts, ys, цвет): вторые кривые полупрозрачным. Их может
    быть две -- другой РЕГУЛЯТОР и другой ОЦЕНИВАТЕЛЬ, -- и цвета у них разные,
    потому что вопросы разные. Масштаб оси берётся по ВСЕМ кривым, иначе
    сравнивать нечего: несколько картинок с разными осями.

    estimate = (ts3, ys3) -- ОЦЕНКА той же величины, тонкой линией цвета
    ESTIMATE. Это не третий регулятор, а то же самое состояние, увиденное
    фильтром; расстояние до основной кривой и есть ошибка оценки.

    legend -- [(подпись, цвет), ...] ровно по нарисованным линиям. Список
    собирает ВЫЗЫВАЮЩИЙ: только он знает, каким регулятором посчитана каждая
    кривая, панель же знает одни массивы. Класса-легенды тут нет и не нужно
    (запрет ER-005): функция и список."""
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
                pg.draw.line(screen, GUIDE, (x, gy), (x + w, gy), 1)
    if limit is not None:
        # ±psi_eq: за этим наклоном из состояния покоя не вернуться никаким
        # управлением. Не путать с psi_fall -- тот просто «считаем упавшим».
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
    _legend_inline(screen, pg, font, x + 6 + font.size(label)[0] + 16, y + 4,
                   x + w - font.size(_fmt(hi))[0] - 16, legend)
    return to_px


def plot_layout():
    """Прямоугольники панели TRAJECTORY: график psi, график u, дорожка ФАЗ,
    дорожка времени.

    Чистая функция, как picker_layout: по одной раскладке и рисуют, и ловят
    мышь. Обе дорожки стоят РОВНО под графиками и той же ширины -- тогда
    ручка времени, вертикальный курсор кадра на графиках и цветная полоса
    фазы это одно и то же место по оси t, и объяснять их словами не нужно.

    Высоту под дорожку фаз отняли у графиков (ER-016, решение Глеба 21.09):
    полоса поверх u(t) сделала бы фон цветным ровно там, где читают кривую.
    """
    x, y, w, h = PLOT
    half = (h - 46 - TIME_H - TIME_LABEL_H - PHASE_H - 10) / 2
    return ((x + 12, y + 10, w - 24, half),
            (x + 12, y + 20 + half, w - 24, half),
            (x + 12, y + 26 + 2 * half, w - 24, PHASE_H),
            (x + 12, y + 36 + 2 * half + PHASE_H, w - 24, TIME_H))


def _ghost_curves(app):
    """Вторые кривые графиков: [(траектория, цвет, подпись), ...].

    Подпись берётся у КНОПКИ, которая эту кривую завела: второй регулятор
    зовётся своим именем, второй оцениватель -- своим. Собранная здесь
    легенда поэтому не может разойтись с тем, что нарисовано.
    """
    out = []
    if app.traj_cmp is not None:
        out.append((app.traj_cmp, GHOST, CONTROLLER_LABEL[app.cmp_key]))
    if app.traj_est_cmp is not None:
        out.append((app.traj_est_cmp, GHOST_EST,
                    "est " + ESTIMATOR_LABEL[app.est_cmp_key]))
    return out


def _phase_rows(app):
    """Строки дорожки фаз: [(сводка, цвет подписи, подпись), ...].

    Порядок тот же, что у кривых: сначала основной, потом призраки в порядке
    легенды. Строка без сводки (траектории нет) в список не попадает -- пустая
    полоса читалась бы как «фазы не было».
    """
    rows = []
    if app.track is not None:
        rows.append((app.track, CURVE, CONTROLLER_LABEL[app.main_key]))
    if app.track_cmp is not None:
        rows.append((app.track_cmp, GHOST, CONTROLLER_LABEL[app.cmp_key]))
    if app.track_est_cmp is not None:
        rows.append((app.track_est_cmp, GHOST_EST,
                     "est " + ESTIMATOR_LABEL[app.est_cmp_key]))
    return rows


def _phase_segments(phases):
    """Отрезки постоянной фазы: [(начало, конец_включительно, код), ...].

    Рисовать поотсчётно нельзя: при 500 отсчётах на 576 пикселей соседние
    прямоугольники накладывались бы, и граница фазы вставала бы не там, где
    она есть. Отрезок же переводится в пиксели ровно теми же формулами, что и
    кривая над ним.
    """
    phases = np.asarray(phases, dtype=int)
    if phases.size == 0:
        return []
    edges = np.flatnonzero(np.diff(phases)) + 1
    starts = np.concatenate([[0], edges])
    stops = np.concatenate([edges - 1, [phases.size - 1]])
    return [(int(a), int(b), int(phases[a])) for a, b in zip(starts, stops)]


def draw_phase_track(app, screen, pg, font, rect, ts):
    """Дорожка фаз (ER-016): кто рулил в каждый момент.

    Строка на систему, цвет на фазу, те же пиксели по t, что у графиков и у
    дорожки времени. Это ответ на вопрос куратора «упал прогон в релейной
    фазе или уже после передачи»: раньше его угадывали по излому u(t).

    Фаза берётся из `app.track*` -- она посчитана один раз при выборе клетки
    (`Explorer._update_tracks`), здесь только рисование.
    """
    x, y, w, h = rect
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1)
    rows = _phase_rows(app)
    if not rows:
        return
    span = max(ts[-1] - ts[0], 1e-9)
    row_h = h / len(rows)
    for i, (summary, _, _) in enumerate(rows):
        phases = summary["phases"]
        ry = y + i * row_h
        for a, b, code in _phase_segments(phases):
            # Правая граница отрезка -- НАЧАЛО следующего отсчёта: последний
            # шаг отрезка действует до него, и без этого между фазами
            # оставалась бы щель в один кадр.
            t0 = ts[a] - ts[0]
            t1 = (ts[b + 1] if b + 1 < len(ts) else ts[-1]) - ts[0]
            px0 = x + t0 / span * w
            px1 = x + t1 / span * w
            pg.draw.rect(screen, PHASE_COLOR[code],
                         (px0, ry + 1, max(px1 - px0, 1.0), row_h - 2))
        # Подпись фазы пишется ВНУТРИ отрезка, когда он шире подписи, и только
        # в одну строку: на трёх строках высоты под текст нет, и там ключом
        # работает легенда графика u.
        if len(rows) == 1:
            for a, b, code in _phase_segments(phases):
                text = PHASE_LABEL[code]
                tw = font.size(text)[0]
                t0, t1 = ts[a] - ts[0], (ts[min(b + 1, len(ts) - 1)] - ts[0])
                px0, px1 = x + t0 / span * w, x + t1 / span * w
                if px1 - px0 > tw + 10:
                    screen.blit(font.render(text, True, BG),
                                (px0 + (px1 - px0 - tw) / 2, ry + (row_h - 14) / 2))


def _handover_lines(app, screen, pg, rect, to_px, ts):
    """Вертикали моментов передачи на графике: одна на фазу, цветом фазы.

    Линии рисуются по ОСНОВНОМУ регулятору: призраков может быть двое, и шесть
    вертикалей на графике -- это уже не подсказка, а сетка. Призрачные
    передачи видно на его строке дорожки.
    """
    if app.track is None:
        return
    for key, code in (("t_handover", PHASE_LQR), ("t_final", PHASE_FINAL)):
        t = app.track[key]
        if not np.isfinite(t):
            continue
        px = to_px(min(max(t, ts[0]), ts[-1]), 0.0)[0]
        pg.draw.line(screen, PHASE_COLOR[code], (px, rect[1]),
                     (px, rect[1] + rect[3]), 1)


def draw_plots(app, screen, pg, font):
    """Два графика (psi и u), легенды к ним и дорожка времени под ними.

    Панель V = x^T P x убрана 21.09 по решению Глеба: в логарифмической шкале
    три кривые и два уровня не читались, а вопрос «почему реле держит V
    наверху» она всё равно не решала. Сами разложения никуда не делись --
    считать их умеет любой скрипт по Trajectory, P и c*, которые окно и так
    отдаёт; просто окно их больше не рисует.
    """
    _panel(screen, pg, PLOT, "TRAJECTORY", font)
    if app.traj is None:
        return
    rect_psi, rect_u, rect_phase, _ = plot_layout()
    ts = app.traj.t - app.traj.t[0]
    ghosts = _ghost_curves(app)
    g_psi = [(g.t - g.t[0], g.x[:, I_PSI], c) for g, c, _ in ghosts]
    g_u = [((g.t - g.t[0])[:-1], g.u[:, 0], c) for g, c, _ in ghosts]

    # Оценка приходит из самой Trajectory (правило 6): окно не пересчитывает
    # фильтр и вообще не знает, какой датчик её породил. x_hat выровнена по u,
    # то есть на один кадр короче x -- отсюда ts[:-1].
    e_psi = None
    if app.traj.x_hat is not None:
        e_psi = (ts[:-1], app.traj.x_hat[:, I_PSI])

    # Порядок легенды -- по важности: сначала кривые (их и спрашивают: «что к
    # чему относится при сравнении двух регуляторов»), потом опорные линии.
    # Если строка не влезет, обрежется хвост из опорных, а не из кривых.
    legend = [(CONTROLLER_LABEL[app.main_key], PSI_CURVE)]
    legend += [(name, color) for _, color, name in ghosts]
    if e_psi is not None:
        legend.append(("x_hat " + ESTIMATOR_LABEL[app.est_key], ESTIMATE))
    if app.psi_eq is not None:
        legend.append(("psi_eq", LIMIT))
    legend.append(("psi_fall", GUIDE))
    to_px1 = _plot(screen, pg, rect_psi,
                   ts, app.traj.x[:, I_PSI], PSI_CURVE, "psi (rad)", font,
                   guides=(app.args.psi_fall,), ghosts=g_psi,
                   limit=app.psi_eq, estimate=e_psi, legend=legend)
    if e_psi is not None:
        err = app.traj.x_hat[:, I_PSI] - app.traj.x[:-1, I_PSI]
        finite = err[np.isfinite(err)]
        rmse = float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")
        # Вторая строка панели: первую занимает легенда, справа стоят числа
        # шкалы, и надпись наезжала бы на верхнюю границу диапазона.
        text = "est RMSE %.4f" % rmse
        screen.blit(font.render(text, True, ESTIMATE),
                    (rect_psi[0] + 6, rect_psi[1] + 22))
        if app.traj_est_cmp is not None and app.traj_est_cmp.x_hat is not None:
            e2 = (app.traj_est_cmp.x_hat[:, I_PSI]
                  - app.traj_est_cmp.x[:-1, I_PSI])
            f2 = e2[np.isfinite(e2)]
            r2 = float(np.sqrt(np.mean(f2 ** 2))) if f2.size else float("nan")
            screen.blit(font.render("vs %.4f" % r2, True, GHOST_EST),
                        (rect_psi[0] + 6 + font.size(text)[0] + 10,
                         rect_psi[1] + 22))

    tx, ty, tw, th = plot_toggle_rect(font)
    live_w = app.w_available and app.w_true is not None
    pg.draw.rect(screen, BTN_ON if app.plot_w else BTN, (tx, ty, tw, th), border_radius=4)
    screen.blit(font.render("[P] u | w", True, (BG if app.plot_w else
                                                (TEXT if live_w else GRID_LINE))),
                (tx + 7, ty + 2))
    if app.plot_w and live_w:
        # Толчок: истина -- из мира (её отдал Explorer.select), оценка -- из
        # Trajectory.w_hat (её записал rollout из Estimator.aux).
        tw_ = ts[:-1]
        w_true = app.w_true
        est = None
        if app.traj.w_hat is not None:
            est = (tw_, app.traj.w_hat[:, 1])
        legend_w = [("w_theta true", WIND), ("w_psi true", _shade(WIND, 0.55))]
        if est is not None:
            legend_w.append(("w_hat " + ESTIMATOR_LABEL[app.est_key], ESTIMATE))
        to_px2 = _plot(screen, pg, rect_u, tw_, w_true[:, 1], WIND, "w (N*m)", font,
                       ghosts=[(tw_, w_true[:, 0], _shade(WIND, 0.55))]
                       + ([] if app.traj.w_hat is None
                          else [(tw_, app.traj.w_hat[:, 0], _shade(ESTIMATE, 0.6))]),
                       estimate=est, legend=legend_w)
        if est is not None:
            err = app.traj.w_hat - w_true
            ok = np.isfinite(err).all(axis=1)
            if ok.any():
                rm = np.sqrt(np.mean(err[ok] ** 2, axis=0))
                screen.blit(font.render(f"w RMSE  psi {rm[0]:.3f}  theta {rm[1]:.3f}",
                                        True, ESTIMATE), (rect_u[0] + 6, rect_u[1] + 22))
        elif app.est_key != "kalman":
            screen.blit(font.render("no estimate: choose kalman+enc", True, DIM),
                        (rect_u[0] + 6, rect_u[1] + 22))
    else:
        legend_u = [(CONTROLLER_LABEL[app.main_key], U_CURVE)]
        legend_u += [(name, color) for _, color, name in ghosts]
        if app.run_u_max is not None:
            legend_u.append(("+-u_max", GUIDE))
        to_px2 = _plot(screen, pg, rect_u, ts[:-1], app.traj.u[:, 0], U_CURVE,
                       "u (N*m)", font,
                       guides=() if app.run_u_max is None else (app.run_u_max,),
                       ghosts=g_u, legend=legend_u)
    # Числа передачи -- второй строкой панели u, как est RMSE у панели psi:
    # они про ту же кривую, и искать их в другом конце окна незачем.
    if app.track is not None and not (app.plot_w and live_w):
        screen.blit(font.render(track_line(app.track), True, DIM),
                    (rect_u[0] + 6, rect_u[1] + 22))
    for to_px, rect in ((to_px1, rect_psi), (to_px2, rect_u)):
        _handover_lines(app, screen, pg, rect, to_px, ts)
        cursor = to_px(ts[min(app.frame, len(ts) - 1)], 0.0)[0]
        pg.draw.line(screen, ACCENT, (cursor, rect[1]),
                     (cursor, rect[1] + rect[3]), 1)
    draw_phase_track(app, screen, pg, font, rect_phase, ts)
    # Курсор кадра проходит и сквозь полосу: «на этом кадре рулит реле» должно
    # читаться в одном месте, а не сопоставлением двух картинок.
    cursor = to_px2(ts[min(app.frame, len(ts) - 1)], 0.0)[0]
    pg.draw.line(screen, ACCENT, (cursor, rect_phase[1]),
                 (cursor, rect_phase[1] + rect_phase[3]), 1)
    draw_time_slider(app, screen, pg, font)


# --- дорожка времени -----------------------------------------------------
#
# Геометрия отдельными чистыми функциями -- как у слайдера u_max: одни и те же
# формулы читают отрисовка, мышь и тест. Разойтись им тут особенно легко,
# потому что дорожка живёт внутри чужой панели.

def time_to_px(app, t: float) -> float:
    """Момент времени -> пиксель на дорожке. Ноль дорожки -- старт прогона."""
    x, _, w, _ = plot_layout()[3]
    span = app.play_span
    return x + float(np.clip(t, 0.0, span)) / max(span, 1e-9) * w


def time_from_px(app, px: float) -> float:
    """Пиксель -> момент времени. Округления до кадра тут НЕТ: кадр выбирает
    frame_of, и делать это в двух местах значило бы разойтись в одном."""
    x, _, w, _ = plot_layout()[3]
    f = float(np.clip((px - x) / w, 0.0, 1.0))
    return f * app.play_span


def time_hit(pos) -> bool:
    """Попали ли мышью в дорожку времени. Область шире на 8 px: ручка
    выступает за дорожку, и промах по собственной ручке -- баг интерфейса."""
    x, y, w, h = plot_layout()[3]
    return x - 8 <= pos[0] <= x + w + 8 and y - 8 <= pos[1] <= y + h + 8


def draw_time_slider(app, screen, pg, font):
    """Дорожка времени: выбор кадра УЖЕ посчитанной траектории.

    Пересчёта здесь нет и быть не может: все кадры лежат в Trajectory, а
    слайдер двигает одно число -- play_time. Поэтому дорожка не «перематывает
    прогон», а выбирает момент, на который смотрят сразу все панели: карта,
    робот, оба графика и обе сравниваемые системы.
    """
    x, y, w, h = plot_layout()[3]
    if app.traj is None:
        return
    span = app.play_span
    t = min(app.play_time, span)
    pg.draw.rect(screen, BTN, (x, y, w, h), border_radius=4)
    # На паузе дорожка ярче, чем на ходу: остановленное время -- это режим, а
    # не случайность, и по одной ручке его не отличить.
    tone = BTN_ON if not app.playing else _shade(BTN_ON, 0.7)
    pg.draw.rect(screen, tone, (x, y, max(time_to_px(app, t) - x, 1.0), h),
                 border_radius=4)
    pg.draw.rect(screen, GRID_LINE, (x, y, w, h), 1, border_radius=4)
    pg.draw.circle(screen, ACCENT, (int(time_to_px(app, t)), int(y + h / 2)), 7)

    ly = y + h + 4
    screen.blit(font.render("0", True, DIM), (x, ly))
    right = "%.2f s" % span
    screen.blit(font.render(right, True, DIM),
                (x + w - font.size(right)[0], ly))
    mid = "t = %.2f s   frame %d/%d   %s" % (
        t, app.frame, len(app.traj) - 1, "playing" if app.playing else "paused")
    screen.blit(font.render(mid, True, DIM if app.playing else ACCENT),
                (x + (w - font.size(mid)[0]) / 2, ly))


#: Подписи рядов кнопок. row 0 -- регулятор, красящий карту; row 1 -- второй
#: регулятор, рисуемый призраком; row 2 -- ОЦЕНИВАТЕЛЬ, который тоже красит
#: карту: исход клетки зависит и от того, что регулятор видит.
PICK_ROWS = ("MAIN", "COMPARE", "ESTIMATOR")


def pick_label(row: int, key) -> str:
    """Подпись кнопки. У рядов разные словари, и путать их нельзя."""
    if key is None:
        return "off"
    return ESTIMATOR_LABEL[key] if row == 2 else CONTROLLER_LABEL[key]


#: Размеры шрифта кнопок, от крупного к мелкому. Ряд кнопок -- единственное
#: место окна, ширина которого растёт от СПИСКА регуляторов, а не от раскладки:
#: каждый новый закон добавляет кнопку. Поэтому размер подбирается, а не задан
#: числом -- иначе очередная запись в CONTROLLERS молча уезжала бы за край
#: (после ER-018 ряд COMPARE при 15 пунктах кончался на 1459 при W = 1280).
PICK_FONT_NAME = "consolas,dejavusansmono,monospace"
PICK_FONT_SIZES = (15, 14, 13, 12, 11, 10)
PICK_MARGIN = 16
_PICK_SIZE_CACHE = {}


def picker_font_size(pg):
    """Самый КРУПНЫЙ размер из PICK_FONT_SIZES, при котором самый длинный ряд
    кнопок влезает в окно. Мельче минимального не опускаемся: нечитаемая
    кнопка -- такой же баг интерфейса, как обрезанная, и если ряд не влез даже
    так, честнее увидеть обрез, чем пиксельную кашу.

    Кэшируется ЧИСЛО, а не объект шрифта. Объект привязан к инициализированному
    pygame: модульный кэш пережил бы `pg.quit()` мёртвой ссылкой, и следующий
    же `render` по нему падает сегфолтом (ловится прогоном виз-тестов подряд).
    """
    rows = (CONTROLLER_KEYS, CONTROLLER_KEYS + [None], ESTIMATOR_KEYS)
    key = tuple(tuple(r) for r in rows)
    if key in _PICK_SIZE_CACHE:
        return _PICK_SIZE_CACHE[key]
    size = PICK_FONT_SIZES[-1]
    for candidate in PICK_FONT_SIZES:
        font = pg.font.SysFont(PICK_FONT_NAME, candidate)
        ends = []
        for row, keys in enumerate(rows):
            bx = PICK[0] + 108
            for k in keys:
                bx += font.size(pick_label(row, k))[0] + 18 + 6
            ends.append(bx - 6)
        if max(ends) <= W - PICK_MARGIN:
            size = candidate
            break
    _PICK_SIZE_CACHE[key] = size
    return size


def picker_font(pg):
    """Шрифт кнопок. Объект живёт не дольше окна -- см. `picker_font_size`."""
    return pg.font.SysFont(PICK_FONT_NAME, picker_font_size(pg))


def picker_layout(app, font):
    """Прямоугольники кнопок: [(rect, row, key), ...]. row 0 -- основной,
    row 1 -- сравнение, row 2 -- оцениватель. Одна и та же раскладка нужна и
    отрисовке, и обработке клика, поэтому она -- чистая функция, а не побочный
    эффект рисования.

    `font` обязан быть тем же, которым кнопки РИСУЮТСЯ (`picker_font`):
    раскладка, посчитанная другим шрифтом, ловила бы мышь не там, где рисует.
    """
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
    if app.est_key == "kalman":
        # tau -- ручка комплементарного фильтра; Калману она ни к чему.
        # Надпись -- только если влезает до кнопок мира.
        text = "kalman: tau unused"
        if hx + font.size(text)[0] <= world_buttons(app, font)[-1][2][0] - 10:
            screen.blit(font.render(text, True, GRID_LINE), (hx, NOISE_Y))
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
        if hx + 12 + font.size(text)[0] > world_buttons(app, font)[-1][2][0] - 10:
            break
        pg.draw.rect(screen, color, (hx, NOISE_Y + 4, 8, 8))
        screen.blit(font.render(text, True, color), (hx + 12, NOISE_Y))
        hx += 12 + font.size(text)[0] + 14
    tail = "tau*, white-accel model vs measured"
    if hx + font.size(tail)[0] <= world_buttons(app, font)[-1][2][0] - 10:
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

    # Отметка предела, за которым карта упирается в psi_fall: цвет тот же,
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
        note, color = "map pinned by --psi-max / --dpsi-max", DIM
    elif app.u_clip is not None and app.u_max > app.u_clip:
        note = f"map clipped by psi_fall = {app.args.psi_fall:g}"
        color = LIMIT
    else:
        note = f"map: psi +-{MAP_FIT:g}*pi/2, dpsi {MAP_FIT:g}x recoverable"
        color = DIM
    screen.blit(font.render(note, True, color), (rect[0] + rect[2] + 16, y - 1))

    srect = settings_button_rect(font)
    pg.draw.rect(screen, BTN_ON if app.settings_open else BTN, srect, border_radius=4)
    pg.draw.rect(screen, GRID_LINE, srect, 1, border_radius=4)
    screen.blit(font.render("settings", True, BG if app.settings_open else TEXT),
                (srect[0] + 9, srect[1] + 3))

    nrect = noise_button_rect(font)
    pg.draw.rect(screen, BTN_ON if app.noise_open else BTN, nrect, border_radius=4)
    pg.draw.rect(screen, GRID_LINE, nrect, 1, border_radius=4)
    screen.blit(font.render("noise", True, BG if app.noise_open else TEXT),
                (nrect[0] + 9, nrect[1] + 3))


def settings_button_rect(font):
    """Кнопка «settings» в правом конце ряда u_max. Чистая функция, как
    limit_button_rect: одна раскладка и на отрисовку, и на обработку клика."""
    w = font.size("settings")[0] + 18
    return (W - 24 - w, SLIDER[1] - 4, w, 22)


def noise_button_rect(font):
    """Кнопка «noise» слева от «settings», в том же ряду."""
    sx = settings_button_rect(font)[0]
    w = font.size("noise")[0] + 18
    return (sx - 10 - w, SLIDER[1] - 4, w, 22)


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
    screen.blit(font.render("SETTINGS  --  LQR cost  Q = diag(q_psi, q_theta, q_dpsi, q_dtheta)",
                            True, TEXT), (x + 24, y + 16))
    screen.blit(font.render("base: LQR and phase 3;  soft: phase 2 of both relay schemes",
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


# --- панель шума ---------------------------------------------------------------
#  Устроена ровно как панель настроек: чистая раскладка, отрисовка, клик,
#  клавиши.  Проверка и пересчёт -- в Explorer.apply_noise.


def noise_panel_layout(font):
    """Раскладка панели шума: [(kind, key, rect)], как settings_layout."""
    x, y, w, h = NOISE_PANEL
    out = []
    for c, (m, _, _) in enumerate(CORR_MATRICES):
        for i in range(len(CORR_PAIRS)):
            out.append(("field", f"{m}.{i}", (x + 170 + c * 165, y + 110 + i * 30, 140, 24)))
    bx = x + 24
    for key in ("apply", "defaults", "close"):
        bw = font.size(key)[0] + 24
        out.append(("button", key, (bx, y + h - 44, bw, 26)))
        bx += bw + 10
    return out


def draw_noise_panel(app, screen, pg, font):
    """Панель шума поверх окна: корреляции каналов ИДУ, три матрицы."""
    if not app.noise_open:
        return
    veil = pg.Surface((W, H), pg.SRCALPHA)
    veil.fill(BG + (150,))
    screen.blit(veil, (0, 0))
    x, y, w, h = NOISE_PANEL
    pg.draw.rect(screen, PANEL, NOISE_PANEL, border_radius=6)
    pg.draw.rect(screen, GRID_LINE, NOISE_PANEL, 1, border_radius=6)
    screen.blit(font.render("NOISE  --  IMU channel correlation  Sigma = D R D",
                            True, TEXT), (x + 24, y + 16))
    screen.blit(font.render("R: 3x3, ones on the diagonal, symmetric -- enter the upper triangle",
                            True, DIM), (x + 24, y + 38))
    screen.blit(font.render("rho", True, DIM), (x + 24, y + 86))
    for c, (_, label, _) in enumerate(CORR_MATRICES):
        screen.blit(font.render(label, True, TEXT), (x + 170 + c * 165, y + 86))
    for i, (_, _, label) in enumerate(CORR_PAIRS):
        screen.blit(font.render(label, True, DIM), (x + 24, y + 114 + i * 30))

    for kind, key, rect in noise_panel_layout(font):
        if kind == "field":
            focused = key == app.noise_focus
            pg.draw.rect(screen, BG, rect, border_radius=3)
            pg.draw.rect(screen, ACCENT if focused else GRID_LINE, rect, 1, border_radius=3)
            text = app.noise_text.get(key, "")
            screen.blit(font.render(text + ("_" if focused else ""), True, TEXT),
                        (rect[0] + 6, rect[1] + 4))
        else:
            fill = BTN_ON if key == "apply" else BTN
            pg.draw.rect(screen, fill, rect, border_radius=4)
            screen.blit(font.render(key, True, BG if key == "apply" else TEXT),
                        (rect[0] + 12, rect[1] + 5))

    # Чем датчик окна посчитан сейчас: у ухода окно не задаёт sigma, берутся
    # значения IMUSensor по умолчанию -- и на горизонте в секунды уход мал,
    # так что корреляция drift почти не видна.  Лучше сказать это прямо, чем
    # оставить гадать, почему столбец ничего не меняет.
    s = IMUSensor(app.system, dt=app.args.dt, mode="imu")
    lines = [
        f"white:   sig_a {app.noise['sigma_a']:.1e}, sig_g {app.noise['sigma_g']:.1e}  (sliders)",
        f"drift:   sig_ba {s.sigma_b[0]:.0e}, sig_bg {s.sigma_b[2]:.0e}, tau inf  (sensor defaults)",
        f"turn-on: b0_g {app.noise['b0_g']:.1e}, b0_a = 9.8 * b0_g  (slider)",
    ]
    if app.est_key == "ideal" and app.est_cmp_key is None:
        lines.append("estimator ideal: the map does not use the IMU")
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, DIM), (x + 24, y + 220 + i * 20))
    if app.noise_error:
        screen.blit(font.render(app.noise_error, True, (235, 110, 90)),
                    (x + 24, y + h - 78))
    hint = "Enter apply   Tab next   Esc close"
    screen.blit(font.render(hint, True, DIM), (x + w - 24 - font.size(hint)[0], y + h - 39))


def noise_click(app, pos, font):
    """Клик при открытой панели шума. Вне панели -- ничего (модальная)."""
    for kind, key, (rx, ry, rw, rh) in noise_panel_layout(font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            if kind == "field":
                app.noise_focus = key
            elif key == "apply":
                noise_submit(app)
            elif key == "defaults":
                for m, _, _ in CORR_MATRICES:
                    for i, v in enumerate(DEFAULT_CORR[m]):
                        app.noise_text[f"{m}.{i}"] = f"{v:g}"
            elif key == "close":
                app.noise_open = False
            return


def noise_submit(app):
    """Применить введённое. Ошибка остаётся на панели, окно -- прежним."""
    error = app.apply_noise(dict(app.noise_text))
    app.noise_error = error or ""
    if error is None:
        app.noise_open = False


def noise_key(app, event, pg):
    """Клавиша при открытой панели шума; то же поведение, что settings_key."""
    if event.key == pg.K_ESCAPE:
        app.noise_open = False
    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
        noise_submit(app)
    elif event.key in (pg.K_TAB, pg.K_DOWN, pg.K_UP):
        step = -1 if (event.key == pg.K_UP or event.mod & pg.KMOD_SHIFT) else 1
        i = NOISE_KEYS.index(app.noise_focus) if app.noise_focus else -1
        app.noise_focus = NOISE_KEYS[(i + step) % len(NOISE_KEYS)]
    elif app.noise_focus is not None:
        text = app.noise_text.get(app.noise_focus, "")
        if event.key == pg.K_BACKSPACE:
            app.noise_text[app.noise_focus] = text[:-1]
        elif event.unicode and event.unicode in SETTINGS_CHARS:
            app.noise_text[app.noise_focus] = text + event.unicode
    return True


# --- виды фильтра: A (эллипс 1σ) и B (измерения) -------------------------------
#  Рисуются поверх правой половины окна (PLANT + графики), до дорожки времени:
#  карта слева и дорожка внизу остаются и управляют видом (решение Глеба 01.10).


def filter_view_rect():
    """Прямоугольник вида: от верха PLANT до дорожки фаз (не включая)."""
    phase = plot_layout()[2]
    return (ANIM[0], ANIM[1], ANIM[2], phase[1] - ANIM[1] - 6)


def view_buttons(app, font):
    """[(key, label, rect)] в строке заголовка PLANT, справа."""
    out, right = [], ANIM[0] + ANIM[2]
    for key, label in (("C", "[B] biases"), ("B", "[H] y vs h(x,u)"), ("A", "[V] P ellipse")):
        w = font.size(label)[0] + 14
        right -= w
        out.append((key, label, (right, ANIM[1] - 22, w, 20)))
        right -= 8
    return out


def draw_view_buttons(app, screen, pg, font):
    for key, label, rect in view_buttons(app, font):
        on = app.view == key
        pg.draw.rect(screen, BTN_ON if on else BTN, rect, border_radius=4)
        screen.blit(font.render(label, True, BG if on else TEXT), (rect[0] + 7, rect[1] + 2))


def view_buttons_click(app, pos, font) -> bool:
    for key, _, (rx, ry, rw, rh) in view_buttons(app, font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            app.toggle_view(key)
            return True
    return False


def _ellipse_1sigma(P2, n=96):
    """Контур e^T P2^-1 e = 1: полуоси sqrt(lambda) по собственным векторам
    (docs/kalman_views.md §9.2, §10).  Возвращает (n, 2)."""
    a, b, d = P2[0, 0], P2[0, 1], P2[1, 1]
    half, root = 0.5 * (a + d), np.sqrt((0.5 * (a - d)) ** 2 + b ** 2)
    l1, l2 = max(half + root, 0.0), max(half - root, 0.0)
    phi = 0.5 * np.arctan2(2 * b, a - d)
    v1 = np.array([np.cos(phi), np.sin(phi)])
    v2 = np.array([-np.sin(phi), np.cos(phi)])
    tt = np.linspace(0, 2 * np.pi, n)
    return (np.sqrt(l1) * np.cos(tt)[:, None] * v1 + np.sqrt(l2) * np.sin(tt)[:, None] * v2)


#: Порог вырождения блока P2 (§17.1): cond выше -- эллипс считаем отрезком.
COND_FLAT = 1e8


def _mahalanobis2(e, P2):
    """d^2 = e^T P2^-1 e по кадрам через спектр, без inv (§17.1).

    При Q = 0 блок P2 к нескольким секундам вырождается (cond ~ 1e16, а
    округление даёт и отрицательное собственное число) -- inv падал с
    Singular matrix.  Собственные числа поднимаем до 1e-12 * tr P2;
    flat[k] = True, если cond P2[k] > COND_FLAT, -- такое d^2 не показатель.
    Возвращает (d2, flat), оба формы (n,)."""
    lam, V = np.linalg.eigh(0.5 * (P2 + np.swapaxes(P2, -1, -2)))
    tr = np.maximum(lam.sum(axis=-1, keepdims=True), np.finfo(float).tiny)
    floor = 1e-12 * tr
    flat = lam[:, 0] < lam[:, -1] / COND_FLAT
    lam = np.maximum(lam, floor)
    z = np.einsum("kij,ki->kj", V, e)
    return np.sum(z * z / lam, axis=-1), flat


def _series(screen, pg, font, rect, t, series, title, cursor_t, band=None, log=False,
            clip_q=99.5):
    """Маленький график: series = [(y, color, label)], band = (lo, hi, color).
    log -- ось log10 для положительных величин (sqrt(P) падает на порядки).
    Масштаб -- по квантилю clip_q, чтобы всплеск первых шагов не съел всё."""
    x, y, w, h = rect
    pg.draw.rect(screen, BG, rect, border_radius=3)
    pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=3)
    tx = x + 6
    screen.blit(font.render(title, True, DIM), (tx, y + 3))
    tx += font.size(title)[0] + 12
    for _, color, label in series:
        screen.blit(font.render(label, True, color), (tx, y + 3))
        tx += font.size(label)[0] + 12
    stack = [np.asarray(v, float) for v, _, _ in series]
    if band is not None:
        stack += [np.asarray(band[0], float), np.asarray(band[1], float)]
    vals = np.concatenate([v[np.isfinite(v)] for v in stack]) if stack else np.zeros(1)
    if log:
        vals = vals[vals > 0]
        if vals.size == 0:
            return
        lo, hi = np.log10(np.percentile(vals, 100 - clip_q)), np.log10(vals.max())
        f = lambda v: np.log10(np.maximum(v, 10 ** lo))
    else:
        m = np.percentile(np.abs(vals), clip_q) if vals.size else 1.0
        lo, hi = -m, m
        f = lambda v: v
    if not np.isfinite(hi - lo) or hi <= lo:
        hi = lo + 1.0
    top, bot = y + 22, y + h - 4
    t0, t1 = float(t[0]), float(t[-1]) if t[-1] > t[0] else float(t[0]) + 1.0
    step = max(1, len(t) // max(1, w))
    tt = np.asarray(t)[::step]

    def px(tv, v):
        vv = np.clip(f(v), lo, hi)
        return (x + 4 + (tv - t0) / (t1 - t0) * (w - 8), bot - (vv - lo) / (hi - lo) * (bot - top))

    if band is not None:
        blo, bhi = np.asarray(band[0])[::step], np.asarray(band[1])[::step]
        pts = [px(a, b) for a, b in zip(tt, bhi)] + [px(a, b) for a, b in zip(tt[::-1], blo[::-1])]
        pts = [p for p in pts if np.isfinite(p[1])]
        if len(pts) > 2:
            surf = pg.Surface((W, H), pg.SRCALPHA)
            pg.draw.polygon(surf, band[2] + (70,), pts)
            screen.blit(surf, (0, 0))
    if not log:
        _, zy = px(t0, 0.0)
        pg.draw.line(screen, GRID_LINE, (x + 4, zy), (x + w - 4, zy), 1)
    for v, color, _ in series:
        vv = np.asarray(v)[::step]
        pts = [px(a, b) for a, b in zip(tt, vv) if np.isfinite(b)]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 1)
    lab_hi = f"{10 ** hi:.1e}" if log else f"{hi:+.2g}"
    lab_lo = f"{10 ** lo:.1e}" if log else f"{lo:+.2g}"
    screen.blit(font.render(lab_hi, True, GRID_LINE), (x + w - 6 - font.size(lab_hi)[0], y + 3))
    screen.blit(font.render(lab_lo, True, GRID_LINE), (x + w - 6 - font.size(lab_lo)[0], bot - 16))
    cx = px(cursor_t, 0.0 if not log else 10 ** hi)[0]
    pg.draw.line(screen, ACCENT, (cx, top), (cx, bot), 1)


#: Окно скользящего среднего на виде B (§17.3 а), с: 100 отсчётов при dt = 1 мс.
MEAN_WINDOW_S = 0.1


def _running_mean(v, n):
    """Скользящее среднее назад по n отсчётам (в начале -- по тому, что есть).
    Назад, а не по центру: линия не знает будущего, как и фильтр."""
    v = np.asarray(v, float)
    n = max(1, int(round(n)))
    c = np.concatenate([[0.0], np.cumsum(v)])
    i = np.arange(1, len(v) + 1)
    lo = np.maximum(i - n, 0)
    return (c[i] - c[lo]) / (i - lo)


def _ellipse_px(center_px, P2, sx, sy, k=1.0):
    """Контур эллипса 1σ (× k) в пикселях: sx, sy -- пикселей на единицу
    по psi и по dpsi (ось dpsi вверх)."""
    pts = _ellipse_1sigma(P2) * k
    return [(center_px[0] + p[0] * sx, center_px[1] - p[1] * sy) for p in pts]


def draw_view_a(app, screen, pg, font):
    """Вид A (§12, §14): фазовая плоскость (psi, dpsi) -- истинная траектория
    и оценка растут до текущего кадра; в текущей точке эллипс 1σ по P,
    увеличенный в k раз (одно k на прогон).  Лупа в углу -- та же точка в
    НАСТОЯЩЕМ масштабе: оценка, истина и эллипс 1σ."""
    x, y, w, h = filter_view_rect()
    d = app.diag
    k = app.diag_frame()
    X = d["x"][:, [I_PSI, I_DPSI]]
    Xh = d["x_hat"][:, [I_PSI, I_DPSI]]
    P2 = d["P"][:, [I_PSI, I_DPSI]][:, :, [I_PSI, I_DPSI]]
    s_psi, s_dpsi = np.sqrt(P2[:, 0, 0]), np.sqrt(P2[:, 1, 1])

    # --- большая плоскость
    plane = (x + 12, y + 34, w - 24, h - 34 - 150)
    pg.draw.rect(screen, BG, plane, border_radius=3)
    pg.draw.rect(screen, GRID_LINE, plane, 1, border_radius=3)
    both = np.vstack([X, Xh])
    both = both[np.isfinite(both).all(axis=1)]
    lo, hi = both.min(axis=0), both.max(axis=0)
    pad = 0.08 * np.maximum(hi - lo, 1e-6)
    lo, hi = lo - pad, hi + pad
    px_per = ((plane[2] - 16) / (hi[0] - lo[0]), (plane[3] - 16) / (hi[1] - lo[1]))

    def to(p):
        return (plane[0] + 8 + (p[0] - lo[0]) * px_per[0],
                plane[1] + plane[3] - 8 - (p[1] - lo[1]) * px_per[1])

    if lo[0] < 0 < hi[0]:
        pg.draw.line(screen, GRID_LINE, to((0, lo[1])), to((0, hi[1])), 1)
    if lo[1] < 0 < hi[1]:
        pg.draw.line(screen, GRID_LINE, to((lo[0], 0)), to((hi[0], 0)), 1)
    step = max(1, (k + 1) // 1500)
    for traj, color in ((X, CURVE), (Xh, ESTIMATE)):
        pts = [to(p) for p in traj[:k + 1:step] if np.all(np.isfinite(p))]
        if len(pts) > 1:
            pg.draw.lines(screen, color, False, pts, 2 if color is CURVE else 1)
    kk = d["scale_k"]
    if np.all(np.isfinite(Xh[k])):
        ell = _ellipse_px(to(Xh[k]), P2[k], px_per[0], px_per[1], kk)
        pg.draw.lines(screen, ESTIMATE, True, ell, 2)
    if np.all(np.isfinite(X[k])):
        tx_, ty_ = to(X[k])
        pg.draw.circle(screen, ACCENT, (int(tx_), int(ty_)), 4)
    screen.blit(font.render("psi [rad] ->", True, DIM),
                (plane[0] + plane[2] - 110, plane[1] + plane[3] - 20))
    screen.blit(font.render("dpsi [rad/s]", True, DIM), (plane[0] + 6, plane[1] + 4))
    legend = [("truth x(t)", CURVE), ("estimate x_hat(t)", ESTIMATE),
              (f"ellipse 1 sigma x{kk:.0e}", ESTIMATE)]
    lx = plane[0] + 120
    for label, color in legend:
        screen.blit(font.render(label, True, color), (lx, plane[1] + 4))
        lx += font.size(label)[0] + 14

    # --- лупа: настоящий масштаб вокруг оценки
    # Лупа -- в тот угол плоскости, где меньше всего кривой: иначе она
    # закрывает ровно то, на что смотрят.
    side = 170
    corners = [(plane[0] + plane[2] - side - 8, plane[1] + 26),
               (plane[0] + 8, plane[1] + 26),
               (plane[0] + plane[2] - side - 8, plane[1] + plane[3] - side - 26),
               (plane[0] + 8, plane[1] + plane[3] - side - 26)]
    pts_all = np.array([to(p) for p in np.vstack([X[::20], Xh[k:k + 1]])
                        if np.all(np.isfinite(p))])

    def crowd(cx, cy):
        if pts_all.size == 0:
            return 0
        inside = ((pts_all[:, 0] > cx - 10) & (pts_all[:, 0] < cx + side + 10)
                  & (pts_all[:, 1] > cy - 10) & (pts_all[:, 1] < cy + side + 10))
        return int(inside.sum())

    lx0, ly0 = min(corners, key=lambda c: crowd(*c))
    lens = (lx0, ly0, side, side)
    pg.draw.rect(screen, PANEL, lens, border_radius=3)
    pg.draw.rect(screen, ACCENT, lens, 1, border_radius=3)
    err = X[k] - Xh[k]
    # Оси лупы ЗАКРЕПЛЕНЫ на весь прогон (решение Глеба 01.10): полуширина --
    # 1.5 sqrt(P) первого кадра.  Тогда видно, как эллипс сжимается со
    # временем; истина за краем прижимается к рамке.
    half = d["lens_half"]
    err = np.clip(err, (-0.98 * half[0], -0.98 * half[1]), (0.98 * half[0], 0.98 * half[1]))
    c = (lens[0] + side / 2, lens[1] + side / 2)
    lsx, lsy = side / 2 / half[0], side / 2 / half[1]
    pg.draw.line(screen, GRID_LINE, (lens[0], c[1]), (lens[0] + side, c[1]), 1)
    pg.draw.line(screen, GRID_LINE, (c[0], lens[1]), (c[0], lens[1] + side), 1)
    pg.draw.lines(screen, ESTIMATE, True, _ellipse_px(c, P2[k], lsx, lsy), 2)
    pg.draw.circle(screen, ESTIMATE, (int(c[0]), int(c[1])), 3)
    if np.all(np.isfinite(err)):
        ex = (c[0] + err[0] * lsx, c[1] - err[1] * lsy)
        pg.draw.circle(screen, ACCENT, (int(ex[0]), int(ex[1])), 4)
    screen.blit(font.render("lens x1, fixed axes", True, ACCENT), (lens[0] + 4, lens[1] + 2))
    screen.blit(font.render(f"+-{half[0]:.0e}", True, DIM),
                (lens[0] + side - 4 - font.size(f"+-{half[0]:.0e}")[0], c[1] + 2))
    screen.blit(font.render(f"+-{half[1]:.0e}", True, DIM), (c[0] + 4, lens[1] + side - 18))

    # --- числа и графики sqrt(P), |e|
    e = Xh - X
    d2, flat = _mahalanobis2(e[:k + 1], P2[:k + 1])
    by = plane[1] + plane[3] + 6
    if flat[-1]:
        # §17.1: при Q = 0 эллипс сплющивается в отрезок вдоль неустойчивого
        # направления; d^2 тогда мерит округление, а не ошибку -- не печатаем.
        second = ("P degenerate (Q = 0?): ellipse is a segment, d^2 not defined", ACCENT)
    else:
        ok = ~flat
        inside = float(np.mean(d2[ok] <= 1.0)) if ok.any() else float("nan")
        second = (f"truth inside 1 sigma: now d^2 = {d2[-1]:.2f}; so far {100 * inside:.0f} % "
                  "(honest ~39 %)", ACCENT)
    lines = [(f"sqrt(P): psi {s_psi[k]:.1e},  dpsi {s_dpsi[k]:.1e}", ESTIMATE), second]
    for i, (line, color) in enumerate(lines):
        screen.blit(font.render(line, True, color), (x + 12, by + i * 18))
    gy = by + 40
    gw = (w - 36) // 2
    t = d["t"]
    _series(screen, pg, font, (x + 12, gy, gw, y + h - 8 - gy), t,
            [(s_psi, ESTIMATE, "sqrt(P)"), (np.abs(e[:, 0]), ACCENT, "|e|")],
            "psi", t[k], log=True, clip_q=100)
    _series(screen, pg, font, (x + 24 + gw, gy, gw, y + h - 8 - gy), t,
            [(s_dpsi, ESTIMATE, "sqrt(P)"), (np.abs(e[:, 1]), ACCENT, "|e|")],
            "dpsi", t[k], log=True, clip_q=100)


VIEW_B_CHANNELS = (("a_x", "m/s^2"), ("a_z", "m/s^2"), ("omega", "rad/s"), ("enc", "rad"))


def draw_view_b(app, screen, pg, font):
    """Вид B: разности относительно чистого показания h(x, u) (§4):
    y - h(x,u) -- смещение + шум датчика (не убывает);
    h(x^-,u) - h(x,u) -- ошибка фильтра в показании (убывает);
    коридор +-sqrt(S) вокруг предсказания: S = C P^- C^T + R."""
    x, y, w, h = filter_view_rect()
    d = app.diag
    k = app.diag_frame()
    t = d["t"]
    m = d["y"].shape[1]
    top = y + 34
    row_h = (h - 44) // m
    for i in range(m):
        name, unit = VIEW_B_CHANNELS[i]
        dy = d["y"][:, i] - d["y_clean"][:, i]
        dp = d["y_pred"][:, i] - d["y_clean"][:, i]
        sd = np.sqrt(d["S"][:, i])
        # Разложение на «bias» и «state» (§13) рисовалось 01.10 и убрано по
        # решению Глеба: график читался хуже.  Данные (b_prior, b_true) в
        # self.diag остаются -- для тестов и анализа.
        # Скользящее среднее серого облака (§17.3 а, решение Глеба 01.10):
        # шум усредняется в sqrt(N) раз, и линия садится на уровень смещения.
        _series(screen, pg, font, (x + 12, top + i * row_h, w - 24, row_h - 6), t,
                [(dy, _shade(CURVE, 0.55), "y - h(x,u)"),
                 (_running_mean(dy, MEAN_WINDOW_S / app.args.dt), ACCENT,
                  f"mean {MEAN_WINDOW_S:g} s"),
                 (dp, ESTIMATE, "h(x^-,u) - h(x,u)")],
                f"{name} [{unit}]", t[k], band=(dp - sd, dp + sd, ESTIMATE))


VIEW_C_CHANNELS = (("b_ax", "m/s^2"), ("b_az", "m/s^2"), ("b_g", "rad/s"))


def draw_view_c(app, screen, pg, font):
    """Вид C (§17.3 б): как фильтр справляется со смещениями.  По каналу:
    |b^ - b| (ошибка оценки), sqrt(P_bb) (что фильтр о ней думает) и |b|
    (само смещение -- для масштаба).  Ось log10, как у графиков вида A:
    ошибка падает на порядки, в линейной оси её конец не виден."""
    x, y, w, h = filter_view_rect()
    d = app.diag
    k = app.diag_frame()
    t = d["t"]
    b_hat, b_true = d["b_hat"], d["b_true"]
    P = d["P"]
    nb = P.shape[1]
    top = y + 34
    row_h = (h - 44) // 4
    for i, (name, unit) in enumerate(VIEW_C_CHANNELS):
        j = nb - 3 + i                     # смещения -- последние три в состоянии
        err = np.abs(b_hat[:, i] - b_true[:, i])
        sp = np.sqrt(np.maximum(P[:, j, j], 0.0))
        # Числа текущего кадра -- в подписях кривых: заголовок короткий,
        # иначе он наезжает на метки оси справа.
        _series(screen, pg, font, (x + 12, top + i * row_h, w - 24, row_h - 6), t,
                [(np.abs(b_true[:, i]), DIM, f"|b| {abs(b_true[k, i]):.1e}"),
                 (sp, ESTIMATE, f"sqrt(P) {sp[k]:.1e}"),
                 (err, ACCENT, f"|b^-b| {err[k]:.1e}")],
                name, t[k], log=True, clip_q=100)

    # Честность по ансамблю (kalman_speed §6.2): одна клетка не говорит,
    # честна ли P, а все клетки карты в один момент -- говорят.
    rect = (x + 12, top + 3 * row_h, w - 24, row_h - 6)
    nees = app.nees
    if nees is None or not len(nees["t"]):
        pg.draw.rect(screen, BG, rect, border_radius=3)
        pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=3)
        screen.blit(font.render("NEES over the map: needs the map computed with kalman+enc",
                                True, DIM), (rect[0] + 6, rect[1] + 3))
        return
    tn, mean, cnt, wild = nees["t"], nees["mean"], nees["count"], nees["wild"]
    lo, hi = _chi2_mean_band(3, cnt)
    j = int(np.clip(np.searchsorted(tn, t[k]), 0, len(tn) - 1))
    _series(screen, pg, font, rect, tn,
            [(np.full(len(tn), 3.0), DIM, f"3 ({lo[j]:.2f}..{hi[j]:.2f})"),
             (mean, ACCENT, f"mean {mean[j]:.3g}"),
             (np.full(len(tn), np.nan), CURVE, f"wild {int(wild[j])}/{int(cnt[j])}")],
            "NEES(b), map", t[k], band=(lo, hi, ESTIMATE), log=True, clip_q=100)


def draw_view(app, screen, pg, font):
    if app.view is None:
        return
    rect = filter_view_rect()
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=6)
    title = {"A": "VIEW A  --  phase plane (psi, dpsi), 1-sigma ellipse of P",
             "B": "VIEW B  --  y and h(x^-,u) minus clean h(x,u);  band +-sqrt(S)",
             "C": "VIEW C  --  biases: estimate error |b^ - b| vs sqrt(P_bb)"}
    screen.blit(font.render(title[app.view], True, TEXT), (rect[0] + 12, rect[1] + 10))
    if app.diag is None:
        screen.blit(font.render(app.diag_note or "select a cell", True, ACCENT),
                    (rect[0] + 12, rect[1] + 40))
        return
    {"A": draw_view_a, "B": draw_view_b, "C": draw_view_c}[app.view](app, screen, pg, font)


# --- кнопки мира, модели и фильтра (ER-025) ----------------------------------
#  Правый конец ряда слайдеров датчика: там свободно, а ряд и так про датчик.


def world_buttons(app, font):
    """[(key, label, rect)] справа налево: kalman, model, world."""
    out, right = [], W - 24
    for key, label in (("kalman", "kalman"), ("model", "model"),
                       ("world", "world: " + ("wind" if app.phys["wind"] else "clean"))):
        w = font.size(label)[0] + 18
        right -= w
        out.append((key, label, (right, NOISE_Y - 3, w, 22)))
        right -= 10
    return out


def draw_world_buttons(app, screen, pg, font):
    on = {"kalman": app.kf_open, "model": app.model_open, "world": app.phys["wind"]}
    for key, label, rect in world_buttons(app, font):
        pg.draw.rect(screen, BTN_ON if on[key] else BTN, rect, border_radius=4)
        pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=4)
        screen.blit(font.render(label, True, BG if on[key] else TEXT),
                    (rect[0] + 9, rect[1] + 3))


def world_buttons_click(app, pos, font) -> bool:
    for key, _, (rx, ry, rw, rh) in world_buttons(app, font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            {"kalman": app.open_kf, "model": app.open_model,
             "world": app.toggle_world}[key]()
            return True
    return False


def plot_toggle_rect(font):
    """Переключатель нижнего графика u | w -- в строке заголовка TRAJECTORY."""
    label = "[P] u | w"
    w = font.size(label)[0] + 14
    return (PLOT[0] + PLOT[2] - w, PLOT[1] - 22, w, 20)


# --- общие части модальных панелей --------------------------------------------


def _draw_field(screen, pg, font, rect, text, focused, enabled=True):
    pg.draw.rect(screen, BG, rect, border_radius=3)
    pg.draw.rect(screen, ACCENT if focused else GRID_LINE, rect, 1, border_radius=3)
    screen.blit(font.render(text + ("_" if focused else ""), True,
                            TEXT if enabled else GRID_LINE), (rect[0] + 6, rect[1] + 4))


def _draw_button(screen, pg, font, rect, label, primary=False):
    pg.draw.rect(screen, BTN_ON if primary else BTN, rect, border_radius=4)
    screen.blit(font.render(label, True, BG if primary else TEXT), (rect[0] + 12, rect[1] + 5))


def _draw_veil(screen, pg, rect, title, subtitle, font):
    veil = pg.Surface((W, H), pg.SRCALPHA)
    veil.fill(BG + (150,))
    screen.blit(veil, (0, 0))
    x, y, _, _ = rect
    pg.draw.rect(screen, PANEL, rect, border_radius=6)
    pg.draw.rect(screen, GRID_LINE, rect, 1, border_radius=6)
    screen.blit(font.render(title, True, TEXT), (x + 24, y + 16))
    screen.blit(font.render(subtitle, True, DIM), (x + 24, y + 38))


def _bottom_buttons(font, rect):
    x, y, w, h = rect
    out, bx = [], x + 24
    for key in ("apply", "defaults", "close"):
        bw = font.size(key)[0] + 24
        out.append(("button", key, (bx, y + h - 44, bw, 26)))
        bx += bw + 10
    return out


def _type_into(text_map, focus, event, pg, chars=SETTINGS_CHARS):
    text = text_map.get(focus, "")
    if event.key == pg.K_BACKSPACE:
        text_map[focus] = text[:-1]
    elif event.unicode and event.unicode in chars:
        text_map[focus] = text + event.unicode


# --- панель model --------------------------------------------------------------


def model_layout(font):
    """[(kind, key, rect)]: поля физики мира и номинала, ветра и кнопки."""
    x, y, w, h = MODEL_PANEL
    out = []
    for i, k in enumerate(PHYS_KEYS):
        out.append(("field", f"world.{k}", (x + 170, y + 104 + i * 30, 140, 24)))
        out.append(("field", f"nominal.{k}", (x + 335, y + 104 + i * 30, 140, 24)))
    out.append(("toggle", "same", (x + 490, y + 104, 180, 24)))
    out.append(("toggle", "wind", (x + 170, y + 250, 140, 24)))
    for i, k in enumerate(DEFAULT_WIND):
        out.append(("field", f"wind.{k}", (x + 170, y + 284 + i * 30, 140, 24)))
    return out + _bottom_buttons(font, MODEL_PANEL)


def draw_model_panel(app, screen, pg, font):
    if not app.model_open:
        return
    x, y, w, h = MODEL_PANEL
    _draw_veil(screen, pg, MODEL_PANEL, "MODEL  --  world (what happens) and nominal "
               "(what controller and filter know)",
               "nominal = world: one machine;  otherwise the controller and the filter "
               "work with a wrong model", font)
    screen.blit(font.render("world", True, TEXT), (x + 170, y + 80))
    screen.blit(font.render("nominal", True, TEXT), (x + 335, y + 80))
    units = {"mb": "m_b, kg", "mw": "m_w, kg", "l": "l, m", "r": "r, m"}
    for i, k in enumerate(PHYS_KEYS):
        screen.blit(font.render(units[k], True, DIM), (x + 24, y + 108 + i * 30))
    screen.blit(font.render("WIND", True, TEXT), (x + 24, y + 254))
    wl = {"sigma_psi": "sigma_psi, N*m", "sigma_theta": "sigma_theta, N*m",
          "tau_w": "tau_w, s", "seed": "seed"}
    for i, k in enumerate(DEFAULT_WIND):
        screen.blit(font.render(wl[k], True, DIM), (x + 24, y + 288 + i * 30))
    same = app.model_text.get("same") == "yes"
    wind = app.model_text.get("wind") == "on"
    for kind, key, rect in model_layout(font):
        if kind == "field":
            enabled = not (same and key.startswith("nominal.")) and \
                (wind or not key.startswith("wind."))
            text = app.model_text.get(key, "")
            if same and key.startswith("nominal."):
                text = app.model_text.get("world." + key.split(".")[1], "")
            _draw_field(screen, pg, font, rect, text, key == app.model_focus, enabled)
        elif kind == "toggle":
            label = {"same": f"nominal = world: {'yes' if same else 'no'}",
                     "wind": f"wind: {'on' if wind else 'off'}"}[key]
            _draw_button(screen, pg, font, rect, label, primary=(same if key == "same" else wind))
        else:
            _draw_button(screen, pg, font, rect, key, primary=key == "apply")
    growth = app.mismatch_growth()
    lines = []
    if growth is not None:
        lines.append(f"now: nominal != world,  max Re eig(A_w - B_w K_n) = {growth:+.3f}"
                     + ("  (UNSTABLE)" if growth >= 0 else ""))
        lines.append("certificate c* and relay curve are the NOMINAL model's")
    if app.phys["wind"]:
        lines.append("recoverable limit on the map is the no-wind one")
    for i, line in enumerate(lines):
        screen.blit(font.render(line, True, DIM), (x + 24, y + 410 + i * 20))
    if app.model_error:
        screen.blit(font.render(app.model_error, True, (235, 110, 90)), (x + 24, y + h - 78))
    hint = "Enter apply   Tab next   Esc close"
    screen.blit(font.render(hint, True, DIM), (x + w - 24 - font.size(hint)[0], y + h - 39))


def model_click(app, pos, font):
    for kind, key, (rx, ry, rw, rh) in model_layout(font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            if kind == "field":
                app.model_focus = key
            elif key == "same":
                app.model_text["same"] = "no" if app.model_text.get("same") == "yes" else "yes"
            elif key == "wind":
                app.model_text["wind"] = "off" if app.model_text.get("wind") == "on" else "on"
            elif key == "apply":
                model_submit(app)
            elif key == "defaults":
                for k in PHYS_KEYS:
                    app.model_text[f"world.{k}"] = f"{DEFAULT_PHYS[k]:g}"
                    app.model_text[f"nominal.{k}"] = f"{DEFAULT_PHYS[k]:g}"
                for k, v in DEFAULT_WIND.items():
                    app.model_text[f"wind.{k}"] = f"{v:g}"
                app.model_text["same"] = "yes"
            elif key == "close":
                app.model_open = False
            return


def model_submit(app):
    error = app.apply_model(dict(app.model_text))
    app.model_error = error or ""
    if error is None:
        app.model_open = False


def model_key(app, event, pg):
    if event.key == pg.K_ESCAPE:
        app.model_open = False
    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
        model_submit(app)
    elif event.key in (pg.K_TAB, pg.K_DOWN, pg.K_UP):
        step = -1 if (event.key == pg.K_UP or event.mod & pg.KMOD_SHIFT) else 1
        i = MODEL_KEYS.index(app.model_focus) if app.model_focus else -1
        app.model_focus = MODEL_KEYS[(i + step) % len(MODEL_KEYS)]
    elif app.model_focus is not None:
        _type_into(app.model_text, app.model_focus, event, pg)
    return True


# --- панель kalman --------------------------------------------------------------


def kf_layout(font):
    x, y, w, h = KF_PANEL
    out = []
    for i, key in enumerate(KF_CHOICES):
        out.append(("choice", key, (x + 150, y + 84 + i * 30, 150, 24)))
    for i, key in enumerate(KF_FIELDS):
        out.append(("field", key, (x + 520, y + 84 + i * 30, 130, 24)))
    return out + _bottom_buttons(font, KF_PANEL)


def draw_kf_panel(app, screen, pg, font):
    if not app.kf_open:
        return
    x, y, w, h = KF_PANEL
    _draw_veil(screen, pg, KF_PANEL, "KALMAN  --  chip, bias prior, disturbance model",
               "chip is the world's: it changes the IMU of EVERY estimator", font)
    cl = {"chip": "chip", "prior": "bias prior", "drift": "bias drift",
          "w_model": "w in filter"}
    for i, key in enumerate(KF_CHOICES):
        screen.blit(font.render(cl[key], True, DIM), (x + 24, y + 88 + i * 30))
    for i, key in enumerate(KF_FIELDS):
        screen.blit(font.render(KF_LABELS[key], True, DIM), (x + 330, y + 88 + i * 30))
    for kind, key, rect in kf_layout(font):
        if kind == "choice":
            _draw_button(screen, pg, font, rect, app.kf_text.get(key, ""),
                         primary=app.kf_text.get(key) not in (DEFAULT_KF[key],))
        elif kind == "field":
            enabled = not key.startswith(("sigma_psi", "sigma_theta", "tau_w")) or \
                app.kf_text.get("w_model") == "own"
            if key in ("T_c", "sigma_jig"):
                enabled = app.kf_text.get("prior") == "calibrate"
            _draw_field(screen, pg, font, rect, app.kf_text.get(key, ""),
                        key == app.kf_focus, enabled)
        else:
            _draw_button(screen, pg, font, rect, key, primary=key == "apply")
    notes = ["raw = MPU-6050 turn-on bias, tolerance/sqrt(3): gyro 0.20 rad/s, accel 0.28 m/s^2",
             "calibrate = stand for T_c before the run (docs/accel_kalman.md §4.3)",
             "matched = the world's wind parameters;  own = the fields on the right"]
    if app.est_key != "kalman" and app.est_cmp_key != "kalman":
        notes.append("kalman+enc is not selected: only the chip changes the map now")
    for i, line in enumerate(notes):
        screen.blit(font.render(line, True, DIM), (x + 24, y + 310 + i * 20))
    if app.kf_error:
        screen.blit(font.render(app.kf_error, True, (235, 110, 90)), (x + 24, y + h - 78))
    hint = "Enter apply   Tab next   Esc close"
    screen.blit(font.render(hint, True, DIM), (x + w - 24 - font.size(hint)[0], y + h - 39))


def kf_click(app, pos, font):
    for kind, key, (rx, ry, rw, rh) in kf_layout(font):
        if rx <= pos[0] < rx + rw and ry <= pos[1] < ry + rh:
            if kind == "field":
                app.kf_focus = key
            elif kind == "choice":
                opts = KF_CHOICES[key]
                cur = app.kf_text.get(key, opts[0])
                app.kf_text[key] = opts[(opts.index(cur) + 1) % len(opts)]
            elif key == "apply":
                kf_submit(app)
            elif key == "defaults":
                app.kf_text = {k: (v if isinstance(v, str) else f"{v:g}")
                               for k, v in DEFAULT_KF.items()}
            elif key == "close":
                app.kf_open = False
            return


def kf_submit(app):
    error = app.apply_kf(dict(app.kf_text))
    app.kf_error = error or ""
    if error is None:
        app.kf_open = False


def kf_key(app, event, pg):
    if event.key == pg.K_ESCAPE:
        app.kf_open = False
    elif event.key in (pg.K_RETURN, pg.K_KP_ENTER):
        kf_submit(app)
    elif event.key in (pg.K_TAB, pg.K_DOWN, pg.K_UP):
        step = -1 if (event.key == pg.K_UP or event.mod & pg.KMOD_SHIFT) else 1
        i = KF_FIELDS.index(app.kf_focus) if app.kf_focus in KF_FIELDS else -1
        app.kf_focus = KF_FIELDS[(i + step) % len(KF_FIELDS)]
    elif app.kf_focus is not None:
        _type_into(app.kf_text, app.kf_focus, event, pg)
    return True


def draw_picker(app, screen, pg, font):
    # Кнопки рисуются шрифтом кнопок: он мельче ровно настолько, насколько
    # нужно, чтобы ряд влез. Кэш -- на окне, а не в модуле (picker_font_size).
    font = app.pick_font(pg)
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
    draw_world_buttons(app, screen, pg, font)


def _legend_row(screen, pg, font, x, y, items):
    """Строка легенды: [(вид, цвет, подпись), ...] -> следующий свободный x.

    Вид -- "box" (цвет клетки карты), "line" (кривая), "dash" (пунктир) или
    "ring" (рамка маркера): расшифровка рисуется тем же значком и тем же
    цветом, каким нарисована сама вещь. Легенда, разошедшаяся с рисунком,
    хуже, чем её отсутствие, поэтому других цветов тут не заводят.
    """
    for kind, color, text in items:
        if x + 20 + font.size(text)[0] > W - 24:
            break                       # обрезанная подпись врёт
        if kind == "box":
            pg.draw.rect(screen, color, (x, y + 2, 14, 12))
        elif kind == "line":
            pg.draw.line(screen, color, (x, y + 8), (x + 14, y + 8), 3)
        elif kind == "dash":
            for dx in (0, 6, 12):
                pg.draw.line(screen, color, (x + dx, y + 8), (x + dx + 3, y + 8), 2)
        else:
            pg.draw.rect(screen, color, (x, y + 2, 14, 12), 2)
        screen.blit(font.render(text, True, color), (x + 20, y))
        x += 20 + font.size(text)[0] + 22
    return x


def draw_header(app, screen, pg, font, big):
    # Заголовок короткий: справа в той же строке стоит строка режима, и на
    # предупреждении о разошедшемся K ("(!) K ушло от константы") они
    # наезжали друг на друга -- ровно в том случае, ради которого
    # предупреждение и написано.
    screen.blit(big.render("wpend explorer", True, TEXT), (24, 12))
    mode = "grid" if app.batch is not None else "on-demand"
    right = (f"horizon={app.args.horizon:g}s  "
             f"dt={app.args.dt:g}  stride={app.args.stride}  {mode}  "
             f"[{app.compute_seconds:.2f}s]  {app.design_note}")
    screen.blit(font.render(right, True, ACCENT if app.design_hint else DIM),
                (W - 24 - font.size(right)[0], 16))
    draw_picker(app, screen, pg, font)

    # Легенда карты -- двумя строками под ней. Верхняя принадлежит ЗАДАЧЕ и не
    # меняется от нажатых кнопок: цвета исходов, граница восстановимости,
    # горизонт корпуса. Нижняя принадлежит ВЫБОРУ и меняется целиком: какая
    # кривая чей регулятор и чей оцениватель. Вопрос куратора «при сравнении
    # двух регуляторов непонятно, что к чему» -- ровно про нижнюю (ER-015).
    row = [("box", OUTCOME_COLOR[HELD], "held"),
           ("box", OUTCOME_COLOR[FELL_FORWARD], "fell forward"),
           ("box", OUTCOME_COLOR[FELL_BACKWARD], "fell backward")]
    if app.psi_eq is None:
        # Свотча нет сознательно: на карте нет цвета, который он объяснял бы.
        row.append(("line", DIM, "no torque limit -- everything recoverable"))
    else:
        row.append(("line", LIMIT,
                    "recoverable limit (psi_eq = %.3f)" % app.psi_eq
                    + (", no wind" if app.phys["wind"] else "")))
    if app.show_switch and app.switch_up is not None:
        row.append(("line", SWITCH, "relay switch sigma = 0"))
    if app.spec.psi_max >= HORIZON_ANGLE:
        row.append(("dash", DIM, "psi = pi/2"))
    _legend_row(screen, pg, font, MAP[0] - 32, LEGEND_Y, row)

    row = [("line", CURVE, "main: " + CONTROLLER_LABEL[app.main_key])]
    if app.phys["wind"]:
        row.append(("line", WIND, "wind %.2g/%.2g N*m, %.2g s" % (
            app.phys["sigma_psi"], app.phys["sigma_theta"], app.phys["tau_w"])))
    if not app.phys["same"]:
        row.append(("line", (235, 110, 90), "nominal != world"))
    if app.cmp_key is not None:
        row.append(("line", GHOST, "compare: " + CONTROLLER_LABEL[app.cmp_key]))
    if app.est_key != "ideal":
        row.append(("line", ESTIMATE, "estimate: " + ESTIMATOR_LABEL[app.est_key]))
    if app.est_cmp_key is not None:
        row.append(("line", GHOST_EST,
                    "vs est: " + ESTIMATOR_LABEL[app.est_cmp_key]))
    row.append(("ring", ACCENT, "selected cell"))
    # Ключ к дорожке фаз (ER-016) -- в ту же строку «выбора»: фаза
    # принадлежит выбранному регулятору, а не задаче, и в строке графика u
    # подписи уже не помещались -- легенда обрывалась на «+1», то есть врала
    # ровно про то новое, ради чего дорожка и заведена.
    if app.track is not None:
        for code in sorted({int(c) for c in app.track["phases"]}):
            row.append(("box", PHASE_COLOR[code], "phase " + PHASE_LABEL[code]))
    _legend_row(screen, pg, font, MAP[0] - 32, LEGEND_Y + LEGEND_ROW, row)

    # Цифр всего девять, а регуляторов может быть больше: подсказка считается
    # по списку, а не написана числом -- иначе очередная запись в CONTROLLERS
    # сделала бы её враньём. Кнопки без цифры доступны мышью.
    n_digits = min(9, len(CONTROLLER_KEYS))
    hint = ("click a cell  |  1-%d main, Shift+1-%d compare  |  E estimator  |  "
            "[ ] u_max, L no limit  |  W M K P V H B  |  time: <- ->  |  "
            "SPACE pause  R restart  ESC quit" % (n_digits, n_digits))
    screen.blit(font.render(hint, True, GRID_LINE),
                (W - 24 - font.size(hint)[0], H - 24))


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
            elif app.noise_open and event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    noise_click(app, canvas_pos(event.pos, view), font)
            elif app.noise_open and event.type == pg.KEYDOWN:
                noise_key(app, event, pg)
            elif app.noise_open and event.type in (pg.MOUSEMOTION, pg.MOUSEBUTTONUP):
                continue
            elif app.model_open and event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    model_click(app, canvas_pos(event.pos, view), font)
            elif app.model_open and event.type == pg.KEYDOWN:
                model_key(app, event, pg)
            elif app.model_open and event.type in (pg.MOUSEMOTION, pg.MOUSEBUTTONUP):
                continue
            elif app.kf_open and event.type == pg.MOUSEBUTTONDOWN:
                if event.button == 1:
                    kf_click(app, canvas_pos(event.pos, view), font)
            elif app.kf_open and event.type == pg.KEYDOWN:
                kf_key(app, event, pg)
            elif app.kf_open and event.type in (pg.MOUSEMOTION, pg.MOUSEBUTTONUP):
                continue
            elif event.type == pg.MOUSEMOTION:
                mx, my = canvas_pos(event.pos, view)
                if app.time_drag:
                    # Пересчёта нет ни на пиксель: кадр уже посчитан.
                    app.scrub(time_from_px(app, mx))
                    continue
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
                nx, ny, nw, nh = noise_button_rect(font)
                if nx <= pos[0] < nx + nw and ny <= pos[1] < ny + nh:
                    app.open_noise()
                    continue
                wx, wy, ww, wh = switch_button_rect(font)
                if wx <= pos[0] < wx + ww and wy <= pos[1] < wy + wh:
                    app.show_switch = not app.show_switch
                    continue
                if world_buttons_click(app, pos, font):
                    continue
                if view_buttons_click(app, pos, font):
                    continue
                px_, py_, pw_, ph_ = plot_toggle_rect(font)
                if px_ <= pos[0] < px_ + pw_ and py_ <= pos[1] < py_ + ph_:
                    app.toggle_plot_w()
                    continue
                if slider_hit(pos):
                    app.dragging = True
                    app.preview_u_max(u_max_from_px(pos[0]))
                    continue
                if app.traj is not None and time_hit(pos):
                    app.time_drag = True
                    app.scrub(time_from_px(app, pos[0]))
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
                # Тем же шрифтом, каким кнопки нарисованы (picker_font):
                # иначе мышь ловится не там, где кнопка видна.
                for rect, row, key in picker_layout(app, app.pick_font(pg)):
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
                if app.time_drag:
                    # У дорожки времени пересчитывать нечего, и отпускание НЕ
                    # запускает воспроизведение: кадр держится до ПРОБЕЛА
                    # (решение Глеба 21.09).
                    app.time_drag = False
                elif app.dragging:
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
                elif event.key == pg.K_n:
                    app.open_noise()
                elif event.key == pg.K_c:
                    app.show_switch = not app.show_switch
                elif event.key == pg.K_w:
                    app.toggle_world()
                elif event.key == pg.K_m:
                    app.open_model()
                elif event.key == pg.K_k:
                    app.open_kf()
                elif event.key == pg.K_p:
                    app.toggle_plot_w()
                elif event.key == pg.K_v:
                    app.toggle_view("A")
                elif event.key == pg.K_h:
                    app.toggle_view("B")
                elif event.key == pg.K_b:
                    app.toggle_view("C")
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
                        app._update_tracks()
                    else:
                        i = ESTIMATOR_KEYS.index(app.est_key)
                        app.set_estimator(ESTIMATOR_KEYS[(i + 1) % len(ESTIMATOR_KEYS)])
                elif event.key in (pg.K_LEFT, pg.K_RIGHT):
                    # Шаг кадра, а не времени: стрелка обязана всегда сдвигать
                    # картинку ровно на один посчитанный кадр.
                    step = 10 if shift else 1
                    app.step_frame(step if event.key == pg.K_RIGHT else -step)
                elif event.key in (pg.K_LEFTBRACKET, pg.K_RIGHTBRACKET):
                    # Шаг при выключенном пределе возвращает его: иначе
                    # клавиша молча ничего не делает.
                    step = U_STEP if event.key == pg.K_RIGHTBRACKET else -U_STEP
                    app.preview_u_max(app.u_finite if app.u_max is None
                                      else app.u_max + step)
                    app.commit_u_max()
                elif shift and digit == 0:
                    app.set_compare(None)
                elif 1 <= digit <= min(9, len(CONTROLLER_KEYS)):
                    key = CONTROLLER_KEYS[digit - 1]
                    (app.set_compare if shift else app.set_main)(key)

        app.advance(dt_wall)

        screen.fill(BG)
        draw_header(app, screen, pg, font, big)
        draw_map(app, screen, pg, font)
        draw_robot(app, screen, pg, font)
        draw_plots(app, screen, pg, font)
        draw_view(app, screen, pg, font)
        draw_view_buttons(app, screen, pg, font)
        draw_settings(app, screen, pg, font)
        draw_noise_panel(app, screen, pg, font)
        draw_model_panel(app, screen, pg, font)
        draw_kf_panel(app, screen, pg, font)
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
    # Умолчания запуска -- решение Глеба 01.10 (kalman_views §17.5): реле с
    # накачкой энергии и фильтр Калмана.  Без scipy окно само откатит реле на lqr.
    p.add_argument("--main", choices=CONTROLLER_KEYS, default="bang-energy-ell",
                   help="основной регулятор: он красит карту")
    p.add_argument("--compare", choices=CONTROLLER_KEYS, default=None,
                   help="регулятор сравнения: рисуется полупрозрачным поверх")
    p.add_argument("--estimator", choices=ESTIMATOR_KEYS, default="kalman",
                   help="что видит регулятор: ideal -- истину; comp, gyro, acc -- "
                        "показания ИДУ и энкодера через комплементарный фильтр; "
                        "kalman -- через фильтр Калмана (по умолчанию)")
    p.add_argument("--estimator-compare", choices=ESTIMATOR_KEYS, default=None,
                   help="второй оцениватель: считается для выбранной клетки и "
                        "накладывается полупрозрачным")
    # Плотности -- по паспорту MPU-6050 (решение Глеба 01.10, kalman_views
    # §17.4 R1): 0.005 °/с/sqrt(Гц) и 400 мкg/sqrt(Гц).  Прежние 1e-4 и 1e-3
    # были мягче паспорта по акселерометру в 3.9 раза.
    p.add_argument("--sigma-g", type=float, default=8.7e-5,
                   help="плотность шума гироскопа, рад/с/sqrt(Гц) (MPU-6050)")
    p.add_argument("--sigma-a", type=float, default=3.9e-3,
                   help="плотность шума акселерометра, м/с^2/sqrt(Гц) (MPU-6050)")
    p.add_argument("--b0-g", type=float, default=1e-2,
                   help="СКО смещения включения гироскопа, рад/с")
    p.add_argument("--tau", type=float, default=0.30,
                   help="постоянная времени комплементарного фильтра, с")
    p.add_argument("--counts", type=int, default=2048,
                   help="меток на оборот у моторного энкодера (он во всех вариантах, кроме ideal)")
    p.add_argument("--seed", type=int, default=0,
                   help="зерно шума датчика (при --estimator, отличном от ideal)")
    # --- мир, номинал, фильтр Калмана (ER-025, docs/explorer_kalman.md §7) ---
    p.add_argument("--world", choices=["clean", "wind"], default="clean",
                   help="мир: чистый или с ветром (DisturbedWheeledPendulum); клавиша W")
    p.add_argument("--sigma-w", type=float, nargs=2, default=None,
                   metavar=("PSI", "THETA"), help="СКО ветра по каналам, Н*м (0.3 0.3)")
    p.add_argument("--tau-w", type=float, default=None, help="время корреляции ветра, с (0.2)")
    p.add_argument("--wind-seed", type=int, default=None, help="зерно ветра (7)")
    p.add_argument("--world-params", default=None,
                   help="физика мира, например mb=10,mw=1,l=1,r=0.3")
    p.add_argument("--nominal-params", default=None,
                   help="физика номинала (что знают регулятор и фильтр); по умолчанию = мир")
    p.add_argument("--chip", choices=list(KF_CHOICES["chip"]), default=None,
                   help="чип ИДУ: calibrated (слайдер b0_g) или raw (MPU-6050)")
    p.add_argument("--bias-prior", choices=list(KF_CHOICES["prior"]), default=None,
                   help="априор смещений фильтра: code или calibrate (стоянка)")
    p.add_argument("--calib-time", type=float, default=None, help="длительность стоянки, с")
    p.add_argument("--sigma-jig", type=float, default=None, help="неточность подставки, рад")
    p.add_argument("--kf-w", choices=list(KF_CHOICES["w_model"]), default=None,
                   help="возмущение в фильтре: off, matched (как у мира), own")
    p.add_argument("--kf-q-acc", type=float, default=None, help="q_acc фильтра")
    p.add_argument("--kf-iter", type=int, default=None, help="итерации (1 -- EKF)")
    p.add_argument("--kf-drift", action="store_true", help="уход смещения в модели фильтра")
    p.add_argument("--eps", type=float, default=0.05,
                   help="bang-eps: реле отдаёт управление ЛКР, когда |psi| < eps, рад")
    p.add_argument("--energy", choices=["tilt", "total"], default="tilt",
                   help="bang-energy: какую энергию качать. tilt -- энергию "
                        "приведённой динамики наклона (dE/dt = u b(psi) dpsi / gamma, "
                        "скорости колеса нет); total -- полную энергию машины "
                        "(dE/dt = u (dpsi - dtheta)). Замер 22.09: tilt 9/55/153, "
                        "total 1/3/11 клеток при 1.5/3/10 Н*м")
    p.add_argument("--eps-e", type=float, default=0.5,
                   help="energy-eps: накачка отдаёт управление, когда |E - E*| < eps_e "
                        "И |psi| < eps. Масштаб: D(1 - cos d) -- энергия, которой не "
                        "хватает корпусу, отклонённому на d; при D = 98 и d = 0.1 рад "
                        "это 0.49")
    p.add_argument("--u-max", type=float, default=3.0,
                   help="предел момента, Н*м -- начальное положение слайдера "
                        f"(диапазон {U_MIN:g}..{U_MAX:g}); inf -- запустить "
                        "без ограничения")
    p.add_argument("--horizon", type=float, default=5.0, help="горизонт прогона, с")
    p.add_argument("--dt", type=float, default=1e-3, help="шаг интегрирования, с")
    p.add_argument("--stride", type=int, default=10,
                   help="сохранять каждый stride-й кадр (счёт идёт с шагом dt)")
    p.add_argument("--psi-max", type=float, default=None,
                   help="границы карты по psi0; по умолчанию подгоняются под "
                        "u_max, явное значение их замораживает")
    p.add_argument("--dpsi-max", type=float, default=None,
                   help="границы карты по dpsi0; по умолчанию подгоняются под "
                        "u_max, явное значение их замораживает")
    p.add_argument("--psi-fall", type=float, default=float(np.pi),
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
