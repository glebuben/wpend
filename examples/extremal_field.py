"""Поле экстремалей: траектории обоих максимальных управлений (ER-023).

Из сетки начальных условий пускаются четыре ветви: u = +u_max и u = -u_max,
каждая вперёд и назад по времени.  Это не иллюстрация, а независимая проверка
трёх вещей, которые в проекте держатся на замкнутых формулах:

  * КРИВАЯ ПЕРЕКЛЮЧЕНИЯ реле (`BangBangLQRController.switching_function`).
    Траектория, приходящая в верхнее положение при постоянном максимальном
    моменте, -- это ровно обратное интегрирование из нуля.  Значит численная
    ветвь обязана лечь на аналитическую sigma = 0.  Если ляжет, формула реле
    проверена не прогоном реле, а интегратором;
  * ГРАНИЦА ВОССТАНОВИМОСТИ (`recoverable_bounds`) -- устойчивое многообразие
    седла при u = -+u_max.  Обратное интегрирование из окрестности седла
    вдоль устойчивого собственного вектора рисует её же;
  * ГДЕ НЕ СПАСАЕТ НИЧТО -- видно глазом: область, из которой ни одна ветвь
    вперёд не приходит в ноль (прямой ответ на вопрос ER-017).

Что на самом деле рисуется
--------------------------
При ПОСТОЯННОМ u сохраняется первый интеграл (Key_Formulas §3)

    H(psi, dpsi; u) = 1/2 Delta(psi) dpsi^2 - u (gamma psi + beta sin psi)
                      + gamma D cos(psi),

поэтому вперёд и назад от одной точки при одном и том же u -- это ОДНА
фазовая кривая, линия уровня H, просто пройденная в обе стороны.  Значит поле
экстремалей -- это два семейства линий уровня: H(.; +u_max) и H(.; -u_max).
Кривая переключения -- член семейства, проходящий через ноль; граница
восстановимости -- член семейства, проходящий через седло.  Обе «особые»
кривые на картинке не дорисованы сверху, а являются частью того же пучка, и
именно это делает наложение аналитики осмысленной проверкой.

Почему плоскость наклона самодостаточна: ни theta, ни dtheta не входят в
правые части (Key_Formulas §1.4), подсистема (psi, dpsi) замкнута сама на
себя.  Колесо -- ведомая ею цепочка интеграторов, поэтому оно на ВТОРОЙ
панели (решение Глеба 21.09), а не подмешано в первую.

Начальные условия (решение Глеба 21.09): dpsi0 = 0, перебирается только угол.
Каждая ветвь стартует с оси dpsi = 0, и пучок читается как «что будет, если
отпустить корпус из такого наклона и выжать мотор в ту или другую сторону».

Две численные тонкости, ради которых написан `branch`
-----------------------------------------------------
  * Назад по времени система та же самая, меняется только знак шага (правая
    часть от времени не зависит).  НО устойчивое и неустойчивое направления
    седла меняются ролями: обратная траектория, выпущенная не точно из седла,
    экспоненциально убегает.  Поэтому ветви ограничиваются не временем, а
    длиной дуги и выходом за рамку картинки.
  * Длина дуги считается в ТОЙ ЖЕ плоскости, в которой рисуем: иначе ветвь с
    большой |dpsi| съедает весь бюджет за пару шагов, а медленная ветвь у
    оси тянется вечно.

Запуск:

    uv run python examples/extremal_field.py
    uv run python examples/extremal_field.py --n-psi 41 --dt 5e-4
    uv run python examples/extremal_field.py --out figures/fig_extremal_field.png

Картинка рисуется, только если доступен matplotlib (в зависимостях проекта
его нет -- PROPOSALS.md §B11).  Без него скрипт печатает те же числа; оракулы
живут в tests/test_extremal_field.py и от картинки не зависят.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import BangBangLQRController, RK4Integrator
from wpend.models import WheeledPendulum

#: Пределы момента, по которым идут колонки картинки (карточка ER-023, шаг 4).
U_MAXES = (1.5, 3.0, 10.0)

#: Стандарт карт (CLAUDE.md, решение Глеба 17.09): интересен psi в [-pi/2, pi/2],
#: сетка и рамка берутся с запасом 1.3 по обеим осям.
PSI_MAP = np.pi / 2
MAP_FIT = 1.3

DT = 1e-3


# --- рамка и сетка -------------------------------------------------------


def frame_limits(wp: WheeledPendulum, u_max: float):
    """(psi_lim, dpsi_lim) -- рамка картинки по стандарту карт.

    dpsi_lim = 1.3 от максимума множества восстановимости на [-pi/2, pi/2]:
    так рамка сама подстраивается под предел момента, и три колонки картинки
    сравнимы не по абсолютной шкале, а по «сколько это от восстановимого».
    Та же формула, что в `examples/bang_eps_map.py`.
    """
    psi = np.linspace(-PSI_MAP, PSI_MAP, 401)
    floor, ceiling = wp.recoverable_bounds(u_max, psi)
    reach = np.abs(np.concatenate([np.atleast_1d(floor), np.atleast_1d(ceiling)]))
    dpsi_lim = MAP_FIT * float(reach[np.isfinite(reach)].max())
    return MAP_FIT * PSI_MAP, dpsi_lim


# --- одна ветвь ----------------------------------------------------------


def branch(wp, x0, u, dt, *, psi_lim, dpsi_lim, max_arc, max_steps=200_000):
    """Одна ветвь поля: RK4 с постоянным u; знак dt задаёт направление времени.

    Возвращает (t, X), X формы (k, 4) в порядке хода времени (для обратной
    ветви t убывает).  Останов -- по длине дуги в плоскости (psi, dpsi), по
    выходу за рамку или по числу шагов; ПО ВРЕМЕНИ НЕ ОСТАНАВЛИВАЕМСЯ
    сознательно, см. докстринг модуля.

    Интегратор -- тот же `RK4Integrator`, что и в прогонах: отрицательный шаг
    для него законен (схема явная и от знака dt не зависит), и именно это
    делает обратную ветвь проверкой прямого прогона, а не отдельной машинерией.
    """
    integ = RK4Integrator()
    u_vec = np.array([float(u)])
    x = np.asarray(x0, dtype=float).copy()
    xs, ts = [x.copy()], [0.0]
    arc, t = 0.0, 0.0
    for _ in range(max_steps):
        x_new = integ.step(wp.f, t, x, u_vec, dt)
        if not np.all(np.isfinite(x_new)):
            break
        arc += float(np.hypot(x_new[0] - x[0], x_new[2] - x[2]))
        t += dt
        x = x_new
        xs.append(x.copy())
        ts.append(t)
        if arc >= max_arc or abs(x[0]) > psi_lim or abs(x[2]) > dpsi_lim:
            break
    return np.asarray(ts), np.asarray(xs)


def extremal_field(wp, u_max, *, n_psi=25, dt=DT, arc=None):
    """Пучок: из каждой точки сетки -- два знака момента, каждый в обе стороны.

    Начальные условия по решению Глеба 21.09: dpsi0 = 0, перебирается только
    psi0; theta0 = dtheta0 = 0.
    """
    psi_lim, dpsi_lim = frame_limits(wp, u_max)
    max_arc = 2.0 * (psi_lim + dpsi_lim) if arc is None else float(arc)
    out = []
    for psi0 in np.linspace(-psi_lim, psi_lim, n_psi):
        x0 = np.array([psi0, 0.0, 0.0, 0.0])
        for u in (+u_max, -u_max):
            lim = dict(psi_lim=psi_lim, dpsi_lim=dpsi_lim, max_arc=max_arc)
            t_f, X_f = branch(wp, x0, u, +dt, **lim)
            t_b, X_b = branch(wp, x0, u, -dt, **lim)
            out.append({"u": u, "psi0": float(psi0),
                        "t_fwd": t_f, "X_fwd": X_f,
                        "t_bwd": t_b, "X_bwd": X_b})
    return {"u_max": u_max, "psi_lim": psi_lim, "dpsi_lim": dpsi_lim,
            "max_arc": max_arc, "branches": out}


def whole_branch(b):
    """(t, X) одной фазовой кривой: обратная ветвь развёрнута и приклеена слева.

    Время идёт по возрастанию и проходит через ноль в исходной точке: t < 0 --
    «как сюда попали», t > 0 -- «что будет дальше».  Именно это и листает
    дорожка времени в окне (`examples/extremal_explorer.py`).
    """
    t = np.concatenate([b["t_bwd"][::-1][:-1], b["t_fwd"]])
    X = np.vstack([b["X_bwd"][::-1][:-1], b["X_fwd"]])
    return t, X


def whole_curve(b):
    """Обратная ветвь + прямая одной строкой: это одна линия уровня H."""
    return whole_branch(b)[1]


# --- аналитика, которую накладываем --------------------------------------


def relay(wp, u_max):
    """Релейный регулятор -- ради одной его функции `switching_function`.

    K и region здесь не работают (регион пуст, до ЛКР дело не доходит): нужна
    именно КОДОВАЯ формула реле, а не её копия в примере, иначе проверка
    выродится в сравнение формулы с самой собой.
    """
    empty = lambda x: np.zeros(np.shape(x)[:-1], dtype=bool)
    return BangBangLQRController(np.zeros((1, 4)), u_max, region=empty, system=wp)


def sigma_zero_dpsi(wp, u_max, psi):
    """dpsi > 0 на кривой sigma = 0, замкнутой формой; NaN там, где ветви нет.

    При dpsi > 0 тормозим моментом u = -u_max, и sigma = 0 означает
    H(psi, dpsi; -u_max) = H(0, 0) = gamma D, откуда

        dpsi^2 = 2 [ gamma D (1 - cos psi) - u_max (gamma psi + beta sin psi) ]
                 / Delta(psi).

    Правая часть положительна только при psi < 0 -- это и есть та половина
    кривой, по которой корпус, наклонённый назад и идущий вперёд, приходит в
    ноль.  Вторая половина получается симметрией (psi, dpsi) -> (-psi, -dpsi).
    """
    p = wp.p
    psi = np.asarray(psi, dtype=float)
    sq = 2.0 * (p.gamma * p.D * (1.0 - np.cos(psi))
                - u_max * (p.gamma * psi + p.beta * np.sin(psi))) / wp.Delta(psi)
    return np.where(sq >= 0.0, np.sqrt(np.clip(sq, 0.0, None)), np.nan)


def saddle_stable_dir(wp, u_max, h=1e-6):
    """(psi_eq, lam, v) в седле при u = -u_max: собственное число и устойчивый вектор.

    Линеаризация приведённой динамики наклона в точке (psi_eq, 0):
    член с dpsi^2 и его производная по dpsi в нуле обращаются в ноль, поэтому
    якобиан -- [[0, 1], [g'(psi_eq), 0]], где

        g(psi) = [ b(psi) u + gamma D sin psi ] / Delta(psi).

    Собственные числа +-sqrt(g'), устойчивый вектор (1, -sqrt(g')).  g' берём
    центральной разностью: это оракул против аналитики, а не ещё одна формула,
    которую потом самим же и проверять.
    """
    p = wp.p
    psi_eq = wp.saddle_angle(u_max)
    g = lambda psi: ((p.gamma + p.beta * np.cos(psi)) * (-u_max)
                     + p.gamma * p.D * np.sin(psi)) / wp.Delta(psi)
    gp = (g(psi_eq + h) - g(psi_eq - h)) / (2.0 * h)
    lam = float(np.sqrt(max(gp, 0.0)))
    return psi_eq, lam, np.array([1.0, -lam])


def stable_manifold(wp, u_max, *, dt=DT, eps=1e-6, **lim):
    """Устойчивое многообразие седла (u = -u_max), нарисованное интегратором.

    Назад по времени устойчивое направление становится неустойчивым, поэтому
    обратная ветвь, выпущенная из седла со сдвигом eps вдоль устойчивого
    вектора, как раз ВЫМЕТАЕТ многообразие.  Две стороны -- два знака eps.
    """
    psi_eq, _, v = saddle_stable_dir(wp, u_max)
    out = []
    for sign in (+1.0, -1.0):
        x0 = np.array([psi_eq + sign * eps * v[0], 0.0, sign * eps * v[1], 0.0])
        out.append(branch(wp, x0, -u_max, -dt, **lim)[1])
    return out


# --- числа ---------------------------------------------------------------


def conservation_error(wp, field):
    """(абсолютное, относительное) максимальное отклонение H вдоль ветвей."""
    worst_abs, worst_rel = 0.0, 0.0
    for b in field["branches"]:
        X = whole_curve(b)
        H = wp.first_integral(X, b["u"])
        H0 = wp.first_integral(np.array([b["psi0"], 0.0, 0.0, 0.0]), b["u"])
        err = float(np.max(np.abs(H - H0)))
        worst_abs = max(worst_abs, err)
        worst_rel = max(worst_rel, err / max(abs(float(H0)), 1e-12))
    return worst_abs, worst_rel


def reversibility_error(wp, u_max, *, n_psi=9, dt=DT, steps=2000):
    """Назад n шагов, потом вперёд n шагов -- насколько вернулись в x0.

    Это проверка не формулы, а схемы: RK4 не обратим точно, и величина
    невязки -- цена численного хода назад, который мы и используем.
    """
    integ = RK4Integrator()
    psi_lim, _ = frame_limits(wp, u_max)
    worst = 0.0
    for psi0 in np.linspace(-psi_lim, psi_lim, n_psi):
        for u in (+u_max, -u_max):
            u_vec = np.array([float(u)])
            x0 = np.array([psi0, 0.0, 0.0, 0.0])
            x, t = x0.copy(), 0.0
            for _ in range(steps):
                x = integ.step(wp.f, t, x, u_vec, -dt)
                t -= dt
            for _ in range(steps):
                x = integ.step(wp.f, t, x, u_vec, +dt)
                t += dt
            worst = max(worst, float(np.max(np.abs(x - x0))))
    return worst


def switching_curve_error(wp, u_max, *, dt=DT, psi_stop=None):
    """Численная кривая переключения против аналитической sigma = 0.

    Численная -- обратное интегрирование ИЗ НУЛЯ при u = -u_max: вперёд по
    времени эта траектория приходит в верхнее положение, то есть она и есть
    тормозная кривая реле.  Сравниваем по dpsi при том же psi и заодно
    смотрим, насколько сама sigma (кодовая, из регулятора) отличается от нуля.

    Возвращает (max |d dpsi|, max |sigma|, psi численной ветви, dpsi ветви).
    """
    psi_lim, dpsi_lim = frame_limits(wp, u_max)
    if psi_stop is not None:
        psi_lim = min(psi_lim, abs(psi_stop))
    _, X = branch(wp, np.zeros(4), -u_max, -dt,
                  psi_lim=psi_lim, dpsi_lim=dpsi_lim,
                  max_arc=4.0 * (psi_lim + dpsi_lim))
    psi, dpsi = X[:, 0], X[:, 2]
    ok = psi < 0.0                      # ветвь sigma = 0 при dpsi > 0 живёт слева
    ref = sigma_zero_dpsi(wp, u_max, psi[ok])
    d_dpsi = float(np.nanmax(np.abs(dpsi[ok] - ref)))
    sigma = relay(wp, u_max).switching_function(X[ok])
    return d_dpsi, float(np.max(np.abs(sigma))), psi[ok], dpsi[ok]


def manifold_error(wp, u_max, *, dt=DT):
    """Численное устойчивое многообразие седла против `recoverable_bounds`."""
    psi_lim, dpsi_lim = frame_limits(wp, u_max)
    lim = dict(psi_lim=psi_lim, dpsi_lim=dpsi_lim,
               max_arc=4.0 * (psi_lim + dpsi_lim))
    worst = 0.0
    for X in stable_manifold(wp, u_max, dt=dt, **lim):
        psi, dpsi = X[:, 0], X[:, 2]
        inside = np.abs(psi) <= PSI_MAP          # сравниваем на рабочем отрезке
        if not np.any(inside):
            continue
        _, ceiling = wp.recoverable_bounds(u_max, psi[inside])
        worst = max(worst, float(np.nanmax(np.abs(dpsi[inside] - ceiling))))
    return worst


# --- картинка ------------------------------------------------------------

#: Палитра та же, что в `examples/imu_allan.py`: различима при дейтеранопии,
#: и к цвету всегда добавлен второй признак (тип линии или подпись).
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c3"
C_PLUS, C_MINUS = "#2a78d6", "#eb6834"      # u = +u_max и u = -u_max


def _arrows(ax, X, color, n=2):
    """Стрелки по ходу времени.

    X идёт в порядке возрастания времени (обратная ветвь развёрнута), поэтому
    стрелка «по индексу» и есть стрелка «по времени».  Ставим по длине дуги, а
    не по индексу: иначе все стрелки скучиваются там, где точки гуще.
    """
    if len(X) < 8:
        return
    s = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(X[:, 0]), np.diff(X[:, 2])))])
    if s[-1] <= 0.0:
        return
    for frac in np.linspace(1.0, n, n) / (n + 1.0):
        k = int(np.searchsorted(s, frac * s[-1]))
        k = min(max(k, 1), len(X) - 2)
        ax.annotate("", xy=(X[k + 1, 0], X[k + 1, 2]), xytext=(X[k, 0], X[k, 2]),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9,
                                    shrinkA=0, shrinkB=0, alpha=0.9))


def _frame(ax, title, xlabel, ylabel):
    ax.set_title(title, color=INK, fontsize=10.5, pad=8)
    ax.set_xlabel(xlabel, color=INK, fontsize=9.5)
    ax.set_ylabel(ylabel, color=INK, fontsize=9.5)
    ax.grid(True, color=GRID, lw=0.6, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8.5)


def draw(wp, fields, errors, out, *, arrow_every=3):
    """Две строки панелей: наклон и колесо, по колонке на предел момента."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(fields)
    fig, axes = plt.subplots(2, n, figsize=(4.9 * n, 8.6))
    axes = np.atleast_2d(axes)

    for j, field in enumerate(fields):
        u_max = field["u_max"]
        ax_t, ax_w = axes[0, j], axes[1, j]

        # 1. Сам пучок. Цвет -- знак момента; прямая и обратная ветви одной
        #    точки -- одна кривая, поэтому рисуем их вместе.
        for i, b in enumerate(field["branches"]):
            color = C_PLUS if b["u"] > 0 else C_MINUS
            X = whole_curve(b)
            ax_t.plot(X[:, 0], X[:, 2], color=color, lw=0.8, alpha=0.5, zorder=2)
            ax_w.plot(X[:, 1], X[:, 3], color=color, lw=0.8, alpha=0.5, zorder=2)
            if i % (2 * arrow_every) < 2:
                _arrows(ax_t, X, color)

        # 2. Аналитика поверх пучка -- в плоскости наклона, где она и живёт.
        psi_lim, dpsi_lim = field["psi_lim"], field["dpsi_lim"]
        grid = np.linspace(-psi_lim, psi_lim, 801)

        sw = sigma_zero_dpsi(wp, u_max, grid)
        ax_t.plot(grid, sw, color=INK, lw=2.0, zorder=4)
        ax_t.plot(-grid, -sw, color=INK, lw=2.0, zorder=4)

        floor, ceiling = wp.recoverable_bounds(u_max, grid)
        # Заливка -- прямой ответ на ER-017: белое поле вокруг неё и есть
        # «откуда не спасает никакое управление», и видно это глазом.
        ax_t.fill_between(grid, floor, ceiling, where=ceiling > floor,
                          color=MUTED, alpha=0.07, lw=0, zorder=0)
        ax_t.plot(grid, ceiling, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=3)
        ax_t.plot(grid, floor, color=MUTED, lw=1.6, ls=(0, (5, 3)), zorder=3)

        psi_eq = wp.saddle_angle(u_max)
        ax_t.plot([psi_eq, -psi_eq], [0.0, 0.0], "o", ms=7, mfc="white",
                  mec=INK, mew=1.4, zorder=6)
        ax_t.annotate(f"седло ±{np.degrees(psi_eq):.1f}°", (psi_eq, 0.0),
                      color=INK, fontsize=8.5, xytext=(6, 8),
                      textcoords="offset points", zorder=6)
        for s in (-1.0, 1.0):
            ax_t.axvline(s * PSI_MAP, color=GRID, lw=1.0, ls=":", zorder=1)

        ax_t.set_xlim(-psi_lim, psi_lim)
        ax_t.set_ylim(-dpsi_lim, dpsi_lim)
        _frame(ax_t, f"u_max = {u_max:g} Н·м   (седло ±{np.degrees(psi_eq):.1f}°)",
               "наклон ψ, рад", "скорость наклона dψ/dt, рад/с")


        # 3. Колесо. Оно ведомое: в поле экстремалей не участвует, но по нему
        #    видно цену манёвра -- сколько оборотов уезжает колесо, пока
        #    наклон идёт по своей линии уровня.
        allc = [whole_curve(b) for b in field["branches"]]
        th = np.concatenate([X[:, 1] for X in allc])
        dth = np.concatenate([X[:, 3] for X in allc])
        m_th = 1.05 * float(np.percentile(np.abs(th), 99.5))
        m_dth = 1.05 * float(np.percentile(np.abs(dth), 99.5))
        ax_w.set_xlim(-m_th, m_th)
        ax_w.set_ylim(-m_dth, m_dth)
        ax_w.plot([0.0], [0.0], "o", ms=5, mfc="white", mec=INK, mew=1.2, zorder=6)
        _frame(ax_w, "колесо вдоль тех же ветвей (ведомая координата)",
               "угол колеса θ, рад", "скорость колеса dθ/dt, рад/с")

    handles = [
        plt.Line2D([], [], color=C_PLUS, lw=1.8),
        plt.Line2D([], [], color=C_MINUS, lw=1.8),
        plt.Line2D([], [], color=INK, lw=2.2),
        plt.Line2D([], [], color=MUTED, lw=1.8, ls=(0, (5, 3))),
        plt.Line2D([], [], marker="o", color="none", mfc="white", mec=INK,
                   mew=1.4, ms=7),
        plt.Line2D([], [], color=GRID, lw=1.2, ls=":"),
    ]
    labels = ["u = +u_max", "u = −u_max", "σ = 0 (кривая переключения реле)",
              "граница восстановимости (серое -- восстановимо)",
              "седло при u = ∓u_max", "ψ = ±π/2"]
    fig.legend(handles, labels, ncol=6, loc="upper center",
               bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=9,
               labelcolor=INK, handlelength=2.0, columnspacing=1.6)

    sub = ("численный пучок против аналитики: σ = 0 -- расхождение "
           f"{errors['sigma']:.1e} рад/с по dψ/dt, "
           f"граница восстановимости -- {errors['manifold']:.1e}; "
           f"H сохраняется до {errors['H_abs']:.1e} "
           f"({errors['H_rel']:.0e} относительных)")
    fig.suptitle("Поле экстремалей: обе ветви максимального момента, вперёд и "
                 "назад по времени\n" + sub, color=INK, fontsize=11.5, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.905))
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)
    return out


# --- запуск --------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--n-psi", type=int, default=25,
                    help="сколько начальных наклонов (dpsi0 = 0 всегда)")
    ap.add_argument("--dt", type=float, default=DT, help="шаг RK4")
    ap.add_argument("--u-max", type=float, nargs="*", default=list(U_MAXES))
    ap.add_argument("--out", default="figures/fig_extremal_field.png")
    ap.add_argument("--no-fig", action="store_true", help="только числа")
    args = ap.parse_args(argv)

    wp = WheeledPendulum()
    fields, errors = [], {"H_abs": 0.0, "H_rel": 0.0, "sigma": 0.0,
                          "manifold": 0.0, "revers": 0.0}

    head = (f"{'u_max':>6} {'седло,°':>8} {'|ΔH|':>10} {'отн.':>10} "
            f"{'σ=0, рад/с':>11} {'граница':>10} {'назад→вперёд':>13}")
    print(f"колёсный маятник, RK4, dt = {args.dt:g} с, "
          f"{args.n_psi} начальных наклонов при dψ/dt = 0")
    print(head)
    print("-" * len(head))
    for u_max in args.u_max:
        field = extremal_field(wp, u_max, n_psi=args.n_psi, dt=args.dt)
        h_abs, h_rel = conservation_error(wp, field)
        d_sigma, _, _, _ = switching_curve_error(wp, u_max, dt=args.dt)
        d_man = manifold_error(wp, u_max, dt=args.dt)
        d_rev = reversibility_error(wp, u_max, n_psi=5, dt=args.dt, steps=1000)
        fields.append(field)
        for key, val in (("H_abs", h_abs), ("H_rel", h_rel), ("sigma", d_sigma),
                         ("manifold", d_man), ("revers", d_rev)):
            errors[key] = max(errors[key], val)
        print(f"{u_max:6g} {np.degrees(wp.saddle_angle(u_max)):8.2f} "
              f"{h_abs:10.2e} {h_rel:10.2e} {d_sigma:11.2e} {d_man:10.2e} "
              f"{d_rev:13.2e}")

    print("\nЧто проверено этими числами:")
    print("  σ=0  -- формула реле `BangBangLQRController.switching_function`: "
          "обратная ветвь из нуля")
    print("          при u = −u_max легла на неё с точностью "
          f"{errors['sigma']:.1e} рад/с по dψ/dt.")
    print("  граница -- `recoverable_bounds`: обратная ветвь из окрестности "
          "седла вдоль устойчивого")
    print(f"          собственного вектора отклонилась на "
          f"{errors['manifold']:.1e} рад/с.")
    print(f"  |ΔH| -- первый интеграл вдоль ветвей (он же оракул шага "
          f"интегрирования): {errors['H_abs']:.1e},")
    print(f"          то есть {errors['H_rel']:.0e} от самой H (H ≈ γD ≈ "
          f"{wp.p.gamma * wp.p.D:.0f}).")

    if args.no_fig:
        return errors
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("\nmatplotlib не установлен -- картинка не рисуется, числа выше "
              "(PROPOSALS.md §B11).")
        return errors
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    print(f"\nкартинка: {draw(wp, fields, errors, args.out)}")
    return errors


if __name__ == "__main__":
    main()
