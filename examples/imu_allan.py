"""Кривая Аллана по синтетическому сигналу IMUSensor (ER-007, шаг 3).

Дисперсия Аллана -- стандартный способ РАЗДЕЛИТЬ компоненты ошибки датчика,
которые во временной области просто складываются в одну кашу.  Идея: усреднить
сигнал по окнам длины tau и посмотреть, как расходятся соседние окна.  Разные
процессы дают разный наклон в двойном логарифме, и наклон не зависит от
амплитуды -- поэтому по нему можно опознать процесс, а не только измерить его.

    наклон -1/2   белый шум          sigma^2(tau) = N^2 / tau,      N при tau=1
    наклон  0     фликкер 1/f        sigma^2(tau) = (2 B^2/pi) ln2, sigma=0.664*B
    наклон +1/2   случайное блуждание sigma^2(tau) = K^2 tau / 3,   K при tau=3

Запуск:

    uv run python examples/imu_allan.py
    uv run python examples/imu_allan.py --n 400000 --out figures/fig_allan.png

Картинка рисуется, только если доступен matplotlib (в зависимостях проекта его
нет -- см. PROPOSALS.md §B11).  Без него скрипт печатает ту же таблицу числами,
и оракул наклонов от картинки не зависит: он живёт в tests/test_imu_sensor.py.

Источники: IEEE Std 952-2020, приложение C; Groves, гл. 4.4;
El-Sheimy, Hou, Niu, "Analysis and Modeling of Inertial Sensors Using Allan
Variance", IEEE Trans. Instrum. Meas. 57 (2008).
"""

from __future__ import annotations

import argparse

import numpy as np

from wpend import IMUSensor
from wpend.models import WheeledPendulum


def allan_deviation(rate, dt: float, n_points: int = 60, min_clusters: int = 20):
    """Перекрывающееся отклонение Аллана для сигнала УГЛОВОЙ СКОРОСТИ.

    Перекрывающаяся оценка использует все возможные положения окна, а не
    только непересекающиеся, и потому при том же сигнале имеет заметно меньшую
    дисперсию -- иначе хвост кривой (большие tau, мало окон) шумит так, что
    наклон по нему не измерить.

        theta(k)      = cumsum(rate) * dt                (проинтегрированный угол)
        sigma^2(tau)  = sum_k [theta_{k+2m} - 2 theta_{k+m} + theta_k]^2
                        / (2 tau^2 (L - 2m)),            tau = m*dt

    Вторая разность здесь не случайна: это в точности разность СРЕДНИХ по двум
    соседним окнам длины tau, умноженная на tau.

    min_clusters обрезает хвост: при tau, сравнимом с длиной записи, в оценку
    входит всего несколько независимых кластеров (их примерно L/m), и точка
    кривой становится шумом, а не измерением.  Считаем только tau, на которых
    независимых кластеров не меньше min_clusters -- иначе по рваному хвосту
    легко "измерить" наклон, которого нет.

    Возвращает (taus, sigmas) -- по одной точке на размер кластера,
    логарифмически равномерно.
    """
    rate = np.asarray(rate, dtype=float)
    theta = np.concatenate(([0.0], np.cumsum(rate) * dt))
    L = theta.size
    m_max = min((L - 1) // 2, L // max(min_clusters, 2))
    if m_max < 1:
        raise ValueError("сигнал слишком короткий для кривой Аллана")
    ms = np.unique(np.floor(np.logspace(0, np.log10(m_max), n_points)).astype(int))

    taus, sigmas = [], []
    for m in ms:
        if 2 * m >= L:
            continue
        diff = theta[2 * m:] - 2.0 * theta[m:-m] + theta[:-2 * m]
        tau = m * dt
        var = np.sum(diff ** 2) / (2.0 * tau ** 2 * diff.size)
        taus.append(tau)
        sigmas.append(np.sqrt(var))
    return np.array(taus), np.array(sigmas)


def fit_slope(taus, sigmas, tau_lo: float, tau_hi: float) -> float:
    """Наклон log-log прямой на участке [tau_lo, tau_hi]."""
    m = (taus >= tau_lo) & (taus <= tau_hi)
    if m.sum() < 3:
        raise ValueError(f"на участке [{tau_lo}, {tau_hi}] всего {m.sum()} точек")
    return float(np.polyfit(np.log10(taus[m]), np.log10(sigmas[m]), 1)[0])


def gyro_signal(n: int, dt: float, seed: int = 0, **kw) -> np.ndarray:
    """n отсчётов гироскопа при неподвижной машине (истинная скорость 0).

    Датчик берётся в режиме "state": нас интересует канал гироскопа, а не
    замкнутый контур, поэтому система стоит в нуле и весь сигнал -- ошибка.
    """
    system = WheeledPendulum()
    sensor = IMUSensor(system, dt=dt, seed=seed, **kw)
    sensor.reset()
    x = np.zeros(system.n_state)
    u = np.zeros(system.n_action)
    i = system.state_names.index("dtheta")
    out = np.empty(n, dtype=float)
    for k in range(n):
        out[k] = sensor.measure(k * dt, x, u)[i]
    return out


# Три конфигурации: в каждой включена РОВНО ОДНА компонента.  Смысл именно в
# изоляции -- на смеси наклон измерить можно, но он уже не оракул.
CONFIGS = {
    "только белый шум": dict(sigma_g=1e-3, sigma_bg=0.0, b0_g=0.0),
    "только блуждание": dict(sigma_g=0.0, sigma_bg=1e-3, b0_g=0.0),
    "марковский, tau=20 с": dict(sigma_g=0.0, sigma_bg=1e-2, tau_g=20.0, b0_g=0.0),
}
ZERO_ACC = dict(sigma_a=0.0, sigma_ba=0.0, b0_a=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, default=200_000, help="число отсчётов")
    ap.add_argument("--dt", type=float, default=0.01, help="шаг датчика, с")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures/fig_allan.png")
    args = ap.parse_args()

    print(f"сигнал: {args.n} отсчётов при dt = {args.dt} с "
          f"({args.n * args.dt:.0f} с модельного времени)\n")

    curves = {}
    for name, kw in CONFIGS.items():
        rate = gyro_signal(args.n, args.dt, seed=args.seed, **ZERO_ACC, **kw)
        taus, sig = allan_deviation(rate, args.dt)
        curves[name] = (taus, sig)

        lo, hi = 10 * args.dt, 100 * args.dt
        print(f"{name}")
        print(f"    наклон на tau в [{lo:g}, {hi:g}] с: {fit_slope(taus, sig, lo, hi):+.3f}")
        t_end = taus[-1]
        print(f"    наклон на tau в [{t_end / 10:g}, {t_end:g}] с: "
              f"{fit_slope(taus, sig, t_end / 10, t_end):+.3f}")
        j = int(np.argmin(np.abs(taus - 1.0)))
        print(f"    sigma(tau=1 с) = {sig[j]:.3e} рад/с\n")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib не установлен -- картинка не рисуется, числа выше.")
        return

    # Палитра различима при дейтеранопии (проверено численно); плюс к цвету
    # каждая кривая имеет свой тип линии и подписана прямо у конца -- опознать
    # её можно, не различая цветов вовсе.
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#c9c8c3"
    STYLE = {
        "только белый шум":     ("#2a78d6", "-"),
        "только блуждание":     ("#eb6834", "--"),
        "марковский, tau=20 с": ("#1baf7a", "-."),
    }

    fig, ax = plt.subplots(figsize=(8.2, 5.4))
    for name, (taus, sig) in curves.items():
        color, ls = STYLE[name]
        ax.loglog(taus, sig, color=color, ls=ls, lw=2.0, solid_capstyle="round")

    # Опорные наклоны, привязанные к своим кривым: если модель верна, кривая
    # ложится на пунктир, а не просто "куда-то растёт".
    t_w, s_w = curves["только белый шум"]
    t_r, s_r = curves["только блуждание"]
    j = int(np.argmin(np.abs(t_w - 1.0)))
    ax.loglog(t_w, s_w[j] * t_w ** -0.5, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    ax.annotate("наклон −1/2\n(белый шум, N при τ=1 с)", (t_w[3], s_w[j] * t_w[3] ** -0.5),
                color=MUTED, fontsize=8.5, va="bottom", ha="left", xytext=(4, 6),
                textcoords="offset points")
    j = int(np.argmin(np.abs(t_r - 3.0)))
    ax.loglog(t_r, s_r[j] * (t_r / 3.0) ** 0.5, color=MUTED, lw=0.9, ls=(0, (2, 3)), zorder=1)
    q = int(np.argmin(np.abs(t_r - 8.0)))
    ax.annotate("наклон +1/2\n(блуждание, K при τ=3 с)", (t_r[q], s_r[j] * (t_r[q] / 3.0) ** 0.5),
                color=MUTED, fontsize=8.5, va="top", ha="left", xytext=(6, -8),
                textcoords="offset points")

    # Полка марковского процесса: максимум кривой при tau ~ 1.9*tau_c
    t_m, s_m = curves["марковский, tau=20 с"]
    k = int(np.argmax(s_m))
    ax.plot([t_m[k]], [s_m[k]], "o", ms=6, mfc="none", mec=MUTED, mew=1.2, zorder=3)
    ax.annotate(f"максимум при τ = {t_m[k]:.0f} с ≈ 1.9·τ_c\n(приближение полки фликкера)",
                (t_m[k], s_m[k]), color=MUTED, fontsize=8.5, va="top", ha="right",
                xytext=(-8, -12), textcoords="offset points")

    ax.set_xlabel("время усреднения τ, с", color=INK)
    ax.set_ylabel(r"отклонение Аллана $\sigma_A(\tau)$, рад/с", color=INK)
    ax.set_title("Компонента ошибки ИДУ узнаётся по наклону, а не по амплитуде",
                 color=INK, fontsize=11.5, pad=12)
    ax.grid(True, which="major", color=GRID, lw=0.6, alpha=0.9)
    ax.grid(True, which="minor", color=GRID, lw=0.4, alpha=0.45)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    handles = [plt.Line2D([], [], color=c, ls=ls, lw=2.0) for c, ls in STYLE.values()]
    ax.legend(handles, list(STYLE), loc="lower left", frameon=False,
              fontsize=9, labelcolor=INK)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, facecolor="white")
    print(f"картинка: {args.out}")


if __name__ == "__main__":
    main()
