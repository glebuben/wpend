"""График переключения между реле и ЛКР: дорожка фаз во времени (ER-016).

Реплика куратора: «рисуй график переключения между бэнг-бэнг и лкр».

Что здесь считается
-------------------
`BangBangLQRController` работает фазами: реле по сепаратрисе -> мягкий ЛКР в
полосе |psi| < eps -> базовый ЛКР в сертифицированном эллипсоиде (A26, A27).
На графике u(t) момент передачи приходится угадывать по излому кривой, и при
сравнении двух прогонов угадывание перестаёт работать. Поэтому под графиками
рисуется ДОРОЖКА ФАЗ -- полоса «кто рулит» во времени, восстановленная
`wpend.viz.phases.phase_track`: регулятор заново прогоняется по записанной
оценке, и фазу называет он сам, а не повторённые здесь предикаты.

Дорожка не бесплатна в одном месте: запись прореживается (`--stride`), и
передача, случившаяся между сохранёнными отсчётами, появится на дорожке
позже. Молчать об этом нельзя, поэтому рядом с числами печатается
`u_mismatch` -- расхождение управления повтора с записанным. Ноль означает,
что повтор воспроизвёл прогон шаг в шаг.

Какие клетки выбираются (не наугад)
-----------------------------------
* ОБЩАЯ клетка -- одна и та же для всех трёх пределов, иначе числа не
  сравнить. Она должна быть восстановима при самом слабом моторе, а
  восстановимое множество при 1.5 Н·м крошечное (седло 0.059 рад), поэтому
  перебор идёт по нему. Из восстановимых берётся та, где передача при слабом
  моторе происходит ПОЗЖЕ всего: ранняя передача не показывает ничего.
* ПОЗДНЯЯ клетка -- тот же перебор при самом сильном моторе, где
  восстановимое множество шире и релейная фаза успевает быть длинной. Это та
  самая «клетка, где передача происходит поздно» из карточки.

Числа на выходе (печатаются всегда): время первой передачи, число смен знака
реле до неё, время передачи третьей фазе, признак «фаза вернулась в реле»
(защёлка этого не допускает -- сработал признак, значит неверна дорожка) и
`u_mismatch`.

Запуск:

    uv run --extra design python examples/switch_track.py
    uv run --extra design python examples/switch_track.py --grid 13 --horizon 10
    uv run --extra design python examples/switch_track.py --out figures/fig_switch_track.png

Картинка рисуется, только если доступен matplotlib (в зависимостях проекта
его нет -- PROPOSALS.md §B11). Без него скрипт печатает те же числа.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    BangBangLQRController,
    RK4Integrator,
    ellipsoid_region,
    psi_band_region,
    rollout_many,
)
from wpend.lqr import certified_level, lqr
from wpend.models import WheeledPendulum
from wpend.viz.phases import (PHASE_BANG, PHASE_FINAL, PHASE_LABEL, PHASE_LQR,
                              track_line, track_summary)

R = np.array([[1.0]])
Q_BASE = np.diag([100.0, 1.0, 10.0, 1.0])
#: Цена второй фазы: веса колеса в сто раз меньше базовых (A26).
Q_SOFT = np.diag([100.0, 1e-2, 10.0, 1e-2])

#: Цвета фаз. Те же три роли, что в окне, но шаг темнее: окно рисует на
#: почти чёрном фоне, картинка -- на белом. Проверены на различимость при
#: дальтонизме (худшая пара по протанопии dE 18.5 при пороге 8); зелёный не
#: дотягивает до контраста 3:1 с белым, поэтому отрезки ещё и подписаны.
PHASE_FACE = {
    PHASE_BANG: "#c67048",
    PHASE_LQR: "#488cc8",
    PHASE_FINAL: "#6ec8a0",
}


def design(u_max, eps):
    """(система, фабрика регулятора) для одного предела момента.

    Фабрика, а не готовый регулятор: у него есть память (защёлки фаз), и один
    объект на два прогона -- это второй прогон, начатый уже переключённым.
    """
    wp = WheeledPendulum(u_max=u_max)
    A, B = wp.linearize_upright()
    K, P = lqr(A, B, Q_BASE, R)
    K_soft, _ = lqr(A, B, Q_SOFT, R)
    # c* зависит от предела мотора: тот же K, но насыщение съедает часть
    # области, где V̇ < 0 (CLAUDE.md, «предел живёт в трёх местах»).
    c_star, _ = certified_level(wp, K, P, u_max=u_max, n_dirs=800, ds=4e-3,
                                s_max=8.0, psi_max=np.pi / 2, seed=0)

    def make():
        return BangBangLQRController(K_soft, u_max, psi_band_region(eps), wp,
                                     K_final=K,
                                     final_region=ellipsoid_region(P, float(c_star)))

    return wp, make


def cells(wp, u_max, n):
    """Сетка (psi0, dpsi0) по восстановимому множеству: только его клетки.

    Границы -- замкнутая форма (`saddle_angle`, `recoverable_bounds`), с
    запасом 1.25: за границей спасения нет вообще, и передачи там не будет ни
    при каком регуляторе -- перебирать такие клетки значит тратить прогоны.
    """
    psi_edge = 1.25 * wp.saddle_angle(u_max)
    _, ceiling = wp.recoverable_bounds(u_max, np.array([0.0]))
    dpsi_edge = 1.25 * float(ceiling[0])
    PSI, DPSI = np.meshgrid(np.linspace(-psi_edge, psi_edge, n),
                            np.linspace(-dpsi_edge, dpsi_edge, n))
    psi, dpsi = PSI.ravel(), DPSI.ravel()
    ok = wp.is_recoverable(u_max, psi, dpsi)
    X0 = np.zeros((int(ok.sum()), 4))
    X0[:, 0], X0[:, 2] = psi[ok], dpsi[ok]
    return X0


def run_batch(wp, make, X0, dt, horizon, stride):
    """Прогон пачки и сводка фаз по каждой строке.

    Пачка считается одним вызовом (векторно), а дорожка восстанавливается
    построчно: повтор -- цикл на Python, и для одной строки это дёшево, а для
    сетки заметно. Поэтому перебор клеток идёт по СВОДКАМ, а не по картинкам.
    """
    batch = rollout_many(wp, make(), RK4Integrator(), X0, dt,
                         int(round(horizon / dt)), stride, record_estimate=True)
    out = []
    for m in range(len(X0)):
        traj = batch[m]
        held = bool(np.nanmax(np.abs(traj.x[:, 0])) < np.pi)
        out.append((traj, track_summary(traj, make(), wp), held))
    return out


def latest_handover(runs, X0):
    """Клетка с самой поздней передачей среди удержанных. (индекс, сводка).

    Падающие клетки не рассматриваются: там «передачи не было» -- это не
    поздняя передача, а её отсутствие, и рисовать надо не это.
    """
    best = None
    for m, (_, s, held) in enumerate(runs):
        if not held or not np.isfinite(s["t_handover"]):
            continue
        if best is None or s["t_handover"] > runs[best][1]["t_handover"]:
            best = m
    if best is None:
        raise SystemExit("ни одна клетка не удержана -- нечего рисовать; "
                         "проверь горизонт и предел момента")
    return best, X0[best]


def report(title, x0, u_max, summary, held):
    print(f"  {title:<22} u_max = {u_max:>4.1f} Н·м   "
          f"psi0 = {x0[0]:+.4f}  dpsi0 = {x0[2]:+.4f}   "
          f"{'удержал' if held else 'УПАЛ'}")
    print(f"      {track_line(summary)}")


def _window(summary, horizon):
    """Сколько секунд показывать в колонке.

    Весь горизонт показывать нельзя: при сильном моторе передача происходит на
    0.07 с, и на десяти секундах от неё остаётся вертикальная черта у оси.
    Окно берётся по последнему СОБЫТИЮ (передаче) с полуторным запасом и не
    короче двух секунд -- чтобы было видно, что после передачи ЛКР делает.
    """
    events = [t for t in (summary["t_handover"], summary["t_final"])
              if np.isfinite(t)]
    last = max(events) if events else horizon
    return float(min(horizon, max(2.0, 1.5 * last)))


def draw(path, columns, u_maxes, eps, horizon):
    """Картинка: колонка на прогон, три строки -- psi(t), u(t), дорожка фаз.

    Одна ось на панель (двух шкал на одной оси нет нигде: psi и u -- разные
    величины и живут в разных панелях). Вертикали передач -- на всех трёх
    строках колонки, цветом той фазы, в которую передали: так момент на
    дорожке и излом на u(t) видны как одно событие.

    У колонок РАЗНЫЕ окна по времени, и это сознательно: сравнивать здесь надо
    не длительности, а устройство переключения, а числа для сравнения
    напечатаны и подписаны на самих панелях.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
    except ImportError:                                  # pragma: no cover
        print("\nmatplotlib не установлен -- картинка не рисуется, числа выше "
              "полны (PROPOSALS.md §B11).")
        return

    n = len(columns)
    fig, axes = plt.subplots(3, n, figsize=(4.6 * n, 7.4),
                             gridspec_kw=dict(height_ratios=[3, 3, 0.55]))
    axes = np.atleast_2d(axes).reshape(3, n)
    for j, (title, traj, summary, u_max) in enumerate(columns):
        ts = traj.t - traj.t[0]
        span = _window(summary, float(ts[-1]))
        ax_psi, ax_u, ax_ph = axes[0, j], axes[1, j], axes[2, j]

        ax_psi.plot(ts, traj.x[:, 0], color="#1f2933", lw=1.4)
        ax_psi.axhline(0.0, color="#b0b7bd", lw=0.8)
        for sign in (+1, -1):
            ax_psi.axhline(sign * eps, color="#7b8794", lw=0.9, ls=":")
        ax_psi.set_title(title, fontsize=10)
        ax_psi.set_ylabel("psi, рад")

        ax_u.plot(ts[:-1], traj.u[:, 0], color="#1f2933", lw=1.2)
        for sign in (+1, -1):
            ax_u.axhline(sign * u_max, color="#7b8794", lw=0.9, ls=":")
        ax_u.set_ylabel("u, Н·м")

        # Дорожка фаз: отрезок постоянной фазы -- один прямоугольник. Подпись
        # пишется внутри, когда отрезок шире неё: зелёный не дотягивает до
        # контраста 3:1 с белым фоном, и цвет один работать не обязан.
        phases = summary["phases"]
        edges = np.flatnonzero(np.diff(phases)) + 1
        starts = np.concatenate([[0], edges])
        stops = np.concatenate([edges, [phases.size]])
        for a, b in zip(starts, stops):
            t0, t1 = ts[a], ts[min(b, len(ts) - 1)]
            code = int(phases[a])
            ax_ph.axvspan(t0, t1, color=PHASE_FACE[code], lw=0)
            if (min(t1, span) - t0) > 0.18 * span:
                ax_ph.text((t0 + min(t1, span)) / 2, 0.5, PHASE_LABEL[code],
                           ha="center", va="center", fontsize=8, color="#10181f")
        ax_ph.set_yticks([])
        ax_ph.set_ylim(0, 1)
        ax_ph.set_ylabel("фаза", rotation=0, ha="right", va="center", fontsize=9)
        ax_ph.set_xlabel("t, с")

        for key, code in (("t_handover", PHASE_LQR), ("t_final", PHASE_FINAL)):
            t = summary[key]
            if not np.isfinite(t):
                continue
            for ax in (ax_psi, ax_u, ax_ph):
                ax.axvline(t, color=PHASE_FACE[code], lw=1.2, ls="--")
        ax_psi.annotate(track_line(summary).replace("   |   ", "\n"),
                        xy=(0.98, 0.95), xycoords="axes fraction", ha="right",
                        va="top", fontsize=7.5, color="#4a5560")
        for ax in (ax_psi, ax_u):
            ax.grid(color="#e6e9ec", lw=0.6)
            ax.set_axisbelow(True)
            ax.tick_params(labelbottom=False)
        for ax in (ax_psi, ax_u, ax_ph):
            ax.set_xlim(0.0, span)

    handles = [Patch(facecolor=PHASE_FACE[c], label="фаза: " + PHASE_LABEL[c])
               for c in (PHASE_BANG, PHASE_LQR, PHASE_FINAL)]
    handles += [
        Line2D([], [], color="#7b8794", ls=":", label="полоса |psi| < eps и ±u_max"),
        Line2D([], [], color=PHASE_FACE[PHASE_LQR], ls="--", label="момент передачи"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9)
    fig.suptitle("Переключение реле -> ЛКР -> жёсткий ЛКР: дорожка фаз (ER-016)",
                 fontsize=12)
    fig.subplots_adjust(left=0.06, right=0.985, top=0.9, bottom=0.11,
                        hspace=0.16, wspace=0.26)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, facecolor="white")
    print(f"\nкартинка: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--u-max", type=float, nargs="+", default=[1.5, 3.0, 10.0],
                    help="пределы момента, Н·м (первый -- самый слабый)")
    ap.add_argument("--eps", type=float, default=0.05,
                    help="полоса передачи ЛКР, рад")
    ap.add_argument("--grid", type=int, default=9,
                    help="сетка N x N для перебора клеток")
    ap.add_argument("--horizon", type=float, default=12.0, help="горизонт прогона, с")
    ap.add_argument("--dt", type=float, default=1e-3, help="шаг интегрирования, с")
    ap.add_argument("--stride", type=int, default=5,
                    help="сохранять каждый stride-й кадр")
    ap.add_argument("--out", type=str, default=None, help="куда сохранить PNG")
    args = ap.parse_args()

    u_weak, u_strong = min(args.u_max), max(args.u_max)
    designs = {u: design(u, args.eps) for u in args.u_max}

    # 1. Общая клетка: перебор по восстановимому множеству СЛАБОГО мотора.
    wp, make = designs[u_weak]
    X0 = cells(wp, u_weak, args.grid)
    runs = run_batch(wp, make, X0, args.dt, args.horizon, args.stride)
    m, x_common = latest_handover(runs, X0)
    print(f"\nобщая клетка (перебор {len(X0)} восстановимых при {u_weak:g} Н·м):")
    report("общая", x_common, u_weak, runs[m][1], runs[m][2])

    columns = []
    for u in args.u_max:
        wp_u, make_u = designs[u]
        traj, summary, held = run_batch(wp_u, make_u, x_common[None, :], args.dt,
                                        args.horizon, args.stride)[0]
        if u != u_weak:
            report("та же клетка", x_common, u, summary, held)
        columns.append((f"общая клетка, u_max = {u:g} Н·м", traj, summary, u))

    # 2. Поздняя клетка: тот же перебор при сильном моторе, где релейная фаза
    #    успевает быть длинной.
    wp_s, make_s = designs[u_strong]
    X0s = cells(wp_s, u_strong, args.grid)
    runs_s = run_batch(wp_s, make_s, X0s, args.dt, args.horizon, args.stride)
    ms, x_late = latest_handover(runs_s, X0s)
    print(f"\nпоздняя передача (перебор {len(X0s)} восстановимых при "
          f"{u_strong:g} Н·м):")
    report("поздняя", x_late, u_strong, runs_s[ms][1], runs_s[ms][2])
    columns.append((f"поздняя передача, u_max = {u_strong:g} Н·м",
                    runs_s[ms][0], runs_s[ms][1], u_strong))

    if args.out:
        draw(args.out, columns, args.u_max, args.eps, args.horizon)
    else:
        print("\n--out не задан -- картинка не рисуется.")


if __name__ == "__main__":
    main()
