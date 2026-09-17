"""Как цена (Q, R) меняет то, где ЛКР и MPC удерживают корпус при пределе момента.

Контекст (15.09, `examples/mpc_clip_map.py`): MPC без предела в планировщике
плюс обрезка в мире держит РОВНО те же клетки, что ЛКР, а часть
восстановимого множества теряют оба. На одной клетке помог мягкий вес
колеса. Этот скрипт проверяет гипотезу «край задаёт цена» на всей карте.

Две команды:

  run   считает и сохраняет результаты в .npz
        uv run python examples/cost_sweep.py run                  # только ЛКР, ~2 мин
        uv run python examples/cost_sweep.py run --mpc base phi=1e-2 R=10
                                                                  # + MPC, ~7 мин на пресет и предел
  show  читает .npz и печатает; ничего не пересчитывает
        uv run python examples/cost_sweep.py show                 # сводная таблица
        uv run python examples/cost_sweep.py show --map phi=1e-2 --u-max 10
        uv run python examples/cost_sweep.py show --map base --u-max 3 --mpc
        uv run python examples/cost_sweep.py cell --preset base --u-max 10 --theta -0.4 --dtheta 0.5
                                                                  # один прогон ЛКР с подробностями

Что сравнивается с чем
----------------------
* `is_recoverable` -- аналитическое множество: снаружи не держит никто.
  Реле по сепаратрисе совпадает с ним на всей карте (A16), это потолок.
* «удержан» -- |theta| ни разу после старта не достиг 1.3 за горизонт
  (тот же критерий, что `wpend.viz.grid.classify`).
* «сошёлся» -- удержан и в конце |theta| < 0.05, |dtheta| < 0.1, |dphi| < 0.1.
* ЦЕНА ЗА УДЕРЖАНИЕ -- колесо. По удержанным клеткам печатается max |phi(T)|
  (куда уехало колесо, рад; метры = рад · r = рад · 0.3) и max |dphi(T)|.

Карта -- как в окне: в 1.3 раза шире множества, обрезана по theta_fall.
Все регуляторы видят истинное состояние; мир обрезает момент (`clip_action`).
MPC: такт 20 мс, горизонт 1 с, Q_f = P своего пресета, предела не знает.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import LinearFeedbackController, MPCController, RK4Integrator, rollout, rollout_many
from wpend.lqr import lqr, lqr_tilt
from wpend.models import WheeledPendulum
from wpend.viz.grid import classify

DT, THETA_FALL, MAP_FIT = 1e-3, 1.3, 1.3
DT_PLAN, PLAN_STEPS = 0.02, 50
HERE = Path(__file__).resolve().parent
DEFAULT_LQR = HERE / "results" / "cost_sweep_lqr.npz"
DEFAULT_MPC = HERE / "results" / "cost_sweep_mpc.npz"

# ---------------------------------------------------------------------------
#  Пресеты цены. Порядок весов: (theta, phi, dtheta, dphi).
#  "base" -- цена, с которой работает весь проект (окно, тесты).
#  phi=... -- меняется ТОЛЬКО вес положения колеса: главный подозреваемый.
#  "tilt" -- крайний случай: колесо не в цене вовсе (`lqr_tilt`, A17).
# ---------------------------------------------------------------------------
PRESETS = {
    "base":       (np.diag([100.0, 1.0,  10.0, 1.0]),  1.0),
    "phi=0.3":    (np.diag([100.0, 0.3,  10.0, 1.0]),  1.0),
    "phi=0.1":    (np.diag([100.0, 0.1,  10.0, 1.0]),  1.0),
    "phi=3e-2":   (np.diag([100.0, 3e-2, 10.0, 1.0]),  1.0),
    "phi=1e-2":   (np.diag([100.0, 1e-2, 10.0, 1.0]),  1.0),
    "phi=1e-3":   (np.diag([100.0, 1e-3, 10.0, 1.0]),  1.0),
    "dphi=1e-2":  (np.diag([100.0, 1.0,  10.0, 1e-2]), 1.0),
    "R=0.1":      (np.diag([100.0, 1.0,  10.0, 1.0]),  0.1),
    "R=10":       (np.diag([100.0, 1.0,  10.0, 1.0]),  10.0),
    "R=100":      (np.diag([100.0, 1.0,  10.0, 1.0]),  100.0),
    "theta x10":  (np.diag([1000.0, 1.0, 100.0, 1.0]), 1.0),
    "tilt":       None,
}


def design(name):
    """(Q, R, K, P) пресета. Для "tilt" Q -- нули на колесе, P вырождена (ранг 2)."""
    model = WheeledPendulum()
    if PRESETS[name] is None:
        K, P = lqr_tilt(model, np.diag([100.0, 10.0]), np.array([[1.0]]))
        return np.diag([100.0, 0.0, 10.0, 0.0]), np.array([[1.0]]), K, P
    Q, r = PRESETS[name]
    R = np.array([[r]])
    A, B = model.linearize_upright()
    K, P = lqr(A, B, Q, R)
    return Q, R, K, P


def grid(u_max, n):
    world = WheeledPendulum(u_max=u_max)
    theta_max = min(MAP_FIT * world.saddle_angle(u_max), 0.98 * THETA_FALL)
    _, ceiling = world.recoverable_bounds(u_max, np.zeros(1))
    dtheta_max = MAP_FIT * float(ceiling[0])
    thetas = np.linspace(-theta_max, theta_max, n)
    dthetas = np.linspace(-dtheta_max, dtheta_max, n)
    TH, DTH = np.meshgrid(thetas, dthetas, indexing="xy")
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = TH.ravel(), DTH.ravel()
    return thetas, dthetas, X0, world.is_recoverable(u_max, X0[:, 0], X0[:, 2])


def outcome_of(t, x_theta):
    """Исход и момент падения -- тем же `classify`, что красит карту окна.

    classify читает у пачки только .t и .x, поэтому хватает простого
    контейнера: MPC считается поклеточно, и TrajectoryBatch у него нет.
    """
    from types import SimpleNamespace
    x = np.zeros(x_theta.shape + (1,))
    x[..., 0] = x_theta
    return classify(SimpleNamespace(t=t, x=x), 0, THETA_FALL)


def summarize(x_theta, x_end):
    """Из траекторий наклона (M, T) и конечных состояний (M, 4) -- метрики клеток."""
    held = ~(np.abs(x_theta[:, 1:]) >= THETA_FALL).any(axis=1)
    conv = (held & (np.abs(x_end[:, 0]) < 0.05) & (np.abs(x_end[:, 2]) < 0.1)
            & (np.abs(x_end[:, 3]) < 0.1))
    return held, conv


# ---------------------------------------------------------------------------
#  run
# ---------------------------------------------------------------------------

def _mpc_cell(job):
    name, u_max, x0, horizon = job
    Q, R, K, P = design(name)
    ctrl = MPCController(WheeledPendulum(), RK4Integrator(), DT_PLAN, PLAN_STEPS,
                         Q, R, P, K_init=K)
    traj = rollout(WheeledPendulum(u_max=u_max), ctrl, RK4Integrator(), x0, DT,
                   int(round(horizon / DT)))
    # Траектория с шагом 10 мс сохраняется целиком: MPC на клетку -- ~8 с, и
    # просмотрщик не должен пересчитывать его по клику.
    return traj.x[::10], traj.u[::10, 0], traj.x[-1]


def cmd_run(args):
    out = {"u_maxes": np.array(args.u_max), "presets": np.array(list(PRESETS)),
           "grid": args.grid, "horizon": args.horizon}
    for u_max in args.u_max:
        thetas, dthetas, X0, rec = grid(u_max, args.grid)
        out[f"{u_max}__thetas"], out[f"{u_max}__dthetas"], out[f"{u_max}__rec"] = thetas, dthetas, rec
        world = WheeledPendulum(u_max=u_max)
        for name in PRESETS:
            _, _, K, _ = design(name)
            t0 = time.perf_counter()
            batch = rollout_many(world, LinearFeedbackController(K), RK4Integrator(), X0,
                                 DT, int(round(args.horizon / DT)), 10)
            held, conv = summarize(batch.x[:, :, 0], batch.x[:, -1])
            key = f"{u_max}__{name}"
            out[key + "__held"], out[key + "__conv"], out[key + "__end"] = held, conv, batch.x[:, -1]
            out[key + "__outcome"], out[key + "__tfall"] = outcome_of(batch.t, batch.x[:, :, 0])
            print(f"ЛКР  u_max={u_max:<5g} {name:<10} удержан {held.sum():4d} из "
                  f"восстановимых {rec.sum():4d}  ({time.perf_counter() - t0:.1f} с)")
    DEFAULT_LQR.parent.mkdir(exist_ok=True)
    np.savez(args.out or DEFAULT_LQR, **out)

    if not args.mpc:
        return
    mpc = {"u_maxes": np.array(args.u_max), "presets": np.array(args.mpc),
           "grid": args.mpc_grid, "horizon": args.horizon}
    for u_max in args.u_max:
        thetas, dthetas, X0, rec = grid(u_max, args.mpc_grid)
        mpc[f"{u_max}__thetas"], mpc[f"{u_max}__dthetas"], mpc[f"{u_max}__rec"] = thetas, dthetas, rec
        for name in args.mpc:
            t0 = time.perf_counter()
            with ProcessPoolExecutor(args.workers) as pool:
                res = list(pool.map(_mpc_cell, [(name, u_max, x, args.horizon) for x in X0]))
            X = np.stack([r[0] for r in res])                   # (M, T, 4)
            U = np.stack([r[1] for r in res])                   # (M, T)
            end = np.stack([r[2] for r in res])
            t = DT * 10 * np.arange(X.shape[1])
            held, conv = summarize(X[:, :, 0], end)
            key = f"{u_max}__{name}"
            mpc[key + "__held"], mpc[key + "__conv"], mpc[key + "__end"] = held, conv, end
            mpc[key + "__outcome"], mpc[key + "__tfall"] = outcome_of(t, X[:, :, 0])
            mpc[key + "__x"], mpc[key + "__u"], mpc["t"] = X.astype(np.float32), U.astype(np.float32), t
            print(f"MPC  u_max={u_max:<5g} {name:<10} удержан {held.sum():4d} из "
                  f"восстановимых {rec.sum():4d}  ({(time.perf_counter() - t0) / 60:.1f} мин)", flush=True)
            # После каждой карты: долгий прогон можно смотреть, не дожидаясь конца.
            np.savez(args.mpc_out or DEFAULT_MPC, **mpc)


# ---------------------------------------------------------------------------
#  show / cell
# ---------------------------------------------------------------------------

def _load(path):
    if not Path(path).exists():
        sys.exit(f"нет файла {path}: сначала `run`")
    return np.load(path, allow_pickle=False)


def _row(d, u_max, name):
    key = f"{u_max}__{name}"
    if key + "__held" not in d:
        return None
    held, conv, end = d[key + "__held"], d[key + "__conv"], d[key + "__end"]
    rec = d[f"{u_max}__rec"]
    phi = np.abs(end[held, 1]).max() if held.any() else np.nan
    dphi = np.abs(end[held, 3]).max() if held.any() else np.nan
    return held.sum(), (rec & ~held).sum(), conv.sum(), phi, dphi


def cmd_show(args):
    d = _load(args.lqr)
    m = _load(args.mpc_file) if Path(args.mpc_file).exists() else None

    if args.map:
        src = m if args.mpc else d
        if src is None:
            sys.exit("MPC-результатов нет: `run --mpc ...`")
        return print_map(src, args.u_max, args.map, args.mpc)

    for u_max in d["u_maxes"]:
        rec = d[f"{u_max}__rec"]
        print(f"\nu_max = {u_max:g} Н·м   ЛКР: сетка {int(d['grid'])}x{int(d['grid'])}, "
              f"горизонт {float(d['horizon']):g} с, восстановимо {rec.sum()}")
        print(f"  {'пресет':<10} {'удержан':>8} {'потерян':>8} {'сошёлся':>8} "
              f"{'max|phi(T)|':>12} {'max|dphi(T)|':>13}")
        for name in d["presets"]:
            r = _row(d, u_max, name)
            print(f"  {name:<10} {r[0]:8d} {r[1]:8d} {r[2]:8d} {r[3]:12.1f} {r[4]:13.2f}")
        if m is not None and u_max in m["u_maxes"]:
            rec_m = m[f"{u_max}__rec"]
            print(f"  MPC (сетка {int(m['grid'])}x{int(m['grid'])}, восстановимо {rec_m.sum()}):")
            for name in m["presets"]:
                r = _row(m, u_max, name)
                if r is not None:
                    print(f"  {name:<10} {r[0]:8d} {r[1]:8d} {r[2]:8d} {r[3]:12.1f} {r[4]:13.2f}")
    print("\n  потерян = восстановимо, но упал. Колесо: рад; метры = рад · 0.3.")


def lqr_on_grid(u_max, name, n, horizon):
    """Удержание ЛКР пресета на сетке n x n -- для сравнения с MPC клетка в клетку.
    Считается на лету: rollout_many на 121 клетке -- около секунды."""
    _, _, X0, _ = grid(u_max, n)
    _, _, K, _ = design(name)
    batch = rollout_many(WheeledPendulum(u_max=u_max), LinearFeedbackController(K),
                         RK4Integrator(), X0, DT, int(round(horizon / DT)), 10)
    return summarize(batch.x[:, :, 0], batch.x[:, -1])[0]


def print_map(src, u_max, name, is_mpc):
    """Карта символами.

    ЛКР:  # удержан   . восстановимо, но упал   пробел -- невосстановимо.
    MPC:  сравнение с ЛКР ТОГО ЖЕ пресета на той же сетке (ЛКР пересчитывается):
          B держат оба   M только MPC   L только ЛКР   . никто, но восстановимо.
    ! -- кто-то удержал невосстановимую клетку: быть не должно, это проверка.
    """
    key = f"{u_max}__{name}"
    if key + "__held" not in src:
        sys.exit(f"нет результата {key}; есть пресеты: {list(src['presets'])}")
    held, rec = src[key + "__held"], src[f"{u_max}__rec"]
    thetas, dthetas = src[f"{u_max}__thetas"], src[f"{u_max}__dthetas"]
    n = thetas.size
    if is_mpc:
        held_lqr = lqr_on_grid(u_max, name, n, float(src["horizon"]))
        print(f"MPC против ЛКР, пресет {name}, u_max = {u_max:g}, сетка {n}x{n}: "
              f"восстановимо {rec.sum()}, MPC {held.sum()}, ЛКР {held_lqr.sum()}, "
              f"только MPC {(held & ~held_lqr).sum()}, только ЛКР {(held_lqr & ~held).sum()}")
        print("  B оба   M только MPC   L только ЛКР   . никто, но восстановимо   ! вне множества")
    else:
        print(f"ЛКР, пресет {name}, u_max = {u_max:g}: удержан {held.sum()}, "
              f"восстановимо {rec.sum()}, потерян {(rec & ~held).sum()}")
        print("  # удержан   . восстановимо, но упал   ! удержан вне множества (быть не должно)")
    print(f"  theta0: {thetas[0]:+.3f} ... {thetas[-1]:+.3f};  строки -- dtheta0 сверху вниз")
    sep = " " if is_mpc else ""
    for iy in range(n - 1, -1, -1):
        cells = []
        for ix in range(n):
            j = iy * n + ix
            if is_mpc:
                a, b = held[j], held_lqr[j]
                c = "!" if (a or b) and not rec[j] else "B" if a and b else \
                    "M" if a else "L" if b else "." if rec[j] else " "
            else:
                c = "#" if held[j] and rec[j] else "!" if held[j] else "." if rec[j] else " "
            cells.append(c)
        print(f"  {dthetas[iy]:+6.2f} |{sep.join(cells)}|")


def cmd_cell(args):
    """Один прогон ЛКР выбранного пресета: когда упал, сколько в насыщении, где колесо."""
    Q, R, K, P = design(args.preset)
    world = WheeledPendulum(u_max=args.u_max)
    x0 = np.array([args.theta, 0.0, args.dtheta, 0.0])
    traj = rollout(world, LinearFeedbackController(K), RK4Integrator(), x0, DT,
                   int(round(args.horizon / DT)))
    theta = traj.x[:, 0]
    fall = np.flatnonzero(np.abs(theta[1:]) >= THETA_FALL)
    sat = np.abs(traj.u[:, 0]) >= args.u_max - 1e-9
    print(f"пресет {args.preset}, K = {np.round(K, 3)}")
    print(f"x0 = {x0}, восстановимо: {bool(world.is_recoverable(args.u_max, x0[0], x0[2]))}")
    print(f"упал: {'нет' if fall.size == 0 else f'в t = {traj.t[fall[0] + 1]:.3f} с'}")
    print(f"в насыщении {sat.mean():.0%} шагов, первое -- "
          f"{'нет' if not sat.any() else f't = {traj.t[np.argmax(sat)]:.3f} с'}")
    print(f"{'t, с':>6} {'theta':>8} {'phi':>8} {'dtheta':>8} {'dphi':>8} {'u':>8}")
    for t_show in np.linspace(0.0, args.horizon, 11):
        k = min(int(round(t_show / DT)), traj.n_steps - 1)
        x = traj.x[k]
        print(f"{traj.t[k]:6.2f} {x[0]:+8.3f} {x[1]:+8.2f} {x[2]:+8.3f} {x[3]:+8.2f} "
              f"{traj.u[k, 0]:+8.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run")
    r.add_argument("--u-max", type=float, nargs="+", default=[3.0, 10.0])
    r.add_argument("--grid", type=int, default=41)
    r.add_argument("--horizon", type=float, default=5.0)
    r.add_argument("--mpc", nargs="*", default=[], choices=list(PRESETS))
    r.add_argument("--mpc-grid", type=int, default=11)
    r.add_argument("--workers", type=int, default=2)
    r.add_argument("--out", default=None)
    r.add_argument("--mpc-out", default=None)

    s = sub.add_parser("show")
    s.add_argument("--lqr", default=str(DEFAULT_LQR))
    s.add_argument("--mpc-file", default=str(DEFAULT_MPC))
    s.add_argument("--map", default=None, help="пресет, карту которого печатать")
    s.add_argument("--u-max", type=float, default=10.0)
    s.add_argument("--mpc", action="store_true", help="карта MPC, а не ЛКР")

    c = sub.add_parser("cell")
    c.add_argument("--preset", default="base", choices=list(PRESETS))
    c.add_argument("--u-max", type=float, default=10.0)
    c.add_argument("--theta", type=float, required=True)
    c.add_argument("--dtheta", type=float, required=True)
    c.add_argument("--horizon", type=float, default=5.0)

    args = ap.parse_args()
    {"run": cmd_run, "show": cmd_show, "cell": cmd_cell}[args.cmd](args)


if __name__ == "__main__":
    main()
