"""Карта: MPC с пределом в планировщике (box-DDP) против «слепого» MPC и ЛКР.

Сетка -- как в `examples/cost_sweep.py` (в 1.3 раза шире множества
восстановимости), мир обрезает момент, «упал» -- |theta| >= 1.3 за 5 с.
Цена -- пять весов, как в `lqr_cost_explorer`, Q_f = P своей цены.

    uv run python examples/box_mpc_map.py --u-max 10 --grid 11 --horizon-s 1 2
    uv run python examples/box_mpc_map.py --q 100 1 10 1 --r 100

Каждая клетка -- отдельный прогон MPC (~8–15 с), поэтому 11x11 на 2 ядрах --
10–25 минут на один горизонт. Результат пишется в examples/results/box_mpc_map.npz
после каждого варианта.
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import LinearFeedbackController, MPCController, RK4Integrator, rollout, rollout_many
from wpend.lqr import lqr
from wpend.models import WheeledPendulum

DT, T_SIM, THETA_FALL, MAP_FIT, DT_PLAN = 1e-3, 5.0, 1.3, 1.3, 0.02
OUT = Path(__file__).resolve().parent / "results" / "box_mpc_map.npz"


def grid(u_max, n):
    world = WheeledPendulum(u_max=u_max)
    th_max = min(MAP_FIT * world.saddle_angle(u_max), 0.98 * THETA_FALL)
    _, ceil = world.recoverable_bounds(u_max, np.zeros(1))
    ths = np.linspace(-th_max, th_max, n)
    dths = np.linspace(-MAP_FIT * ceil[0], MAP_FIT * ceil[0], n)
    TH, DTH = np.meshgrid(ths, dths, indexing="xy")
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = TH.ravel(), DTH.ravel()
    return X0, world.is_recoverable(u_max, X0[:, 0], X0[:, 2])


def cell(job):
    """Один прогон MPC. model_limited -- знает ли планировщик предел."""
    x0, u_max, q, r, steps, model_limited = job
    model = WheeledPendulum()
    A, B = model.linearize_upright()
    Q, R = np.diag(q), np.array([[r]])
    K, P = lqr(A, B, Q, R)
    planner = WheeledPendulum(u_max=u_max) if model_limited else WheeledPendulum()
    ctrl = MPCController(planner, RK4Integrator(), DT_PLAN, steps, Q, R, P, K_init=K)
    traj = rollout(WheeledPendulum(u_max=u_max), ctrl, RK4Integrator(), x0, DT,
                   int(round(T_SIM / DT)))
    return not (np.abs(traj.x[1:, 0]) >= THETA_FALL).any()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-max", type=float, default=10.0)
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--q", type=float, nargs=4, default=[100.0, 1.0, 10.0, 1.0])
    ap.add_argument("--r", type=float, default=1.0)
    ap.add_argument("--horizon-s", type=float, nargs="+", default=[1.0, 2.0])
    ap.add_argument("--blind", action="store_true", help="добавить MPC без предела в плане")
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()

    X0, rec = grid(args.u_max, args.grid)
    A, B = WheeledPendulum().linearize_upright()
    K, _ = lqr(A, B, np.diag(args.q), np.array([[args.r]]))
    batch = rollout_many(WheeledPendulum(u_max=args.u_max), LinearFeedbackController(K),
                         RK4Integrator(), X0, DT, int(round(T_SIM / DT)), 10)
    held = {"lqr": ~(np.abs(batch.x[:, 1:, 0]) >= THETA_FALL).any(axis=1)}
    print(f"u_max = {args.u_max}, Q = diag{tuple(args.q)}, R = {args.r}, сетка "
          f"{args.grid}x{args.grid}: восстановимо {rec.sum()}")
    print(f"  ЛКР + clip:                   удержан {held['lqr'].sum():3d}", flush=True)

    variants = [(h, True) for h in args.horizon_s]
    if args.blind:
        variants += [(h, False) for h in args.horizon_s]
    out = {"X0": X0, "rec": rec, "lqr": held["lqr"]}
    OUT.parent.mkdir(exist_ok=True)
    for horizon_s, limited in variants:
        steps = int(round(horizon_s / DT_PLAN))
        t0 = time.perf_counter()
        with ProcessPoolExecutor(args.workers) as pool:
            res = np.array(list(pool.map(cell, [(x, args.u_max, args.q, args.r, steps, limited)
                                                for x in X0])))
        name = f"{'box' if limited else 'blind'}_{horizon_s:g}s"
        held[name] = res
        out[name] = res
        np.savez(OUT, **out)
        print(f"  MPC {'box  ' if limited else 'blind'} горизонт {horizon_s:g} с:  "
              f"удержан {res.sum():3d}, только он {(res & ~held['lqr']).sum()}, "
              f"только ЛКР {(held['lqr'] & ~res).sum()}, вне множества {(res & ~rec).sum()}  "
              f"({(time.perf_counter() - t0) / 60:.1f} мин)", flush=True)


if __name__ == "__main__":
    main()
