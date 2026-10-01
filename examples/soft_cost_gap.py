"""ER-017: сколько восстановимого теряет вторая фаза при разных ценах мягкого ЛКР.

Запуск:  uv run --extra design python examples/soft_cost_gap.py [--grid 31] [--horizon 30]

Схема одна -- реле по сепаратрисе -> |psi| < eps -> мягкий ЛКР (`bang-eps`
без третьей фазы: третья взводится только в сертифицированном эллипсоиде и
на удержание не влияет). Меняется только цена мягкого ЛКР. Карта -- стандарт
(CLAUDE.md): psi до +-1.3*pi/2, dpsi до 1.3 от максимума восстановимого,
theta0 = dtheta0 = 0. «Удержал» -- |psi| ни разу не достиг pi. Потолок --
`is_recoverable` (A16), считаем удержанные ТОЛЬКО среди восстановимых.

Замер 24.09 (31x31, 30 с, eps = 0.05), удержано / восстановимо:

    q_psi, q_theta, q_dpsi, q_dtheta, R   K_dth   pole      1     1.5      2      3      5      10
    100, 0.01, 10, 0.01, 1   (было)       0.532  -0.216   7/19   9/27  25/45  53/57  95/97 173/173
    100, 0.01, 10, 0.01, 250              0.118  -0.055  19/19  27/27  45/45  57/57  97/97 173/173
    100, 0.02, 10, 0.01, 250 (окно, A38)  0.141  -0.065  19/19  27/27  45/45  57/57  97/97 173/173
    100, 0.03, 10, 0.01, 250              0.157  -0.072  19/19  27/27  45/45  57/57  97/97 173/173
    100, 0.05, 10, 0.01, 250              0.179  -0.082  17/19  27/27  45/45  57/57  97/97 173/173
    100, 1,    10, 0.01, 250              0.394  -0.174   9/19  11/27  37/45  57/57  97/97 173/173

На сетке 61x61 при 1 Н*м q_theta = 3e-2 теряет 2 из 91, 2e-2 -- 0 (при 1.5 оба
117 из 117), поэтому в окно взято 2e-2.

Решает отношение q_theta / R: при R >> q_psi наклон почти не меняется,
q_dtheta почти не влияет (проверено 1e-3 ... 1 при R = 250), а q_theta / R
задаёт и гейн по скорости колеса, и медленную пару полюсов (столбец pole).
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import BangBangLQRController, RK4Integrator, psi_band_region, rollout_many
from wpend.lqr import lqr
from wpend.models import WheeledPendulum

MAP_FIT = 1.3
PSI_MAP = np.pi / 2
DT = 1e-3

#: (q_psi, q_theta, q_dpsi, q_dtheta, R) -- порядок как в панели окна.
COSTS = [
    (100.0, 1e-2, 10.0, 1e-2, 1.0),
    (100.0, 1e-2, 10.0, 1e-2, 250.0),
    (100.0, 2e-2, 10.0, 1e-2, 250.0),
    (100.0, 3e-2, 10.0, 1e-2, 250.0),
    (100.0, 5e-2, 10.0, 1e-2, 250.0),
    (100.0, 1.0, 10.0, 1e-2, 250.0),
]


def grid(wp, u_max, n):
    psi = np.linspace(-PSI_MAP, PSI_MAP, 401)
    floor, ceiling = wp.recoverable_bounds(u_max, psi)
    reach = np.abs(np.concatenate([floor, ceiling]))
    dpsi_max = MAP_FIT * float(reach[np.isfinite(reach)].max())
    P, D = np.meshgrid(np.linspace(-MAP_FIT * PSI_MAP, MAP_FIT * PSI_MAP, n),
                       np.linspace(-dpsi_max, dpsi_max, n))
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = P.ravel(), D.ravel()
    return X0


def gain(wp, cost):
    A, B = wp.linearize_upright()
    K, _ = lqr(A, B, np.diag(cost[:4]), np.array([[cost[4]]]))
    ev = np.linalg.eigvals(A - B @ K)
    return K, float(ev.real.max())          # медленный полюс -- ближайший к нулю


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--u-max", type=float, nargs="+", default=[1, 1.5, 2, 3, 5, 10])
    args = ap.parse_args()

    heads = "  ".join(f"{u:>6g}" for u in args.u_max)
    print(f"{'q_psi, q_theta, q_dpsi, q_dtheta, R':38s} {'K_dth':>6s} {'pole':>7s}  {heads}")
    for cost in COSTS:
        cells = []
        for u_max in args.u_max:
            wp = WheeledPendulum(u_max=u_max)
            K, pole = gain(wp, cost)
            X0 = grid(wp, u_max, args.grid)
            rec = wp.is_recoverable(u_max, X0[:, 0], X0[:, 2])
            ctrl = BangBangLQRController(K, u_max, psi_band_region(args.eps), wp)
            b = rollout_many(wp, ctrl, RK4Integrator(), X0[rec], dt=DT,
                             n_steps=int(round(args.horizon / DT)), stride=20)
            held = (np.abs(b.x[:, :, 0]) < np.pi).all(axis=1)
            cells.append(f"{held.sum()}/{rec.sum()}")
        label = ", ".join(f"{c:g}" for c in cost)
        print(f"{label:38s} {K[0, 3]:6.3f} {pole:7.3f}  "
              + "  ".join(f"{c:>6s}" for c in cells), flush=True)


if __name__ == "__main__":
    main()
