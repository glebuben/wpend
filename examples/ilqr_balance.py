"""iLQR против ЛКР на колёсном маятнике: цена, момент, время расчёта.

Для каждого начального наклона:
  1. прогон ЛКР (он же начальное приближение для iLQR);
  2. iLQR с той же ценой: Q, R и терминальной Q_f = P;
  3. прогон TrajectoryTrackingController по плану -- на НЕЛИНЕЙНОЙ модели.

Цена J считается одинаково для обоих (`wpend.ddp.trajectory_cost`), поэтому
числа сравнимы. Предела момента нет: iLQR его не учитывает (см. wpend/ddp.py).

Запуск:  uv run python examples/ilqr_balance.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    LinearFeedbackController,
    RK4Integrator,
    TrajectoryTrackingController,
    rollout,
)
from wpend.ddp import ilqr, trajectory_cost
from wpend.lqr import lqr
from wpend.models import WheeledPendulum

Q = np.diag([100.0, 1.0, 10.0, 1.0])
R = np.array([[1.0]])


def main():
    wp = WheeledPendulum()
    A, B = wp.linearize_upright()
    K, P = lqr(A, B, Q, R)
    rk4 = RK4Integrator()
    dt, T = 0.01, 3.0
    N = int(round(T / dt))

    print(f"dt = {dt}, горизонт {T} с ({N} шагов), Q_f = P")
    print(f"{'theta0':>7} | {'J ЛКР':>9} {'J iLQR':>9} {'выигрыш':>8} | "
          f"{'max|u| ЛКР':>10} {'max|u| iLQR':>11} | {'итер.':>5} {'время, с':>8} | "
          f"{'|theta(T)|':>10}")
    for theta0 in (0.1, 0.3, 0.6, 0.9):
        x0 = np.array([theta0, 0.0, 0.0, 0.0])
        base = rollout(wp, LinearFeedbackController(K), rk4, x0, dt, N)
        J_lqr = trajectory_cost(base.x, base.u, Q, R, P, dt)

        t_start = time.perf_counter()
        x_bar, u_bar, K_seq, costs = ilqr(wp, rk4, x0, dt, N, Q, R, P,
                                          u_init=base.u)
        elapsed = time.perf_counter() - t_start

        tracked = rollout(wp, TrajectoryTrackingController(x_bar, u_bar, K_seq, dt),
                          rk4, x0, dt, N)
        J_ilqr = trajectory_cost(tracked.x, tracked.u, Q, R, P, dt)
        gain = 100.0 * (J_lqr - J_ilqr) / J_lqr
        print(f"{theta0:7.2f} | {J_lqr:9.2f} {J_ilqr:9.2f} {gain:7.2f}% | "
              f"{np.abs(base.u).max():10.2f} {np.abs(tracked.u).max():11.2f} | "
              f"{len(costs) - 1:5d} {elapsed:8.2f} | {abs(tracked.x[-1, 0]):10.2e}")


if __name__ == "__main__":
    main()
