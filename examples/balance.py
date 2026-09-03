"""Замкнутый контур на обеих моделях: удержание в верхнем положении.

Запуск:  python examples/balance.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # запуск без pip install -e .

import numpy as np

from wpend import (
    GaussianNoiseSensor,
    LinearFeedbackController,
    RK4Integrator,
    ZeroController,
    rollout,
)
from wpend.models import Pendulum, WheeledPendulum

# Матрицы обратной связи -- ЛКР для линеаризации в верхнем положении.
# Маятник:          Q = diag(10, 1),         R = 1
# Колёсный маятник: Q = diag(100, 1, 10, 1), R = 1
K_PENDULUM = np.array([[20.11709, 6.421385]])
K_WHEELED = np.array([[95.122221, 1.0, 19.807871, 1.449624]])


def report(name, system, traj):
    x_end = traj.x[-1]
    print(f"{name:<34} t={traj.t[-1]:5.2f}s  "
          + "  ".join(f"{n}={v:+8.4f}" for n, v in zip(system.state_names, x_end))
          + f"  max|u|={np.max(np.abs(traj.u)):6.3f}")


def main():
    dt, T = 1e-3, 5.0
    n = int(T / dt)

    pend = Pendulum()
    report("pendulum, no control", pend,
           rollout(pend, ZeroController(1), RK4Integrator(), [0.3, 0.0], dt, n))
    report("pendulum, LQR feedback", pend,
           rollout(pend, LinearFeedbackController(K_PENDULUM), RK4Integrator(),
                   [0.3, 0.0], dt, n))
    report("pendulum, LQR + noisy sensor", pend,
           rollout(pend, LinearFeedbackController(K_PENDULUM), RK4Integrator(),
                   [0.3, 0.0], dt, n,
                   sensor=GaussianNoiseSensor(sigma=[0.01, 0.05], seed=0)))

    wp = WheeledPendulum(u_max=15.0)
    report("wheeled pendulum, no control", wp,
           rollout(wp, ZeroController(1), RK4Integrator(), [0.15, 0, 0, 0], dt, n))
    report("wheeled pendulum, LQR feedback", wp,
           rollout(wp, LinearFeedbackController(K_WHEELED), RK4Integrator(),
                   [0.15, 0, 0, 0], dt, n))

    weak = WheeledPendulum(u_max=0.5)
    report("wheeled pendulum, weak motor", weak,
           rollout(weak, LinearFeedbackController(K_WHEELED), RK4Integrator(),
                   [0.15, 0, 0, 0], dt, n))


if __name__ == "__main__":
    main()
