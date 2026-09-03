"""Замкнутый контур целиком: линейная обратная связь действительно стабилизирует."""

import numpy as np

from wpend import LinearFeedbackController, RK4Integrator, ZeroController, rollout
from wpend.models import Pendulum, WheeledPendulum

# Матрицы получены как ЛКР для линеаризации в верхнем положении:
#   маятник:         Q = diag(10, 1),          R = 1
#   колёсный маятник: Q = diag(100, 1, 10, 1),  R = 1
# Синтез вынесен за пределы библиотеки, здесь они -- просто числа.
K_PENDULUM = np.array([[20.11709, 6.421385]])
K_WHEELED = np.array([[95.122221, 1.0, 19.807871, 1.449624]])


def test_pendulum_is_stabilised():
    sys_ = Pendulum()
    traj = rollout(sys_, LinearFeedbackController(K_PENDULUM), RK4Integrator(),
                   x0=[0.3, 0.0], dt=1e-3, n_steps=5000)
    assert np.linalg.norm(traj.x[-1]) < 1e-3
    assert np.linalg.norm(traj.x[-1]) < np.linalg.norm(traj.x[0])


def test_pendulum_falls_without_control():
    traj = rollout(Pendulum(), ZeroController(1), RK4Integrator(),
                   x0=[0.3, 0.0], dt=1e-3, n_steps=5000)
    assert np.linalg.norm(traj.x[-1]) > 1.0


def test_wheeled_pendulum_is_stabilised():
    sys_ = WheeledPendulum()
    traj = rollout(sys_, LinearFeedbackController(K_WHEELED), RK4Integrator(),
                   x0=[0.1, 0.0, 0.0, 0.0], dt=1e-3, n_steps=8000)
    assert abs(traj.x[-1, 0]) < 1e-3          # корпус вертикально
    assert abs(traj.x[-1, 2]) < 1e-3          # и не качается


def test_torque_limit_can_break_stabilisation():
    """Тот же регулятор при слишком малом пределе момента не удерживает корпус.
    Это проверка, что u_bounds действительно влияет на замкнутый контур."""
    weak = WheeledPendulum(u_max=0.05)
    traj = rollout(weak, LinearFeedbackController(K_WHEELED), RK4Integrator(),
                   x0=[0.3, 0.0, 0.0, 0.0], dt=1e-3, n_steps=3000)
    assert abs(traj.x[-1, 0]) > 0.3
