"""wpend -- минимальная лаборатория динамики и управления.

Пять слоёв, один прогон:

    System      dx/dt = f(t, x, u), плюс ограничение мотора u_bounds
    Sensor      x -> y          что реально видно
    Estimator   y -> x_hat      что мы думаем о состоянии
    Controller  x_hat -> u      что мы хотим приложить
    Integrator  (f, x, u, dt) -> x_next

    rollout(...) -> Trajectory(t, x, u)

Всё, что идёт дальше (графики, анимация, метрики), читает только Trajectory.
"""

from .controller import (
    ConstantController,
    Controller,
    LinearFeedbackController,
    ZeroController,
)
from .estimator import Estimator, PassthroughEstimator
from .integrator import EulerIntegrator, Integrator, RK4Integrator
from .rollout import Trajectory, rollout
from .sensor import FullStateSensor, GaussianNoiseSensor, Sensor
from .system import System

__all__ = [
    "System",
    "Sensor",
    "FullStateSensor",
    "GaussianNoiseSensor",
    "Estimator",
    "PassthroughEstimator",
    "Controller",
    "ZeroController",
    "ConstantController",
    "LinearFeedbackController",
    "Integrator",
    "EulerIntegrator",
    "RK4Integrator",
    "rollout",
    "Trajectory",
]

__version__ = "0.1.0"
