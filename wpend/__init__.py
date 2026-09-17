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
    BangBangLQRController,
    ConstantController,
    Controller,
    LinearFeedbackController,
    MPCController,
    TrajectoryTrackingController,
    ZeroController,
    ellipsoid_region,
    grid_region,
    theta_band_region,
)
from .estimator import (
    ComplementaryEstimator,
    Estimator,
    PassthroughEstimator,
    optimal_tau,
    tilt_error_model,
)
from .integrator import EulerIntegrator, Integrator, RK4Integrator
from .rollout import Trajectory, TrajectoryBatch, rollout, rollout_many
from .sensor import (
    EncoderSensor,
    FullStateSensor,
    GaussianNoiseSensor,
    IMUSensor,
    Sensor,
    StackedSensor,
)
from .system import System

__all__ = [
    "System",
    "Sensor",
    "FullStateSensor",
    "GaussianNoiseSensor",
    "IMUSensor",
    "EncoderSensor",
    "StackedSensor",
    "Estimator",
    "PassthroughEstimator",
    "ComplementaryEstimator",
    "optimal_tau",
    "tilt_error_model",
    "Controller",
    "ZeroController",
    "ConstantController",
    "LinearFeedbackController",
    "BangBangLQRController",
    "TrajectoryTrackingController",
    "MPCController",
    "ellipsoid_region",
    "grid_region",
    "theta_band_region",
    "Integrator",
    "EulerIntegrator",
    "RK4Integrator",
    "rollout",
    "rollout_many",
    "Trajectory",
    "TrajectoryBatch",
]

__version__ = "0.1.0"
