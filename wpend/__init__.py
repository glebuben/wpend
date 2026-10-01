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
    EnergyBangBangController,
    LinearFeedbackController,
    MPCController,
    TrajectoryTrackingController,
    ZeroController,
    all_regions,
    ellipsoid_region,
    energy_band_region,
    grid_region,
    psi_band_region,
)
from .estimator import (
    ComplementaryEstimator,
    Estimator,
    KalmanEstimator,
    PassthroughEstimator,
    calibrate_on_stand,
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
    MPU6050_RAW_B0,
    Sensor,
    StackedSensor,
    draw_turn_on_bias,
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
    "MPU6050_RAW_B0",
    "draw_turn_on_bias",
    "Estimator",
    "PassthroughEstimator",
    "ComplementaryEstimator",
    "KalmanEstimator",
    "calibrate_on_stand",
    "optimal_tau",
    "tilt_error_model",
    "Controller",
    "ZeroController",
    "ConstantController",
    "LinearFeedbackController",
    "BangBangLQRController",
    "EnergyBangBangController",
    "TrajectoryTrackingController",
    "MPCController",
    "ellipsoid_region",
    "energy_band_region",
    "all_regions",
    "grid_region",
    "psi_band_region",
    "Integrator",
    "EulerIntegrator",
    "RK4Integrator",
    "rollout",
    "rollout_many",
    "Trajectory",
    "TrajectoryBatch",
]

__version__ = "0.1.0"
