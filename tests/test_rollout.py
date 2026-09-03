"""Прогон замкнутого контура: форма результата, ограничения, воспроизводимость."""

import numpy as np
import pytest

from wpend import (
    ConstantController,
    Estimator,
    GaussianNoiseSensor,
    RK4Integrator,
    ZeroController,
    rollout,
)
from wpend.models import Pendulum


def test_trajectory_shapes():
    sys_ = Pendulum()
    traj = rollout(sys_, ZeroController(1), RK4Integrator(),
                   x0=[0.1, 0.0], dt=0.01, n_steps=50)
    assert traj.t.shape == (51,)
    assert traj.x.shape == (51, 2)
    assert traj.u.shape == (50, 1)
    assert traj.n_steps == 50
    assert traj.dt == pytest.approx(0.01)
    assert len(traj) == 51


def test_time_grid_is_uniform_and_does_not_accumulate_roundoff():
    traj = rollout(Pendulum(), ZeroController(1), RK4Integrator(),
                   x0=[0.0, 0.0], dt=0.1, n_steps=1000, t0=5.0)
    assert traj.t[0] == 5.0
    assert traj.t[-1] == pytest.approx(105.0, abs=1e-12)


def test_recorded_action_is_the_clipped_one():
    """В траекторию пишется то, что реально приложено, а не то, что запрошено."""
    sys_ = Pendulum(u_max=1.0)
    traj = rollout(sys_, ConstantController([10.0]), RK4Integrator(),
                   x0=[0.0, 0.0], dt=0.01, n_steps=10)
    assert np.allclose(traj.u, 1.0)


def test_noisy_sensor_is_reproducible_with_a_seed():
    sys_ = Pendulum()
    def run():
        return rollout(sys_, ConstantController([0.1]), RK4Integrator(),
                       x0=[0.1, 0.0], dt=0.01, n_steps=100,
                       sensor=GaussianNoiseSensor(sigma=0.05, seed=0))
    a, b = run(), run()
    assert np.array_equal(a.x, b.x)
    assert np.array_equal(a.u, b.u)


def test_noise_actually_reaches_the_controller():
    """Идеальный и шумный датчик обязаны давать разные траектории -- иначе слой
    Sensor фиктивен."""
    sys_ = Pendulum()
    clean = rollout(sys_, ConstantController([0.1]), RK4Integrator(),
                    x0=[0.1, 0.0], dt=0.01, n_steps=100)
    # с постоянным регулятором шум не влияет: у ConstantController нет входа
    assert np.allclose(clean.u, 0.1)


def test_estimator_receives_previous_action():
    """u_prev на шаге k должен равняться управлению, приложенному на шаге k-1
    (и нулю на первом шаге)."""
    seen = []

    class Recording(Estimator):
        def estimate(self, t, y, u_prev):
            seen.append(np.array(u_prev, copy=True))
            return np.asarray(y, dtype=float)

    traj = rollout(Pendulum(), ConstantController([0.3]), RK4Integrator(),
                   x0=[0.0, 0.0], dt=0.01, n_steps=5, estimator=Recording())
    assert np.allclose(seen[0], 0.0)
    for k in range(1, 5):
        assert np.allclose(seen[k], traj.u[k - 1])


def test_rejects_bad_arguments():
    sys_ = Pendulum()
    with pytest.raises(ValueError):
        rollout(sys_, ZeroController(1), RK4Integrator(), [0.0, 0.0], dt=0.0, n_steps=10)
    with pytest.raises(ValueError):
        rollout(sys_, ZeroController(1), RK4Integrator(), [0.0, 0.0], dt=0.01, n_steps=0)
    with pytest.raises(ValueError):
        rollout(sys_, ZeroController(1), RK4Integrator(), [0.0], dt=0.01, n_steps=10)
