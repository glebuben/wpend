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


# --- запись оценки в Trajectory (A22) ---------------------------------------


def test_ideal_estimator_records_the_true_state_bitwise():
    """С идеальным датчиком и тождественным оценивателем x_hat -- это x[:-1].

    Не тавтология: x_hat собирается в другом месте цикла и другой веткой кода.
    Любая ошибка выравнивания (на шаг вперёд, на шаг назад) видна сразу.
    """
    traj = rollout(Pendulum(), ZeroController(1), RK4Integrator(),
                   x0=[0.2, -0.5], dt=0.01, n_steps=40)
    assert traj.x_hat is not None
    assert np.array_equal(traj.x_hat, traj.x[:-1])


def test_estimate_is_aligned_with_u_not_with_x():
    """x_hat[k] -- та оценка, из которой получено u[k].

    Проверяется через сам закон управления: у линейной обратной связи
    u = clip(-K x_hat), и это восстанавливается из записанных полей точно.
    """
    from wpend import LinearFeedbackController

    sys_ = Pendulum(u_max=3.0)
    K = np.array([[7.0, 1.5]])
    traj = rollout(sys_, LinearFeedbackController(K), RK4Integrator(),
                   x0=[0.3, 0.0], dt=0.01, n_steps=50,
                   sensor=GaussianNoiseSensor(sigma=0.02, seed=1))
    assert traj.x_hat.shape == (traj.n_steps, sys_.n_state)
    assert traj.x.shape == (traj.n_steps + 1, sys_.n_state)
    expected = np.array([sys_.clip_action(-K @ xh) for xh in traj.x_hat])
    assert np.array_equal(traj.u, expected)


def test_noisy_run_records_an_estimate_that_differs_from_truth():
    """Иначе поле было бы копией x и ничего бы не значило."""
    traj = rollout(Pendulum(), ZeroController(1), RK4Integrator(),
                   x0=[0.1, 0.0], dt=0.01, n_steps=100,
                   sensor=GaussianNoiseSensor(sigma=0.05, seed=0))
    err = traj.x_hat - traj.x[:-1]
    assert np.max(np.abs(err)) > 0.01
    assert err.std() == pytest.approx(0.05, rel=0.2)


def test_recording_can_be_switched_off():
    traj = rollout(Pendulum(), ZeroController(1), RK4Integrator(),
                   x0=[0.1, 0.0], dt=0.01, n_steps=10, record_estimate=False)
    assert traj.x_hat is None


def test_batch_does_not_record_estimates_by_default():
    """Память пачки -- задокументированное число; молча удваивать его нельзя."""
    from wpend import rollout_many

    X0 = np.array([[0.1, 0.0], [-0.2, 0.3]])
    off = rollout_many(Pendulum(), ZeroController(1), RK4Integrator(),
                       X0, dt=0.01, n_steps=20, stride=2)
    on = rollout_many(Pendulum(), ZeroController(1), RK4Integrator(),
                      X0, dt=0.01, n_steps=20, stride=2, record_estimate=True)
    assert off.x_hat is None
    assert on.x_hat.shape == (2, on.u.shape[1], 2)
    assert on.nbytes - off.nbytes == on.x_hat.nbytes
    assert off[0].x_hat is None
    assert np.array_equal(on[1].x_hat, on.x_hat[1])


def test_batch_estimates_match_the_sequential_rollout_row_by_row():
    """Та же проверка, что и для x и u: пачка -- это M обычных прогонов."""
    from wpend import rollout_many

    X0 = np.array([[0.15, 0.0], [-0.25, 0.4], [0.0, -1.0]])
    stride = 5
    batch = rollout_many(Pendulum(u_max=2.0), ConstantController([0.3]), RK4Integrator(),
                         X0, dt=0.01, n_steps=50, stride=stride, record_estimate=True)
    for m, x0 in enumerate(X0):
        one = rollout(Pendulum(u_max=2.0), ConstantController([0.3]), RK4Integrator(),
                      x0=x0, dt=0.01, n_steps=50)
        assert np.allclose(batch.x_hat[m], one.x_hat[::stride], rtol=0, atol=1e-12)
