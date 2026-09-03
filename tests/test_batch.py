"""Соглашение о пачках: векторный прогон обязан совпадать с последовательным."""

import numpy as np
import pytest

from wpend import (
    ConstantController,
    LinearFeedbackController,
    RK4Integrator,
    ZeroController,
    rollout,
    rollout_many,
)
from wpend.models import Pendulum, WheeledPendulum

K_WHEELED = np.array([[95.122221, 1.0, 19.807871, 1.449624]])


@pytest.mark.parametrize(
    "system, X0",
    [
        (Pendulum(), np.array([[0.1, 0.0], [-0.3, 1.0], [0.0, -2.0]])),
        (WheeledPendulum(), np.array([[0.1, 0.0, 0.0, 0.0],
                                      [-0.2, 1.0, 0.5, -3.0],
                                      [0.4, -2.0, -1.0, 7.0]])),
    ],
)
def test_f_on_a_batch_equals_f_row_by_row(system, X0):
    """Главная проверка соглашения: векторная формула -- та же формула."""
    U = np.array([[0.3], [-1.0], [0.7]])
    batched = system.f(0.0, X0, U)
    for i in range(X0.shape[0]):
        assert np.allclose(batched[i], system.f(0.0, X0[i], U[i]))


def test_rollout_many_matches_sequential_rollout():
    """Векторный прогон не имеет права отличаться от M отдельных прогонов --
    ни на бит физики, только на округление порядка суммирования."""
    system = WheeledPendulum(u_max=15.0)
    controller = LinearFeedbackController(K_WHEELED)
    X0 = np.array([[0.10, 0.0, 0.0, 0.0],
                   [-0.25, 0.0, 1.0, 0.0],
                   [0.40, 0.0, -2.0, 0.0]])

    batch = rollout_many(system, controller, RK4Integrator(), X0,
                         dt=1e-3, n_steps=2000)
    for i in range(X0.shape[0]):
        one = rollout(system, controller, RK4Integrator(), X0[i],
                      dt=1e-3, n_steps=2000)
        assert np.allclose(batch[i].x, one.x, atol=1e-12)
        assert np.allclose(batch[i].u, one.u, atol=1e-12)
        assert np.allclose(batch.t, one.t)


def test_stride_decimates_frames_without_changing_them():
    """Прореживание меняет только то, ЧТО сохранено, а не то, что посчитано:
    сохранённые кадры обязаны совпасть с каждым 10-м кадром полного прогона."""
    system = Pendulum()
    controller = ConstantController([0.2])
    X0 = np.array([[0.1, 0.0], [0.3, -1.0]])

    dense = rollout_many(system, controller, RK4Integrator(), X0,
                         dt=1e-3, n_steps=1000, stride=1)
    thin = rollout_many(system, controller, RK4Integrator(), X0,
                        dt=1e-3, n_steps=1000, stride=10)

    assert thin.x.shape == (2, 101, 2)
    assert thin.u.shape == (2, 100, 1)
    assert np.allclose(thin.x, dense.x[:, ::10])
    assert np.allclose(thin.t, dense.t[::10])


def test_batch_element_is_an_ordinary_trajectory():
    batch = rollout_many(Pendulum(), ZeroController(1), RK4Integrator(),
                         np.array([[0.1, 0.0], [0.2, 0.0]]),
                         dt=1e-3, n_steps=100)
    traj = batch[1]
    assert traj.x.shape == (101, 2)
    assert traj.u.shape == (100, 1)
    assert traj.dt == pytest.approx(1e-3)
    assert len(batch) == 2


def test_rollout_many_rejects_bad_arguments():
    system = Pendulum()
    args = (system, ZeroController(1), RK4Integrator(), np.zeros((3, 2)))
    with pytest.raises(ValueError):
        rollout_many(*args, dt=1e-3, n_steps=1000, stride=7)   # не делится
    with pytest.raises(ValueError):
        rollout_many(system, ZeroController(1), RK4Integrator(),
                     np.zeros((3, 5)), dt=1e-3, n_steps=10)     # не та ширина
