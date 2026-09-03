"""Маятник: сохранение энергии и корректность линеаризации."""

import numpy as np

from wpend import RK4Integrator, ZeroController, rollout
from wpend.models import Pendulum


def test_energy_is_conserved_without_torque_and_friction():
    """При u = 0 и c = 0 система гамильтонова: E = const.

    RK4 не симплектичен, поэтому E дрейфует, но как O(dt^4) -- на 5 секундах
    с dt = 1 мс дрейф должен быть исчезающе мал по сравнению с самой энергией.
    """
    sys_ = Pendulum(c=0.0)
    traj = rollout(sys_, ZeroController(1), RK4Integrator(),
                   x0=[0.5, 0.0], dt=1e-3, n_steps=5000)
    E = np.array([sys_.energy(x) for x in traj.x])
    assert np.max(np.abs(E - E[0])) < 1e-8 * max(1.0, abs(E[0]))


def test_energy_drift_shrinks_as_dt_to_the_fourth():
    sys_ = Pendulum(c=0.0)
    drift = []
    for dt in (4e-2, 2e-2):
        traj = rollout(sys_, ZeroController(1), RK4Integrator(),
                       x0=[1.0, 0.0], dt=dt, n_steps=int(2.0 / dt))
        E = np.array([sys_.energy(x) for x in traj.x])
        drift.append(np.max(np.abs(E - E[0])))
    # уменьшение шага вдвое должно снизить дрейф примерно в 16 раз
    assert drift[0] / drift[1] > 8.0


def test_friction_dissipates_energy():
    sys_ = Pendulum(c=0.5)
    traj = rollout(sys_, ZeroController(1), RK4Integrator(),
                   x0=[0.5, 0.0], dt=1e-3, n_steps=5000)
    assert sys_.energy(traj.x[-1]) < sys_.energy(traj.x[0])


def test_linearization_matches_finite_differences():
    sys_ = Pendulum(c=0.3)
    A, B = sys_.linearize_upright()
    eps = 1e-6
    x0, u0 = np.zeros(2), np.zeros(1)
    A_num = np.column_stack([
        (sys_.f(0.0, x0 + eps * e, u0) - sys_.f(0.0, x0 - eps * e, u0)) / (2 * eps)
        for e in np.eye(2)
    ])
    B_num = ((sys_.f(0.0, x0, u0 + eps) - sys_.f(0.0, x0, u0 - eps)) / (2 * eps)).reshape(2, 1)
    assert np.allclose(A, A_num, atol=1e-6)
    assert np.allclose(B, B_num, atol=1e-6)


def test_upright_equilibrium_is_unstable():
    """Собственные числа A при c = 0 равны +-sqrt(g/l): есть положительное."""
    sys_ = Pendulum(c=0.0)
    A, _ = sys_.linearize_upright()
    eigs = np.linalg.eigvals(A)
    assert np.max(eigs.real) > 0.0
    assert np.allclose(np.sort(eigs.real), np.sort([-np.sqrt(9.81), np.sqrt(9.81)]))
