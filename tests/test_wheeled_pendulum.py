"""Колёсный маятник: проверки против формул из docs/Key_Formulas.md (master)."""

import numpy as np
import pytest

from wpend import ConstantController, RK4Integrator, ZeroController, rollout
from wpend.models import WheeledPendulum


def test_upright_is_an_equilibrium():
    sys_ = WheeledPendulum()
    assert np.allclose(sys_.f(0.0, np.zeros(4), np.zeros(1)), np.zeros(4))


def test_determinant_is_positive_everywhere():
    """Delta = alpha gamma - beta^2 cos^2 > 0: матрица масс положительно
    определена, уравнения всегда разрешимы относительно ускорений."""
    sys_ = WheeledPendulum()
    thetas = np.linspace(-np.pi, np.pi, 401)
    assert np.all(sys_.Delta(thetas) > 0.0)


def test_tilt_dynamics_does_not_depend_on_wheel_state():
    """Key_Formulas §1.4: phi -- циклическая координата, поэтому ddtheta и ddphi
    не зависят ни от phi, ни от dphi (галилеева инвариантность идеального
    качения).  Меняем колесо -- ускорения обязаны остаться теми же."""
    sys_ = WheeledPendulum()
    u = np.array([0.7])
    base = sys_.f(0.0, np.array([0.2, 0.0, 0.5, 0.0]), u)
    for phi, dphi in [(3.0, 0.0), (0.0, 12.0), (-7.0, -4.0)]:
        other = sys_.f(0.0, np.array([0.2, phi, 0.5, dphi]), u)
        assert np.allclose(base[2:], other[2:])


def test_first_integral_is_conserved_under_constant_torque():
    """Key_Formulas §3: при постоянном u величина H сохраняется вдоль траекторий.
    Это самый сильный оракул модели: он завязан одновременно на все члены f."""
    sys_ = WheeledPendulum()
    for u_const in (0.0, 1.5, -2.0):
        traj = rollout(sys_, ConstantController([u_const]), RK4Integrator(),
                       x0=[0.15, 0.0, 0.0, 0.0], dt=1e-4, n_steps=20000)
        H = np.array([sys_.first_integral(x, u_const) for x in traj.x])
        assert np.max(np.abs(H - H[0])) < 1e-8 * max(1.0, abs(H[0]))


def test_linearization_matches_finite_differences():
    sys_ = WheeledPendulum()
    A, B = sys_.linearize_upright()
    eps = 1e-6
    x0, u0 = np.zeros(4), np.zeros(1)
    A_num = np.column_stack([
        (sys_.f(0.0, x0 + eps * e, u0) - sys_.f(0.0, x0 - eps * e, u0)) / (2 * eps)
        for e in np.eye(4)
    ])
    B_num = ((sys_.f(0.0, x0, u0 + eps) - sys_.f(0.0, x0, u0 - eps)) / (2 * eps)).reshape(4, 1)
    assert np.allclose(A, A_num, atol=1e-6)
    assert np.allclose(B, B_num, atol=1e-6)


def test_torque_sign_pushes_body_and_wheel_opposite_ways():
    """Момент мотора действует на корпус и колесо в противоположные стороны
    (третий закон Ньютона): знаки ddtheta и ddphi от u должны быть разными."""
    sys_ = WheeledPendulum()
    dx = sys_.f(0.0, np.zeros(4), np.array([1.0]))
    assert dx[2] > 0.0
    assert dx[3] < 0.0


def test_body_falls_without_torque():
    sys_ = WheeledPendulum()
    traj = rollout(sys_, ZeroController(1), RK4Integrator(),
                   x0=[0.05, 0.0, 0.0, 0.0], dt=1e-3, n_steps=1000)
    assert abs(traj.x[-1, 0]) > abs(traj.x[0, 0])
