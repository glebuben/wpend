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


def test_first_integral_broadcasts_u():
    """u_const может быть массивом: у релейного закона знак момента свой в
    каждой строке пачки. Формула написана через x[..., i], поэтому это должно
    работать без оговорок -- пачка обязана совпасть с отдельными вызовами."""
    wp = WheeledPendulum()
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, size=(9, 4))
    u = rng.choice([-1.5, 1.5], size=9)
    batch = wp.first_integral(X, u)
    one = np.array([wp.first_integral(x, float(ui)) for x, ui in zip(X, u)])
    assert np.allclose(batch, one, atol=1e-15)


# --- множество восстановимости (Key_Formulas §3.2) -------------------------

U_MAX = 3.0


def test_saddle_angle_is_an_equilibrium():
    """theta_eq определён как угол, где предельный момент ровно держит вес.
    Оракул прямой: в этой точке ускорение наклона обязано быть нулём."""
    wp = WheeledPendulum()
    th = wp.saddle_angle(U_MAX)
    x = np.array([th, 0.0, 0.0, 0.0])
    assert abs(wp.f(0.0, x, np.array([-U_MAX]))[2]) < 1e-12


def test_saddle_is_unstable_not_a_centre():
    """Он именно СЕДЛО: чуть ниже момент возвращает, чуть выше тяжесть уносит.
    Если бы знак не менялся, вся конструкция границы была бы бессмысленной."""
    wp = WheeledPendulum()
    th = wp.saddle_angle(U_MAX)
    u = np.array([-U_MAX])
    below = wp.f(0.0, np.array([th - 1e-3, 0.0, 0.0, 0.0]), u)[2]
    above = wp.f(0.0, np.array([th + 1e-3, 0.0, 0.0, 0.0]), u)[2]
    assert below < 0.0 < above


def test_boundary_lies_on_the_separatrix_level():
    """Верхняя граница -- множество уровня H(-u_max) = K(u_max). Проверяется
    подстановкой: сохраняющаяся величина обязана равняться уровню седла."""
    wp = WheeledPendulum()
    K = wp.separatrix_level(U_MAX)
    th = np.linspace(-0.5, wp.saddle_angle(U_MAX) - 1e-3, 60)
    _, ceiling = wp.recoverable_bounds(U_MAX, th)
    x = np.stack([th, np.zeros_like(th), ceiling, np.zeros_like(th)], axis=-1)
    assert np.allclose(wp.first_integral(x, -U_MAX), K, atol=1e-12)


def test_sign_flip_past_the_saddle_keeps_the_tongues():
    """Граница -- устойчивое многообразие седла, а не просто линия уровня:
    за theta_eq нужная ветвь имеет ОТРИЦАТЕЛЬНУЮ dtheta. Поэтому корпус,
    наклонённый дальше седла, всё ещё восстановим, если качается назад."""
    wp = WheeledPendulum()
    th_eq = wp.saddle_angle(U_MAX)
    th = th_eq + 0.05
    floor, ceiling = wp.recoverable_bounds(U_MAX, th)
    assert ceiling < 0.0                       # только назад
    assert floor < ceiling
    assert wp.is_recoverable(U_MAX, th, 0.5 * (floor + ceiling))
    assert not wp.is_recoverable(U_MAX, th, 0.0)   # из покоя -- уже нет


def test_strong_motor_has_no_saddle_at_all():
    """Если момент сильнее веса, равновесия не существует и корпус
    удерживается на любом наклоне до горизонтали."""
    wp = WheeledPendulum()
    assert wp.saddle_angle(wp.p.D * 1.01) == pytest.approx(np.pi / 2)
    assert wp.saddle_angle(0.0) == 0.0


def test_recoverable_bounds_respect_the_batch_convention():
    wp = WheeledPendulum()
    th = np.linspace(-0.7, 0.7, 11)
    fl, ce = wp.recoverable_bounds(U_MAX, th)
    for i, t in enumerate(th):
        f1, c1 = wp.recoverable_bounds(U_MAX, t)
        assert float(f1) == pytest.approx(fl[i], abs=1e-15)
        assert float(c1) == pytest.approx(ce[i], abs=1e-15)
