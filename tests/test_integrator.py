"""Численная проверка ПОРЯДКА схем.

Идея: для схемы порядка p глобальная ошибка на фиксированном горизонте ведёт
себя как C * dt^p.  Значит log(ошибка) от log(dt) -- прямая с наклоном p.
Меряем наклон по двум измельчениям сетки и сравниваем с заявленным order.
Тест ловит и опечатку в коэффициентах Рунге--Кутты, и подмену схемы.
"""

import numpy as np
import pytest

from wpend import EulerIntegrator, RK4Integrator


def _f_linear(t, x, u):
    """dx/dt = -2x + u; точное решение при u = const известно аналитически."""
    return -2.0 * x + u


def _exact(t, x0, u):
    """x(t) = (x0 - u/2) e^{-2t} + u/2."""
    return (x0 - u / 2.0) * np.exp(-2.0 * t) + u / 2.0


def _global_error(integrator, dt, T=1.0, x0=1.0, u=0.5):
    n = int(round(T / dt))
    x = np.array([x0])
    t = 0.0
    for k in range(n):
        x = integrator.step(_f_linear, t, x, np.array([u]), dt)
        t = (k + 1) * dt
    return abs(x[0] - _exact(T, x0, u))


@pytest.mark.parametrize("integrator", [EulerIntegrator(), RK4Integrator()])
def test_observed_order_matches_declared_order(integrator):
    dts = [1e-2, 5e-3, 2.5e-3]
    errs = [_global_error(integrator, dt) for dt in dts]
    slopes = [
        np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1])
        for i in range(len(dts) - 1)
    ]
    observed = float(np.mean(slopes))
    assert abs(observed - integrator.order) < 0.2, (
        f"{type(integrator).__name__}: заявлен порядок {integrator.order}, "
        f"измерен {observed:.2f}"
    )


def test_rk4_is_much_more_accurate_than_euler():
    dt = 1e-2
    assert _global_error(RK4Integrator(), dt) < 1e-6 * _global_error(EulerIntegrator(), dt)


def test_step_does_not_mutate_input_state():
    x = np.array([1.0])
    x_copy = x.copy()
    RK4Integrator().step(_f_linear, 0.0, x, np.array([0.0]), 0.01)
    assert np.array_equal(x, x_copy)
