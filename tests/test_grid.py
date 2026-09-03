"""Сетка и классификация исходов -- без pygame."""

import numpy as np

from wpend import LinearFeedbackController, RK4Integrator, ZeroController, rollout_many
from wpend.models import WheeledPendulum
from wpend.viz.grid import FELL_BACKWARD, FELL_FORWARD, HELD, GridSpec, classify

K_WHEELED = np.array([[95.122221, 1.0, 19.807871, 1.449624]])


def test_grid_layout():
    spec = GridSpec(n=5, theta_max=0.4, dtheta_max=2.0)
    X0 = spec.initial_states(n_state=4, i_theta=0, i_dtheta=2)
    assert X0.shape == (25, 4)
    assert np.allclose(X0[:, 1], 0.0) and np.allclose(X0[:, 3], 0.0)
    # клетка (ix, iy) обязана нести именно ту пару (theta0, dtheta0)
    for ix in range(5):
        for iy in range(5):
            m = spec.index(ix, iy)
            assert X0[m, 0] == spec.thetas[ix]
            assert X0[m, 2] == spec.dthetas[iy]


def test_free_fall_is_classified_by_direction():
    """Без управления знак начального наклона решает, в какую сторону падать."""
    system = WheeledPendulum()
    X0 = np.array([[0.2, 0.0, 0.0, 0.0], [-0.2, 0.0, 0.0, 0.0]])
    batch = rollout_many(system, ZeroController(1), RK4Integrator(), X0,
                         dt=1e-3, n_steps=2000, stride=10)
    outcome, t_fall = classify(batch, i_theta=0, theta_fall=1.0)
    assert outcome[0] == FELL_FORWARD
    assert outcome[1] == FELL_BACKWARD
    assert np.all(np.isfinite(t_fall))
    assert np.allclose(t_fall[0], t_fall[1])   # симметрия задачи


def test_stabilised_states_are_held():
    system = WheeledPendulum(u_max=15.0)
    X0 = np.array([[0.05, 0.0, 0.0, 0.0]])
    batch = rollout_many(system, LinearFeedbackController(K_WHEELED),
                         RK4Integrator(), X0, dt=1e-3, n_steps=3000, stride=10)
    outcome, t_fall = classify(batch, i_theta=0, theta_fall=1.0)
    assert outcome[0] == HELD
    assert np.isnan(t_fall[0])


def test_weak_motor_shrinks_the_held_region():
    """Физическая проверка карты: чем слабее мотор, тем меньше клеток удержано."""
    spec = GridSpec(n=11, theta_max=0.6, dtheta_max=3.0)
    X0 = spec.initial_states(4, 0, 2)
    held = []
    for u_max in (15.0, 1.0):
        system = WheeledPendulum(u_max=u_max)
        batch = rollout_many(system, LinearFeedbackController(K_WHEELED),
                             RK4Integrator(), X0, dt=1e-3, n_steps=2000, stride=20)
        outcome, _ = classify(batch, i_theta=0, theta_fall=1.0)
        held.append(int((outcome == HELD).sum()))
    assert held[0] > held[1]
