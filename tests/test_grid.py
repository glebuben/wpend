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


def test_starting_beyond_theta_fall_is_not_itself_a_fall():
    """Начальный отсчёт -- условие задачи, а не пересечение порога.

    Пара, а не одна проверка: иначе «не смотреть на отсчёт 0» прошло бы и в
    вырожденном виде «никогда не падать». Первая клетка стартует за порогом с
    броском обратно, после старта порога не касается и обязана быть HELD.
    Вторая стартует за порогом и уходит дальше -- обязана быть FELL, причём
    сразу, на первом же отсчёте после старта.

    Числа взяты с приколоченной карты (--theta-max 1.7 --dtheta-max 5.0), где
    ошибка и вылезла: там 1620 клеток красились «упал» в нулевой момент.
    """
    system = WheeledPendulum(u_max=3.0)
    X0 = np.array([[-1.3175, 0.0, 3.125, 0.0],     # летит обратно, ловится
                   [-1.40, 0.0, 0.0, 0.0]])        # лежит и лежит
    batch = rollout_many(system, LinearFeedbackController(K_WHEELED),
                         RK4Integrator(), X0, dt=1e-3, n_steps=5000, stride=20)
    outcome, t_fall = classify(batch, i_theta=0, theta_fall=1.3)

    theta = batch.x[:, :, 0]
    assert np.abs(theta[0, 1:]).max() < 1.3        # после старта порог цел
    assert outcome[0] == HELD and np.isnan(t_fall[0])
    assert abs(theta[0, -1]) < 0.05                # и правда стоит вертикально

    assert outcome[1] == FELL_BACKWARD
    assert t_fall[1] == batch.t[1]                 # первый отсчёт после старта


def test_the_initial_sample_never_fires_inside_auto_fitted_bounds():
    """Правка выше обязана быть НИЧЕМ, пока границы карты подгоняются сами.

    Автоподгонка режет theta_max по 0.98 * theta_fall, так что за порогом не
    может оказаться ни один узел сетки. Проверяется именно это неравенство --
    оно и есть причина, по которой старые карты и картинки в figures/ не
    поехали.
    """
    theta_fall = 1.3
    spec = GridSpec(n=41, theta_max=0.98 * theta_fall, dtheta_max=5.0)
    assert np.abs(spec.thetas).max() < theta_fall
