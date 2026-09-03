"""Контракт слоёв: что обязан и чего не обязан уметь каждый абстрактный класс."""

import numpy as np
import pytest

from wpend import (
    ConstantController,
    Controller,
    Estimator,
    FullStateSensor,
    Integrator,
    PassthroughEstimator,
    Sensor,
    System,
    ZeroController,
)
from wpend.models import Pendulum


@pytest.mark.parametrize("cls", [System, Sensor, Estimator, Controller, Integrator])
def test_abstract_classes_are_not_instantiable(cls):
    with pytest.raises(TypeError):
        cls()


def test_clip_action_respects_u_bounds():
    sys_ = Pendulum(u_max=2.0)
    assert np.allclose(sys_.clip_action(np.array([5.0])), [2.0])
    assert np.allclose(sys_.clip_action(np.array([-5.0])), [-2.0])
    assert np.allclose(sys_.clip_action(np.array([1.0])), [1.0])


def test_clip_action_is_identity_without_bounds():
    sys_ = Pendulum()
    assert sys_.u_bounds is None
    assert np.allclose(sys_.clip_action(np.array([1e6])), [1e6])


def test_clip_action_rejects_wrong_shape():
    with pytest.raises(ValueError):
        Pendulum().clip_action(np.array([1.0, 2.0]))


def test_default_sensor_and_estimator_are_transparent():
    x = np.array([0.3, -1.2])
    y = FullStateSensor().measure(0.0, x)
    x_hat = PassthroughEstimator().estimate(0.0, y, np.zeros(1))
    assert np.allclose(x_hat, x)
    # копия, а не ссылка: изменение оценки не должно портить истинное состояние
    x_hat[0] = 99.0
    assert x[0] == 0.3


def test_controllers_return_right_shape():
    assert ZeroController(1).act(0.0, np.zeros(2)).shape == (1,)
    assert ConstantController([0.5]).act(0.0, np.zeros(2)).shape == (1,)
