"""Маятник на оси с моментом мотора -- простейшая модель для обкатки каркаса.

Состояние x = (theta, dtheta), где theta -- отклонение от ВЕРХНЕЙ вертикали
(theta = 0 -- стойка вверх, как у колёсного маятника; theta = pi -- висит вниз).

Уравнение движения (стержень массы m с центром масс на расстоянии l,
момент инерции I = m l^2, вязкое трение c, момент мотора u):

    I * ddtheta = m g l sin(theta) - c * dtheta + u

Знак "+" перед sin неслучаен: при отсчёте от верхней вертикали гравитация
УВОДИТ от равновесия, поэтому верхнее положение неустойчиво -- ровно тот
случай, ради которого существует регулятор.

Полная энергия E = 1/2 I dtheta^2 + m g l cos(theta) удовлетворяет
    dE/dt = dtheta * u - c * dtheta^2,
то есть при u = 0 и c = 0 сохраняется.  Это даёт бесплатный оракул для тестов:
любая ошибка в f или в интеграторе немедленно ломает сохранение E.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..system import System


@dataclass
class PendulumParams:
    m: float = 1.0      # масса                       [кг]
    l: float = 1.0      # расстояние до центра масс   [м]
    g: float = 9.81     # ускорение свободного падения [м/с^2]
    c: float = 0.0      # вязкое трение в оси         [Н·м·с]
    u_max: float | None = None   # предел момента мотора [Н·м], None = без предела

    @property
    def I(self) -> float:
        """Момент инерции относительно оси подвеса (точечная масса на стержне)."""
        return self.m * self.l ** 2


class Pendulum(System):
    """dx/dt = f(t, x, u) для маятника, x = (theta, dtheta), u = (torque,)."""

    def __init__(self, params: PendulumParams | None = None, **kwargs):
        self.p = params if params is not None else PendulumParams(**kwargs)

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_action(self) -> int:
        return 1

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("theta", "dtheta")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ("torque",)

    @property
    def u_bounds(self):
        if self.p.u_max is None:
            return None
        return (np.array([-self.p.u_max]), np.array([self.p.u_max]))

    def f(self, t, x, u):
        # Индексы x[..., i], а не распаковка: так одна и та же формула считает
        # и одно состояние (2,), и пачку (M, 2) -- см. System.f.
        theta = x[..., 0]
        dtheta = x[..., 1]
        p = self.p
        ddtheta = (p.m * p.g * p.l * np.sin(theta) - p.c * dtheta + u[..., 0]) / p.I
        return np.stack([dtheta, ddtheta], axis=-1)

    # --- оракулы для тестов и анализа ------------------------------------

    def energy(self, x) -> float:
        """E = 1/2 I dtheta^2 + m g l cos(theta); сохраняется при u = 0, c = 0."""
        x = np.asarray(x, dtype=float)
        theta, dtheta = x[..., 0], x[..., 1]
        p = self.p
        return 0.5 * p.I * dtheta ** 2 + p.m * p.g * p.l * np.cos(theta)

    def linearize_upright(self):
        """(A, B) линеаризации в верхнем положении x = 0, u = 0.

            A = [[0, 1], [m g l / I, -c / I]],   B = [[0], [1 / I]]

        Собственные числа A при c = 0 равны +-sqrt(g/l): одно строго
        положительное, откуда и берётся неустойчивость.
        """
        p = self.p
        A = np.array([[0.0, 1.0], [p.m * p.g * p.l / p.I, -p.c / p.I]])
        B = np.array([[0.0], [1.0 / p.I]])
        return A, B
