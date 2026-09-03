"""Слой 5 из 5: ИНТЕГРАТОР.

Интегратор -- это численная схема: как из (t, x, u) получить состояние через
шаг dt.  Он ничего не знает о конкретной системе: получает f как аргумент.

На протяжении одного шага управление u считается ПОСТОЯННЫМ.  Это не
упрощение ради удобства, а модель реального контроллера: цифровой регулятор
выдаёт значение на ЦАП и держит его до следующего такта (zero-order hold).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

VectorField = Callable[[float, np.ndarray, np.ndarray], np.ndarray]


class Integrator(ABC):
    """Одношаговая схема для dx/dt = f(t, x, u) при постоянном u на шаге."""

    #: Теоретический порядок точности схемы (локальная ошибка O(dt^{order+1}),
    #: глобальная O(dt^order)).  Используется тестами сходимости.
    order: int

    @abstractmethod
    def step(
        self,
        f: VectorField,
        t: float,
        x: np.ndarray,
        u: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Вернуть состояние в момент t + dt."""


class EulerIntegrator(Integrator):
    """Явный метод Эйлера: x_{k+1} = x_k + dt * f(t_k, x_k, u).

    Порядок 1.  Держим его не ради точности, а как эталон для теста сходимости:
    если численно измеренный порядок RK4 равен 4, а Эйлера -- 1, значит тест
    действительно меряет порядок схемы, а не что-то другое.
    """

    order = 1

    def step(self, f, t, x, u, dt):
        x = np.asarray(x, dtype=float)
        return x + dt * np.asarray(f(t, x, u), dtype=float)


class RK4Integrator(Integrator):
    """Классический метод Рунге--Кутты 4-го порядка.

        k1 = f(t,        x)
        k2 = f(t + dt/2, x + dt/2 * k1)
        k3 = f(t + dt/2, x + dt/2 * k2)
        k4 = f(t + dt,   x + dt   * k3)
        x_{k+1} = x_k + dt/6 * (k1 + 2 k2 + 2 k3 + k4)

    Явная схема, порядок 4.  Не симплектическая: на консервативной системе
    полная энергия медленно дрейфует (для маятника -- убывает как O(dt^4) на
    шаг).  Для задач "продержаться несколько секунд" это несущественно, для
    длинных прогонов гамильтоновой динамики -- существенно; см. PROPOSALS.md.
    """

    order = 4

    def step(self, f, t, x, u, dt):
        x = np.asarray(x, dtype=float)
        h = 0.5 * dt
        k1 = np.asarray(f(t, x, u), dtype=float)
        k2 = np.asarray(f(t + h, x + h * k1, u), dtype=float)
        k3 = np.asarray(f(t + h, x + h * k2, u), dtype=float)
        k4 = np.asarray(f(t + dt, x + dt * k3, u), dtype=float)
        return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
