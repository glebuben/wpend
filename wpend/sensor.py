"""Слой 2 из 5: ДАТЧИК.

Датчик превращает истинное состояние x в измерение y -- то, что реально
доступно бортовому компьютеру.  Это единственное место, где живёт разрыв
между "что есть" и "что видно": шум, неполнота измерений, дискретизация.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Sensor(ABC):
    """Отображение измерения y = h(t, x)."""

    @abstractmethod
    def measure(self, t: float, x: np.ndarray) -> np.ndarray:
        """Вернуть измерение y в момент t при истинном состоянии x."""

    def reset(self) -> None:
        """Сбросить внутреннее состояние (например, генератор шума).

        Вызывается rollout-ом один раз перед началом прогона.
        """


class FullStateSensor(Sensor):
    """Идеальный датчик: y = x.

    Полезен как база отсчёта: с ним оценка совпадает с истиной, и любое
    расхождение траекторий объясняется только регулятором или интегратором.
    """

    def measure(self, t: float, x: np.ndarray) -> np.ndarray:
        return np.array(x, dtype=float, copy=True)


class GaussianNoiseSensor(Sensor):
    """y = x + w, где w ~ N(0, diag(sigma^2)), независимо на каждом шаге.

    Параметры
    ---------
    sigma : float или массив (n_state,)
        Среднеквадратичное отклонение шума покомпонентно.
    seed : int | None
        Зерно генератора.  Фиксированное зерно делает прогон воспроизводимым:
        два вызова rollout с одним seed дадут побитово одинаковые траектории.

    Замечание.  Шум здесь белый и дискретный по шагам: это шум ИЗМЕРЕНИЯ, а не
    шум процесса.  Он не входит в динамику и потому не требует стохастических
    интеграторов (Ито/Стратонович) -- система остаётся детерминированным ОДУ,
    случайность живёт только в канале наблюдения.
    """

    def __init__(self, sigma, seed: int | None = None):
        self.sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def measure(self, t: float, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return x + self.sigma * self._rng.standard_normal(x.shape)
