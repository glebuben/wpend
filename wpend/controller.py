"""Слой 4 из 5: РЕГУЛЯТОР.

Регулятор превращает оценку состояния x_hat в требуемое управление u.
Он НЕ знает, откуда взялась оценка, и НЕ знает про ограничения мотора --
за проекцию на допустимое множество отвечает System.clip_action.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Controller(ABC):
    """Закон управления u = pi(t, x_hat)."""

    @abstractmethod
    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        """Вернуть требуемое управление формы (n_action,)."""

    def reset(self) -> None:
        """Сбросить внутреннее состояние (например, интегральную часть)."""


class ZeroController(Controller):
    """u = 0. Свободное движение системы -- база для проверки динамики."""

    def __init__(self, n_action: int = 1):
        self.n_action = int(n_action)

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        return np.zeros(self.n_action)


class ConstantController(Controller):
    """u = const. Нужен, в частности, для проверки первых интегралов:
    у колёсного маятника при постоянном u сохраняется H (см. Key_Formulas §3).
    """

    def __init__(self, u):
        self.u = np.atleast_1d(np.asarray(u, dtype=float))

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        return self.u.copy()


class LinearFeedbackController(Controller):
    """Линейная обратная связь по состоянию: u = -K (x_hat - x_ref).

    Одна формула покрывает и ПД-регулятор (K = [kp, kd] для системы
    (угол, скорость)), и ЛКР (K из уравнения Риккати), и любое размещение
    полюсов: различается только способ получить K, а не закон управления.
    Синтез K в библиотеке сознательно отсутствует -- матрица передаётся снаружи.

    Параметры
    ---------
    K : массив (n_action, n_state)
    x_ref : массив (n_state,) | None
        Целевое состояние; None означает нулевое.
    """

    def __init__(self, K, x_ref=None):
        self.K = np.atleast_2d(np.asarray(K, dtype=float))
        self.x_ref = None if x_ref is None else np.asarray(x_ref, dtype=float)

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x_hat = np.asarray(x_hat, dtype=float)
        error = x_hat if self.x_ref is None else x_hat - self.x_ref
        return -self.K @ error
