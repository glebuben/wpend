"""Слой 1 из 5: СИСТЕМА.

Система отвечает ровно на один вопрос: "если сейчас время t, состояние x и
приложено управление u, чему равно dx/dt?".  Плюс объявляет, какие управления
физически допустимы (ограничение мотора).

Система НЕ знает про:
  - интегратор (как решается ОДУ),
  - датчики (что из состояния видно),
  - регулятор (кто выбирает u),
  - время моделирования, шаг dt, траектории.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class System(ABC):
    """Непрерывная управляемая система dx/dt = f(t, x, u)."""

    # --- размерности -----------------------------------------------------

    @property
    @abstractmethod
    def n_state(self) -> int:
        """Размерность вектора состояния x."""

    @property
    @abstractmethod
    def n_action(self) -> int:
        """Размерность вектора управления u."""

    # --- динамика --------------------------------------------------------

    @abstractmethod
    def f(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Векторное поле: возвращает dx/dt формы (n_state,).

        Реализация обязана быть чистой функцией: никакого внутреннего
        состояния, никаких побочных эффектов.  Интегратор вызывает f
        несколько раз внутри одного шага (RK4 -- четыре раза) с разными
        промежуточными x, и любая память сломала бы схему.
        """

    # --- допустимые управления -------------------------------------------

    @property
    def u_bounds(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Ограничение мотора: (u_min, u_max), каждый формы (n_action,).

        None -- управление не ограничено.  Ограничение живёт здесь, а не в
        регуляторе, потому что это свойство железа, а не алгоритма: любой
        регулятор, подключённый к этой системе, обязан ему подчиняться.
        """
        return None

    def clip_action(self, u: np.ndarray) -> np.ndarray:
        """Спроецировать u на допустимое множество и проверить форму.

        Вызывается rollout-ом перед подстановкой u в f, так что в траекторию
        записывается реально приложенное управление, а не запрошенное.
        """
        u = np.atleast_1d(np.asarray(u, dtype=float))
        if u.shape != (self.n_action,):
            raise ValueError(
                f"{type(self).__name__}: ожидалось управление формы "
                f"({self.n_action},), получено {u.shape}"
            )
        bounds = self.u_bounds
        if bounds is None:
            return u
        u_min, u_max = bounds
        return np.clip(u, u_min, u_max)

    # --- подписи (для графиков; чистая косметика) --------------------------

    @property
    def state_names(self) -> tuple[str, ...]:
        return tuple(f"x{i}" for i in range(self.n_state))

    @property
    def action_names(self) -> tuple[str, ...]:
        return tuple(f"u{i}" for i in range(self.n_action))
