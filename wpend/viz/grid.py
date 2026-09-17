"""Сетка начальных условий и классификация исходов.

Читает ТОЛЬКО траектории: на вход -- TrajectoryBatch, на выход -- по какому
сценарию закончилось каждое начальное условие.  Ни физики, ни pygame здесь нет,
поэтому модуль тестируется отдельно от окна.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..rollout import TrajectoryBatch

#: Коды исходов (значения выбраны так, чтобы годиться в индекс палитры).
HELD = 0          #: корпус не упал за горизонт
FELL_FORWARD = 1  #: theta вышла за +theta_fall
FELL_BACKWARD = 2 #: theta вышла за -theta_fall


@dataclass(frozen=True)
class GridSpec:
    """Прямоугольная сетка начальных условий в плоскости (theta0, dtheta0).

    Остальные компоненты состояния берутся нулевыми: колесо в нуле и покоится.
    Это не потеря общности для колёсного маятника -- ускорения не зависят ни от
    phi, ни от dphi (Key_Formulas §1.4), так что плоскость (theta, dtheta)
    исчерпывает вопрос "упадёт ли".
    """

    n: int = 21
    theta_max: float = 0.6
    dtheta_max: float = 3.0

    @property
    def thetas(self) -> np.ndarray:
        """Ось theta0, по возрастанию (слева направо на карте)."""
        return np.linspace(-self.theta_max, self.theta_max, self.n)

    @property
    def dthetas(self) -> np.ndarray:
        """Ось dtheta0, по возрастанию (снизу вверх на карте)."""
        return np.linspace(-self.dtheta_max, self.dtheta_max, self.n)

    def initial_states(self, n_state: int, i_theta: int, i_dtheta: int) -> np.ndarray:
        """Матрица начальных условий (n*n, n_state).

        Порядок строк: m = iy * n + ix, где ix -- индекс по theta,
        iy -- индекс по dtheta.  Тот же порядок использует index().
        """
        X0 = np.zeros((self.n * self.n, n_state))
        TH, DTH = np.meshgrid(self.thetas, self.dthetas, indexing="xy")
        X0[:, i_theta] = TH.ravel()
        X0[:, i_dtheta] = DTH.ravel()
        return X0

    def index(self, ix: int, iy: int) -> int:
        """Номер строки в пачке для клетки (ix по theta, iy по dtheta)."""
        return iy * self.n + ix


def classify(
    batch: TrajectoryBatch,
    i_theta: int,
    theta_fall: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Определить исход каждой траектории пачки.

    Возвращает (outcome, t_fall):
      outcome : (M,) int   -- HELD / FELL_FORWARD / FELL_BACKWARD
      t_fall  : (M,) float -- момент первого выхода за theta_fall, NaN если не упал

    "Упал" определяется как первое ПЕРЕСЕЧЕНИЕ |theta| = theta_fall, а не по
    состоянию в конце горизонта: траектория может пройти через большой угол и
    формально вернуться, но физический робот к этому моменту уже лежит.
    Направление берётся по тому, какое пересечение случилось РАНЬШЕ.

    Начальный отсчёт в проверку не входит: он не пересечение, а условие задачи.
    Пока границы карты подгоняются автоматически, разницы нет -- theta_max там
    не превышает 0.98 * theta_fall, и отсчёт 0 не срабатывает никогда. Но
    приколоченная карта (--theta-max) заходит и дальше, и тогда включённый
    отсчёт 0 красит "упал" целую полосу клеток в нулевой момент, до того как
    регулятор хоть что-то сделал. Замер: --theta-max 1.7 --dtheta-max 5.0
    --grid 81 даёт 1620 таких клеток (24.7% карты), и 24 из них к концу
    горизонта стоят вертикально, ни разу после старта не выйдя за порог.
    Ошибка была тихая: карта просто врала цветом.
    """
    theta = batch.x[:, :, i_theta]
    forward = theta >= theta_fall
    backward = theta <= -theta_fall
    forward[:, 0] = False
    backward[:, 0] = False

    big = theta.shape[1] + 1
    first_f = np.where(forward.any(axis=1), forward.argmax(axis=1), big)
    first_b = np.where(backward.any(axis=1), backward.argmax(axis=1), big)
    first = np.minimum(first_f, first_b)

    outcome = np.full(theta.shape[0], HELD, dtype=int)
    fell = first < big
    outcome[fell & (first_f <= first_b)] = FELL_FORWARD
    outcome[fell & (first_b < first_f)] = FELL_BACKWARD

    t_fall = np.full(theta.shape[0], np.nan)
    t_fall[fell] = batch.t[first[fell]]
    return outcome, t_fall
