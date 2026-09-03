"""Прогон замкнутого контура: единственная точка, где встречаются все пять слоёв.

Результат прогона -- Trajectory (t, x, u).  Всё дальнейшее (графики, анимация,
метрики, обучение) читает ТОЛЬКО её и ничего не знает про то, какие датчик,
оцениватель и регулятор её породили.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .controller import Controller
from .estimator import Estimator, PassthroughEstimator
from .integrator import Integrator
from .sensor import FullStateSensor, Sensor
from .system import System


@dataclass(frozen=True)
class Trajectory:
    """Результат прогона.

    Атрибуты
    --------
    t : (N+1,)      моменты времени t_0 .. t_N
    x : (N+1, n_x)  состояния в эти моменты
    u : (N, n_u)    управления, реально приложенные на интервалах [t_k, t_{k+1})

    Управлений на одно меньше, чем состояний, и это не небрежность: u_k -- это
    то, что действовало НА ИНТЕРВАЛЕ между x_k и x_{k+1}.  У последнего
    состояния интервала впереди нет.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray

    @property
    def n_steps(self) -> int:
        return int(self.u.shape[0])

    @property
    def dt(self) -> float:
        return float(self.t[1] - self.t[0])

    def __len__(self) -> int:
        return int(self.x.shape[0])


def rollout(
    system: System,
    controller: Controller,
    integrator: Integrator,
    x0,
    dt: float,
    n_steps: int,
    sensor: Sensor | None = None,
    estimator: Estimator | None = None,
    t0: float = 0.0,
) -> Trajectory:
    """Прогнать замкнутый контур n_steps шагов и вернуть Trajectory.

    Порядок одного шага k (единый шаг dt на всё -- и на управление, и на
    интегрирование):

        y_k     = sensor.measure(t_k, x_k)              # что видно
        xh_k    = estimator.estimate(t_k, y_k, u_{k-1}) # что думаем
        u_k     = controller.act(t_k, xh_k)             # что хотим
        u_k     = system.clip_action(u_k)               # что можем
        x_{k+1} = integrator.step(system.f, t_k, x_k, u_k, dt)

    Обратите внимание: в f подставляется истинное x_k, а не оценка.  Мир
    развивается по истинному состоянию; оценка влияет на него только через u.
    Именно это разделение делает осмысленным вопрос "насколько плохая оценка
    ломает регулятор".

    По умолчанию датчик идеальный (y = x), оцениватель -- тождественный
    (x_hat = y), то есть регулятор видит истинное состояние.
    """
    if dt <= 0:
        raise ValueError("dt должен быть положительным")
    if n_steps < 1:
        raise ValueError("n_steps должен быть >= 1")

    sensor = FullStateSensor() if sensor is None else sensor
    estimator = PassthroughEstimator() if estimator is None else estimator

    x = np.atleast_1d(np.asarray(x0, dtype=float)).copy()
    if x.shape != (system.n_state,):
        raise ValueError(
            f"x0 должен быть формы ({system.n_state},), получено {x.shape}"
        )

    sensor.reset()
    y0 = sensor.measure(t0, x)
    estimator.reset(y0)
    controller.reset()

    ts = np.empty(n_steps + 1, dtype=float)
    xs = np.empty((n_steps + 1, system.n_state), dtype=float)
    us = np.empty((n_steps, system.n_action), dtype=float)

    t = float(t0)
    u_prev = np.zeros(system.n_action)
    ts[0] = t
    xs[0] = x

    for k in range(n_steps):
        y = sensor.measure(t, x)
        x_hat = estimator.estimate(t, y, u_prev)
        u = system.clip_action(controller.act(t, x_hat))

        x = integrator.step(system.f, t, x, u, dt)
        t = t0 + (k + 1) * dt   # накопление по формуле, а не t += dt: не копит ошибку округления

        us[k] = u
        ts[k + 1] = t
        xs[k + 1] = x
        u_prev = u

    return Trajectory(t=ts, x=xs, u=us)


@dataclass(frozen=True)
class TrajectoryBatch:
    """Пачка траекторий, посчитанных одним прогоном из M начальных условий.

    Атрибуты
    --------
    t : (S,)            моменты сохранённых кадров
    x : (M, S, n_x)     состояния
    u : (M, S-1, n_u)   управления в начале каждого сохранённого интервала

    S -- число СОХРАНЁННЫХ кадров, а не шагов интегрирования: при stride > 1
    считается каждый шаг dt, а записывается каждый stride-й.  Точность счёта
    от этого не страдает, память -- уменьшается в stride раз.

    batch[m] возвращает обычную Trajectory для m-го начального условия, так что
    всё, что умеет читать Trajectory, умеет читать и элемент пачки.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, m: int) -> Trajectory:
        return Trajectory(t=self.t, x=self.x[m], u=self.u[m])

    @property
    def n_trajectories(self) -> int:
        return int(self.x.shape[0])

    @property
    def nbytes(self) -> int:
        return int(self.x.nbytes + self.u.nbytes + self.t.nbytes)


def rollout_many(
    system: System,
    controller: Controller,
    integrator: Integrator,
    X0,
    dt: float,
    n_steps: int,
    stride: int = 1,
    sensor: Sensor | None = None,
    estimator: Estimator | None = None,
    t0: float = 0.0,
) -> TrajectoryBatch:
    """Прогнать M начальных условий ОДНОВРЕМЕННО и вернуть TrajectoryBatch.

    Цикл ровно тот же, что в rollout, но состояние -- матрица (M, n_state), и
    каждый шаг это одна операция над массивом вместо M вызовов из Python.
    Для колёсного маятника на сетке 41x41 (5 с, dt = 1 мс) это 1.8 с вместо
    461 с -- ускорение в ~250 раз.  Никакой другой физики здесь нет: тест
    сравнивает результат с последовательным rollout построчно.

    Требование: все пять слоёв должны соблюдать соглашение о пачках
    (см. докстринг System.f).

    Параметры
    ---------
    X0 : (M, n_state)
        Начальные условия по строкам.
    stride : int
        Сохранять каждый stride-й кадр; n_steps должно делиться на stride.
    """
    if dt <= 0:
        raise ValueError("dt должен быть положительным")
    if n_steps < 1:
        raise ValueError("n_steps должен быть >= 1")
    if stride < 1:
        raise ValueError("stride должен быть >= 1")
    if n_steps % stride != 0:
        raise ValueError(f"n_steps ({n_steps}) должно делиться на stride ({stride})")

    sensor = FullStateSensor() if sensor is None else sensor
    estimator = PassthroughEstimator() if estimator is None else estimator

    X = np.atleast_2d(np.asarray(X0, dtype=float)).copy()
    if X.ndim != 2 or X.shape[1] != system.n_state:
        raise ValueError(
            f"X0 должен быть формы (M, {system.n_state}), получено {X.shape}"
        )
    M = X.shape[0]
    n_frames = n_steps // stride + 1

    sensor.reset()
    estimator.reset(sensor.measure(t0, X))
    controller.reset()

    ts = np.empty(n_frames, dtype=float)
    xs = np.empty((M, n_frames, system.n_state), dtype=float)
    us = np.empty((M, n_frames - 1, system.n_action), dtype=float)

    t = float(t0)
    U_prev = np.zeros((M, system.n_action))

    for k in range(n_steps):
        Y = sensor.measure(t, X)
        X_hat = estimator.estimate(t, Y, U_prev)
        U = system.clip_action(controller.act(t, X_hat))

        if k % stride == 0:
            j = k // stride
            ts[j] = t
            xs[:, j] = X
            us[:, j] = U

        X = integrator.step(system.f, t, X, U, dt)
        t = t0 + (k + 1) * dt
        U_prev = U

    ts[-1] = t
    xs[:, -1] = X
    return TrajectoryBatch(t=ts, x=xs, u=us)
