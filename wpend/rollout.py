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
    t : (N+1,)          моменты времени t_0 .. t_N
    x : (N+1, n_x)      ИСТИННЫЕ состояния в эти моменты
    u : (N, n_u)        управления, реально приложенные на интервалах [t_k, t_{k+1})
    x_hat : (N, n_x) | None   ОЦЕНКИ состояния, по которым выбирались управления

    Управлений на одно меньше, чем состояний, и это не небрежность: u_k -- это
    то, что действовало НА ИНТЕРВАЛЕ между x_k и x_{k+1}.  У последнего
    состояния интервала впереди нет.

    x_hat выровнена по u, а НЕ по x: x_hat[k] -- это то, что оцениватель думал
    в момент t_k, и именно из неё регулятор получил u_k.  Поэтому длина та же,
    что у u.  Пара (x[k], x_hat[k]) -- истина и оценка в один и тот же момент.

    Зачем оценка лежит здесь.  Правило 6 говорит, что всё после прогона читает
    только Trajectory.  Значит окно, которое хочет показать работу фильтра,
    обязано взять оценку отсюда: восстанавливать её повторным прогоном датчика
    оно не может, не узнав про датчики, а это и было бы нарушением правила 6.
    Поле не расширяет ответственность структуры -- Trajectory как была записью
    прогона, так и осталась, просто запись стала полной.

    None означает "не записывали".  У rollout запись включена по умолчанию
    (5000 шагов x 4 компоненты -- 160 КБ, шум на фоне всего остального), у
    rollout_many -- ВЫКЛЮЧЕНА: там оценка добавляет +80 % к памяти пачки
    (замер: сетка 41x41 при stride=10 -- 60.6 МБ вместо 33.7, 81x81 -- 236 МБ
    вместо 131).  Асимметрия осознанная и стоит одного флага.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    x_hat: np.ndarray | None = None

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
    record_estimate: bool = True,
) -> Trajectory:
    """Прогнать замкнутый контур n_steps шагов и вернуть Trajectory.

    Порядок одного шага k (единый шаг dt на всё -- и на управление, и на
    интегрирование):

        y_k     = sensor.measure(t_k, x_k, u_{k-1})     # что видно
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

    record_estimate=True (по умолчанию) кладёт оценки в traj.x_hat.  Для
    одиночного прогона это дёшево, а без них нельзя ни нарисовать работу
    фильтра, ни померить ошибку оценки -- на железе такой величины не
    существует, и симуляция ровно этим и ценна.
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
    y0 = sensor.measure(t0, x, np.zeros(system.n_action))
    estimator.reset(y0)
    controller.reset()

    ts = np.empty(n_steps + 1, dtype=float)
    xs = np.empty((n_steps + 1, system.n_state), dtype=float)
    us = np.empty((n_steps, system.n_action), dtype=float)
    xhs = np.empty((n_steps, system.n_state), dtype=float) if record_estimate else None

    t = float(t0)
    u_prev = np.zeros(system.n_action)
    ts[0] = t
    xs[0] = x

    for k in range(n_steps):
        y = sensor.measure(t, x, u_prev)
        x_hat = estimator.estimate(t, y, u_prev)
        if xhs is not None:
            x_hat_arr = np.asarray(x_hat, dtype=float)
            if x_hat_arr.shape != (system.n_state,):
                raise ValueError(
                    f"{type(estimator).__name__} вернул оценку формы "
                    f"{x_hat_arr.shape}, а регулятору нужно состояние "
                    f"({system.n_state},).  Записать такую в Trajectory.x_hat "
                    "нельзя; если оцениватель намеренно неполон, гоняйте с "
                    "record_estimate=False."
                )
            xhs[k] = x_hat_arr
        u = system.clip_action(controller.act(t, x_hat))

        x = integrator.step(system.f, t, x, u, dt)
        t = t0 + (k + 1) * dt   # накопление по формуле, а не t += dt: не копит ошибку округления

        us[k] = u
        ts[k + 1] = t
        xs[k + 1] = x
        u_prev = u

    return Trajectory(t=ts, x=xs, u=us, x_hat=xhs)


@dataclass(frozen=True)
class TrajectoryBatch:
    """Пачка траекторий, посчитанных одним прогоном из M начальных условий.

    Атрибуты
    --------
    t : (S,)                     моменты сохранённых кадров
    x : (M, S, n_x)              истинные состояния
    u : (M, S-1, n_u)            управления в начале каждого сохранённого интервала
    x_hat : (M, S-1, n_x) | None оценки, по которым эти управления выбраны

    S -- число СОХРАНЁННЫХ кадров, а не шагов интегрирования: при stride > 1
    считается каждый шаг dt, а записывается каждый stride-й.  Точность счёта
    от этого не страдает, память -- уменьшается в stride раз.

    batch[m] возвращает обычную Trajectory для m-го начального условия, так что
    всё, что умеет читать Trajectory, умеет читать и элемент пачки.

    x_hat по умолчанию НЕ записывается (record_estimate=False): она добавляет
    +80 % к памяти пачки, а сетка нужна почти всегда ради исхода, а не ради
    оценки.  Замер при stride=10 и 5000 шагах: 41x41 -- 60.6 МБ вместо 33.7,
    81x81 -- 236 МБ вместо 131.  Включать осмысленно, когда сетка строится
    именно про фильтр.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    x_hat: np.ndarray | None = None

    def __len__(self) -> int:
        return int(self.x.shape[0])

    def __getitem__(self, m: int) -> Trajectory:
        return Trajectory(t=self.t, x=self.x[m], u=self.u[m],
                          x_hat=None if self.x_hat is None else self.x_hat[m])

    @property
    def n_trajectories(self) -> int:
        return int(self.x.shape[0])

    @property
    def nbytes(self) -> int:
        extra = 0 if self.x_hat is None else self.x_hat.nbytes
        return int(self.x.nbytes + self.u.nbytes + self.t.nbytes + extra)


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
    record_estimate: bool = False,
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
    estimator.reset(sensor.measure(t0, X, np.zeros((X.shape[0], system.n_action))))
    controller.reset()

    ts = np.empty(n_frames, dtype=float)
    xs = np.empty((M, n_frames, system.n_state), dtype=float)
    us = np.empty((M, n_frames - 1, system.n_action), dtype=float)
    # ВЫКЛЮЧЕНО по умолчанию: +80 % памяти (см. докстринг TrajectoryBatch)
    xhs = (np.empty((M, n_frames - 1, system.n_state), dtype=float)
           if record_estimate else None)

    t = float(t0)
    U_prev = np.zeros((M, system.n_action))

    for k in range(n_steps):
        Y = sensor.measure(t, X, U_prev)
        X_hat = estimator.estimate(t, Y, U_prev)
        U = system.clip_action(controller.act(t, X_hat))

        if k % stride == 0:
            j = k // stride
            ts[j] = t
            xs[:, j] = X
            us[:, j] = U
            if xhs is not None:
                xhs[:, j] = X_hat

        X = integrator.step(system.f, t, X, U, dt)
        t = t0 + (k + 1) * dt
        U_prev = U

    ts[-1] = t
    xs[:, -1] = X
    return TrajectoryBatch(t=ts, x=xs, u=us, x_hat=xhs)
