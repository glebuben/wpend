"""Условия мира как модели (ER-025, docs/accel_kalman.md §4.3, §6).

Решение Глеба 30.09: шум процесса и крепление машины -- это НЕ новые входы
слоёв, а другие модели того же колёсного маятника.  Какую модель отдать в
rollout -- такой и мир:

    WheeledPendulum()                  -- как было, без возмущений;
    DisturbedWheeledPendulum(...)      -- на машину действует случайная
                                          внешняя сила («погода прогона»);
    StandFixture(model)                -- машина жёстко закреплена на
                                          подставке (калибровка датчика).

System, rollout, Sensor, Integrator при этом не меняются.  Датчик видит ту
же силу автоматически: specific_force зовёт self.f.
"""

from __future__ import annotations

import copy

import numpy as np

from .wheeled_pendulum import WheeledPendulum, WheeledPendulumParams


class DisturbedWheeledPendulum(WheeledPendulum):
    """Колёсный маятник под случайной внешней обобщённой силой w(t).

    ФИЗИКА (§6.1).  К моменту мотора (u, -u) добавляется пара
    w = (w_psi, w_theta): момент на корпус (толчок, ветер) и момент на колесо
    (дорога).  Ускорения получают добавку M(psi)^-1 w:

        dx/dt = f(x, u) + [0; 0; M(psi)^-1 w(t)].

    Момент мотора по-прежнему известен точно; предел u_max на w не действует
    -- это не мотор.

    ПРОЦЕСС (§6.2).  Каждый канал -- Гаусс--Марков с установившимся СКО
    sigma_w [Н·м] и временем корреляции tau_w [с].  Узлы пути
    t_i = t0 + i*dt, точная дискретизация

        w_{i+1} = phi w_i + sigma_w sqrt(1 - phi^2) xi_i,  phi = exp(-dt/tau_w),
        w_0 ~ N(0, sigma_w^2)   (сразу стационарный).

    Белый шум силы сюда сознательно не взят: акселерометр меряет силу
    напрямую, и белая сила заметной величины топит его отсчёт (§6.5).

    ПУТЬ -- ФУНКЦИЯ ВРЕМЕНИ (§6.4).  Между узлами путь продолжается
    ЛИНЕЙНО.  Три следствия:
      * у RK4 нет ловушки с индексом: в t + dt ступенька уже сменила бы
        номер, а непрерывный путь в узле однозначен;
      * если узлы пути совпадают с узлами прогона (dt модели = dt прогона,
        общий t0), внутри шага сила линейна по t, и RK4 сохраняет порядок 4
        (оракул -- tests/test_accel_kalman.py).  При несовпадении узлов
        излом силы попадает внутрь шага, и порядок падает до 3 (замер: 3.00;
        не до 2 -- изломов на отрезке фиксированное число, оно не растёт при
        уменьшении шага).  Следите, чтобы dt совпадал;
      * провал дисперсии посередине интервала -- (1+phi)/2, при dt = 1 мс и
        tau_w = 0.2 с это 0.25 %.
    По теореме Вонга--Закаи гладкие приближения сходятся к решению
    Стратоновича, а оно здесь совпадает с решением Ито (§6.3): Gamma(psi)
    зависит только от psi, а шум входит только в скорости.

    ПРАВИЛО 1 ARCHITECTURE.md СОБЛЮДЕНО.  f -- чистая функция (t, x, u):
    путь целиком определён seed-ом, а внутренний массив узлов -- лишь
    кэш уже посчитанных значений.  Узлы каждой строки считаются кусками по
    _CHUNK, и шум куска c строки m берётся из SeedSequence(entropy,
    spawn_key=(m, c)).  Поэтому значение в узле не зависит ни от порядка
    вызовов, ни от размера пачки: строка m пачки получает тот же путь, что
    строка m любой другой пачки, одиночный прогон -- путь строки 0.

    ЛОВУШКА.  Оценивателю эту модель отдавать нельзя: он узнал бы погоду
    заранее.  Фильтр прогнозирует по номинальной WheeledPendulum.

    Параметры
    ---------
    params : WheeledPendulumParams | None
        Физика машины, как у WheeledPendulum.
    sigma_w : float | (2,)
        Установившееся СКО (w_psi, w_theta) [Н·м].  0 -- канал выключен;
        при нулях в обоих каналах прогон совпадает с WheeledPendulum бит в бит.
    tau_w : float | (2,)
        Время корреляции [с], > 0.
    dt : float
        Шаг узлов пути -- должен совпадать с шагом прогона.
    t0 : float
        Время первого узла -- должно совпадать с t0 прогона.
    seed : int | None
        Зерно пути.  None -- случайное, но фиксируется при создании модели,
        так что два прогона на одной модели видят одну и ту же погоду.
    row_offset : int
        Сдвиг номера строки: строка k пачки получает путь строки
        k + row_offset.  Нужен окну исследователя (решение Глеба 30.09): карта
        считается пачкой, а одну клетку m окно пересчитывает отдельно
        (второй регулятор, второй оцениватель), и без сдвига она получила бы
        ветер строки 0, а не своей.  Удобнее -- with_row_offset(m).
    """

    _CHUNK = 1024

    def __init__(self, params: WheeledPendulumParams | None = None, *,
                 sigma_w=(0.3, 0.3), tau_w=0.2, dt: float, t0: float = 0.0,
                 seed: int | None = None, row_offset: int = 0, **kwargs):
        super().__init__(params, **kwargs)
        if dt <= 0:
            raise ValueError("dt должен быть положительным")
        self.sigma_w = np.broadcast_to(np.asarray(sigma_w, dtype=float), (2,)).copy()
        self.tau_w = np.broadcast_to(np.asarray(tau_w, dtype=float), (2,)).copy()
        if np.any(self.sigma_w < 0):
            raise ValueError("sigma_w не может быть отрицательным")
        if np.any(self.tau_w <= 0):
            raise ValueError("tau_w должно быть положительным")
        self.dt = float(dt)
        self.t0 = float(t0)
        self.seed = seed
        self._entropy = np.random.SeedSequence(seed).entropy
        if row_offset < 0:
            raise ValueError("row_offset не может быть отрицательным")
        self.row_offset = int(row_offset)
        self._phi = np.exp(-self.dt / self.tau_w)
        self._kick = self.sigma_w * np.sqrt(-np.expm1(-2.0 * self.dt / self.tau_w))
        self._silent = bool(np.all(self.sigma_w == 0.0))
        self._nodes = np.zeros((0, 0, 2))      # кэш: (строки, узлы, 2)

    def with_row_offset(self, m: int) -> "DisturbedWheeledPendulum":
        """Та же погода (seed, физика, sigma_w, tau_w), но строка 0 -- это
        строка m исходной пачки.  Кэш узлов у копии свой."""
        twin = copy.copy(self)
        twin.row_offset = self.row_offset + int(m)
        twin._nodes = np.zeros((0, 0, 2))
        return twin

    # --- путь возмущения ----------------------------------------------------

    def _chunk_noise(self, row: int, chunk: int) -> np.ndarray:
        ss = np.random.SeedSequence(self._entropy,
                                    spawn_key=(row + self.row_offset, chunk))
        return np.random.default_rng(ss).standard_normal((self._CHUNK, 2))

    def _grow(self, rows: range, k_from: int, k_to: int, w_prev) -> np.ndarray:
        """Узлы k_from..k_to-1 для строк rows (k_from, k_to кратны _CHUNK).
        w_prev -- узел k_from-1 этих строк или None (начало пути).
        Рекурсия идёт по узлам, векторно по строкам."""
        rows = list(rows)
        out = np.empty((len(rows), k_to - k_from, 2))
        w = None if w_prev is None else np.array(w_prev, dtype=float)
        for c in range(k_from // self._CHUNK, k_to // self._CHUNK):
            xi = np.stack([self._chunk_noise(m, c) for m in rows])     # (R, CHUNK, 2)
            base = c * self._CHUNK - k_from
            for j in range(self._CHUNK):
                if w is None:
                    w = self.sigma_w * xi[:, 0]          # стационарный старт
                else:
                    w = self._phi * w + self._kick * xi[:, j]
                out[:, base + j] = w
        return out

    def _ensure(self, n_rows: int, n_nodes: int) -> None:
        """Достроить кэш до n_rows строк и n_nodes узлов, кусками целиком.
        Новые строки считаются с узла 0, старые -- продолжаются: значение в
        узле от порядка вызовов не зависит (см. докстринг класса)."""
        R, K = self._nodes.shape[:2]
        if n_rows <= R and n_nodes <= K:
            return
        K_new = max(K, -(-n_nodes // self._CHUNK) * self._CHUNK)
        R_new = max(R, n_rows)
        nodes = np.empty((R_new, K_new, 2))
        nodes[:R, :K] = self._nodes
        if R > 0 and K_new > K:
            nodes[:R, K:] = self._grow(range(R), K, K_new, self._nodes[:, K - 1])
        if R_new > R:
            nodes[R:] = self._grow(range(R, R_new), 0, K_new, None)
        self._nodes = nodes

    def disturbance(self, t, rows: int | None = None) -> np.ndarray:
        """Истинное w(t).  rows=None -- путь строки 0, форма t.shape + (2,);
        rows=M -- пути строк 0..M-1, форма (M,) + t.shape + (2,).

        Это то, с чем сравнивается оценка фильтра (Trajectory.w_hat):
        model.disturbance(traj.t)."""
        n_rows = 1 if rows is None else int(rows)
        if np.ndim(t) == 0:
            # быстрый путь для скаляра: f зовёт это 4 раза на шаг RK4
            s = (float(t) - self.t0) / self.dt
            if s < -1e-9:
                raise ValueError(f"возмущение определено с t0 = {self.t0}, запрошено t < t0")
            i = int(s) if s > 0.0 else 0
            frac = s - i if s > 0.0 else 0.0
            if n_rows > self._nodes.shape[0] or i + 2 > self._nodes.shape[1]:
                self._ensure(n_rows, i + 2)
            nodes = self._nodes
            w = nodes[:n_rows, i] if frac == 0.0 else \
                (1.0 - frac) * nodes[:n_rows, i] + frac * nodes[:n_rows, i + 1]
            return w[0] if rows is None else w
        t = np.asarray(t, dtype=float)
        s = (t - self.t0) / self.dt
        if np.any(s < -1e-9):
            raise ValueError(f"возмущение определено с t0 = {self.t0}, запрошено t < t0")
        s = np.maximum(s, 0.0)
        i = np.floor(s).astype(int)
        frac = (s - i)[..., None]
        self._ensure(n_rows, int(np.max(i)) + 2)
        w = ((1.0 - frac) * self._nodes[:n_rows, i] + frac * self._nodes[:n_rows, i + 1])
        return w[0] if rows is None else w

    # --- динамика ---------------------------------------------------------------

    def f(self, t, x, u):
        dx = super().f(t, x, u)
        if self._silent:
            return dx
        x = np.asarray(x, dtype=float)
        prefix = x.shape[:-1]
        n_rows = int(np.prod(prefix)) if prefix else 1
        w = self.disturbance(t, rows=n_rows).reshape(prefix + (2,))
        qdd = self.accel_from_force(x[..., 0], w)
        out = dx.copy()
        out[..., 2:] += qdd
        return out


class StandFixture(WheeledPendulum):
    """Машина на подставке: корпус и колесо жёстко закреплены (§4.3).

    f = 0: состояние заморожено, ускорений нет.  Поэтому унаследованный
    specific_force (он зовёт self.f) выдаёт СТАТИЧЕСКОЕ показание

        (a_x, a_z) = (-g sin psi, g cos psi)     при dpsi = 0,

    а не свободное падение, которое увидел бы датчик на обычной модели при
    psi != 0.  Нужна только калибровке (calibrate_on_stand): в rollout её
    не отдают.
    """

    def __init__(self, model: WheeledPendulum):
        super().__init__(model.p)

    def f(self, t, x, u):
        x = np.asarray(x, dtype=float)
        u = np.asarray(u, dtype=float)
        prefix = np.broadcast_shapes(x.shape[:-1], u.shape[:-1])
        return np.zeros(prefix + (x.shape[-1],))
