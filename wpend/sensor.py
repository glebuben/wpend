"""Слой 2 из 5: ДАТЧИК.

Датчик превращает истинное состояние x в измерение y -- то, что реально
доступно бортовому компьютеру.  Это единственное место, где живёт разрыв
между "что есть" и "что видно": шум, неполнота измерений, дискретизация.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Sensor(ABC):
    """Отображение измерения y = h(t, x, u_prev)."""

    @abstractmethod
    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
        """Вернуть измерение y в момент t при истинном состоянии x.

        Параметры
        ---------
        t : float
            Текущее время.
        x : np.ndarray
            Истинное состояние (n_state,) или пачка (M, n_state).
        u_prev : np.ndarray | None
            Управление, приложенное на ПРЕДЫДУЩЕМ шаге (нули на первом).
            Симметрично третьему аргументу Estimator.estimate (см. A2 в
            PROPOSALS.md) и нужно по той же причине -- только для датчика она
            физическая, а не алгоритмическая.  Акселерометр меряет не
            положение, а УДЕЛЬНУЮ СИЛУ, в которую входит ускорение корпуса,
            то есть приложенный момент.  В момент t_k на корпус действовал
            момент, поданный на интервале [t_{k-1}, t_k), то есть u_{k-1}:
            rollout зовёт measure ДО регулятора, и u_k ещё не существует.
            Датчик без акселерометра аргумент просто игнорирует.
        """

    def reset(self) -> None:
        """Сбросить внутреннее состояние (генератор шума, смещение нуля).

        Вызывается rollout-ом один раз перед началом прогона.
        """


class FullStateSensor(Sensor):
    """Идеальный датчик: y = x.

    Полезен как база отсчёта: с ним оценка совпадает с истиной, и любое
    расхождение траекторий объясняется только регулятором или интегратором.
    """

    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
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

    Замечание 2.  sigma здесь -- СКО на шаг, а не спектральная плотность: от
    dt оно не зависит.  Это удобная игрушка, а не модель железа; физическую
    модель с правильной зависимостью от dt см. в IMUSensor.
    """

    def __init__(self, sigma, seed: int | None = None):
        self.sigma = np.atleast_1d(np.asarray(sigma, dtype=float))
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        return x + self.sigma * self._rng.standard_normal(x.shape)


class IMUSensor(Sensor):
    """Инерциальный датчик: белый шум + смещение нуля + его уход.

    Отличие от GaussianNoiseSensor не в количестве шума, а в том, что ошибка
    НЕ БЕЛАЯ.  Три компоненты, и только первая усредняется:

        измеренная скорость = w(t) + b(t) + n(t)

      * n(t)  -- белый шум, гауссов и стационарный.  Задаётся спектральной
        ПЛОТНОСТЬЮ sigma_g [рад/с/sqrt(Гц)]; СКО одного отсчёта равно
        sigma_g / sqrt(dt).  То есть при уменьшении шага шум РАСТЁТ -- это не
        опечатка, а определение: белый шум имеет постоянную плотность, и чем
        короче окно усреднения, тем хуже отдельный отсчёт.
      * b(0)  -- смещение включения.  Гауссово ПО АНСАМБЛЮ прогонов
        (СКО b0_g), но внутри одного прогона -- константа, не случайная
        величина вовсе.  Ошибка угла от него растёт ЛИНЕЙНО: b*t.
      * b(t)  -- уход смещения.  Нестационарный процесс: дисперсия растёт со
        временем, усреднением не убирается.  Именно из-за него нельзя
        интегрировать гироскоп, и именно его лечит расширенное состояние в
        ER-008.

    Модель ухода -- марковский процесс первого порядка

        db/dt = -b/tau + sigma_b * w(t),
        b_k   = phi*b_{k-1} + sigma_b*sqrt(tau(1-phi^2)/2) * w_k,
        phi   = exp(-dt/tau),

    при tau = inf (по умолчанию) вырождающийся ТОЧНО в случайное блуждание
    b_k = b_{k-1} + sigma_b*sqrt(dt)*w_k -- ту самую модель, которую пишут
    Kalibr и вся литература по визуально-инерциальной одометрии.  Проверка
    предела: tau(1-exp(-2dt/tau))/2 -> dt при tau -> inf.

    Почему блуждание, если в железе меряется фликкер.  Реальная кривая Аллана
    даёт ПОЛКУ (спектр 1/f), а не наклон +1/2 (спектр 1/f^2).  Но у фликкера
    нет конечномерной реализации в пространстве состояний: спектр 1/f требует
    передаточной функции порядка f^(-1/2), а это не рациональная функция от s,
    то есть ни интегрирование, ни любая их конечная комбинация.  В состояние
    фильтра его засунуть НЕЛЬЗЯ -- доказуемо.  Блуждание же переоценивает рост
    ошибки (дисперсия ~ t против ~ ln t), то есть ошибается в безопасную
    сторону.  Конечное tau даёт полку и нужно ровно для того, чтобы увидеть её
    на кривой Аллана; на горизонте 5-20 с при tau ~ 100 с разницы нет.
    Подробности и ссылки -- docs/imu_noise.md.

    Два режима.
      * mode="state" (по умолчанию) -- y имеет форму состояния: шум и смещение
        кладутся на theta и dtheta, колесо (phi, dphi) проходит без ошибки,
        как от идеального энкодера.  Контур замыкается через
        PassthroughEstimator, и пороги срыва меряются уже сейчас.
        Ошибка угла получается из ошибки акселерометра статической
        линеаризацией theta_acc = atan2(-f_x, f_z): d(theta) = -d(f_x)/g.
      * mode="imu" -- y = (a_x, a_z, omega), настоящие показания в м/с^2 и
        рад/с.  Это НЕ состояние: PassthroughEstimator на нём бессмыслен,
        достать theta из показаний -- работа оценивателя (ER-008).  Удельную
        силу считает модель (System.specific_force), а не датчик: это физика.

    Параметры (значения по умолчанию -- круглые порядки величин для дешёвого
    MEMS, не паспорт конкретной микросхемы; см. таблицу в docs/imu_noise.md)
    ---------
    system : System
        Нужна для g, для индексов theta/dtheta в векторе состояния и -- в
        режиме "imu" -- для удельной силы.
    dt : float
        Шаг дискретизации датчика [с].  ЯВНЫЙ параметр, а не выведенный из t:
        от него зависят и шум (1/sqrt(dt)), и уход (sqrt(dt)), и ошибка в нём
        тихо меняет обе модели.  measure сверяет его с реальной разностью
        времён и падает при расхождении.
    sigma_g, sigma_a : float
        Плотность белого шума [рад/с/sqrt(Гц)], [м/с^2/sqrt(Гц)].
    sigma_bg, sigma_ba : float
        Интенсивность ухода смещения [рад/с^2/sqrt(Гц)], [м/с^3/sqrt(Гц)].
    b0_g, b0_a : float
        СКО смещения ВКЛЮЧЕНИЯ по ансамблю прогонов [рад/с], [м/с^2].
    tau_g, tau_a : float
        Время корреляции ухода [с]; inf -- случайное блуждание.
    d : float
        Вынос датчика от оси колеса вдоль корпуса [м].  Нужен только в режиме
        "imu".
    seed : int | None
        Зерно.  reset() возвращает генератор в начало, поэтому два прогона с
        одним seed дают одинаковое смещение включения и одинаковый шум.
    """

    _N_BIAS = 3   # [b_ax, b_az, b_g] -- одно состояние ошибки на оба режима

    def __init__(
        self,
        system,
        *,
        dt: float,
        mode: str = "state",
        sigma_g: float = 1e-4,
        sigma_a: float = 1e-3,
        sigma_bg: float = 1e-5,
        sigma_ba: float = 1e-4,
        b0_g: float = 1e-2,
        b0_a: float = 1e-1,
        tau_g: float = np.inf,
        tau_a: float = np.inf,
        d: float = 0.20,
        seed: int | None = None,
    ):
        if mode not in ("state", "imu"):
            raise ValueError(f"mode должен быть 'state' или 'imu', получено {mode!r}")
        if dt <= 0:
            raise ValueError("dt должен быть положительным")
        if tau_g <= 0 or tau_a <= 0:
            raise ValueError("время корреляции tau должно быть положительным (inf -- блуждание)")

        self.system = system
        self.mode = mode
        self.dt = float(dt)
        self.d = float(d)
        self.seed = seed

        # порядок компонент состояния ошибки: [a_x, a_z, gyro]
        self.sigma_w = np.array([sigma_a, sigma_a, sigma_g], dtype=float)
        self.sigma_b = np.array([sigma_ba, sigma_ba, sigma_bg], dtype=float)
        self.b0 = np.array([b0_a, b0_a, b0_g], dtype=float)
        self.tau = np.array([tau_a, tau_a, tau_g], dtype=float)

        names = tuple(system.state_names)
        if "theta" not in names or "dtheta" not in names:
            raise ValueError(
                f"IMUSensor не понимает состояние {names}: нужны компоненты "
                "'theta' и 'dtheta' (гироскоп меряет dtheta, акселерометр -- наклон)"
            )
        if mode == "imu" and not hasattr(system, "specific_force"):
            raise ValueError(
                f"{type(system).__name__} не умеет specific_force: режим 'imu' "
                "требует модели, знающей, где стоит датчик и что он чувствует"
            )
        self.i_theta = names.index("theta")
        self.i_dtheta = names.index("dtheta")
        self.g = float(getattr(system.p, "g", 9.8))

        self._rng = np.random.default_rng(seed)
        self._b = None        # состояние ошибки, форма prefix + (3,)
        self._t_prev = None

    # --- жизненный цикл ---------------------------------------------------

    def reset(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._b = None
        self._t_prev = None

    # --- медленное состояние ошибки ---------------------------------------

    def _advance_bias(self, prefix: tuple[int, ...], t: float) -> np.ndarray:
        """Продвинуть смещение до момента t и вернуть его.

        Первый вызов разыгрывает смещение ВКЛЮЧЕНИЯ.  Форма берётся из формы
        состояния: у пачки (M, n) каждая строка -- отдельный прогон, и своё
        смещение включения у неё быть ОБЯЗАНО, иначе "20 реализаций шума на
        клетку" окажутся одной реализацией, повторённой 20 раз.
        """
        shape = prefix + (self._N_BIAS,)
        if self._b is None or self._b.shape != shape:
            self._b = self.b0 * self._rng.standard_normal(shape)
            self._t_prev = t
            return self._b

        dt = t - self._t_prev
        if dt <= 0:                      # повторный вызов в тот же момент
            return self._b
        if not np.isclose(dt, self.dt, rtol=1e-6, atol=0.0):
            raise ValueError(
                f"IMUSensor: шаг по времени {dt!r} не совпадает с dt={self.dt!r}, "
                "заданным датчику.  И шум (1/sqrt(dt)), и уход (sqrt(dt)) зависят "
                "от шага -- расхождение молча испортило бы обе модели."
            )
        tau_safe = np.where(np.isinf(self.tau), 1.0, self.tau)
        phi = np.exp(-dt / self.tau)                     # при tau=inf ровно 1.0
        var = np.where(
            np.isinf(self.tau),
            dt,                                          # предел блуждания
            -tau_safe * np.expm1(-2.0 * dt / tau_safe) / 2.0,   # expm1: без потери точности при tau >> dt
        )
        self._b = phi * self._b + self.sigma_b * np.sqrt(var) * self._rng.standard_normal(shape)
        self._t_prev = t
        return self._b

    def _white(self, prefix: tuple[int, ...]) -> np.ndarray:
        """Белый шум одного отсчёта: плотность / sqrt(dt)."""
        return (self.sigma_w / np.sqrt(self.dt)) * self._rng.standard_normal(
            prefix + (self._N_BIAS,)
        )

    # --- измерение --------------------------------------------------------

    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        prefix = x.shape[:-1]
        e = self._advance_bias(prefix, float(t)) + self._white(prefix)

        if self.mode == "state":
            y = x.copy()
            # гироскоп меряет dtheta напрямую
            y[..., self.i_dtheta] = x[..., self.i_dtheta] + e[..., 2]
            # акселерометр даёт наклон: theta_acc = atan2(-f_x, f_z),
            # в окрестности покоя d(theta) = -d(f_x)/g
            y[..., self.i_theta] = x[..., self.i_theta] - e[..., 0] / self.g
            return y

        f_x, f_z = self.system.specific_force(t, x, u_prev, d=self.d)
        return np.stack(
            [f_x + e[..., 0], f_z + e[..., 1], x[..., self.i_dtheta] + e[..., 2]],
            axis=-1,
        )


class EncoderSensor(Sensor):
    """Моторный энкодер: относительный угол и относительная скорость.

    ЧТО ОН МЕРЯЕТ И ПОЧЕМУ ИМЕННО ЭТО.  Датчик стоит на валу мотора, между
    корпусом и колесом, и видит их ВЗАИМНОЕ вращение:

        y = ( phi - theta,  dphi - dtheta )

    а вовсе не phi.  Абсолютного поворота колеса относительно земли на валу
    не видно ниоткуда: чтобы получить phi, нужно прибавить наклон, а наклон
    даёт ИДУ.  Путать эти две величины -- обычная и дорогая ошибка: модель с
    «энкодером, меряющим phi» выглядит правдоподобно и завышает то, что
    измерение даёт на самом деле.

    ЧТО ЭТО ДАЁТ.  При одном ИДУ колесо ненаблюдаемо (rank 2 из 4).  Замер
    матрицы наблюдаемости в вертикали:

        только ИДУ                       rank 2   (ядро: phi, dphi)
        + тахометр колеса (dphi)         rank 3   (ядро: phi)
        ИДУ + этот энкодер               rank 4
        ТОЛЬКО этот энкодер, без ИДУ     rank 4

    Последняя строка не опечатка: относительное ускорение выдаёт наклон,
    потому что и ddtheta, и ddphi зависят от theta, поэтому за два
    дифференцирования восстанавливаются и theta, и dtheta.  Формально --
    да; практически такая оценка держится целиком на точности модели.

    МОДЕЛЬ ОШИБКИ -- КВАНТОВАНИЕ, А НЕ ШУМ.  У энкодера N меток на оборот, и
    угол округляется до ближайшей: шаг q = 2*pi/N.  Эта ошибка НЕ гауссова и
    не белая -- она ограничена по модулю (|e| <= q/2), детерминирована при
    данном угле и повторяется, если вернуться в то же положение.  Фильтр
    Калмана предполагает ровно противоположное, и это осознанное
    несоответствие модели, а не недосмотр (ER-009).

    СКОРОСТЬ БЕРЁТСЯ РАЗНОСТЬЮ СЧЁТЧИКА, как в железе:

        rate_k = (angle_k - angle_{k-1}) / dt

    и здесь квантование бьёт больнее всего: шаг q, делённый на dt, даёт
    ступеньку q/dt.  При 2048 метках и dt = 1 мс это 3.07 рад/с -- сравнимо
    с самой измеряемой величиной.  Это не дефект модели, а причина, по
    которой на реальных роботах контур скорости не гоняют на килогерце:
    либо больше меток, либо реже опрос, либо измерение времени между
    метками.  Увидеть это в замере полезнее, чем прочитать.

    Память здесь -- предыдущий отсчёт счётчика (правило 2 ARCHITECTURE.md
    разрешает `Sensor` медленное состояние).  На ПЕРВОМ вызове предыдущего
    отсчёта нет, и скорость возвращается нулевой: настоящий энкодер тоже не
    знает её до второго отсчёта.

    Параметры
    ---------
    system : System
        Нужна для индексов theta, phi, dtheta, dphi в векторе состояния.
    dt : float
        Шаг опроса [с].  ЯВНЫЙ: скорость считается разностью, делённой на
        него.  measure сверяет его с реальной разностью времён.
    counts_per_rev : int
        Меток на оборот.  2048 -- приличный оптический энкодер.  None или 0
        выключает квантование (идеальный отсчёт) -- нужно, чтобы отделить
        вклад квантования от всего остального.
    """

    def __init__(self, system, *, dt: float, counts_per_rev: int | None = 2048):
        if dt <= 0:
            raise ValueError("dt должен быть положительным")
        if counts_per_rev is not None and counts_per_rev < 0:
            raise ValueError("counts_per_rev не может быть отрицательным")
        names = tuple(system.state_names)
        for need in ("theta", "phi", "dtheta", "dphi"):
            if need not in names:
                raise ValueError(
                    f"EncoderSensor не понимает состояние {names}: нужны "
                    "theta, phi, dtheta, dphi"
                )
        self.i_theta = names.index("theta")
        self.i_phi = names.index("phi")
        self.i_dtheta = names.index("dtheta")
        self.i_dphi = names.index("dphi")
        self.dt = float(dt)
        self.counts_per_rev = counts_per_rev
        self.q = (0.0 if not counts_per_rev
                  else 2.0 * np.pi / float(counts_per_rev))

        self._prev_angle = None
        self._t_prev = None

    def reset(self) -> None:
        self._prev_angle = None
        self._t_prev = None

    def _quantize(self, angle):
        """Округление до ближайшей метки. q = 0 -- квантования нет."""
        if self.q == 0.0:
            return angle
        return np.round(angle / self.q) * self.q

    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
        x = np.asarray(x, dtype=float)
        angle = self._quantize(x[..., self.i_phi] - x[..., self.i_theta])

        first = (self._prev_angle is None
                 or self._t_prev is None
                 or np.shape(self._prev_angle) != np.shape(angle))
        if first:
            rate = np.zeros_like(angle)
        else:
            dt = float(t) - self._t_prev
            if dt <= 0:
                # Повторный вызов в тот же момент: счётчик не двигался.
                return np.stack([angle, np.zeros_like(angle)], axis=-1)
            if not np.isclose(dt, self.dt, rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"EncoderSensor: шаг {dt!r} не совпадает с dt={self.dt!r}. "
                    "Скорость считается разностью счётчика, делённой на шаг, "
                    "и от него зависит напрямую."
                )
            rate = (angle - self._prev_angle) / self.dt

        self._prev_angle = angle
        self._t_prev = float(t)
        return np.stack([angle, rate], axis=-1)


class StackedSensor(Sensor):
    """Несколько датчиков как один: показания склеиваются по последней оси.

    Настоящая машина несёт несколько независимых приборов, и складывать их в
    один класс значило бы смешивать разные железки.  Здесь каждый датчик
    остаётся собой, а склейка -- отдельная, тривиальная реализация того же
    интерфейса: ни нового слоя, ни нового понятия.

    Порядок датчиков -- часть контракта: оцениватель разбирает y по позициям.
    StackedSensor(imu, encoder) даёт (a_x, a_z, omega, phi-theta, dphi-dtheta).

    reset() пробрасывается всем, поэтому воспроизводимость по seed сохраняется.
    """

    def __init__(self, *sensors: Sensor):
        if not sensors:
            raise ValueError("StackedSensor нужен хотя бы один датчик")
        self.sensors = tuple(sensors)

    def reset(self) -> None:
        for s in self.sensors:
            s.reset()

    def measure(self, t: float, x: np.ndarray, u_prev: np.ndarray | None = None):
        parts = [np.atleast_1d(np.asarray(s.measure(t, x, u_prev), dtype=float))
                 for s in self.sensors]
        return np.concatenate(parts, axis=-1)
