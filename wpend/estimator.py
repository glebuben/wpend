"""Слой 3 из 5: ОЦЕНИВАТЕЛЬ.

Оцениватель превращает поток измерений y в оценку состояния x_hat, которую
получает регулятор.  Это единственное место, где разрешено иметь память о
прошлом (фильтр, интегратор ошибки, наблюдатель).
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np


class Estimator(ABC):
    """Отображение (история измерений) -> оценка состояния x_hat."""

    @abstractmethod
    def estimate(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        """Вернуть оценку x_hat формы (n_state,).

        Параметры
        ---------
        t : float
            Текущее время.
        y : np.ndarray
            Измерение, полученное от датчика в момент t.
        u_prev : np.ndarray
            Управление, приложенное на ПРЕДЫДУЩЕМ шаге (нули на первом шаге).
            Нужно любому наблюдателю с моделью (Люэнбергер, фильтр Калмана):
            предсказание x_hat_{k|k-1} строится по модели, в которую входит
            прошлое управление.  Оцениватель без модели его просто игнорирует.
        """

    def reset(self, y0: np.ndarray | None = None) -> None:
        """Инициализировать состояние фильтра перед прогоном."""

    @property
    def aux(self) -> dict:
        """То, что оцениватель знает сверх x_hat (оценки смещений,
        возмущения).  По умолчанию пусто.  rollout при record_estimate
        записывает aux["w_hat"], если он есть, в Trajectory.w_hat
        (ER-025, решение Глеба 30.09)."""
        return {}


class PassthroughEstimator(Estimator):
    """x_hat = y: измерение считается оценкой как есть.

    Корректно ровно тогда, когда датчик даёт полное состояние.  С шумным
    датчиком это "нулевой фильтр" -- база, относительно которой меряют выигрыш
    настоящего наблюдателя.
    """

    def estimate(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        return np.array(y, dtype=float, copy=True)


class ComplementaryEstimator(Estimator):
    """Комплементарный фильтр: гироскоп на высоких частотах, акселерометр на низких.

    Читает показания ИДУ y = (a_x, a_z, omega) -- то, что даёт IMUSensor в
    режиме "imu".  Наклон по акселерометру берётся как направление кажущейся
    вертикали, psi_acc = atan2(-a_x, a_z), после чего измерение становится
    ЛИНЕЙНЫМ по состоянию (y = psi + шум), и расширенный фильтр Калмана для
    этой задачи не нужен -- достаточно обычного.

        psi_hat_k = a*(psi_hat_{k-1} + omega_k*dt) + (1-a)*psi_acc_k,
        a = tau/(tau + dt)

    ПОЧЕМУ "комплементарный".  В непрерывном времени это
    psi_hat' = omega + k*(psi_acc - psi_hat), k = 1/tau, откуда

        psi_hat = [ s/(s+k) ]*psi  +  [ k/(s+k) ]*psi_acc
                      гироскоп             акселерометр
                    ФВЧ                  ФНЧ

    и две передаточные функции в сумме дают РОВНО единицу на всех частотах.
    Отсюда название: каналы дополняют друг друга.  Практическое следствие --
    истинный сигнал проходит без искажения и без запаздывания; фильтр не
    сглаживает угол, он лишь решает, какому датчику верить на какой частоте.
    Оракул на это тождество -- tests/test_complementary.py.

    ЧТО ОН НЕ УМЕЕТ.  Со смещением гироскопа b в установившемся режиме

        остаточная ошибка = b/k = b*tau

    Смещение ослабляется, но не исчезает: у фильтра одно состояние, и про
    существование смещения он не знает.  Уменьшать tau нельзя даром -- шум
    акселерометра проходит с полосой ~1/tau, его вклад растёт как 1/tau.
    Складывая независимые вклады, RMSE^2 ~ b^2 tau^2 + C/tau, откуда оптимум
    tau* = (C/2b^2)^(1/3).  Убирает остаток только расширенное состояние
    (ER-008.2), где смещение оценивается наравне с углом.

    Скорость наклона фильтр НЕ фильтрует: dpsi_hat = omega, то есть в
    регулятор уходит показание гироскопа вместе с его смещением.  Это тоже
    лечится расширением состояния, а не подбором tau.

    КОЛЕСО.  Из ИДУ оно не достаётся никак.  Не "плохо достаётся" -- никак:
    ddpsi и ddtheta зависят только от (psi, dpsi, u) (Key_Formulas §1.4,
    theta циклична), поэтому и a_x, a_z, omega зависят только от них.  Два
    состояния, отличающиеся лишь (theta, dtheta), дают ПОБИТОВО одинаковые
    показания -- это проверено тестом, и это сильнее, чем rank(O) = 2 из 4:
    утверждение точное, а не про линеаризацию.  Ненаблюдаемая подсистема при
    этом ещё и недетектируема: A, суженная на ядро, -- нильпотентная жорданова
    клетка (двойной интегратор), значит ошибка колеса растёт линейно даже без
    шума.

    Отсюда два режима, и оба честны по-своему:

    * wheel="zero" -- theta_hat = dtheta_hat = 0.  Законно ровно в паре с
      регулятором, у которого нули в этих столбцах (lqr_tilt, A17): выдуманное
      число тогда физически не может попасть в управление.  Это база отсчёта.
    * wheel="encoder" -- колесо приходит от МОТОРНОГО ЭНКОДЕРА, и тогда
      система наблюдаема (rank 4 против 2 при одном ИДУ).  Датчик даёт
      ОТНОСИТЕЛЬНЫЕ величины, поэтому оценка складывается:

          theta_hat  = (theta - psi)_изм  + psi_hat
          dtheta_hat = (dtheta - dpsi)_изм + omega

      Ошибка наклона протекает в колесо целиком -- это не дефект, а плата за
      то, что вал не видит землю.  Зато она ОГРАНИЧЕНА ошибкой наклона, а не
      растёт со временем, как при счислении пути.  Ждёт y из пяти чисел:
      StackedSensor(IMUSensor(mode="imu"), EncoderSensor).
    * wheel="dead_reckon" (по умолчанию) -- счисление пути: ddtheta берётся из
      ПРАВОЙ ЧАСТИ МОДЕЛИ по текущей оценке наклона и прошлому управлению и
      дважды интегрируется явным Эйлером.  Оценка обязана разойтись, и в этом
      смысл режима: ошибка ускорения колеса равна c*eps, где eps -- ошибка
      наклона, а c = d(ddtheta)/d(psi) = -350 для параметров по умолчанию.
      Значит постоянный остаток eps_0 = b*tau даёт ошибку dtheta = c*eps_0*t
      (линейно) и theta = c*eps_0*t^2/2 (квадратично), а белый шум -- рост как
      sqrt(t) и t^(3/2).  Замер показывает, что убивает оценку колеса не шум,
      а остаток фильтра, усиленный в 350 раз.

    Модель внутри оценивателя -- не нарушение слоёв: наблюдатель С МОДЕЛЬЮ
    предусмотрен пунктом A2 реестра, ради него у estimate и есть u_prev.

    Параметры
    ---------
    system : System
        Нужна для правой части при счислении пути и для индексов состояния.
    dt : float
        Шаг фильтра [с].  ЯВНЫЙ параметр: от него зависит и a = tau/(tau+dt),
        и оба интегрирования.  estimate сверяет его с реальной разностью
        времён и падает при расхождении.
    tau : float или массив (M,)
        Постоянная времени [с].  tau -> 0 -- чистый акселерометр, tau -> inf --
        чистый гироскоп (оба режима поддержаны и проверены тестами).

        МАССИВ -- своё tau на каждую строку пачки.  Это продолжение правила 7
        на ПАРАМЕТР: состояние бывает одно или пачкой, и постоянная времени
        тоже.  Нужно ради развёртки: семь значений tau -- это семь прогонов, а
        цена прогона сидит в питоновском цикле по шагам, а не в числе строк.
        Замер: семь отдельных прогонов -- 9.1 с, одна пачка из семи строк --
        около 1.3 с.  Побочно честнее: у всех строк один и тот же датчик, то
        есть одна реализация шума на все tau.
    wheel : {"dead_reckon", "zero", "encoder"}
        Откуда брать колесо.  "encoder" требует пяти чисел в y.
    accel_offset : float | None
        Вынос акселерометра от оси колеса вдоль корпуса [м] -- тот же d, что
        у IMUSensor.  None (по умолчанию) -- акселерометр берётся как есть.
        Число включает КОМПЕНСАЦИЮ КАЖУЩЕЙСЯ ВЕРТИКАЛИ, см. ниже.

    КОМПЕНСАЦИЯ КАЖУЩЕЙСЯ ВЕРТИКАЛИ.  Акселерометр меряет не наклон, а
    удельную силу (WheeledPendulum.specific_force):

        f_x = -g sin(psi) + d*ddpsi + r cos(psi)*ddtheta
        f_z =  g cos(psi) - d*dpsi^2 + r sin(psi)*ddtheta

    Всё, кроме членов с g, -- инерционная часть, и atan2(-f_x, f_z) врёт
    ровно на неё.  Размер вранья задают r и d: при r = 0.3 (параметры модели
    с 17.09) член r*ddtheta вшестеро больше, чем был при r = 0.05, и замер
    показывает, что без компенсации ЛКР с этим фильтром раскачивается при
    любом tau (0.1...3 с), а со старыми l, r держал при tau = 0.3.  Причина --
    обратная связь: ошибка наклона -> момент -> ddtheta -> ошибка наклона.

    Инерционная часть предсказуема: ddpsi и ddtheta -- правая часть модели по
    (psi, dpsi, u), а u_prev -- ровно тот момент, что действовал на
    интервале, кончающемся в t (A20).  Поэтому перед atan2 из показаний
    вычитается предсказание

        a_x' = a_x - (f_x(x_hat, u_prev) + g sin(psi_hat)),
        a_z' = a_z - (f_z(x_hat, u_prev) - g cos(psi_hat)),

    где x_hat собирается из ПРОШЛОЙ оценки наклона, текущего omega и колеса.
    Ошибка компенсации -- второго порядка: она пропорциональна ошибке
    наклона, помноженной на d(ddtheta)/d(psi), и сама проходит через ФНЧ
    фильтра.  Это наблюдатель с моделью в смысле A2, как и счисление пути.

    На первом вызове прошлой оценки нет, а у покоящейся в наклоне машины
    ddpsi != 0 уже от силы тяжести.  Поэтому начальный наклон -- корень
    psi = atan2(-a_x'(psi), a_z'(psi)), найденный методом Ньютона
    (_initial_tilt).  Простая итерация здесь не годится: замер при
    psi = 0.05 даёт psi_acc = 0.117, производная отображения по модулю
    около 1.3 > 1, и итерация расходится.  По той же причине и на каждом
    шаге берётся корень (один шаг Ньютона от прогноза гироскопом), а не
    подстановка прошлой оценки: иначе при tau = 0 фильтр -- это та самая
    расходящаяся итерация.

    Источники: Higgins, "A Comparison of Complementary and Kalman Filtering",
    IEEE Trans. AES 11 (1975) -- комплементарный фильтр как установившийся
    фильтр Калмана; Mahony, Hamel, Pflimlin, IEEE TAC 53 (2008) -- явный
    комплементарный фильтр и оценка смещения; Simon, *Optimal State
    Estimation*, гл. 5; docs/imu_noise.md §1 (кажущаяся вертикаль).
    """

    def __init__(self, system, *, dt: float, tau: float = 0.5,
                 wheel: str = "dead_reckon", accel_offset: float | None = None):
        if dt <= 0:
            raise ValueError("dt должен быть положительным")
        tau = np.asarray(tau, dtype=float)
        if tau.ndim > 1:
            raise ValueError(f"tau должно быть числом или массивом (M,), получено {tau.shape}")
        if np.any(tau < 0):
            raise ValueError("tau должно быть неотрицательным (0 -- чистый акселерометр)")
        if wheel not in ("dead_reckon", "zero", "encoder"):
            raise ValueError(
                "wheel должен быть 'dead_reckon', 'zero' или 'encoder', "
                f"получено {wheel!r}"
            )
        names = tuple(system.state_names)
        for need in ("psi", "theta", "dpsi", "dtheta"):
            if need not in names:
                raise ValueError(
                    f"ComplementaryEstimator не понимает состояние {names}: "
                    "нужны psi, theta, dpsi, dtheta"
                )
        if accel_offset is not None and not hasattr(system, "specific_force"):
            raise ValueError(
                f"{type(system).__name__} не умеет specific_force: компенсации "
                "кажущейся вертикали нечего вычитать"
            )
        self.system = system
        self.accel_offset = None if accel_offset is None else float(accel_offset)
        self.dt = float(dt)
        self.tau = tau if tau.ndim else float(tau)
        self.wheel = wheel
        self.i = tuple(names.index(n) for n in ("psi", "theta", "dpsi", "dtheta"))
        # a = tau/(tau+dt); при tau = inf это ровно 1 (чистый гироскоп), но
        # inf/(inf+dt) в плавающей точке даёт nan -- поэтому предел явно.
        # np.where вместо if: tau может быть массивом, и тогда бесконечность
        # стоит лишь в части строк.
        safe = np.where(np.isinf(tau), 1.0, tau)
        self.a = np.where(np.isinf(tau), 1.0, safe / (safe + self.dt))
        if not tau.ndim:
            self.a = float(self.a)

        self._psi = None
        self._theta = None
        self._dtheta = None
        self._t_prev = None

    # --- вспомогательное -------------------------------------------------

    @staticmethod
    def tilt_from_accelerometer(a_x, a_z):
        """Кажущаяся вертикаль: psi_acc = atan2(-a_x, a_z).

        В покое f = (-g sin psi, g cos psi), и формула возвращает ровно
        psi.  При разгоне -- систематически смещённое значение: это не шум,
        а физика (docs/imu_noise.md §1, WheeledPendulum.specific_force).
        """
        return np.arctan2(-np.asarray(a_x, dtype=float), np.asarray(a_z, dtype=float))

    # --- жизненный цикл --------------------------------------------------

    def reset(self, y0: np.ndarray | None = None) -> None:
        if y0 is None:
            self._psi = None
            self._theta = self._dtheta = None
        else:
            y0 = np.asarray(y0, dtype=float)
            u0 = np.zeros(y0.shape[:-1] + (self.system.n_action,))
            self._psi = self._initial_tilt(0.0, y0, u0)
            zeros = np.zeros_like(self._psi)
            self._theta, self._dtheta = zeros.copy(), zeros.copy()
        self._t_prev = None

    # --- оценка ----------------------------------------------------------

    def estimate(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n_need = 5 if self.wheel == "encoder" else 3
        if y.shape[-1] != n_need:
            what = ("(a_x, a_z, omega, theta-psi, dtheta-dpsi)"
                    if n_need == 5 else "(a_x, a_z, omega)")
            hint = ("Нужен StackedSensor(IMUSensor(mode='imu'), EncoderSensor)."
                    if n_need == 5 else "Датчику нужен mode='imu'.")
            raise ValueError(
                f"ComplementaryEstimator при wheel={self.wheel!r} ждёт {n_need} "
                f"числа {what}, получено {y.shape}.  {hint}"
            )
        omega = y[..., 2]
        if self.accel_offset is None or self._psi is None \
                or np.shape(self._psi) != np.shape(omega):
            psi_acc = self.tilt_from_accelerometer(y[..., 0], y[..., 1])
        else:
            # Корень, а не одна подстановка прошлой оценки: у отображения
            # psi -> compensated_tilt(psi) производная около -1.3, и при
            # tau = 0 (a = 0) простая подстановка -- расходящаяся итерация.
            # Старт -- прогноз гироскопом, одного шага Ньютона хватает.
            psi_acc = self._initial_tilt(t, y, u_prev,
                                           start=self._psi + omega * self.dt, n_iter=1)
        if np.ndim(self.a) and np.shape(self.a) != np.shape(psi_acc):
            raise ValueError(
                f"tau задано массивом {np.shape(self.a)}, а пачка измерений "
                f"имеет форму {np.shape(psi_acc)}: на каждую строку нужно "
                "ровно одно tau."
            )

        # Первый вызов: reset мог задать psi по y0, но времени он не знает,
        # поэтому _t_prev is None -- такой же признак первого шага, как и
        # отсутствие оценки.  Шага фильтра здесь нет, только инициализация.
        if (self._psi is None or self._t_prev is None
                or np.shape(self._psi) != np.shape(psi_acc)):
            if self._psi is None or np.shape(self._psi) != np.shape(psi_acc):
                if self.accel_offset is not None:
                    psi_acc = self._initial_tilt(t, y, u_prev)
                self._psi = np.array(psi_acc, dtype=float, copy=True)
                self._theta = np.zeros_like(self._psi)
                self._dtheta = np.zeros_like(self._psi)
            self._t_prev = float(t)
            return self._assemble(self._psi, omega, y)

        dt = float(t) - self._t_prev
        if dt > 0:
            if not abs(dt - self.dt) <= 1e-6 * abs(self.dt):   # = np.isclose(rtol=1e-6, atol=0), но без его накладных расходов (kalman_speed §5)
                raise ValueError(
                    f"ComplementaryEstimator: шаг {dt!r} не совпадает с dt={self.dt!r}. "
                    "От шага зависят и a = tau/(tau+dt), и оба интегрирования."
                )
            # комплементарный шаг: прогноз гироскопом + подтяжка к акселерометру
            self._psi = self.a * (self._psi + omega * self.dt) \
                + (1.0 - self.a) * psi_acc
            if self.wheel == "dead_reckon":
                self._integrate_wheel(t, omega, u_prev)
            self._t_prev = float(t)

        return self._assemble(self._psi, omega, y)

    def _compensated_tilt(self, t, y, u_prev, psi_guess):
        """atan2 по показаниям, из которых вычтена предсказанная инерционная
        часть удельной силы (вывод -- докстринг класса)."""
        if u_prev is None:
            raise ValueError(
                "accel_offset требует u_prev: инерционная часть удельной силы "
                "зависит от приложенного момента"
            )
        psi_guess = np.asarray(psi_guess, dtype=float)
        x_hat = self._assemble(psi_guess, y[..., 2], y)
        u = np.atleast_1d(np.asarray(u_prev, dtype=float))
        f_x, f_z = self.system.specific_force(t, x_hat, u, d=self.accel_offset)
        g = float(self.system.p.g)
        a_x = y[..., 0] - (f_x + g * np.sin(psi_guess))
        a_z = y[..., 1] - (f_z - g * np.cos(psi_guess))
        return self.tilt_from_accelerometer(a_x, a_z)

    def _initial_tilt(self, t, y, u_prev, start=None, n_iter: int = 8):
        """Наклон по акселерометру.  Без компенсации -- просто atan2.  С
        компенсацией -- корень h(psi) = compensated_tilt(psi) - psi = 0
        методом Ньютона с центральной разностью; старт -- сырой atan2 (первый
        вызов) или прогноз гироскопом (шаг фильтра).  Пачка решается
        построчно-независимо тем же векторным кодом."""
        y = np.asarray(y, dtype=float)
        psi = self.tilt_from_accelerometer(y[..., 0], y[..., 1])
        if self.accel_offset is None:
            return psi
        if start is not None:
            psi = np.asarray(start, dtype=float)
        h = 1e-6
        for _ in range(n_iter):
            r0 = self._compensated_tilt(t, y, u_prev, psi) - psi
            rp = self._compensated_tilt(t, y, u_prev, psi + h) - (psi + h)
            rm = self._compensated_tilt(t, y, u_prev, psi - h) - (psi - h)
            slope = (rp - rm) / (2.0 * h)
            # Вырожденный наклон (slope ~ 0) -- шаг не делаем: лучше сырой
            # atan2, чем деление на ноль.  Шаг ограничен, чтобы из дальнего
            # старта не перепрыгнуть через pi.
            ok = np.abs(slope) > 1e-9
            step = np.where(ok, -r0 / np.where(ok, slope, 1.0), 0.0)
            psi = psi + np.clip(step, -0.5, 0.5)
        return psi

    def _integrate_wheel(self, t: float, omega, u_prev) -> None:
        """Счисление пути по колесу: ddtheta из правой части модели, явный Эйлер.

        ddtheta не зависит ни от theta, ни от dtheta (§1.4), поэтому подстановка
        собственных -- заведомо неверных -- оценок колеса на результат не
        влияет: считается функция только от (psi_hat, omega, u_prev).
        Обратной связи здесь нет вовсе, и ошибка ничем не ограничена.
        """
        if u_prev is None:
            raise ValueError(
                "wheel='dead_reckon' требует u_prev: ускорение колеса зависит "
                "от приложенного момента"
            )
        x_hat = self._assemble(self._psi, omega, None)
        ddtheta = self.system.f(t, x_hat, np.atleast_1d(np.asarray(u_prev, float)))[..., self.i[3]]
        self._theta = self._theta + self._dtheta * self.dt      # явный Эйлер: theta по СТАРОЙ скорости
        self._dtheta = self._dtheta + ddtheta * self.dt

    def _assemble(self, psi, omega, y) -> np.ndarray:
        """Собрать x_hat = (psi, theta, dpsi, dtheta) в порядке state_names.

        y нужен только режиму "encoder": колесо там не хранится в состоянии
        фильтра, а каждый раз складывается из показания и наклона.
        """
        psi = np.asarray(psi, dtype=float)
        omega = np.asarray(omega, dtype=float)
        if self.wheel == "encoder" and y is not None:
            # Энкодер даёт ОТНОСИТЕЛЬНЫЕ величины -- прибавляем наклон.
            theta = y[..., 3] + psi
            dtheta = y[..., 4] + omega
        elif self.wheel == "zero" or self._theta is None:
            theta = np.zeros_like(psi)
            dtheta = np.zeros_like(psi)
        else:
            theta, dtheta = self._theta, self._dtheta
        parts = [None] * 4
        parts[self.i[0]], parts[self.i[1]] = psi, theta
        parts[self.i[2]], parts[self.i[3]] = omega, dtheta
        return np.stack(parts, axis=-1)


# ---------------------------------------------------------------------------
#  Оптимальное tau: замкнутая форма и её граница применимости
# ---------------------------------------------------------------------------
#  Не слой и не класс -- две функции, считающие числа до прогона, как lqr().


def tilt_error_model(tau, *, sigma_g=0.0, sigma_a=0.0, b=0.0, g=9.8):
    """СКО ошибки наклона комплементарного фильтра -- замкнутая форма.

    Три независимых вклада, каждый выводится из передаточных функций
    (см. докстринг ComplementaryEstimator):

        E[e^2](tau) = (b*tau)^2  +  sigma_g^2 * tau/2  +  sigma_a^2 / (2 g^2 tau)
                       смещение     шум гироскопа        шум акселерометра

    Первый растёт с tau (дольше верим гироскопу -- больше просачивается
    смещения), третий падает (дольше усредняем акселерометр).  Второй -- шум
    гироскопа, проходящий тем же путём, что и смещение.  Обрати внимание: dt
    из ответа выпадает, хотя в СКО одного отсчёта он входит; так и должно
    быть -- фильтр усредняет по времени, а не по отсчётам.

    ГДЕ ЭТО ВЕРНО.  Формула считает ошибку акселерометра БЕЛОЙ.  Проверено
    на стоящей машине по каждому вкладу отдельно: совпадение 1-3 %, а
    предсказанный минимум (0.0296 с) попал в измеренный (0.03 с).

    ГДЕ НЕВЕРНО.  В замкнутом контуре при манёвре главная ошибка
    акселерометра -- не шум, а КАЖУЩАЯСЯ ВЕРТИКАЛЬ: систематическая, вместе
    с моментом, и усреднением не убирается.  Замер: истинный оптимум там
    около 0.3 с, то есть в десять раз больше предсказанного.  Поэтому число
    из этой формулы -- нижняя оценка и опорная точка, а не рекомендация;
    рядом всегда должен стоять измеренный оптимум.

    Источники: Higgins, IEEE Trans. AES 11 (1975); docs/imu_noise.md §1.
    """
    tau = np.asarray(tau, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        var = ((b * tau) ** 2 + sigma_g ** 2 * tau / 2.0
               + sigma_a ** 2 / (2.0 * g ** 2 * tau))
    return np.sqrt(var)


def optimal_tau(*, sigma_g=0.0, sigma_a=0.0, b=0.0, g=9.8):
    """tau, минимизирующее tilt_error_model.  None, если минимума нет.

    Приравнивая производную нулю, получаем кубическое уравнение

        2 b^2 tau^3 + (sigma_g^2 / 2) tau^2 - sigma_a^2/(2 g^2) = 0

    и берём его единственный положительный корень.  При sigma_g = 0 оно
    вырождается в тот самый кубический корень

        tau* = ( sigma_a^2 / (4 g^2 b^2) )^(1/3),

    из-за которого оптимум ПОЛОГИЙ: ошибиться в tau вдвое стоит немного, и
    это хорошая новость для настройки.

    Вырожденные случаи честные, а не подогнанные: без шума акселерометра
    (sigma_a = 0) выгодно tau -> 0, без смещения и шума гироскопа --
    tau -> inf; в обоих случаях конечного минимума нет и возвращается None.
    """
    A = sigma_a ** 2 / (2.0 * g ** 2)
    if A <= 0.0:
        return None                       # акселерометру можно верить всегда
    if b == 0.0 and sigma_g == 0.0:
        return None                       # гироскопу можно верить всегда
    if b == 0.0:
        return float(np.sqrt(2.0 * A / sigma_g ** 2))
    roots = np.roots([2.0 * b ** 2, sigma_g ** 2 / 2.0, 0.0, -A])
    real = [float(r.real) for r in roots
            if abs(r.imag) < 1e-9 * max(1.0, abs(r.real)) and r.real > 0]
    return min(real) if real else None


# ---------------------------------------------------------------------------
#  ER-025: фильтр Калмана с уравнениями акселерометра и калибровка на подставке
# ---------------------------------------------------------------------------
#  Выкладки -- docs/accel_kalman.md.  Решения Глеба 30.09: a_x, a_z двумя
#  каналами напрямую; смещения ИДУ в состоянии; момент известен точно;
#  энкодер (только угол) в том же фильтре; возмущение -- по флагу.


def calibrate_on_stand(model, imu_params: dict, *, b_init, T_c: float = 1.0,
                       sigma_jig: float = 1e-2, psi_jig=None,
                       seed: int | None = None):
    """Калибровка ИДУ на подставке: вернуть (b_hat, P_bb) для фильтра.

    Функция, а не слой -- как tilt_error_model и optimal_tau: считает числа
    ДО прогона (docs/accel_kalman.md §4.3).  Вынесена из rollout по решению
    Глеба 30.09.

    Что происходит.  Машина жёстко стоит на подставке (StandFixture) в
    наклоне psi_jig -- номинально 0, на деле с ошибкой ~ N(0, sigma_jig^2),
    неизвестной фильтру.  Датчик того же чипа (b_init -- то же смещение, что
    будет в прогоне) пишет T_c секунд со СВОИМ шумом (свой seed).
    Показания усредняются в предположении psi = 0:

        b_g  = mean(omega),   b_ax = mean(a_x),
        b_az = mean(a_z) - g E[cos psi_jig] = mean(a_z) - g exp(-sigma_jig^2/2).

    Поправка exp(-sigma^2/2) снимает среднее смещение g psi^2/2 от наклона
    подставки -- остаётся только его разброс.

    Остаточная ошибка (P_bb, диагональ) -- §4.3:

        var b_g  = sigma_g^2/T_c + sigma_bg^2 T_c/3
        var b_ax = sigma_a^2/T_c + g^2 sigma_jig^2 + sigma_ba^2 T_c/3
        var b_az = sigma_a^2/T_c + g^2 Var(cos psi_jig) + sigma_ba^2 T_c/3,
                   Var(cos) = (1 - e^{-s^2})^2/2 ~ sigma_jig^4/2

    Первое слагаемое -- усреднённый белый шум, sigma^2 T_c/3 -- разность
    между смещением в конце стоянки и его средним (блуждание).  Член
    g^2 sigma_jig^2 у b_ax -- главный урок: неподвижный акселерометр не
    отличает наклон подставки от смещения, и калибровка b_ax почти ничего не
    даёт, пока подставка не точнее ~0.06 градуса.  У b_az наклон входит во
    втором порядке, g(1 - cos) ~ g psi^2/2: его среднее вычтено, дисперсия
    ~ g^2 sigma^4/2.  Корреляции каналов (corr_w) в P_bb не
    переносятся: диагональ -- сознательное упрощение.

    Параметры
    ---------
    model : WheeledPendulum
        Номинальная машина: подставка берёт её параметры.
    imu_params : dict
        Аргументы IMUSensor без system, mode, seed, b_init (dt, sigma_*,
        tau_*, d, corr_w, ...) -- ТЕ ЖЕ, что у датчика прогона.
    b_init : (3,) | (M, 3)
        Смещение включения чипа (draw_turn_on_bias).  У пачки -- своё на
        каждую строку, и калибровка тоже построчная.
    T_c : float
        Длительность стоянки [с].
    sigma_jig : float
        СКО неточности подставки [рад].
    psi_jig : массив | None
        Наклон подставки явно (для тестов); None -- розыгрыш N(0, sigma_jig^2).
    seed : int | None
        Зерно стоянки: наклон подставки и шум датчика на ней.

    Возвращает
    ----------
    b_hat : (3,) | (M, 3)
    P_bb : (3, 3)
    """
    from .models.disturbed import StandFixture
    from .sensor import IMUSensor

    if T_c <= 0:
        raise ValueError("T_c должно быть положительным")
    b_init = np.asarray(b_init, dtype=float)
    prefix = b_init.shape[:-1]
    rng = np.random.default_rng(seed)
    if psi_jig is None:
        psi_jig = sigma_jig * rng.standard_normal(prefix)
    sensor_seed = int(rng.integers(2 ** 63 - 1))
    sensor = IMUSensor(StandFixture(model), mode="imu", b_init=b_init,
                       seed=sensor_seed, **imu_params)
    n = max(1, int(round(T_c / sensor.dt)))
    x = np.zeros(prefix + (4,))
    x[..., 0] = psi_jig
    u = np.zeros(prefix + (1,))
    acc = np.zeros(prefix + (3,))
    for k in range(n):
        acc += sensor.measure(k * sensor.dt, x, u)
    mean = acc / n
    g = float(model.p.g)
    s2 = sigma_jig ** 2
    b_hat = np.stack([mean[..., 0], mean[..., 1] - g * np.exp(-s2 / 2.0), mean[..., 2]],
                     axis=-1)

    T = n * sensor.dt
    sa, sg = sensor.sigma_w[0], sensor.sigma_w[2]
    sba, sbg = sensor.sigma_b[0], sensor.sigma_b[2]
    P_bb = np.diag([
        sa ** 2 / T + g ** 2 * sigma_jig ** 2 + sba ** 2 * T / 3.0,
        sa ** 2 / T + g ** 2 * 0.5 * (-np.expm1(-s2)) ** 2 + sba ** 2 * T / 3.0,
        sg ** 2 / T + sbg ** 2 * T / 3.0,
    ])
    return b_hat, P_bb


class KalmanEstimator(Estimator):
    """Расширенный фильтр Калмана (EKF/IEKF) с уравнениями акселерометра.

    ЧТО ЧИТАЕТ.  y = (a_x, a_z, omega[, e, de]) -- IMUSensor(mode="imu"),
    при желании StackedSensor(imu, EncoderSensor).  Акселерометр входит
    ДВУМЯ каналами как есть, а не углом atan2: модель наблюдения -- сама
    удельная сила (docs/accel_kalman.md §3),

        a_x = -g s + d ddpsi + r c ddtheta + b_ax,
        a_z =  g c - d dpsi^2 + r s ddtheta + b_az,
        omega = dpsi + b_g,        e = theta - psi.

    Кажущаяся вертикаль здесь не помеха, а часть h(x, u): «оценка зависит от
    себя» из A31 -- это просто линеаризация h в текущей оценке.  Шаг Ньютона
    A31 -- частный случай этого фильтра (§9).  Скорость энкодера de НЕ
    используется: это разность того же угла, двойной счёт (§7.3).

    СОСТОЯНИЕ ФИЛЬТРА.  z = (psi, theta, dpsi, dtheta, [w_psi, w_theta,]
    b_ax, b_az, b_g).  Регулятору отдаются первые четыре.
      * Смещения ИДУ -- постоянные неизвестные (Q_b = 0); уход -- флаг
        bias_drift (только если есть кривая Аллана своего датчика, §4.2).
      * Начальная неуверенность смещений -- флаг bias_prior: None --
        режим «code» (P_bb = diag(b0_a, b0_a, b0_g)^2, ровно распределение
        датчика), (b_hat, P_bb) -- режим «calibrate» (calibrate_on_stand).
      * Возмущение w -- флаг disturbance (Гаусс--Марков, §6.2, §7.4).  При
        включённом флаге фильтр оценивает толчок, оценка -- aux["w_hat"].

    ШАГ (§8).  Прогноз: RK4 по номинальной модели с w, держащимся на шаге;
    ковариация -- F = I + J dt + J^2 dt^2/2, J -- аналитический якобиан
    (WheeledPendulum.jacobian, вариант А решения Глеба).  Коррекция: h и C по
    specific_force и specific_force_jacobian, форма Джозефа.  iterations > 1
    -- итерированный фильтр (IEKF, метод Гаусса--Ньютона).

    Момент известен точно (решение Глеба 3): D_u входит только в
    предсказание h, в ковариацию -- нет.

    ПОДГЛЯДЫВАНИЕ ЗАПРЕЩЕНО.  Модели с возмущением (DisturbedWheeledPendulum)
    фильтр не принимает: она знает погоду прогона заранее.

    Параметры
    ---------
    system : WheeledPendulum
        НОМИНАЛЬНАЯ модель.
    dt : float
        Шаг [с]; сверяется с разностью времён.
    d : float
        Вынос акселерометра [м] -- как у датчика.
    sigma_a, sigma_g, corr_w
        Плотности белого шума и корреляция каналов [a_x, a_z, gyro] -- как у
        датчика; R = D corr_w D, D = diag(sigma)/sqrt(dt) (§7.1).
    counts_per_rev : int | None
        Метки энкодера; дисперсия угла q^2/12, q = 2 pi / N.
    b0_a, b0_g : float
        Для режима «code»: СКО смещения включения, как у датчика.
    bias_prior : (b_hat, P_bb) | None
        Режим «calibrate»: результат calibrate_on_stand.
    bias_drift : dict(sigma_ba=, sigma_bg=) | None
        Уход смещения как блуждание: Q_b = diag(sigma^2) dt.
    disturbance : dict(sigma_w=, tau_w=) | None
        Модель возмущения в фильтре.  Согласованный фильтр -- те же числа,
        что у DisturbedWheeledPendulum.
    q_acc : float
        Настроечная неуверенность в ускорениях (белый шум на ddpsi, ddtheta)
        [рад/с^2/sqrt(Гц)].  0 -- модели верим полностью.
    psi0_std, dtheta0_std : float
        Начальная неуверенность наклона сверх ошибки смещения (статический
        atan2 врёт при движении) и скорости колеса (энкодер не знает её до
        второго отсчёта).
    iterations : int
        1 -- EKF, больше -- IEKF.
    diagnostics : bool
        Отдавать в aux ещё и P, y, y_pred = h(x^-, u) и диагональ S (виды окна
        исследователя, docs/kalman_views.md).  Выключено по умолчанию: P на
        каждом шаге -- это копия (M, n, n), и для карты она не нужна.
    backend : "auto" | "numpy" | "numba"
        Чем считать шаг (docs/kalman_speed.md §7).  "numba" -- скомпилированное
        ядро wpend/_kf_kernel.py, для одной клетки и для пачки (с потоками).
        Нужны пакет numba (extra `fast`), модель -- WheeledPendulum, а
        iterations == 1.  "numpy" -- прежние пути: батч и скалярный (S1).
        "auto" -- numba, если она есть и условия выполнены, иначе numpy.
        Результат один и тот же до округления.
    """

    def __init__(self, system, *, dt: float, d: float = 0.20,
                 sigma_a: float = 1e-3, sigma_g: float = 1e-4, corr_w=None,
                 counts_per_rev: int | None = 2048,
                 b0_a: float = 1e-1, b0_g: float = 1e-2, bias_prior=None,
                 bias_drift: dict | None = None, disturbance: dict | None = None,
                 q_acc: float = 0.0, psi0_std: float = 0.1,
                 dtheta0_std: float = 1.0, iterations: int = 1,
                 diagnostics: bool = False, backend: str = "auto"):
        from .models.disturbed import DisturbedWheeledPendulum
        from .sensor import _cholesky, _correlation_matrix

        if isinstance(system, DisturbedWheeledPendulum):
            raise ValueError(
                "KalmanEstimator получил DisturbedWheeledPendulum: так фильтр "
                "знал бы погоду прогона заранее.  Отдайте ему номинальную "
                "WheeledPendulum с теми же параметрами (WheeledPendulum(model.p))."
            )
        for need in ("jacobian", "specific_force_jacobian", "mass_matrix_inverse"):
            if not hasattr(system, need):
                raise ValueError(f"{type(system).__name__} не умеет {need}")
        if tuple(system.state_names) != ("psi", "theta", "dpsi", "dtheta"):
            raise ValueError(f"ожидалось состояние (psi, theta, dpsi, dtheta), "
                             f"получено {system.state_names}")
        if dt <= 0:
            raise ValueError("dt должен быть положительным")
        if iterations < 1:
            raise ValueError("iterations >= 1")
        self.system = system
        self.dt = float(dt)
        self.d = float(d)
        self.iterations = int(iterations)
        self.diagnostics = bool(diagnostics)
        self._diag_rec = None
        self.q_acc = float(q_acc)
        self.psi0_std = float(psi0_std)
        self.dtheta0_std = float(dtheta0_std)

        Rw = _correlation_matrix(corr_w, "corr_w", 3)
        D = np.diag([sigma_a, sigma_a, sigma_g]) / np.sqrt(self.dt)
        _cholesky(Rw, "corr_w")                       # проверка допустимости
        self.R_imu = D @ Rw @ D
        q = 0.0 if not counts_per_rev else 2.0 * np.pi / counts_per_rev
        # q = 0 (без квантования) даёт вырожденную R -- оставляем крошечную
        # дисперсию, чтобы S оставалась обратимой
        self.R_enc = max(q ** 2 / 12.0, 1e-14)
        self.sigma_g = float(sigma_g)

        if bias_prior is None:
            self._b_hat0 = np.zeros(3)
            self._P_bb0 = np.diag([b0_a ** 2, b0_a ** 2, b0_g ** 2])
        else:
            b_hat, P_bb = bias_prior
            self._b_hat0 = np.asarray(b_hat, dtype=float)
            self._P_bb0 = np.asarray(P_bb, dtype=float)
            if self._P_bb0.shape != (3, 3):
                raise ValueError(f"P_bb должна быть 3x3, получено {self._P_bb0.shape}")

        self.bias_drift = bias_drift
        self._Q_b = np.zeros(3) if bias_drift is None else self.dt * np.array([
            bias_drift["sigma_ba"] ** 2, bias_drift["sigma_ba"] ** 2,
            bias_drift["sigma_bg"] ** 2])

        self.has_w = disturbance is not None
        if self.has_w:
            self.sigma_w = np.broadcast_to(np.asarray(disturbance["sigma_w"], float), (2,)).copy()
            self.tau_w = np.broadcast_to(np.asarray(disturbance["tau_w"], float), (2,)).copy()
            self._phi_w = np.exp(-self.dt / self.tau_w)
            self._Q_w = self.sigma_w ** 2 * (-np.expm1(-2.0 * self.dt / self.tau_w))
        # раскладка состояния
        self.i_w = slice(4, 6) if self.has_w else slice(4, 4)
        self.i_b = slice(6, 9) if self.has_w else slice(4, 7)
        self.n = 9 if self.has_w else 7
        self._ib = np.arange(self.n)[self.i_b]
        Qd = np.zeros(self.n)
        Qd[2] = Qd[3] = self.q_acc ** 2 * self.dt
        if self.has_w:
            Qd[self.i_w] = self._Q_w
        Qd[self.i_b] = self._Q_b
        self._diag = np.flatnonzero(Qd)
        self._Q_diag = Qd[self._diag]
        self._R3 = self.R_imu.copy()
        self._L_inv = np.linalg.inv(np.linalg.cholesky(self.R_imu))
        self._E_cols = np.array([0, 2, 3, 4, 5] if self.has_w else [0, 2, 3])
        self._R4 = np.zeros((4, 4))
        self._R4[:3, :3] = self.R_imu
        self._R4[3, 3] = self.R_enc

        self._z = None           # (M, n)
        self._P = None           # (M, n, n)
        self._prefix = None
        self._enc = None
        self._t_prev = None

        # Скалярный путь одной клетки (docs/kalman_speed.md §3, S1): та же
        # математика, но на float и плотных 7x7 вместо батч-индексации.  Только
        # для «настоящей» WheeledPendulum: подкласс с другой f считался бы не
        # по своей модели.
        from .models.wheeled_pendulum import WheeledPendulum
        cls = type(system)
        self._scalar_ok = (self.iterations == 1 and all(
            getattr(cls, name) is getattr(WheeledPendulum, name)
            for name in ("f", "jacobian", "specific_force_with_jacobian")))
        pp = system.p
        self._pp = (float(pp.alpha), float(pp.beta), float(pp.gamma), float(pp.D),
                    float(pp.g), float(pp.r))
        Qfull = np.zeros(self.n)
        Qfull[self._diag] = self._Q_diag
        self._Q_full = Qfull
        self._inv_tau = (1.0 / self.tau_w) if self.has_w else None
        self._dg = np.diag_indices(self.n)
        self._ib_list = [int(i) for i in self._ib]
        self._R_diag = np.diag(self._R4)

        from . import _kf_kernel as kk
        if backend not in ("auto", "numpy", "numba"):
            raise ValueError(f"backend: auto, numpy или numba, получено {backend!r}")
        if backend == "numba":
            if not kk.HAVE_NUMBA:
                raise ValueError("backend='numba': пакет numba не установлен "
                                 "(uv sync --extra fast)")
            if not self._scalar_ok:
                raise ValueError("backend='numba': ядро умеет только WheeledPendulum "
                                 "и iterations == 1")
        self.backend = ("numba" if backend != "numpy" and kk.HAVE_NUMBA and self._scalar_ok
                        else "numpy")
        self._kk = kk
        # Аргументы ядра, которые не меняются от шага к шагу.
        self._k_pp = np.array(self._pp)
        self._k_phi = (np.ascontiguousarray(self._phi_w, float) if self.has_w
                       else np.zeros(2))
        self._k_itau = (np.ascontiguousarray(self._inv_tau, float) if self.has_w
                        else np.zeros(2))
        self._k_Linv = np.ascontiguousarray(self._L_inv)
        self._k_Rdiag = np.ascontiguousarray(self._R_diag)

    # --- наружу --------------------------------------------------------------

    @property
    def aux(self) -> dict:
        """Что фильтр знает сверх x_hat: оценки смещений и возмущения.
        rollout кладёт aux["w_hat"] в Trajectory.w_hat."""
        if self._z is None:
            return {}
        out = {"b_hat": self._z[:, self.i_b].reshape(self._prefix + (3,))}
        if self.has_w:
            out["w_hat"] = self._z[:, self.i_w].reshape(self._prefix + (2,))
        if self.diagnostics and self._diag_rec is not None:
            m = self._diag_rec["y"].shape[-1]
            out["P"] = self._P.reshape(self._prefix + (self.n, self.n)).copy()
            for key in ("y", "y_pred", "S_diag"):
                out[key] = self._diag_rec[key].reshape(self._prefix + (m,))
        return out

    @property
    def state(self) -> np.ndarray:
        """Полное состояние фильтра z формы prefix + (n,)."""
        return self._z.reshape(self._prefix + (self.n,))

    @property
    def covariance(self) -> np.ndarray:
        """Ковариация P формы prefix + (n, n) -- для NEES и графиков."""
        return self._P.reshape(self._prefix + (self.n, self.n))

    # --- жизненный цикл ------------------------------------------------------

    def reset(self, y0: np.ndarray | None = None) -> None:
        self._z = self._P = self._prefix = self._enc = self._t_prev = None
        if y0 is not None:
            self._init_from(np.asarray(y0, dtype=float))

    def _init_from(self, y: np.ndarray) -> None:
        """Начальная оценка по первому отсчёту и её ковариация.

        Машина в t0 может стоять с наклоном и сразу падать: тогда статический
        atan2 врёт на kappa_psi psi (кажущаяся вертикаль, §3.5).  Поэтому
        наклон берётся как корень уравнения акселерометра по модели --
        метод Гаусса--Ньютона по (a_x, a_z) при dpsi = omega - b_g, u = 0,
        w = 0 (§9: это и есть шаг A31, только по обоим каналам):

            psi <- psi + H^T r / H^T H,   r = y_acc - b - h(psi),  H = dh/dpsi.

        Ковариация -- линейное отображение независимых источников ошибки
        (ошибка смещений d_b, возмущения e_w, шум отсчёта v) в ошибку
        состояния.  Из h(psi_hat) - h(psi) = -d_b_acc - C_w e_w + v_acc -
        C_dpsi e_dpsi:

            e_dpsi = -d_b_g + v_g,
            e_psi  = H^T (-d_b_acc - C_w e_w + v_acc - C_dpsi e_dpsi) / H^T H,
            e_theta = e_psi + v_e   (энкодер: theta = e + psi).

        Сверх этого -- psi0_std (то, чего модель в t0 не знает) и
        dtheta0_std (скорость колеса энкодер до второго отсчёта не знает).
        """
        if y.shape[-1] not in (3, 5):
            raise ValueError(
                "KalmanEstimator ждёт (a_x, a_z, omega) или (a_x, a_z, omega, e, de) "
                f"-- IMUSensor(mode='imu') [+ EncoderSensor]; получено {y.shape}"
            )
        self._enc = y.shape[-1] == 5
        self._prefix = y.shape[:-1]
        Y = y.reshape(-1, y.shape[-1])
        M, n = Y.shape[0], self.n
        b = (np.broadcast_to(self._b_hat0, (M, 3)) if self._b_hat0.ndim == 1
             else self._b_hat0.reshape(-1, 3))
        u0 = np.zeros((M, 1))
        x = np.zeros((M, 4))
        x[:, 0] = np.arctan2(-(Y[:, 0] - b[:, 0]), Y[:, 1] - b[:, 1])
        x[:, 2] = Y[:, 2] - b[:, 2]
        acc = Y[:, :2] - b[:, :2]
        for _ in range(8):
            h = np.stack(self.system.specific_force(0.0, x, u0, self.d), axis=-1)
            C, _, _ = self.system.specific_force_jacobian(0.0, x, u0, self.d)
            H = C[:, :, 0]
            HH = np.sum(H * H, axis=-1)
            step = np.where(HH > 1e-12, np.sum(H * (acc - h), axis=-1) / np.maximum(HH, 1e-12), 0.0)
            x[:, 0] += np.clip(step, -0.5, 0.5)
        C, _, Cw = self.system.specific_force_jacobian(0.0, x, u0, self.d)
        H = C[:, :, 0]
        g_inv = H / np.sum(H * H, axis=-1, keepdims=True)          # H^T / H^T H, (M, 2)
        C_dpsi = C[:, :, 2]

        z = np.zeros((M, n))
        z[:, :4] = x
        z[:, 1] = Y[:, 3] + x[:, 0] if self._enc else 0.0
        z[:, self.i_b] = b

        # источники: d_b (3), v = (v_ax, v_az, v_g) (3), [e_w (2)]
        n_src = 6 + (2 if self.has_w else 0)
        Sig = np.zeros((n_src, n_src))
        Sig[:3, :3] = self._P_bb0
        Sig[3:6, 3:6] = self.R_imu
        T = np.zeros((M, n, n_src))
        ib = np.arange(n)[self.i_b]
        T[:, ib, [0, 1, 2]] = 1.0                                     # b <- d_b
        T[:, 2, 2] = -1.0                                             # dpsi <- d_b_g
        T[:, 2, 5] = 1.0                                              # dpsi <- v_g
        # psi <- через g_inv: (-d_b_acc + v_acc - C_dpsi e_dpsi - C_w e_w)
        k_dpsi = np.sum(g_inv * C_dpsi, axis=-1)                      # (M,)
        T[:, 0, 0] = -g_inv[:, 0]
        T[:, 0, 1] = -g_inv[:, 1]
        T[:, 0, 3] = g_inv[:, 0]
        T[:, 0, 4] = g_inv[:, 1]
        T[:, 0, :] -= k_dpsi[:, None] * T[:, 2, :]
        if self.has_w:
            Sig[6:, 6:] = np.diag(self.sigma_w ** 2)
            T[:, 0, 6:] = -np.einsum("mi,mij->mj", g_inv, Cw)
            T[:, self.i_w, 6:] = np.eye(2)
        if self._enc:
            T[:, 1, :] = T[:, 0, :]
        P = T @ Sig @ np.swapaxes(T, -1, -2)
        P[:, 0, 0] += self.psi0_std ** 2
        if self._enc:
            P[:, 0, 1] += self.psi0_std ** 2
            P[:, 1, 0] += self.psi0_std ** 2
            P[:, 1, 1] += self.psi0_std ** 2 + self.R_enc
        else:
            P[:, 1, 1] += 1.0                         # колесо без энкодера не видно
        P[:, 3, 3] += self.dtheta0_std ** 2
        self._z = z
        self._P = 0.5 * (P + np.swapaxes(P, -1, -2))

    # --- модель фильтра ------------------------------------------------------

    def _w(self, z):
        return z[:, self.i_w] if self.has_w else None

    def _field(self, t, x, u, w):
        dx = self.system.f(t, x, u)
        if w is None:
            return dx
        dx = dx.copy()
        dx[:, 2:] += self.system.accel_from_force(x[:, 0], w)
        return dx

    # Ускорение (docs/explorer_kalman.md §10.1).  Математика прежняя, но в
    # F и C почти всё -- нули и единицы, а батч-произведения маленьких
    # матриц в numpy стоят ~70 мкс независимо от размера (замер: M = 441,
    # 7x7).  Поэтому произведения собраны из столбцов, где они ненулевые, а
    # коррекция идёт по одному скалярному измерению (§«_update»).  Оракул --
    # совпадение с прежней плотной реализацией до округления и NEES-тест.

    def _predict(self, t_prev, u):
        """Прогноз: RK4 по номинальной модели, P <- F P F^T + Q.

        F = I + E, E = J dt + J^2 dt^2/2.  У E ненулевые только строки
        (psi, theta, dpsi, dtheta[, w]) -- у смещений Q_b-блуждание, F = I, --
        и только столбцы (psi, dpsi, dtheta[, w]): J[0] = e_dpsi, J[1] =
        e_dtheta, строки ускорений зависят лишь от psi, dpsi и w (§1.4).
        Поэтому

            F P F^T = P + E P + (E P)^T + E P E^T,

        и каждое произведение -- сумма по 3..5 ненулевым столбцам E.
        """
        z, P, dt = self._z, self._P, self.dt
        M, n = z.shape
        x = z[:, :4]
        w = None
        if self.has_w:
            w = self._phi_w * z[:, self.i_w]
        h = dt
        k1 = self._field(t_prev, x, u, w)
        k2 = self._field(t_prev + h / 2, x + h / 2 * k1, u, w)
        k3 = self._field(t_prev + h / 2, x + h / 2 * k2, u, w)
        k4 = self._field(t_prev + h, x + h * k3, u, w)
        x_new = x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        A, _, G = self.system.jacobian(t_prev, x, u, w)
        rows = 6 if self.has_w else 4
        J = np.zeros((M, rows, n))
        J[:, :4, :4] = A
        if self.has_w:
            J[:, :4, 4:6] = G
            J[:, 4, 4] = -1.0 / self.tau_w[0]
            J[:, 5, 5] = -1.0 / self.tau_w[1]
        cols = self._E_cols
        # J^2 по строкам, из структуры J: J[0] = e_dpsi, J[1] = e_dtheta, у
        # строк ускорений ненулевые столбцы -- psi, dpsi и w, у строк w --
        # только диагональ -1/tau.  Поэтому
        #   (J^2)[0] = J[2],  (J^2)[1] = J[3],
        #   (J^2)[i] = J[i,0] e_dpsi + J[i,2] J[2] + sum_w J[i,w] J[w]  (i = 2, 3),
        #   (J^2)[w] = J[w,w]^2 e_w.
        J2 = np.zeros_like(J)
        J2[:, 0] = J[:, 2]
        J2[:, 1] = J[:, 3]
        for i in (2, 3):
            J2[:, i] = J[:, i, 2, None] * J[:, 2]
            J2[:, i, 2] += J[:, i, 0]
            if self.has_w:
                J2[:, i, 4] += J[:, i, 4] * J[:, 4, 4]
                J2[:, i, 5] += J[:, i, 5] * J[:, 5, 5]
        if self.has_w:
            J2[:, 4, 4] = J[:, 4, 4] ** 2
            J2[:, 5, 5] = J[:, 5, 5] ** 2
        E = J * dt + J2 * (dt * dt / 2.0)
        Ec = E[:, :, cols]                                              # (M, rows, |cols|)
        EP = Ec @ P[:, cols, :]                                         # (M, rows, n)
        EPE = EP[:, :, cols] @ np.swapaxes(Ec, -1, -2)                  # (M, rows, rows)
        P = P.copy()
        P[:, :rows, :] += EP
        P[:, :, :rows] += np.swapaxes(EP, -1, -2)
        P[:, :rows, :rows] += EPE
        P[:, self._diag, self._diag] += self._Q_diag
        z = z.copy()
        z[:, :4] = x_new
        if self.has_w:
            z[:, self.i_w] = w
        self._z, self._P = z, P

    def _h_and_C(self, t, z, u):
        """Предсказанное измерение и его якобиан по полному состоянию."""
        M, n = z.shape
        x, w = z[:, :4], self._w(z)
        b = z[:, self.i_b]
        a_x, a_z, Cx, _, Cw = self.system.specific_force_with_jacobian(t, x, u, self.d, w)
        m = 4 if self._enc else 3
        h = np.empty((M, m))
        h[:, 0] = a_x + b[:, 0]
        h[:, 1] = a_z + b[:, 1]
        h[:, 2] = x[:, 2] + b[:, 2]
        C = np.zeros((M, m, n))
        C[:, :2, :4] = Cx
        if self.has_w:
            C[:, :2, self.i_w] = Cw
        ib = self._ib
        C[:, 0, ib[0]] = C[:, 1, ib[1]] = C[:, 2, ib[2]] = 1.0
        C[:, 2, 2] = 1.0
        if self._enc:
            h[:, 3] = x[:, 1] - x[:, 0]
            C[:, 3, 0], C[:, 3, 1] = -1.0, 1.0
        return h, C

    def _record_diag(self, Y, h, C, P):
        """Для видов окна: показание, предсказание h(x^-, u) по ПРОГНОЗУ (до
        коррекции) и диагональ S = C P^- C^T + R -- в исходных каналах
        (a_x, a_z, omega[, e]), а не в «обелённых»."""
        m = Y.shape[1]
        R = self._R4 if self._enc else self._R3
        S = np.einsum("mik,mkl,mil->mi", C, P, C) + np.diag(R)[:m]
        self._diag_rec = {"y": Y.copy(), "y_pred": h.copy(), "S_diag": S}

    def _update(self, t, y, u):
        """Коррекция.

        EKF (iterations = 1) -- ПОСЛЕДОВАТЕЛЬНО по скалярным измерениям
        (Simon, Optimal State Estimation, §6.1): каналы ИДУ сначала
        «обеляются» -- y' = L^-1 y, C' = L^-1 C, L L^T = R_imu, -- после
        чего шумы независимы с единичной дисперсией, и обработка по одному
        каналу при линеаризации в ОДНОЙ точке (априорной) даёт ровно тот же
        результат, что общий шаг с матрицей S^-1.  Зато вместо обращения
        S 4x4 и батч-произведений -- векторные операции над (M, n):

            Pc = P c,   s = c^T P c + r,   K = Pc / s,
            z <- z + K (y_k - h_k - c^T (z - z_prior)),   P <- P - K Pc^T.

        IEKF (iterations > 1) -- общий шаг с перелинеаризацией:
        P+ = P- - K (C P-), что при оптимальном K равно форме Джозефа.
        """
        Y = y.reshape(-1, y.shape[-1])[:, : (4 if self._enc else 3)]
        z_prior, P = self._z, self._P
        if self.iterations == 1:
            h, C = self._h_and_C(t, z_prior, u)
            if self.diagnostics:
                self._record_diag(Y, h, C, P)
            innov = Y - h
            Li = self._L_inv
            Cw = [sum(Li[i, j] * C[:, j] for j in range(i + 1)) for i in range(3)]
            nw = [sum(Li[i, j] * innov[:, j] for j in range(i + 1)) for i in range(3)]
            r = [1.0, 1.0, 1.0]
            if self._enc:
                Cw.append(C[:, 3])
                nw.append(innov[:, 3])
                r.append(self.R_enc)
            z = z_prior.copy()
            P = P.copy()
            for c, v, rk in zip(Cw, nw, r):
                # c разрежена: у гироскопа и энкодера два ненулевых столбца,
                # у акселерометра -- пять-семь; P c -- сумма этих столбцов P
                nz = np.flatnonzero(np.any(c != 0.0, axis=0))
                Pc = sum(P[:, :, k] * c[:, k, None] for k in nz)   # (M, n)
                s = sum(c[:, k] * Pc[:, k] for k in nz) + rk
                K = Pc / s[:, None]
                dz = z - z_prior
                resid = v - sum(c[:, k] * dz[:, k] for k in nz)
                z = z + K * resid[:, None]
                P = P - K[:, :, None] * Pc[:, None, :]
            self._P = 0.5 * (P + np.swapaxes(P, -1, -2))
            self._z = z
            return

        m = Y.shape[1]
        R = self._R4 if self._enc else self._R3
        z_i = z_prior
        for it in range(self.iterations):
            h, C = self._h_and_C(t, z_i, u)
            if self.diagnostics and it == 0:
                self._record_diag(Y, h, C, P)
            CP = C @ P
            S = CP @ np.swapaxes(C, -1, -2) + R
            K = np.swapaxes(np.linalg.solve(S, CP), -1, -2)      # P C^T S^-1
            innov = Y - h - np.einsum("mij,mj->mi", C, z_prior - z_i)
            z_i = z_prior + np.einsum("mij,mj->mi", K, innov)
        P = P - K @ CP
        self._P = 0.5 * (P + np.swapaxes(P, -1, -2))
        self._z = z_i

    def estimate(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        if self._z is None or y.shape[:-1] != self._prefix:
            self._init_from(y)
        if self.backend == "numba":
            return self._estimate_jit(t, y, u_prev)
        if self._prefix == () and self._scalar_ok:
            return self._estimate1(t, y, u_prev)
        M = self._z.shape[0]
        u = (np.zeros((M, 1)) if u_prev is None
             else np.broadcast_to(np.asarray(u_prev, dtype=float).reshape(-1, 1), (M, 1)))
        if self._t_prev is not None:
            dt = float(t) - self._t_prev
            if dt > 0:
                if abs(dt - self.dt) > 1e-6 * self.dt:
                    raise ValueError(
                        f"KalmanEstimator: шаг {dt!r} не совпадает с dt={self.dt!r}"
                    )
                self._predict(self._t_prev, u)
        self._update(float(t), y, u)
        self._t_prev = float(t)
        return self._z[:, :4].reshape(self._prefix + (4,))

    # --- скалярный путь одной клетки (docs/kalman_speed.md §3, S1) ------------
    #
    # Одна клетка (виды окна A/B/C, одиночный пересчёт) упиралась не в
    # арифметику, а в накладные расходы Python: батч-путь делает ~800 мелких
    # вызовов numpy на шаг над массивами формы (1, ...).  Здесь та же
    # математика (прогноз RK4 + F P F^T + Q, последовательная скалярная
    # коррекция с «обелением»), но модель считается на float через math, а
    # матрицы -- плотные n x n.  Оракул: совпадение с батч-путём до
    # округления на одной и той же последовательности измерений
    # (tests/test_accel_kalman.py).

    def _terms1(self, psi, dpsi, u, w_psi, w_th):
        """Ускорения и их производные в точке, на float (Key_Formulas §2).

        Возвращает (ddpsi, ddth, A20, A22, A30, A32, Mi00, Mi01, Mi11): то же,
        что строки 3-4 WheeledPendulum.jacobian и M^-1, но без массивов."""
        al, be, ga, D, _, _ = self._pp
        s, c = math.sin(psi), math.cos(psi)
        s2, c2 = 2.0 * s * c, c * c - s * s                 # sin 2psi, cos 2psi
        dp2 = dpsi * dpsi
        N = (ga + be * c) * u + ga * D * s - be * be * s * c * dp2 + ga * w_psi - be * c * w_th
        R = al * be * s * dp2 - be * D * s * c - (al + be * c) * u - be * c * w_psi + al * w_th
        N_psi = -be * s * u + ga * D * c - be * be * c2 * dp2 + be * s * w_th
        R_psi = al * be * c * dp2 - be * D * c2 + be * s * u + be * s * w_psi
        De = al * ga - be * be * c * c
        dDe = be * be * s2
        De2 = De * De
        return (N / De, R / De,
                (N_psi * De - N * dDe) / De2, -2.0 * be * be * s * c * dpsi / De,
                (R_psi * De - R * dDe) / De2, 2.0 * al * be * s * dpsi / De,
                ga / De, -be * c / De, al / De)

    def _field1(self, x0, x2, x3, u, w_psi, w_th):
        al, be, ga, D, _, _ = self._pp
        s, c = math.sin(x0), math.cos(x0)
        De = al * ga - be * be * c * c
        dp2 = x2 * x2
        ddpsi = ((ga + be * c) * u + ga * D * s - be * be * s * c * dp2) / De
        ddth = (al * be * s * dp2 - be * D * s * c - (al + be * c) * u) / De
        if w_psi or w_th:
            ddpsi += (ga * w_psi - be * c * w_th) / De
            ddth += (al * w_th - be * c * w_psi) / De
        return x2, x3, ddpsi, ddth

    def _predict1(self, u):
        z, P, dt = self._z[0], self._P[0], self.dt
        n = self.n
        w_psi = w_th = 0.0
        if self.has_w:
            w_psi = float(self._phi_w[0] * z[4])
            w_th = float(self._phi_w[1] * z[5])
        x0, x1, x2, x3 = (float(v) for v in z[:4])
        h, h2 = dt, dt / 2.0
        k1 = self._field1(x0, x2, x3, u, w_psi, w_th)
        k2 = self._field1(x0 + h2 * k1[0], x2 + h2 * k1[2], x3 + h2 * k1[3], u, w_psi, w_th)
        k3 = self._field1(x0 + h2 * k2[0], x2 + h2 * k2[2], x3 + h2 * k2[3], u, w_psi, w_th)
        k4 = self._field1(x0 + h * k3[0], x2 + h * k3[2], x3 + h * k3[3], u, w_psi, w_th)
        x_new = [xi + h / 6.0 * (a + 2.0 * b + 2.0 * c + d)
                 for xi, a, b, c, d in zip((x0, x1, x2, x3), k1, k2, k3, k4)]

        # Якобиан в начале шага, как у батч-пути: F = I + J dt + J^2 dt^2/2.
        _, _, A20, A22, A30, A32, Mi00, Mi01, Mi11 = self._terms1(x0, x2, u, w_psi, w_th)
        J = np.zeros((n, n))
        J[0, 2] = 1.0
        J[1, 3] = 1.0
        J[2, 0], J[2, 2], J[3, 0], J[3, 2] = A20, A22, A30, A32
        if self.has_w:
            J[2, 4], J[2, 5], J[3, 4], J[3, 5] = Mi00, Mi01, Mi01, Mi11
            J[4, 4] = -self._inv_tau[0]
            J[5, 5] = -self._inv_tau[1]
        F = J * dt + (J @ J) * (dt * dt / 2.0)
        F[self._dg] += 1.0
        P = F @ P @ F.T
        P[self._dg] += self._Q_full
        z = z.copy()
        z[:4] = x_new
        if self.has_w:
            z[4], z[5] = w_psi, w_th
        self._z[0] = z
        self._P[0] = P

    def _update1(self, y, u):
        z_prior, P = self._z[0].copy(), self._P[0].copy()
        n, d = self.n, self.d
        _, _, _, _, g, r = self._pp
        zl = z_prior.tolist()                     # float, а не скаляры numpy
        x0, x1, x2 = zl[0], zl[1], zl[2]
        w_psi = zl[4] if self.has_w else 0.0
        w_th = zl[5] if self.has_w else 0.0
        ib = self._ib_list
        b = [zl[i] for i in ib]
        ddpsi, ddth, A20, A22, A30, A32, Mi00, Mi01, Mi11 = self._terms1(x0, x2, u, w_psi, w_th)
        s, c = math.sin(x0), math.cos(x0)
        m = 4 if self._enc else 3
        h = np.empty(m)
        h[0] = -g * s + d * ddpsi + r * c * ddth + b[0]
        h[1] = g * c - d * x2 * x2 + r * s * ddth + b[1]
        h[2] = x2 + b[2]
        C = np.zeros((m, n))
        C[0, 0] = -g * c + d * A20 - r * s * ddth + r * c * A30
        C[0, 2] = d * A22 + r * c * A32
        C[1, 0] = -g * s + r * c * ddth + r * s * A30
        C[1, 2] = -2.0 * d * x2 + r * s * A32
        if self.has_w:
            C[0, 4], C[0, 5] = d * Mi00 + r * c * Mi01, d * Mi01 + r * c * Mi11
            C[1, 4], C[1, 5] = r * s * Mi01, r * s * Mi11
        C[0, ib[0]] = C[1, ib[1]] = C[2, ib[2]] = 1.0
        C[2, 2] = 1.0
        if self._enc:
            h[3] = x1 - x0
            C[3, 0], C[3, 1] = -1.0, 1.0
        Y = y[:m]
        if self.diagnostics:
            S = np.einsum("ik,kl,il->i", C, P, C) + self._R_diag[:m]
            self._diag_rec = {"y": Y[None].copy(), "y_pred": h[None].copy(), "S_diag": S[None]}
        innov = Y - h
        Cw = np.empty((m, n))
        nw = np.empty(m)
        Cw[:3] = self._L_inv @ C[:3]
        nw[:3] = self._L_inv @ innov[:3]
        rr = [1.0, 1.0, 1.0]
        if self._enc:
            Cw[3], nw[3] = C[3], innov[3]
            rr.append(self.R_enc)
        z = z_prior.copy()
        for k in range(m):
            ck = Cw[k]
            Pc = P @ ck
            sk = float(ck @ Pc) + rr[k]
            K = Pc / sk
            resid = float(nw[k]) - float(ck @ (z - z_prior))
            z += K * resid
            P -= K[:, None] * Pc
        self._P[0] = 0.5 * (P + P.T)
        self._z[0] = z

    def _estimate1(self, t, y, u_prev):
        u = 0.0 if u_prev is None else float(np.asarray(u_prev, dtype=float).reshape(-1)[0])
        if self._t_prev is not None:
            dt = float(t) - self._t_prev
            if dt > 0:
                if abs(dt - self.dt) > 1e-6 * self.dt:
                    raise ValueError(
                        f"KalmanEstimator: шаг {dt!r} не совпадает с dt={self.dt!r}"
                    )
                self._predict1(u)
        self._update1(y.reshape(-1), u)
        self._t_prev = float(t)
        return self._z[0, :4].copy()

    # --- numba-ядро (docs/kalman_speed.md §7, S3) ------------------------------

    def _estimate_jit(self, t, y, u_prev):
        """Шаг через wpend/_kf_kernel.py: то же, что _estimate1 / _predict +
        _update, но скомпилированное; пачка -- по потокам, если она большая.
        Z и P меняются на месте."""
        kk = self._kk
        M = self._z.shape[0]
        m = 4 if self._enc else 3
        Y = np.ascontiguousarray(y.reshape(-1, y.shape[-1])[:, :m], dtype=float)
        if u_prev is None:
            U = np.zeros(M)
        else:
            U = np.ascontiguousarray(np.broadcast_to(
                np.asarray(u_prev, dtype=float).reshape(-1), (M,)))
        predict = False
        if self._t_prev is not None:
            dt = float(t) - self._t_prev
            if dt > 0:
                if abs(dt - self.dt) > 1e-6 * self.dt:
                    raise ValueError(
                        f"KalmanEstimator: шаг {dt!r} не совпадает с dt={self.dt!r}"
                    )
                predict = True
        out_h = np.empty((M, m))
        out_S = np.empty((M, m))
        step = kk.step_parallel if M >= kk.PARALLEL_FROM else kk.step_serial
        step(self._z, self._P, Y, U, predict, self._k_pp, self.dt, self.d, self.has_w,
             self._k_phi, self._k_itau, self._Q_full, int(self._ib[0]), bool(self._enc),
             self._k_Linv, float(self.R_enc), self._k_Rdiag, bool(self.diagnostics),
             out_h, out_S)
        if self.diagnostics:
            self._diag_rec = {"y": Y.copy(), "y_pred": out_h, "S_diag": out_S}
        self._t_prev = float(t)
        return self._z[:, :4].reshape(self._prefix + (4,)).copy()
