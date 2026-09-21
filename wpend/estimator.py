"""Слой 3 из 5: ОЦЕНИВАТЕЛЬ.

Оцениватель превращает поток измерений y в оценку состояния x_hat, которую
получает регулятор.  Это единственное место, где разрешено иметь память о
прошлом (фильтр, интегратор ошибки, наблюдатель).
"""

from __future__ import annotations

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
            if not np.isclose(dt, self.dt, rtol=1e-6, atol=0.0):
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
