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
    вертикали, theta_acc = atan2(-a_x, a_z), после чего измерение становится
    ЛИНЕЙНЫМ по состоянию (y = theta + шум), и расширенный фильтр Калмана для
    этой задачи не нужен -- достаточно обычного.

        theta_hat_k = a*(theta_hat_{k-1} + omega_k*dt) + (1-a)*theta_acc_k,
        a = tau/(tau + dt)

    ПОЧЕМУ "комплементарный".  В непрерывном времени это
    theta_hat' = omega + k*(theta_acc - theta_hat), k = 1/tau, откуда

        theta_hat = [ s/(s+k) ]*theta  +  [ k/(s+k) ]*theta_acc
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

    Скорость наклона фильтр НЕ фильтрует: dtheta_hat = omega, то есть в
    регулятор уходит показание гироскопа вместе с его смещением.  Это тоже
    лечится расширением состояния, а не подбором tau.

    КОЛЕСО.  Из ИДУ оно не достаётся никак.  Не "плохо достаётся" -- никак:
    ddtheta и ddphi зависят только от (theta, dtheta, u) (Key_Formulas §1.4,
    phi циклична), поэтому и a_x, a_z, omega зависят только от них.  Два
    состояния, отличающиеся лишь (phi, dphi), дают ПОБИТОВО одинаковые
    показания -- это проверено тестом, и это сильнее, чем rank(O) = 2 из 4:
    утверждение точное, а не про линеаризацию.  Ненаблюдаемая подсистема при
    этом ещё и недетектируема: A, суженная на ядро, -- нильпотентная жорданова
    клетка (двойной интегратор), значит ошибка колеса растёт линейно даже без
    шума.

    Отсюда два режима, и оба честны по-своему:

    * wheel="zero" -- phi_hat = dphi_hat = 0.  Законно ровно в паре с
      регулятором, у которого нули в этих столбцах (lqr_tilt, A17): выдуманное
      число тогда физически не может попасть в управление.  Это база отсчёта.
    * wheel="encoder" -- колесо приходит от МОТОРНОГО ЭНКОДЕРА, и тогда
      система наблюдаема (rank 4 против 2 при одном ИДУ).  Датчик даёт
      ОТНОСИТЕЛЬНЫЕ величины, поэтому оценка складывается:

          phi_hat  = (phi - theta)_изм  + theta_hat
          dphi_hat = (dphi - dtheta)_изм + omega

      Ошибка наклона протекает в колесо целиком -- это не дефект, а плата за
      то, что вал не видит землю.  Зато она ОГРАНИЧЕНА ошибкой наклона, а не
      растёт со временем, как при счислении пути.  Ждёт y из пяти чисел:
      StackedSensor(IMUSensor(mode="imu"), EncoderSensor).
    * wheel="dead_reckon" (по умолчанию) -- счисление пути: ddphi берётся из
      ПРАВОЙ ЧАСТИ МОДЕЛИ по текущей оценке наклона и прошлому управлению и
      дважды интегрируется явным Эйлером.  Оценка обязана разойтись, и в этом
      смысл режима: ошибка ускорения колеса равна c*eps, где eps -- ошибка
      наклона, а c = d(ddphi)/d(theta) = -350 для параметров по умолчанию.
      Значит постоянный остаток eps_0 = b*tau даёт ошибку dphi = c*eps_0*t
      (линейно) и phi = c*eps_0*t^2/2 (квадратично), а белый шум -- рост как
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

    Источники: Higgins, "A Comparison of Complementary and Kalman Filtering",
    IEEE Trans. AES 11 (1975) -- комплементарный фильтр как установившийся
    фильтр Калмана; Mahony, Hamel, Pflimlin, IEEE TAC 53 (2008) -- явный
    комплементарный фильтр и оценка смещения; Simon, *Optimal State
    Estimation*, гл. 5; docs/imu_noise.md §1 (кажущаяся вертикаль).
    """

    def __init__(self, system, *, dt: float, tau: float = 0.5,
                 wheel: str = "dead_reckon"):
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
        for need in ("theta", "phi", "dtheta", "dphi"):
            if need not in names:
                raise ValueError(
                    f"ComplementaryEstimator не понимает состояние {names}: "
                    "нужны theta, phi, dtheta, dphi"
                )
        self.system = system
        self.dt = float(dt)
        self.tau = tau if tau.ndim else float(tau)
        self.wheel = wheel
        self.i = tuple(names.index(n) for n in ("theta", "phi", "dtheta", "dphi"))
        # a = tau/(tau+dt); при tau = inf это ровно 1 (чистый гироскоп), но
        # inf/(inf+dt) в плавающей точке даёт nan -- поэтому предел явно.
        # np.where вместо if: tau может быть массивом, и тогда бесконечность
        # стоит лишь в части строк.
        safe = np.where(np.isinf(tau), 1.0, tau)
        self.a = np.where(np.isinf(tau), 1.0, safe / (safe + self.dt))
        if not tau.ndim:
            self.a = float(self.a)

        self._theta = None
        self._phi = None
        self._dphi = None
        self._t_prev = None

    # --- вспомогательное -------------------------------------------------

    @staticmethod
    def tilt_from_accelerometer(a_x, a_z):
        """Кажущаяся вертикаль: theta_acc = atan2(-a_x, a_z).

        В покое f = (-g sin theta, g cos theta), и формула возвращает ровно
        theta.  При разгоне -- систематически смещённое значение: это не шум,
        а физика (docs/imu_noise.md §1, WheeledPendulum.specific_force).
        """
        return np.arctan2(-np.asarray(a_x, dtype=float), np.asarray(a_z, dtype=float))

    # --- жизненный цикл --------------------------------------------------

    def reset(self, y0: np.ndarray | None = None) -> None:
        if y0 is None:
            self._theta = None
            self._phi = self._dphi = None
        else:
            y0 = np.asarray(y0, dtype=float)
            self._theta = self.tilt_from_accelerometer(y0[..., 0], y0[..., 1])
            zeros = np.zeros_like(self._theta)
            self._phi, self._dphi = zeros.copy(), zeros.copy()
        self._t_prev = None

    # --- оценка ----------------------------------------------------------

    def estimate(self, t: float, y: np.ndarray, u_prev: np.ndarray) -> np.ndarray:
        y = np.asarray(y, dtype=float)
        n_need = 5 if self.wheel == "encoder" else 3
        if y.shape[-1] != n_need:
            what = ("(a_x, a_z, omega, phi-theta, dphi-dtheta)"
                    if n_need == 5 else "(a_x, a_z, omega)")
            hint = ("Нужен StackedSensor(IMUSensor(mode='imu'), EncoderSensor)."
                    if n_need == 5 else "Датчику нужен mode='imu'.")
            raise ValueError(
                f"ComplementaryEstimator при wheel={self.wheel!r} ждёт {n_need} "
                f"числа {what}, получено {y.shape}.  {hint}"
            )
        theta_acc = self.tilt_from_accelerometer(y[..., 0], y[..., 1])
        omega = y[..., 2]
        if np.ndim(self.a) and np.shape(self.a) != np.shape(theta_acc):
            raise ValueError(
                f"tau задано массивом {np.shape(self.a)}, а пачка измерений "
                f"имеет форму {np.shape(theta_acc)}: на каждую строку нужно "
                "ровно одно tau."
            )

        # Первый вызов: reset мог задать theta по y0, но времени он не знает,
        # поэтому _t_prev is None -- такой же признак первого шага, как и
        # отсутствие оценки.  Шага фильтра здесь нет, только инициализация.
        if (self._theta is None or self._t_prev is None
                or np.shape(self._theta) != np.shape(theta_acc)):
            if self._theta is None or np.shape(self._theta) != np.shape(theta_acc):
                self._theta = np.array(theta_acc, dtype=float, copy=True)
                self._phi = np.zeros_like(self._theta)
                self._dphi = np.zeros_like(self._theta)
            self._t_prev = float(t)
            return self._assemble(self._theta, omega, y)

        dt = float(t) - self._t_prev
        if dt > 0:
            if not np.isclose(dt, self.dt, rtol=1e-6, atol=0.0):
                raise ValueError(
                    f"ComplementaryEstimator: шаг {dt!r} не совпадает с dt={self.dt!r}. "
                    "От шага зависят и a = tau/(tau+dt), и оба интегрирования."
                )
            # комплементарный шаг: прогноз гироскопом + подтяжка к акселерометру
            self._theta = self.a * (self._theta + omega * self.dt) \
                + (1.0 - self.a) * theta_acc
            if self.wheel == "dead_reckon":
                self._integrate_wheel(t, omega, u_prev)
            self._t_prev = float(t)

        return self._assemble(self._theta, omega, y)

    def _integrate_wheel(self, t: float, omega, u_prev) -> None:
        """Счисление пути по колесу: ddphi из правой части модели, явный Эйлер.

        ddphi не зависит ни от phi, ни от dphi (§1.4), поэтому подстановка
        собственных -- заведомо неверных -- оценок колеса на результат не
        влияет: считается функция только от (theta_hat, omega, u_prev).
        Обратной связи здесь нет вовсе, и ошибка ничем не ограничена.
        """
        if u_prev is None:
            raise ValueError(
                "wheel='dead_reckon' требует u_prev: ускорение колеса зависит "
                "от приложенного момента"
            )
        x_hat = self._assemble(self._theta, omega, None)
        ddphi = self.system.f(t, x_hat, np.atleast_1d(np.asarray(u_prev, float)))[..., self.i[3]]
        self._phi = self._phi + self._dphi * self.dt      # явный Эйлер: phi по СТАРОЙ скорости
        self._dphi = self._dphi + ddphi * self.dt

    def _assemble(self, theta, omega, y) -> np.ndarray:
        """Собрать x_hat = (theta, phi, dtheta, dphi) в порядке state_names.

        y нужен только режиму "encoder": колесо там не хранится в состоянии
        фильтра, а каждый раз складывается из показания и наклона.
        """
        theta = np.asarray(theta, dtype=float)
        omega = np.asarray(omega, dtype=float)
        if self.wheel == "encoder" and y is not None:
            # Энкодер даёт ОТНОСИТЕЛЬНЫЕ величины -- прибавляем наклон.
            phi = y[..., 3] + theta
            dphi = y[..., 4] + omega
        elif self.wheel == "zero" or self._phi is None:
            phi = np.zeros_like(theta)
            dphi = np.zeros_like(theta)
        else:
            phi, dphi = self._phi, self._dphi
        parts = [None] * 4
        parts[self.i[0]], parts[self.i[1]] = theta, phi
        parts[self.i[2]], parts[self.i[3]] = omega, dphi
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
