"""Слой 4 из 5: РЕГУЛЯТОР.

Регулятор превращает оценку состояния x_hat в требуемое управление u.
Он НЕ знает, откуда взялась оценка, и НЕ знает про ограничения мотора --
за проекцию на допустимое множество отвечает System.clip_action.
"""

from __future__ import annotations

import time
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
        return np.zeros(np.shape(x_hat)[:-1] + (self.n_action,))


class ConstantController(Controller):
    """u = const. Нужен, в частности, для проверки первых интегралов:
    у колёсного маятника при постоянном u сохраняется H (см. Key_Formulas §3).
    """

    def __init__(self, u):
        self.u = np.atleast_1d(np.asarray(u, dtype=float))

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        shape = np.shape(x_hat)[:-1] + (self.u.shape[-1],)
        return np.broadcast_to(self.u, shape).copy()


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
        # error @ K.T, а не K @ error: так формула одинаково работает и для
        # одного состояния (n,) -> (n_u,), и для пачки (M, n) -> (M, n_u).
        return -error @ self.K.T


# ---------------------------------------------------------------------------
# Регион, в котором ЛКР можно доверять
# ---------------------------------------------------------------------------
# Оба предиката ниже -- обычные функции x -> bool, а не классы. Регион нужен
# ровно в одном месте (условие переключения), поэтому он передаётся аргументом,
# и сравнение двух ответов на вопрос "где ЛКР работает" -- это подмена одного
# аргумента, а не второй регулятор.


def ellipsoid_region(P, c):
    """Предикат «x внутри сертифицированной области притяжения x^T P x <= c».

    P -- решение уравнения Риккати, c -- уровень из `wpend.lqr.certified_level`:
    наибольший, при котором весь эллипсоид лежит в {V̇ < 0}. Внутри этого
    множества сходимость ЛКР на НЕЛИНЕЙНОЙ системе доказана (Khalil, гл. 8.2),
    а не измерена. Цена -- консервативность: настоящий бассейн заметно шире.

    Проверка идёт по ПОЛНОМУ состоянию, включая колесо: c* сертифицирован для
    4-мерного эллипсоида, и выбрасывать из проверки phi/dphi значит потерять
    ровно ту гарантию, ради которой этот вариант и берут.
    """
    P = np.asarray(P, dtype=float)
    c = float(c)

    def inside(x):
        x = np.asarray(x, dtype=float)
        return np.einsum("...i,ij,...j->...", x, P, x) <= c

    return inside


def _nearest_index(grid, values):
    """Индекс ближайшего узла возрастающей сетки для каждого значения."""
    j = np.clip(np.searchsorted(grid, values), 1, len(grid) - 1)
    return np.where(values - grid[j - 1] <= grid[j] - values, j - 1, j)


def grid_region(mask, thetas, dthetas, i_theta=0, i_dtheta=2):
    """Предикат «клетка карты (theta, dtheta) помечена как удержанная».

    `mask` -- булева матрица формы (len(dthetas), len(thetas)); порядок осей
    тот же, что у `wpend.viz.grid.GridSpec`: строка -- dtheta, столбец -- theta.
    Обычно это результат прогона сетки под чистым ЛКР: `classify(...) == HELD`.

    Это измеренный бассейн, а не доказанный: клетка означает «за горизонт
    прогона корпус не упал», и между узлами ничего не гарантировано. Зато он
    заметно шире эллипсоида -- в этом и смысл сравнения.

    Читает только (theta, dtheta). Для вопроса «упадёт ли» это не потеря
    общности: ускорения колёсного маятника не зависят ни от phi, ни от dphi
    (Key_Formulas §1.4), поэтому карта и строится в этой плоскости. Но для
    вопроса «сойдётся ли ЛКР к нулю ПОЛНОСТЬЮ» -- потеря: колесо, укатившееся
    далеко, из проверки выпадает, и регулятор может переключиться раньше, чем
    имеет право. Именно это и надо увидеть в сравнении.
    """
    mask = np.asarray(mask, dtype=bool)
    thetas = np.asarray(thetas, dtype=float)
    dthetas = np.asarray(dthetas, dtype=float)
    if mask.shape != (dthetas.size, thetas.size):
        raise ValueError(f"mask должна быть формы {(dthetas.size, thetas.size)}, "
                         f"а не {mask.shape}")

    def inside(x):
        x = np.asarray(x, dtype=float)
        th, dth = x[..., i_theta], x[..., i_dtheta]
        on_grid = ((th >= thetas[0]) & (th <= thetas[-1])
                   & (dth >= dthetas[0]) & (dth <= dthetas[-1]))
        ix = _nearest_index(thetas, np.clip(th, thetas[0], thetas[-1]))
        iy = _nearest_index(dthetas, np.clip(dth, dthetas[0], dthetas[-1]))
        return mask[iy, ix] & on_grid

    return inside


class BangBangLQRController(Controller):
    """Релейное управление до входа в регион, ЛКР -- внутри.

    Идея. ЛКР доказуемо работает только вблизи верхнего положения: он построен
    по линеаризации, и на большом наклоне выброшенные нелинейности перевешивают.
    Релейное же управление u = ±u_max не претендует на точность -- оно просто
    выжимает из мотора всё и приводит состояние туда, где ЛКР законен. Отсюда
    двухфазная схема: сначала догнать регион, потом отдать управление ЛКР.

    Закон переключения релейной фазы -- ТОЧНАЯ сепаратриса, а не подгонка
    -------------------------------------------------------------------------
    Наклон колёсного маятника живёт своей жизнью: phi циклична, поэтому
    (Key_Formulas §1.3-1.4)

        Δ(θ) θ̈ = b(θ) u + γ D sin θ − β² sin θ cos θ θ̇² ,

    и при ПОСТОЯННОМ u сохраняется первый интеграл (§3)

        H(θ, θ̇; u) = ½ Δ(θ) θ̇² − u (γ θ + β sin θ) + γ D cos θ .

    Значит фазовые кривые при u = const -- в точности линии уровня H, и
    траекторию, приходящую в верхнее положение, можно выписать, а не подобрать:

      * при θ̇ > 0 корпус уезжает вперёд, тормозим моментом u = −u_max, и
        нужная кривая -- ветвь уровня H(·; −u_max) = H(0,0) = γ D;
      * при θ̇ < 0 -- зеркально, u = +u_max на уровне H(·; +u_max) = γ D.

    Обозначив d = sign θ̇ (а при θ̇ = 0 -- sign θ), обе ветви складываются в одну
    переключающую функцию

        σ(θ, θ̇) = d · [ H(θ, θ̇; −u_max·d) − γ D ] ,      u = −u_max · sign σ.

    Читается так: σ > 0 -- «энергии больше, чем на тормозной кривой, ведущей в
    ноль», надо тормозить; σ < 0 -- «меньше», надо разгоняться к нулю.

    Проверка знака в пределе двойного интегратора (β = 0, D = 0, то есть
    θ̈ = u/α): при θ̇ > 0 получается σ = γ(½ α θ̇² + u_max θ) -- это, с точностью
    до положительного множителя, классическая переключающая функция
    θ + α θ̇|θ̇| / (2 u_max) с u = −u_max sign(·).

    Чего эта формула НЕ обещает
    ---------------------------
    * Оптимальность «одного переключения» -- свойство участка внутри
      восстановимого множества. Дальше, за седлом u = −u_max (уровень K(u_max),
      см. `src/recoverable.py`), спасения нет вообще, и формула вырождается в
      «тормозить изо всех сил» -- лучшее из доступного, но не возврат.
    * Колесо. Релейная фаза считает только наклон; phi за это время уезжает, и
      возвращает его уже ЛКР. Если фаза длинная, состояние может не попасть в
      4-мерный эллипсоид именно из-за phi -- это видно в `examples/bangbang_lqr.py`.
    * На самой кривой σ = 0 управление дребезжит с частотой шага интегратора.
      Это неустранимое свойство релейного закона и одна из причин передавать
      управление ЛКР, а не досиживать до нуля на реле.

    Параметры
    ---------
    K : (n_action, n_state)
        Матрица ЛКР. Синтез -- `wpend.lqr.lqr` или числа снаружи.
    u_max : float
        Модуль релейного момента. Это НЕ дубликат `System.u_bounds`: система
        обрезает то, что регулятор просит, а здесь записано, на что регулятор
        рассчитывает. Если поставить больше системного предела, релейная фаза
        будет считать сепаратрису для мотора, которого нет, и промахнётся;
        `Trajectory.u` при этом честно покажет обрезанное значение.
    region : callable
        Предикат x -> bool (или (M,) bool для пачки). См. `ellipsoid_region`
        и `grid_region`.
    system : WheeledPendulum
        Нужен ровно ради `first_integral` и параметров γ, D -- то есть ради
        формулы сепаратрисы. Никакого состояния из системы не читается.
    wheel_ref : bool
        False (по умолчанию) -- ЛКР гонит колесо в phi = 0, а регион
        проверяется по полному состоянию. True -- ПРИВЯЗКА К КОЛЕСУ (§B8):
        целью становится не phi = 0, а то положение колеса, в котором
        произошла передача.

        Почему это законно, а не подгонка. Координата phi циклична: правая
        часть f от неё не зависит вовсе (Key_Formulas §1.4). Значит для
        e = x - (0, phi_ref, 0, 0) выполняется ė = f(e, u) с той же самой
        функцией f -- сдвиг вдоль phi не меняет векторное поле. Поэтому
        сертификат «V̇ < 0 внутри x^T P x <= c*» переносится на e ДОСЛОВНО,
        без нового доказательства (Khalil, гл. 8.2 + инвариантность поля).

        Что меняется на практике: до передачи регион проверяется по состоянию
        с обнулённой phi -- то есть задаётся вопрос «поместились бы мы в
        эллипсоид, если объявить колесо стоящим там, где оно есть». В момент
        передачи phi_ref защёлкивается, и дальше ЛКР работает как u = -K e,
        то есть выпрямляет корпус и останавливает колесо ТАМ, ГДЕ ОНО
        ОКАЗАЛОСЬ. Цель прогона меняется с «приехать в ноль» на
        «остановиться где стоишь» -- это надо признавать вслух.

        Скорость колеса dphi при этом из проверки НЕ уходит: она не циклична,
        и вклад P[3,3] dphi^2 остаётся. Привязка снимает вклад phi, но не
        вклад dphi -- см. замер в PROPOSALS.md §B8.
    latch : bool
        True (по умолчанию) -- вошёл в регион и остался в ЛКР до `reset()`.
        Сертифицированный эллипсоид инвариантен (внутри V̇ < 0, значит V
        убывает и наружу состояние не выйдет), поэтому защёлка не отнимает
        корректности, но снимает дребезг на границе от численного шума.
        False -- регион проверяется заново каждый шаг.

    Память (флаг защёлки) живёт в регуляторе -- правило 2 архитектуры; для
    пачки (M, n) это вектор из M флагов, независимых по строкам. `reset()`
    вызывается `rollout`-ом один раз перед прогоном.
    """

    def __init__(self, K, u_max, region, system, *, i_theta=0, i_phi=1,
                 i_dtheta=2, latch=True, wheel_ref=False):
        self.K = np.atleast_2d(np.asarray(K, dtype=float))
        self.u_max = float(u_max)
        self.region = region
        self.system = system
        self.i_theta = int(i_theta)
        self.i_phi = int(i_phi)
        self.i_dtheta = int(i_dtheta)
        self.latch = bool(latch)
        self.wheel_ref = bool(wheel_ref)
        # H(0, 0; u) = γ D при любом u -- уровень, на котором лежит кривая,
        # приходящая в верхнее положение.
        self.H_upright = float(system.first_integral(
            np.zeros(system.n_state), 0.0))
        self._engaged = None
        self._phi_ref = None

    def reset(self) -> None:
        self._engaged = None
        self._phi_ref = None

    def switching_function(self, x_hat):
        """σ(θ, θ̇) из докстринга класса. Вынесено отдельно ради теста:
        вдоль траектории с постоянным u = −u_max, стартующей с σ = 0,
        величина σ обязана оставаться нулём -- это сохранение H, а не
        сравнение с прошлым прогоном."""
        x = np.asarray(x_hat, dtype=float)
        theta, dtheta = x[..., self.i_theta], x[..., self.i_dtheta]
        # При θ̇ = 0 направление задаёт сам наклон: из (θ > 0, θ̇ = 0) надо
        # толкать назад, а не «в сторону, куда летим», -- лететь ещё некуда.
        d = np.where(dtheta != 0.0, np.sign(dtheta), np.sign(theta))
        return d * (self.system.first_integral(x, -self.u_max * d)
                    - self.H_upright)

    def _probe(self, x):
        """Состояние, по которому проверяется регион.

        При привязке к колесу вопрос звучит как «поместились бы мы, если
        объявить колесо стоящим там, где оно есть», поэтому phi обнуляется.
        Скорость колеса не трогаем: она не циклична и из проверки не уходит.
        """
        if not self.wheel_ref:
            return x
        probe = np.array(x, dtype=float, copy=True)
        probe[..., self.i_phi] = 0.0
        return probe

    def _error(self, x):
        """Состояние, которое видит ЛКР: x либо x - (0, phi_ref, 0, 0)."""
        if not self.wheel_ref:
            return x
        err = np.array(x, dtype=float, copy=True)
        err[..., self.i_phi] = x[..., self.i_phi] - self._phi_ref
        return err

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_hat, dtype=float)

        inside = np.asarray(self.region(self._probe(x)), dtype=bool)
        if self._engaged is None or self._engaged.shape != inside.shape:
            self._engaged = np.zeros(inside.shape, dtype=bool)
            self._phi_ref = np.zeros(inside.shape, dtype=float)
        # phi_ref защёлкивается на переднем фронте: цель фиксируется один раз,
        # в момент передачи, и дальше не уползает вслед за колесом.
        newly = inside & ~self._engaged
        self._phi_ref = np.where(newly, x[..., self.i_phi], self._phi_ref)
        self._engaged = (self._engaged | inside) if self.latch else inside

        # σ >= 0 -> тормозим. Равенство попадает в ту же ветвь сознательно:
        # sign(0) = 0 дало бы u = 0, то есть «отпустить мотор» ровно на той
        # кривой, вдоль которой и надо ехать.
        u_bang = np.where(self.switching_function(x) >= 0.0,
                          -self.u_max, self.u_max)[..., None]
        u_lqr = -self._error(x) @ self.K.T
        return np.where(self._engaged[..., None], u_lqr, u_bang)


class TrajectoryTrackingController(Controller):
    """Слежение за заранее рассчитанной траекторией: ЛКР, меняющийся во времени.

        k = round((t - t0) / dt),
        u = u_bar_k - K_k (x_hat - x_bar_k).

    Откуда числа. `wpend.ddp.ilqr` один раз до прогона считает номинальную
    траекторию (x_bar, u_bar) и матрицы K_k. Синтеза здесь нет, как и у
    `LinearFeedbackController`: регулятор только исполняет план.

    Зачем обратная связь, если план оптимален. Без неё (u = u_bar_k) это
    разомкнутое управление: любое отклонение -- другой x0, шум, неточная
    модель -- растёт вдоль неустойчивой траектории маятника. K_k -- это
    линеаризация оптимального закона вокруг плана: оно говорит, как исправлять
    отклонение δx на шаге k. Это та же K, что в обратном проходе iLQR
    (Tassa, Erez & Todorov, IROS 2012, §II; Tedrake, *Underactuated Robotics*,
    глава про LQR-слежение за траекторией).

    Что происходит за горизонтом. Шаг k обрезается до N-1, и используется
    последняя точка плана x_bar_N. Закон тогда постоянный:
    u = -K_{N-1} x + (u_bar_{N-1} + K_{N-1} x_bar_N). Если цель плана --
    ноль, а Q_f = P из `lqr()`, то конец плана уже живёт в режиме ЛКР,
    u_bar_{N-1} ≈ -K_{N-1} x_bar_N, скобка ≈ 0, и регулятор превращается в
    обычный ЛКР с K_{N-1} ≈ K. Замер (theta0 = 0.3, план 3 с, dt = 0.01):
    скобка ≈ 0.004 Н·м, к 9 с |theta| ≈ 1e-4. Для других целей это не
    гарантировано, и прогон длиннее плана -- отдельный вопрос.

    Время. Индекс считается из t, а не счётчиком вызовов: так регулятор не
    хранит памяти, и `reset()` ему не нужен. `rollout` даёт t = t0 + k·dt
    точно (формулой, без накопления), поэтому round безопасен. dt регулятора
    ОБЯЗАН совпадать с dt прогона: иначе план исполняется в другом темпе.

    Пачка (M, n) поддерживается: t общее для всех строк, поэтому k одно и то
    же, а x_hat - x_bar_k транслируется по строкам.

    Параметры
    ---------
    x_bar : (N+1, n)
    u_bar : (N, m)
    K : (N, m, n)
    dt : float
    t0 : float
        Момент, которому соответствует x_bar_0.
    """

    def __init__(self, x_bar, u_bar, K, dt, t0=0.0):
        self.x_bar = np.asarray(x_bar, dtype=float)
        self.u_bar = np.asarray(u_bar, dtype=float)
        self.K = np.asarray(K, dtype=float)
        self.dt = float(dt)
        self.t0 = float(t0)
        n_steps = self.u_bar.shape[0]
        if self.x_bar.shape[0] != n_steps + 1 or self.K.shape[0] != n_steps:
            raise ValueError(
                f"форма плана не согласована: x_bar {self.x_bar.shape}, "
                f"u_bar {self.u_bar.shape}, K {self.K.shape} -- нужно "
                "(N+1, n), (N, m), (N, m, n)")

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_hat, dtype=float)
        n_steps = self.u_bar.shape[0]
        k = int(round((t - self.t0) / self.dt))
        k_u = min(max(k, 0), n_steps - 1)
        k_x = min(max(k, 0), n_steps)       # за горизонтом держим x_bar_N
        error = x - self.x_bar[k_x]
        return self.u_bar[k_u] - error @ self.K[k_u].T


class MPCController(Controller):
    """Управление с прогнозирующей моделью (MPC) на основе `wpend.ddp.ilqr`.

    Раз в такт управления dt_plan регулятор:
      1. берёт текущую оценку x_hat как начальное состояние;
      2. решает задачу iLQR на горизонте N шагов dt_plan по СВОЕЙ модели,
         начиная с прошлого плана, сдвинутого на шаг (тёплый старт);
      3. прикладывает первое управление плана u_bar_0 и держит его до
         следующего такта (zero-order hold).
    Остальной план выбрасывается; от него остаётся только тёплый старт.

    Где здесь обратная связь. Явной матрицы K в законе нет: обратная связь --
    это само перепланирование из измеренного состояния. На линейной системе с
    квадратичной ценой это можно проверить точно: решение задачи из любого x
    -- линейная функция u = -K_0 x, и MPC обязан совпасть с этой постоянной
    обратной связью (`tests/test_mpc.py`).

    Почему это законно в пяти слоях
    -------------------------------
    * Память (прошлый план, момент следующего такта, удерживаемое u) живёт
      в регуляторе -- правило 2. `reset()` её стирает.
    * Модель внутри регулятора -- это модель РЕГУЛЯТОРА, а не мир: `rollout`
      по-прежнему двигает истинную систему. Их можно намеренно сделать
      разными (другая масса, другой интегратор) -- опыт «MPC с неточной
      моделью» получается подменой аргумента.
    * Правило 5 (один dt) не нарушено: rollout зовёт `act` каждый шаг, а
      регулятор сам решает, что между тактами управление не меняется. dt_plan
      должен быть кратен dt прогона, иначе такт «плывёт» относительно сетки
      прогона (такт срабатывает на первом шаге, где t >= t_next). Честное
      разделение dt_control / dt_sim -- по-прежнему PROPOSALS.md §B2.
    * Предел момента планировщик берёт из СВОЕЙ модели: `model.u_bounds`.
      Модель с `u_max` -- планирование с пределом (box-DDP, `wpend/ddp.py`),
      план допустим, и `System.clip_action` мира его не меняет. Модель без
      предела -- прежний опыт «планировщик не знает, мир обрезает».

    Устойчивость. Терминальная цена Q_f = P из `lqr()` -- классическое
    условие устойчивости MPC: хвост за горизонтом оценён ценой ЛКР, который
    доведёт систему до нуля (Rawlings, Mayne & Diehl, *Model Predictive
    Control*, 2-е изд., гл. 2). Строго это доказано для точного решения и
    терминального множества; здесь ни того, ни другого нет, так что это
    ориентир, а не сертификат.

    Параметры
    ---------
    model, integrator
        Модель планировщика (обычно та же система, что в прогоне).
    dt_plan : float
        Шаг плана и такт перепланирования.
    horizon : int
        N -- число шагов плана. Горизонт в секундах -- N·dt_plan.
    Q, R, Q_f
        Веса цены, как в `ilqr`.
    x_goal : (n,) | None
    n_iter : int
        Итераций iLQR на такт. Тёплый старт уже почти оптимален, поэтому
        хватает 1-2 (Tassa, Erez & Todorov, IROS 2012, §III).
    first_iter : int
        Итераций на ПЕРВОМ такте, где тёплого старта ещё нет.
    K_init : (m, n) | None
        Как получить начальное приближение на первом такте: прогон модели
        под u = -K_init x. None -- нули. Для неустойчивой системы нули плохи:
        прогон падает, и iLQR застревает в локальном минимуме (замер в
        `tests/test_ddp.py`: цена 3653 против 84.1).
    record_plans : bool
        Писать в `self.plans` каждый план (t, x_bar, u_bar, итерации, время
        расчёта). Только для просмотра поведения; на управление не влияет.
    mu : float
        Регуляризация Q_uu в iLQR (см. `ilqr`). Она сдвигает не только шаг, но
        и матрицы G: на линейной задаче при mu = 1e-6 управление отличается от
        точного на 4e-5 относительно (Q_uu ≈ 0.05). Для прогонов это шум, для
        точного оракула в тестах ставится mu = 0.

    Пачка (M, n): каждая строка -- независимая задача со своим планом; цена
    растёт в M раз. Для сетки начальных условий это самая дорогая вещь в
    проекте, окно так её вызывать не должно.
    """

    def __init__(self, model, integrator, dt_plan, horizon, Q, R, Q_f, *,
                 x_goal=None, n_iter=2, first_iter=50, K_init=None, mu=1e-6,
                 record_plans=False):
        self.model = model
        self.integrator = integrator
        self.dt_plan = float(dt_plan)
        self.horizon = int(horizon)
        self.Q, self.R, self.Q_f = Q, R, Q_f
        self.x_goal = x_goal
        self.n_iter = int(n_iter)
        self.first_iter = int(first_iter)
        self.K_init = None if K_init is None else np.atleast_2d(
            np.asarray(K_init, dtype=float))
        self.mu = float(mu)
        self.record_plans = bool(record_plans)
        self.reset()

    def reset(self) -> None:
        self._plans = None        # (M, N, m): u_bar последнего плана по строкам
        self._u_hold = None       # (M, m): управление, которое держим до такта
        self._t_next = None       # момент следующего перепланирования
        self.n_replans = 0        # счётчик для замеров
        # Журнал планов (record_plans=True): что регулятор ПРЕДСКАЗЫВАЛ на каждом
        # такте. Нужен окну, которое показывает поведение алгоритма; для
        # пачки пишется только строка 0, иначе память росла бы в M раз.
        self.plans = []

    def _initial_guess(self, x0):
        """Прогон модели под u = -K_init x -- начальное приближение."""
        m = self.model.n_action
        u = np.zeros((self.horizon, m))
        if self.K_init is None:
            return u
        x = np.array(x0, dtype=float)
        for k in range(self.horizon):
            # clip_action модели: если модель знает предел, начальный план
            # стартует допустимым -- это ЛКР с насыщением, а не без него.
            u[k] = self.model.clip_action(-self.K_init @ x)
            x = self.integrator.step(self.model.f, k * self.dt_plan, x, u[k],
                                     self.dt_plan)
        return u

    def _replan(self, t, x_row, i):
        from .ddp import ilqr     # отложенный импорт: ddp -- не слой, см. модуль

        if self._plans[i] is None:
            u_init, max_iter = self._initial_guess(x_row), self.first_iter
        else:
            # Тёплый старт: план сдвигается на шаг, последнее управление
            # повторяется. На конце плана система уже в режиме ЛКР, поэтому
            # повтор -- разумное продолжение.
            prev = self._plans[i]
            u_init = np.concatenate([prev[1:], prev[-1:]], axis=0)
            max_iter = self.n_iter
        t_wall = time.perf_counter()
        x_bar, u_bar, _, costs = ilqr(self.model, self.integrator, x_row, self.dt_plan,
                                      self.horizon, self.Q, self.R, self.Q_f,
                                      x_goal=self.x_goal, u_init=u_init, t0=t,
                                      max_iter=max_iter, mu=self.mu,
                                      u_bounds=self.model.u_bounds)
        if self.record_plans and i == 0:
            self.plans.append(dict(t=t, x=x_bar, u=u_bar[:, 0], iters=len(costs) - 1,
                                   wall=time.perf_counter() - t_wall))
        self._plans[i] = u_bar
        return u_bar[0]

    def act(self, t: float, x_hat: np.ndarray) -> np.ndarray:
        x = np.asarray(x_hat, dtype=float)
        rows = x.reshape(-1, x.shape[-1])
        M = rows.shape[0]
        if self._plans is None or len(self._plans) != M:
            self._plans = [None] * M
            self._u_hold = np.zeros((M, self.model.n_action))
            self._t_next = None

        # 1e-9 -- допуск на округление: t приходит как t0 + k·dt, а t_next
        # копится суммой, и точное равенство может не выполниться.
        if self._t_next is None or t >= self._t_next - 1e-9:
            for i in range(M):
                self._u_hold[i] = self._replan(t, rows[i], i)
            self.n_replans += 1
            base = t if self._t_next is None else self._t_next
            self._t_next = base + self.dt_plan
        return self._u_hold.reshape(x.shape[:-1] + (self.model.n_action,)).copy()
