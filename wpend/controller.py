"""Слой 4 из 5: РЕГУЛЯТОР.

Регулятор превращает оценку состояния x_hat в требуемое управление u.
Он НЕ знает, откуда взялась оценка, и НЕ знает про ограничения мотора --
за проекцию на допустимое множество отвечает System.clip_action.
"""

from __future__ import annotations

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
