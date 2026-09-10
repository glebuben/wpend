"""Колёсный маятник (segway): корпус на одном колесе, привод -- момент мотора.

Состояние x = (theta, phi, dtheta, dphi):
    theta -- наклон корпуса от вертикали (0 -- стойка вверх),
    phi   -- угол поворота колеса в мировой системе (качение без проскальзывания).
Управление u -- момент мотора между корпусом и колесом; на корпус действует
+u, на колесо -u (третий закон Ньютона).

Все формулы -- из docs/Key_Formulas.md (ветка master), §1-2.  Лумпированные
параметры:

    alpha = I_b + m_b l^2,   beta = m_b r l,
    gamma = I_w + (m_w + m_b) r^2,   D = m_b g l,
    b(theta)     = gamma + beta cos(theta),
    Delta(theta) = alpha gamma - beta^2 cos^2(theta) > 0.

Уравнения Эйлера--Лагранжа (§1.1):
    alpha ddtheta + beta cos(theta) ddphi - D sin(theta) =  u
    beta cos(theta) ddtheta + gamma ddphi - beta sin(theta) dtheta^2 = -u

Разрешая относительно ускорений (определитель = Delta), получаем §1.3:

    ddtheta = [ b(theta) u + gamma D sin(theta)
                - beta^2 sin(theta) cos(theta) dtheta^2 ] / Delta
    ddphi   = [ alpha beta sin(theta) dtheta^2 - beta D sin(theta) cos(theta)
                - (alpha + beta cos(theta)) u ] / Delta

Ни phi, ни dphi в правую часть не входят: phi -- циклическая координата
(§1.4, галилеева инвариантность идеального качения).  Это встроенный тест
корректности модели -- см. tests/test_wheeled_pendulum.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..system import System


@dataclass
class WheeledPendulumParams:
    """Физические параметры. Значения по умолчанию -- из отчёта (тонкий стержень
    как корпус, малое колесо)."""

    mb: float = 10.0    # масса корпуса                   [кг]
    mw: float = 1.0     # масса колеса                    [кг]
    l: float = 0.30     # центр масс корпуса над осью     [м]
    r: float = 0.05     # радиус колеса                   [м]
    g: float = 9.8      # ускорение свободного падения    [м/с^2]
    u_max: float | None = None   # предел момента мотора  [Н·м], None = без предела

    @property
    def Ib(self) -> float:
        """Центральный момент инерции корпуса (тонкий стержень)."""
        return self.mb * self.l ** 2 / 3.0

    @property
    def Iw(self) -> float:
        """Момент инерции колеса относительно его центра."""
        return 0.7 * self.mw * self.r ** 2

    @property
    def alpha(self) -> float:
        return self.Ib + self.mb * self.l ** 2

    @property
    def beta(self) -> float:
        return self.mb * self.r * self.l

    @property
    def gamma(self) -> float:
        return self.Iw + (self.mw + self.mb) * self.r ** 2

    @property
    def D(self) -> float:
        return self.mb * self.g * self.l


class WheeledPendulum(System):
    """dx/dt = f(t, x, u), x = (theta, phi, dtheta, dphi), u = (torque,)."""

    def __init__(self, params: WheeledPendulumParams | None = None, **kwargs):
        self.p = params if params is not None else WheeledPendulumParams(**kwargs)

    @property
    def n_state(self) -> int:
        return 4

    @property
    def n_action(self) -> int:
        return 1

    @property
    def state_names(self) -> tuple[str, ...]:
        return ("theta", "phi", "dtheta", "dphi")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ("torque",)

    @property
    def u_bounds(self):
        if self.p.u_max is None:
            return None
        return (np.array([-self.p.u_max]), np.array([self.p.u_max]))

    # --- вспомогательные скаляры ------------------------------------------

    def b(self, theta) -> float:
        """b(theta) = gamma + beta cos(theta) -- множитель при u в ddtheta."""
        return self.p.gamma + self.p.beta * np.cos(theta)

    def Delta(self, theta) -> float:
        """Delta(theta) = det M = alpha gamma - beta^2 cos^2(theta).

        Строго положителен для физичных параметров (неравенство Коши--Буняковского
        для матрицы масс), поэтому уравнения всегда разрешимы относительно
        ускорений.
        """
        p = self.p
        return p.alpha * p.gamma - p.beta ** 2 * np.cos(theta) ** 2

    # --- динамика ----------------------------------------------------------

    def f(self, t, x, u):
        # Индексы x[..., i], а не распаковка: так одна и та же формула считает
        # и одно состояние (4,), и пачку (M, 4) -- см. System.f.
        theta = x[..., 0]
        dtheta = x[..., 2]
        dphi = x[..., 3]
        p = self.p
        s, c = np.sin(theta), np.cos(theta)
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        torque = u[..., 0]

        ddtheta = ((p.gamma + p.beta * c) * torque
                   + p.gamma * p.D * s
                   - p.beta ** 2 * s * c * dtheta ** 2) / Delta

        ddphi = (p.alpha * p.beta * s * dtheta ** 2
                 - p.beta * p.D * s * c
                 - (p.alpha + p.beta * c) * torque) / Delta

        return np.stack([dtheta, dphi, ddtheta, ddphi], axis=-1)

    # --- что видит акселерометр (физика датчика, а не датчик) ---------------

    def specific_force(self, t, x, u, d: float = 0.0):
        """Удельная сила на датчике, стоящем на корпусе на выносе d от оси колеса.

        Акселерометр меряет НЕ ускорение, а удельную силу f = R^T (a - g_world):
        в свободном падении он показывает ноль, в покое -- вектор длиной g,
        направленный вверх.  Именно поэтому по нему вообще можно судить о
        наклоне, и именно поэтому при разгоне он врёт.

        Датчик стоит в точке p_s = (r*phi + d*sin(theta), r + d*cos(theta)).
        Оси корпуса: e_z = (sin theta, cos theta) -- вдоль корпуса вверх,
        e_x = (cos theta, -sin theta) -- вперёд.  Дважды дифференцируя p_s,
        проецируя (d2p_s/dt2 - g_world) на эти оси и сокращая, получаем

            f_x = -g sin(theta) + d*ddtheta + r cos(theta)*ddphi
            f_z =  g cos(theta) - d*dtheta^2 + r sin(theta)*ddphi

        В покое f = (-g sin theta, g cos theta): модуль ровно g, и
        atan2(-f_x, f_z) = theta.  Отсюда "кажущаяся вертикаль": как только
        корпус разгоняется, члены с ddphi и ddtheta сдвигают этот угол, и
        ошибка наклона коррелирует с моментом.  Это не шум датчика, а его
        физика -- усреднением она не убирается, нужен гироскоп (ER-008).

        Ускорения берутся из собственной правой части при управлении u,
        поэтому u обязателен: без него удельная сила не определена.  Формула
        живёт в МОДЕЛИ, а не в датчике: это свойство машины и места установки,
        как first_integral или recoverable_bounds (A7, A16).

        Соглашение о пачках соблюдено: x формы (4,) или (M, 4), результат --
        пара массивов той же ведущей формы.

        Источники: Groves, "Principles of GNSS, Inertial, and Multisensor
        Integrated Navigation Systems", 2-е изд., гл. 2.4 (удельная сила) и
        гл. 4.1; docs/Key_Formulas.md §1.3 (откуда ddtheta и ddphi).
        """
        if u is None:
            raise ValueError(
                "specific_force: нужен момент u -- в удельную силу входят "
                "ускорения корпуса, а они зависят от приложенного момента"
            )
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        dx = self.f(t, x, u)
        theta = x[..., 0]
        dtheta = x[..., 2]
        ddtheta = dx[..., 2]
        ddphi = dx[..., 3]
        s, c = np.sin(theta), np.cos(theta)
        g, r = self.p.g, self.p.r
        f_x = -g * s + d * ddtheta + r * c * ddphi
        f_z = g * c - d * dtheta ** 2 + r * s * ddphi
        return f_x, f_z

    # --- оракулы для тестов и анализа --------------------------------------

    def first_integral(self, x, u_const) -> float:
        """H(theta, dtheta; u) = 1/2 Delta dtheta^2 - u (gamma theta + beta sin theta)
        + gamma D cos(theta).

        При ПОСТОЯННОМ u выполняется dH/dt = 0 (Key_Formulas §3, вывод в
        приложении B): приведённая динамика наклона одномерна и потому
        интегрируема.  Именно на уровнях H строится множество восстановимых
        состояний.  Для тестов это точный оракул: RK4 обязан сохранять H с
        точностью O(dt^4).

        u_const -- скаляр ЛИБО массив, транслируемый на пачку состояний: у
        релейного закона знак момента свой в каждой строке пачки (см.
        BangBangLQRController.switching_function).  Формула написана через
        x[..., i], поэтому это работает без оговорок -- пинится тестом
        tests/test_wheeled_pendulum.py::test_first_integral_broadcasts_u.
        """
        x = np.asarray(x, dtype=float)
        theta, dtheta = x[..., 0], x[..., 2]
        p = self.p
        return (0.5 * self.Delta(theta) * dtheta ** 2
                - u_const * (p.gamma * theta + p.beta * np.sin(theta))
                + p.gamma * p.D * np.cos(theta))

    # --- множество восстановимости (Key_Formulas §3.2) ----------------------
    #
    # Всё, что ниже, НЕ зависит от регулятора. Это ответ на вопрос «может ли
    # хоть какое-нибудь управление |u| <= u_max вернуть корпус в вертикаль»,
    # и он существует в замкнутом виде ровно потому, что приведённая динамика
    # наклона одномерна (phi циклична, §1.4) и при постоянном u интегрируема.

    def saddle_angle(self, u_max: float) -> float:
        """Угол theta_eq > 0, при котором предельный момент u = -u_max ровно
        уравновешивает силу тяжести:

            gamma D sin(theta_eq) = u_max (gamma + beta cos(theta_eq)).      (*)

        Это положение равновесия приведённой динамики, и оно СЕДЛОВОЕ: при
        theta < theta_eq момент пересиливает тяжесть и возвращает корпус, при
        theta > theta_eq тяжесть пересиливает и корпус уходит. Дальше этого
        угла из состояния покоя не спастись никаким управлением.

        Корень ищется делением пополам: (*) монотонна на (0, pi/2), а тащить
        ради одного скаляра scipy в ядро незачем. Если мотор сильнее веса
        (u_max >= D), равновесия нет вовсе и возвращается pi/2 -- корпус
        удерживается на любом наклоне до горизонтали.
        """
        p = self.p
        if u_max <= 0.0:
            return 0.0
        if u_max >= p.D:
            return np.pi / 2.0
        f = lambda th: u_max * (p.gamma + p.beta * np.cos(th)) - p.gamma * p.D * np.sin(th)
        lo, hi = 0.0, np.pi / 2.0 - 1e-7
        if f(lo) * f(hi) > 0.0:
            return np.pi / 2.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if f(lo) * f(mid) <= 0.0:
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    def separatrix_level(self, u_max: float) -> float:
        """K(u_max) = H(theta_eq, 0; -u_max) -- уровень первого интеграла,
        проходящий через седло. Граница восстановимости лежит на нём."""
        p = self.p
        th = self.saddle_angle(u_max)
        return (u_max * (p.gamma * th + p.beta * np.sin(th))
                + p.gamma * p.D * np.cos(th))

    def recoverable_bounds(self, u_max: float, theta):
        """(floor, ceiling) -- границы по dtheta множества восстановимости.

        Состояние (theta, dtheta) восстановимо тогда и только тогда, когда
        floor(theta) < dtheta < ceiling(theta). Снаружи не помогает НИКАКОЙ
        регулятор: это свойство системы и предела мотора, а не закона
        управления.

        Из H(theta, dtheta; -+u_max) = K(u_max) получается

            dtheta^2 = 2 [ K - gamma D cos(theta) -+ u_max (gamma theta
                           + beta sin(theta)) ] / Delta(theta).

        ЛОВУШКА, ради которой это написано отдельным методом. Граница -- не
        линия уровня H = K, а УСТОЙЧИВОЕ МНОГООБРАЗИЕ седла, и оно составляет
        лишь часть этой линии. У седла устойчивый собственный вектор имеет
        отрицательный наклон, поэтому нужная ветвь меняет знак dtheta при
        переходе через theta_eq:

            ceiling(theta) = +sqrt(sq_minus)  при theta <  theta_eq,
                             -sqrt(sq_minus)  при theta >  theta_eq;
            floor(theta)   = -sqrt(sq_plus)   при theta > -theta_eq,
                             +sqrt(sq_plus)   при theta < -theta_eq.

        Смена знака -- это не косметика: она добавляет «языки» за +-theta_eq.
        Корпус, наклонённый почти горизонтально, ВСЁ ЕЩЁ восстановим, если
        качается назад достаточно быстро -- но не слишком быстро, иначе
        перелетит в другую сторону. Взять просто линию уровня значит потерять
        эти области и объявить невосстановимым то, что восстановимо.

        Работает и со скаляром, и с массивом theta -- соглашение A8.

        Источник: `docs/Key_Formulas.md` §3.2 и приложение C (ветка master);
        Khalil, *Nonlinear Systems*, 3-е изд., гл. 4 (области притяжения) и
        гл. 8 (седловые многообразия). Численно это же считает
        `src/recoverable.py` со второго семестра -- совпадение проверено
        тестом.
        """
        p = self.p
        theta = np.asarray(theta, dtype=float)
        K = self.separatrix_level(u_max)
        th_eq = self.saddle_angle(u_max)

        work = u_max * (p.gamma * theta + p.beta * np.sin(theta))
        gap = K - p.gamma * p.D * np.cos(theta)
        Delta = self.Delta(theta)
        sq_minus = 2.0 * (gap - work) / Delta      # H(-u_max) = K
        sq_plus = 2.0 * (gap + work) / Delta       # H(+u_max) = K

        sm = np.sqrt(np.clip(sq_minus, 0.0, None))
        sp = np.sqrt(np.clip(sq_plus, 0.0, None))
        ceiling = np.where(theta < th_eq, sm, -sm)
        floor = np.where(theta > -th_eq, -sp, sp)
        return floor, ceiling

    def is_recoverable(self, u_max: float, theta, dtheta):
        """Булев ответ «отсюда можно вернуться при |u| <= u_max»."""
        floor, ceiling = self.recoverable_bounds(u_max, theta)
        return (np.asarray(dtheta) > floor) & (np.asarray(dtheta) < ceiling)

    def linearize_tilt(self):
        """(A_t, B_t) приведённой подсистемы наклона z = (theta, dtheta).

            A_t = [[0, 1], [gamma D / Delta_0, 0]],   B_t = [0, (gamma+beta)/Delta_0]^T

        Почему такая подсистема вообще существует. В правые части НЕ входят ни
        phi, ни dphi (§1.4): наклон живёт своей жизнью, а колесо -- это цепочка
        интеграторов, ведомая наклоном и моментом. Система треугольная, и
        верхнее звено этой цепочки замкнуто само на себя.

        Это ровно строки и столбцы (theta, dtheta) из `linearize_upright` --
        тест сверяет их поэлементно. Отдельный метод нужен потому, что синтез
        ЛКР по этой паре даёт СУЩЕСТВЕННО другой регулятор: четырёхмерная
        задача обязана возвращать колесо, а единственный рычаг для этого --
        наклон, поэтому она вынуждена держать наклон жёстко. Убрав колесо из
        цели, получаем гейны примерно в шесть раз мягче, меньше насыщения и
        заметно более широкую сертифицированную область.

        ВАЖНО: сертификат, построенный по этой паре, доказывает сходимость
        наклона при управлении u = -K_t z И НИЧЕГО НЕ ГОВОРИТ о колесе. Если
        переключиться по нему, а подать полный четырёхмерный ЛКР, сертификат
        недействителен -- и это не формальность, замер в PROPOSALS.md §B8
        показывает потерю 292 клеток из 1081.
        """
        p = self.p
        D0 = p.alpha * p.gamma - p.beta ** 2
        A = np.array([[0.0, 1.0], [p.gamma * p.D / D0, 0.0]])
        B = np.array([[0.0], [(p.gamma + p.beta) / D0]])
        return A, B

    def linearize_upright(self):
        """(A, B) в верхнем положении x = 0, u = 0 (Key_Formulas §2.2), где
        Delta_0 = alpha gamma - beta^2:

            A = [[0,0,1,0],[0,0,0,1],[gamma D/Delta_0,0,0,0],[-beta D/Delta_0,0,0,0]]
            B = [0, 0, (gamma+beta)/Delta_0, -(alpha+beta)/Delta_0]^T
        """
        p = self.p
        D0 = p.alpha * p.gamma - p.beta ** 2
        A = np.zeros((4, 4))
        A[0, 2] = 1.0
        A[1, 3] = 1.0
        A[2, 0] = p.gamma * p.D / D0
        A[3, 0] = -p.beta * p.D / D0
        B = np.array([[0.0], [0.0], [(p.gamma + p.beta) / D0], [-(p.alpha + p.beta) / D0]])
        return A, B
