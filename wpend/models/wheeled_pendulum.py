"""Колёсный маятник (segway): корпус на одном колесе, привод -- момент мотора.

Состояние x = (psi, theta, dpsi, dtheta):
    psi -- наклон корпуса от вертикали (0 -- стойка вверх),
    theta   -- угол поворота колеса в мировой системе (качение без проскальзывания).
Управление u -- момент мотора между корпусом и колесом; на корпус действует
+u, на колесо -u (третий закон Ньютона).

Все формулы -- из docs/Key_Formulas.md (ветка master), §1-2.  Лумпированные
параметры:

    alpha = I_b + m_b l^2,   beta = m_b r l,
    gamma = I_w + (m_w + m_b) r^2,   D = m_b g l,
    b(psi)     = gamma + beta cos(psi),
    Delta(psi) = alpha gamma - beta^2 cos^2(psi) > 0.

Уравнения Эйлера--Лагранжа (§1.1):
    alpha ddpsi + beta cos(psi) ddtheta - D sin(psi) =  u
    beta cos(psi) ddpsi + gamma ddtheta - beta sin(psi) dpsi^2 = -u

Разрешая относительно ускорений (определитель = Delta), получаем §1.3:

    ddpsi = [ b(psi) u + gamma D sin(psi)
                - beta^2 sin(psi) cos(psi) dpsi^2 ] / Delta
    ddtheta   = [ alpha beta sin(psi) dpsi^2 - beta D sin(psi) cos(psi)
                - (alpha + beta cos(psi)) u ] / Delta

Ни theta, ни dtheta в правую часть не входят: theta -- циклическая координата
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
    l: float = 1.0     # центр масс корпуса над осью     [м]
    r: float = 0.3     # радиус колеса                   [м]
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
    """dx/dt = f(t, x, u), x = (psi, theta, dpsi, dtheta), u = (torque,)."""

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
        return ("psi", "theta", "dpsi", "dtheta")

    @property
    def action_names(self) -> tuple[str, ...]:
        return ("torque",)

    @property
    def u_bounds(self):
        if self.p.u_max is None:
            return None
        return (np.array([-self.p.u_max]), np.array([self.p.u_max]))

    # --- вспомогательные скаляры ------------------------------------------

    def b(self, psi) -> float:
        """b(psi) = gamma + beta cos(psi) -- множитель при u в ddpsi."""
        return self.p.gamma + self.p.beta * np.cos(psi)

    def Delta(self, psi) -> float:
        """Delta(psi) = det M = alpha gamma - beta^2 cos^2(psi).

        Строго положителен для физичных параметров (неравенство Коши--Буняковского
        для матрицы масс), поэтому уравнения всегда разрешимы относительно
        ускорений.
        """
        p = self.p
        return p.alpha * p.gamma - p.beta ** 2 * np.cos(psi) ** 2

    # --- динамика ----------------------------------------------------------

    def f(self, t, x, u):
        if np.ndim(x) == 1 and np.ndim(u) == 1:
            return self._f1(x, u)
        # Индексы x[..., i], а не распаковка: так одна и та же формула считает
        # и одно состояние (4,), и пачку (M, 4) -- см. System.f.
        psi = x[..., 0]
        dpsi = x[..., 2]
        dtheta = x[..., 3]
        p = self.p
        s, c = np.sin(psi), np.cos(psi)
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        torque = u[..., 0]

        ddpsi = ((p.gamma + p.beta * c) * torque
                   + p.gamma * p.D * s
                   - p.beta ** 2 * s * c * dpsi ** 2) / Delta

        ddtheta = (p.alpha * p.beta * s * dpsi ** 2
                 - p.beta * p.D * s * c
                 - (p.alpha + p.beta * c) * torque) / Delta

        return np.stack([dpsi, dtheta, ddpsi, ddtheta], axis=-1)

    def _f1(self, x, u):
        """f для одного состояния (4,) на float (docs/kalman_speed.md §3, S4).

        Одна клетка звала f с массивами формы (4,): ~20 операций над
        скалярами numpy и np.stack -- около 18 мкс на вызов, четыре вызова
        на шаг RK4.  Здесь та же формула в том же порядке операций на float.
        sin и cos -- те же np.sin/np.cos, квадраты -- умножением (numpy так и
        считает x**2), поэтому результат совпадает с батч-формулой бит в бит
        (оракул -- tests/test_accel_kalman.py)."""
        p = self.p
        psi, dpsi, dtheta = float(x[0]), float(x[2]), float(x[3])
        torque = float(u[0])
        s, c = float(np.sin(psi)), float(np.cos(psi))
        al, be, ga, D = p.alpha, p.beta, p.gamma, p.D
        be2 = be ** 2
        Delta = al * ga - be2 * (c * c)
        dp2 = dpsi * dpsi
        ddpsi = ((ga + be * c) * torque + ga * D * s - be2 * s * c * dp2) / Delta
        ddtheta = (al * be * s * dp2 - be * D * s * c - (al + be * c) * torque) / Delta
        return np.array([dpsi, dtheta, ddpsi, ddtheta])

    # --- что видит акселерометр (физика датчика, а не датчик) ---------------

    def specific_force(self, t, x, u, d: float = 0.0, w=None):
        """Удельная сила на датчике, стоящем на корпусе на выносе d от оси колеса.

        Акселерометр меряет НЕ ускорение, а удельную силу f = R^T (a - g_world):
        в свободном падении он показывает ноль, в покое -- вектор длиной g,
        направленный вверх.  Именно поэтому по нему вообще можно судить о
        наклоне, и именно поэтому при разгоне он врёт.

        Датчик стоит в точке p_s = (r*theta + d*sin(psi), r + d*cos(psi)).
        Оси корпуса: e_z = (sin psi, cos psi) -- вдоль корпуса вверх,
        e_x = (cos psi, -sin psi) -- вперёд.  Дважды дифференцируя p_s,
        проецируя (d2p_s/dt2 - g_world) на эти оси и сокращая, получаем

            f_x = -g sin(psi) + d*ddpsi + r cos(psi)*ddtheta
            f_z =  g cos(psi) - d*dpsi^2 + r sin(psi)*ddtheta

        В покое f = (-g sin psi, g cos psi): модуль ровно g, и
        atan2(-f_x, f_z) = psi.  Отсюда "кажущаяся вертикаль": как только
        корпус разгоняется, члены с ddtheta и ddpsi сдвигают этот угол, и
        ошибка наклона коррелирует с моментом.  Это не шум датчика, а его
        физика -- усреднением она не убирается, нужен гироскоп (ER-008).

        Ускорения берутся из собственной правой части при управлении u,
        поэтому u обязателен: без него удельная сила не определена.  Формула
        живёт в МОДЕЛИ, а не в датчике: это свойство машины и места установки,
        как first_integral или recoverable_bounds (A7, A16).

        Соглашение о пачках соблюдено: x формы (4,) или (M, 4), результат --
        пара массивов той же ведущей формы.

        w -- необязательная внешняя обобщённая сила (w_psi, w_theta) формы
        (..., 2): её вклад M^-1 w прибавляется к ускорениям.  Модель с
        возмущением (DisturbedWheeledPendulum) её не передаёт -- у неё сила
        уже внутри f; параметр нужен фильтру, который подставляет ОЦЕНКУ w.

        Источники: Groves, "Principles of GNSS, Inertial, and Multisensor
        Integrated Navigation Systems", 2-е изд., гл. 2.4 (удельная сила) и
        гл. 4.1; docs/Key_Formulas.md §1.3 (откуда ddpsi и ddtheta).
        """
        if u is None:
            raise ValueError(
                "specific_force: нужен момент u -- в удельную силу входят "
                "ускорения корпуса, а они зависят от приложенного момента"
            )
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        dx = self.f(t, x, u)
        psi = x[..., 0]
        dpsi = x[..., 2]
        ddpsi = dx[..., 2]
        ddtheta = dx[..., 3]
        if w is not None:
            # Внешняя обобщённая сила (w_psi, w_theta): добавка к ускорениям
            # M^-1 w.  Нужна фильтру, который оценивает возмущение и должен
            # предсказать, что покажет датчик (docs/accel_kalman.md §8).
            qdd = self.accel_from_force(psi, w)
            ddpsi = ddpsi + qdd[..., 0]
            ddtheta = ddtheta + qdd[..., 1]
        s, c = np.sin(psi), np.cos(psi)
        g, r = self.p.g, self.p.r
        f_x = -g * s + d * ddpsi + r * c * ddtheta
        f_z = g * c - d * dpsi ** 2 + r * s * ddtheta
        return f_x, f_z

    # --- внешние обобщённые силы (ER-025, docs/accel_kalman.md §6.1) --------

    def mass_matrix_inverse(self, psi):
        """M(psi)^-1 формы (..., 2, 2), где M = [[alpha, beta c], [beta c, gamma]].

        Нужна там, где на машину действует обобщённая сила w = (w_psi, w_theta)
        помимо мотора: добавка к ускорениям равна M^-1 w (Key_Formulas §1.1).
        Явная формула обращения 2x2, а не np.linalg.inv: так она сама
        соблюдает соглашение о пачках и точна до округления.
        """
        psi = np.asarray(psi, dtype=float)
        p = self.p
        c = np.cos(psi)
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        row0 = np.stack([p.gamma / Delta, -p.beta * c / Delta], axis=-1)
        row1 = np.stack([-p.beta * c / Delta, p.alpha / Delta], axis=-1)
        return np.stack([row0, row1], axis=-2)

    def accel_from_force(self, psi, w):
        """Добавка к ускорениям (ddpsi, ddtheta) = M(psi)^-1 w, форма (..., 2).

        То же, что mass_matrix_inverse(psi) @ w, но без сборки матрицы: эту
        добавку модель с возмущением и фильтр считают по нескольку раз на
        шаг для каждой клетки карты, и сборка 2x2 через np.stack стоила
        дороже самого умножения (docs/explorer_kalman.md §10.1).
        """
        psi = np.asarray(psi, dtype=float)
        w = np.asarray(w, dtype=float)
        p = self.p
        c = np.cos(psi)
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        w_psi, w_th = w[..., 0], w[..., 1]
        return np.stack([(p.gamma * w_psi - p.beta * c * w_th) / Delta,
                         (p.alpha * w_th - p.beta * c * w_psi) / Delta], axis=-1)

    def _numerators(self, x, u, w):
        """Числители N, R ускорений ddpsi = N/Delta, ddtheta = R/Delta
        (Key_Formulas §2) с необязательной внешней силой w и их производные
        по psi и dpsi.  Общая заготовка для jacobian и specific_force_jacobian.

        С силой w числители получают (правило Крамера для M q'' = rhs):
            N += gamma w_psi - beta c w_theta,
            R += -beta c w_psi + alpha w_theta.
        """
        p = self.p
        psi = x[..., 0]
        dpsi = x[..., 2]
        s, c = np.sin(psi), np.cos(psi)
        torque = u[..., 0]
        N = (p.gamma + p.beta * c) * torque + p.gamma * p.D * s - p.beta ** 2 * s * c * dpsi ** 2
        R = p.alpha * p.beta * s * dpsi ** 2 - p.beta * p.D * s * c - (p.alpha + p.beta * c) * torque
        # d/dpsi -- Key_Formulas §2: N_psi, R_psi
        N_psi = -p.beta * s * torque + p.gamma * p.D * c - p.beta ** 2 * np.cos(2 * psi) * dpsi ** 2
        R_psi = p.alpha * p.beta * c * dpsi ** 2 - p.beta * p.D * np.cos(2 * psi) + p.beta * s * torque
        # d/d(dpsi) и d/du
        N_dpsi = -2.0 * p.beta ** 2 * s * c * dpsi
        R_dpsi = 2.0 * p.alpha * p.beta * s * dpsi
        N_u = p.gamma + p.beta * c
        R_u = -(p.alpha + p.beta * c)
        if w is not None:
            w = np.asarray(w, dtype=float)
            w_psi, w_th = w[..., 0], w[..., 1]
            N = N + p.gamma * w_psi - p.beta * c * w_th
            R = R - p.beta * c * w_psi + p.alpha * w_th
            N_psi = N_psi + p.beta * s * w_th
            R_psi = R_psi + p.beta * s * w_psi
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        dDelta = p.beta ** 2 * np.sin(2 * psi)
        return dict(N=N, R=R, N_psi=N_psi, R_psi=R_psi, N_dpsi=N_dpsi,
                    R_dpsi=R_dpsi, N_u=N_u, R_u=R_u, Delta=Delta, dDelta=dDelta)

    def jacobian(self, t, x, u, w=None):
        """Аналитический якобиан поля: (A, B, G) = (df/dx, df/du, df/dw).

        A формы (..., 4, 4), B -- (..., 4, 1), G -- (..., 4, 2).  Формулы --
        Key_Formulas §2, §2.1: столбцы theta и dtheta нулевые (§1.4), ненулевые
        элементы -- производные дробей N/Delta и R/Delta по правилу частного

            d(N/Delta)/dpsi = (N_psi Delta - N Delta') / Delta^2,
            Delta' = beta^2 sin(2 psi).

        w -- необязательная внешняя обобщённая сила (w_psi, w_theta), в
        которой линеаризуется поле; без неё w = 0.  G = [0; M(psi)^-1] от w
        не зависит.  Нужен фильтру Калмана (прогноз ковариации) и якобиану
        датчика (specific_force_jacobian).  Оракул -- конечные разности f,
        tests/test_accel_kalman.py.
        """
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        k = self._numerators(x, u, w)
        De, dDe = k["Delta"], k["dDelta"]
        ddpsi_psi = (k["N_psi"] * De - k["N"] * dDe) / De ** 2
        ddth_psi = (k["R_psi"] * De - k["R"] * dDe) / De ** 2
        prefix = np.broadcast_shapes(x.shape[:-1], u.shape[:-1])
        A = np.zeros(prefix + (4, 4))
        A[..., 0, 2] = 1.0
        A[..., 1, 3] = 1.0
        A[..., 2, 0] = ddpsi_psi
        A[..., 3, 0] = ddth_psi
        A[..., 2, 2] = k["N_dpsi"] / De
        A[..., 3, 2] = k["R_dpsi"] / De
        B = np.zeros(prefix + (4, 1))
        B[..., 2, 0] = k["N_u"] / De
        B[..., 3, 0] = k["R_u"] / De
        G = np.zeros(prefix + (4, 2))
        G[..., 2:, :] = self.mass_matrix_inverse(x[..., 0])
        return A, B, G

    @property
    def center_of_percussion(self) -> float:
        """d* = alpha/(m_b l) -- центр качания корпуса относительно оси колеса.

        Горизонтальный канал акселерометра на выносе d удовлетворяет ТОЧНОМУ
        тождеству (docs/accel_kalman.md §3.2, из первого уравнения §1.1):

            a_x = (u + w_psi)/(m_b l) - (d* - d) ddpsi.

        В точке d = d* датчик видит только момент, а не состояние.
        """
        return self.p.alpha / (self.p.mb * self.p.l)

    def specific_force_jacobian(self, t, x, u, d: float = 0.0, w=None):
        """Якобиан удельной силы: (C, D_u, C_w) = d(f_x, f_z)/d(x, u, w).

        C -- (..., 2, 4), D_u -- (..., 2, 1), C_w -- (..., 2, 2).  Это цепное
        правило от кинематики датчика (specific_force)

            f_x = -g s + d ddpsi + r c ddtheta,
            f_z =  g c - d dpsi^2 + r s ddtheta,

        где производные ddpsi, ddtheta -- строки 3 и 4 якобиана поля
        (docs/accel_kalman.md §3.4).  По theta и dtheta производные нулевые.
        """
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        A, B, G = self.jacobian(t, x, u, w)
        k = self._numerators(x, u, w)
        ddth = k["R"] / k["Delta"]
        psi, dpsi = x[..., 0], x[..., 2]
        s, c = np.sin(psi), np.cos(psi)
        g, r = self.p.g, self.p.r
        C = np.zeros(A.shape[:-2] + (2, 4))
        C[..., 0, 0] = -g * c + d * A[..., 2, 0] - r * s * ddth + r * c * A[..., 3, 0]
        C[..., 0, 2] = d * A[..., 2, 2] + r * c * A[..., 3, 2]
        C[..., 1, 0] = -g * s + r * c * ddth + r * s * A[..., 3, 0]
        C[..., 1, 2] = -2.0 * d * dpsi + r * s * A[..., 3, 2]
        Du = np.stack([d * B[..., 2, :] + r * c[..., None] * B[..., 3, :],
                       r * s[..., None] * B[..., 3, :]], axis=-2)
        Cw = np.stack([d * G[..., 2, :] + r * c[..., None] * G[..., 3, :],
                       r * s[..., None] * G[..., 3, :]], axis=-2)
        return C, Du, Cw

    def specific_force_with_jacobian(self, t, x, u, d: float = 0.0, w=None):
        """(f_x, f_z, C, D_u, C_w) за ОДИН расчёт числителей.

        То же, что specific_force(..., w=w) и specific_force_jacobian вместе,
        но N, R, Delta и их производные считаются один раз: фильтр Калмана
        зовёт это на каждом шаге для каждой клетки карты, и тройной пересчёт
        стоил ~8 % его времени (docs/explorer_kalman.md §10.1).  Оракул --
        совпадение с двумя отдельными методами (tests/test_accel_kalman.py).
        """
        x = np.asarray(x, dtype=float)
        u = np.atleast_1d(np.asarray(u, dtype=float))
        p = self.p
        k = self._numerators(x, u, w)
        De, dDe = k["Delta"], k["dDelta"]
        ddpsi = k["N"] / De
        ddth = k["R"] / De
        ddpsi_psi = (k["N_psi"] * De - k["N"] * dDe) / De ** 2
        ddth_psi = (k["R_psi"] * De - k["R"] * dDe) / De ** 2
        ddpsi_dpsi = k["N_dpsi"] / De
        ddth_dpsi = k["R_dpsi"] / De
        psi, dpsi = x[..., 0], x[..., 2]
        s, c = np.sin(psi), np.cos(psi)
        g, r = p.g, p.r
        f_x = -g * s + d * ddpsi + r * c * ddth
        f_z = g * c - d * dpsi ** 2 + r * s * ddth
        prefix = f_x.shape
        C = np.zeros(prefix + (2, 4))
        C[..., 0, 0] = -g * c + d * ddpsi_psi - r * s * ddth + r * c * ddth_psi
        C[..., 0, 2] = d * ddpsi_dpsi + r * c * ddth_dpsi
        C[..., 1, 0] = -g * s + r * c * ddth + r * s * ddth_psi
        C[..., 1, 2] = -2.0 * d * dpsi + r * s * ddth_dpsi
        Du = np.zeros(prefix + (2, 1))
        Du[..., 0, 0] = (d * k["N_u"] + r * c * k["R_u"]) / De
        Du[..., 1, 0] = r * s * k["R_u"] / De
        # d(ddpsi, ddtheta)/dw = M^-1 = [[gamma, -beta c], [-beta c, alpha]]/Delta
        Mi00, Mi01, Mi11 = p.gamma / De, -p.beta * c / De, p.alpha / De
        Cw = np.zeros(prefix + (2, 2))
        Cw[..., 0, 0] = d * Mi00 + r * c * Mi01
        Cw[..., 0, 1] = d * Mi01 + r * c * Mi11
        Cw[..., 1, 0] = r * s * Mi01
        Cw[..., 1, 1] = r * s * Mi11
        return f_x, f_z, C, Du, Cw

    # --- оракулы для тестов и анализа --------------------------------------

    def first_integral(self, x, u_const) -> float:
        """H(psi, dpsi; u) = 1/2 Delta dpsi^2 - u (gamma psi + beta sin psi)
        + gamma D cos(psi).

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
        psi, dpsi = x[..., 0], x[..., 2]
        p = self.p
        return (0.5 * self.Delta(psi) * dpsi ** 2
                - u_const * (p.gamma * psi + p.beta * np.sin(psi))
                + p.gamma * p.D * np.cos(psi))

    # --- полная энергия (Key_Formulas §4) -----------------------------------

    @property
    def E_upright(self) -> float:
        """E в верхнем положении при нулевых скоростях: E* = D.

        Именно к этому уровню качает энергетическое реле (ER-018).  Значение
        равно D ровно потому, что из потенциала отброшена постоянная m_b g r
        (см. `energy`): начало отсчёта выбрано так, чтобы уровень цели был
        коротким выражением, а не суммой с нефизичным слагаемым.
        """
        return float(self.p.D)

    def energy(self, x):
        """E(x) = 1/2 alpha dpsi^2 + beta cos(psi) dpsi dtheta
                  + 1/2 gamma dtheta^2 + D cos(psi)      (Key_Formulas §4).

        Полная механическая энергия машины.  Постоянная m_b g r из потенциала
        отброшена: на динамику она не влияет, а уровень верхнего положения при
        этом равен ровно D (`E_upright`).

        Оракул, НЕЗАВИСИМЫЙ от H.  При u = 0 система консервативна, dE/dt = 0;
        при постоянном u != 0 сохраняется H (§3), но НЕ E.  Две величины ловят
        разные ошибки в правой части: H слепа к колесу (его координаты в неё
        не входят вовсе), E видит и наклон, и колесо, и перекрёстный член.

        theta в E не входит -- та же цикличность, что в §1.4: энергия не
        зависит от того, ГДЕ стоит колесо, только от того, как быстро оно
        крутится.

        Соглашение о пачках (A8): x формы (4,) или (M, 4).
        """
        x = np.asarray(x, dtype=float)
        psi, dpsi, dtheta = x[..., 0], x[..., 2], x[..., 3]
        p = self.p
        return (0.5 * p.alpha * dpsi ** 2
                + p.beta * np.cos(psi) * dpsi * dtheta
                + 0.5 * p.gamma * dtheta ** 2
                + p.D * np.cos(psi))

    def wheel_momentum(self, x):
        """p_theta = dT/d(dtheta) = beta cos(psi) dpsi + gamma dtheta.

        Обобщённый импульс колеса.  Нужен ради ТОЧНОГО разложения (§4.2)

            E = H(psi, dpsi; 0) / gamma + p_theta^2 / (2 gamma),

        которое и объясняет, чем накачка по E отличается от слежения за H:
        при p_theta != 0 один и тот же уровень E набирается при МЕНЬШЕЙ H,
        то есть часть энергии крутит колесо, а не поднимает корпус.  Формула
        живёт в модели, как `first_integral` (A7) и `specific_force` (A21):
        это свойство машины, а не регулятора.

        При u = 0 сохраняется (theta циклична, §1.4); при u != 0 выполняется
        dp_theta/dt = -u -- второе уравнение §1.1, записанное как теорема об
        изменении импульса.
        """
        x = np.asarray(x, dtype=float)
        psi, dpsi, dtheta = x[..., 0], x[..., 2], x[..., 3]
        p = self.p
        return p.beta * np.cos(psi) * dpsi + p.gamma * dtheta

    def power(self, x, u):
        """dE/dt = u (dpsi - dtheta) -- мощность, вводимая мотором (§4.1).

        Момент приложен МЕЖДУ корпусом и колесом (Q_psi = u, Q_theta = -u),
        поэтому в мощность входит ОТНОСИТЕЛЬНАЯ скорость, а не скорость
        наклона.  Практическое следствие, на котором стоит энергетическое
        реле: при dpsi = dtheta мотор не вводит энергии ВОВСЕ, каким бы ни был
        момент, и знак закона накачки в этом месте не определён -- см.
        `EnergyBangBangController`.

        u -- скаляр либо массив, транслируемый на ведущую форму x (как у
        `first_integral`): у релейного закона знак момента свой в каждой
        строке пачки.

        Оракул: конечная разность E вдоль прогона совпадает с этой формулой
        (tests/test_energy_bang.py).
        """
        x = np.asarray(x, dtype=float)
        dpsi, dtheta = x[..., 2], x[..., 3]
        return np.asarray(u, dtype=float) * (dpsi - dtheta)

    def tilt_energy(self, x):
        """E_psi(x) = H(psi, dpsi; 0) / gamma = 1/2 (Delta(psi)/gamma) dpsi^2 + D cos psi.

        Энергия ПРИВЕДЁННОЙ динамики наклона -- то, что остаётся от полной
        энергии, если отдать колесу его долю (Key_Formulas §4.2):

            E_psi = E - p_theta^2 / (2 gamma) .

        Уровень верхнего положения у неё ТОТ ЖЕ, что у полной энергии:
        E_psi(0, *, 0, *) = D = `E_upright`. Разница в том, ЧТО именно
        считается: `energy` спрашивает «сколько энергии в машине», а
        `tilt_energy` -- «сколько её досталось корпусу». Поскольку
        p_theta^2/(2 gamma) >= 0, всегда E_psi <= E, и равенство только при
        нулевом импульсе колеса.

        Это НЕ новая величина: H -- первый интеграл §3, по уровням которого
        построен сепаратрисный закон. Деление на gamma сделано ради того,
        чтобы промах по энергии мерился в тех же джоулях, что у `energy`, и
        eps_E у обоих значил одно и то же.
        """
        return self.first_integral(x, 0.0) / self.p.gamma

    def tilt_power(self, x, u):
        """dE_psi/dt = b(psi) u dpsi / gamma,  b(psi) = gamma + beta cos(psi).

        Баланс мощности ПРИВЕДЁННОЙ динамики -- вывод в Key_Formulas §4.3.
        Главное отличие от `power`: скорости колеса здесь НЕТ ВООБЩЕ. Мотор
        толкает колесо, колесо через инерционную связь толкает корпус, и с
        точки зрения наклона всё это сводится к множителю b(psi).

        ЛОВУШКА: b(psi) МЕНЯЕТ ЗНАК. При параметрах по умолчанию
        (gamma = 1.053, beta = 3) это происходит на |psi| = arccos(-gamma/beta)
        = 1.929 рад, то есть ВНУТРИ стандартной карты (|psi| до 2.04). За этим
        углом тот же момент качает наклон в другую сторону: корпус завален
        так далеко, что отталкивание колеса назад роняет его вперёд. Закон,
        который смотрит только на знак dpsi, за этой границей работает
        наоборот -- поэтому знак берётся у произведения b(psi) dpsi, а не у
        одной скорости.

        u -- скаляр либо массив, транслируемый на ведущую форму x (как у
        `first_integral` и `power`).

        Оракул: конечная разность tilt_energy вдоль прогона с постоянным u
        совпадает с этой формулой со вторым порядком (измерено 2.0).
        """
        x = np.asarray(x, dtype=float)
        psi, dpsi = x[..., 0], x[..., 2]
        return np.asarray(u, dtype=float) * self.b(psi) * dpsi / self.p.gamma

    # --- множество восстановимости (Key_Formulas §3.2) ----------------------
    #
    # Всё, что ниже, НЕ зависит от регулятора. Это ответ на вопрос «может ли
    # хоть какое-нибудь управление |u| <= u_max вернуть корпус в вертикаль»,
    # и он существует в замкнутом виде ровно потому, что приведённая динамика
    # наклона одномерна (theta циклична, §1.4) и при постоянном u интегрируема.

    def saddle_angle(self, u_max: float) -> float:
        """Угол psi_eq > 0, при котором предельный момент u = -u_max ровно
        уравновешивает силу тяжести:

            gamma D sin(psi_eq) = u_max (gamma + beta cos(psi_eq)).      (*)

        Это положение равновесия приведённой динамики, и оно СЕДЛОВОЕ: при
        psi < psi_eq момент пересиливает тяжесть и возвращает корпус, при
        psi > psi_eq тяжесть пересиливает и корпус уходит. Дальше этого
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
        """K(u_max) = H(psi_eq, 0; -u_max) -- уровень первого интеграла,
        проходящий через седло. Граница восстановимости лежит на нём."""
        p = self.p
        th = self.saddle_angle(u_max)
        return (u_max * (p.gamma * th + p.beta * np.sin(th))
                + p.gamma * p.D * np.cos(th))

    def recoverable_bounds(self, u_max: float, psi):
        """(floor, ceiling) -- границы по dpsi множества восстановимости.

        Состояние (psi, dpsi) восстановимо тогда и только тогда, когда
        floor(psi) < dpsi < ceiling(psi). Снаружи не помогает НИКАКОЙ
        регулятор: это свойство системы и предела мотора, а не закона
        управления.

        Из H(psi, dpsi; -+u_max) = K(u_max) получается

            dpsi^2 = 2 [ K - gamma D cos(psi) -+ u_max (gamma psi
                           + beta sin(psi)) ] / Delta(psi).

        ЛОВУШКА, ради которой это написано отдельным методом. Граница -- не
        линия уровня H = K, а УСТОЙЧИВОЕ МНОГООБРАЗИЕ седла, и оно составляет
        лишь часть этой линии. У седла устойчивый собственный вектор имеет
        отрицательный наклон, поэтому нужная ветвь меняет знак dpsi при
        переходе через psi_eq:

            ceiling(psi) = +sqrt(sq_minus)  при psi <  psi_eq,
                             -sqrt(sq_minus)  при psi >  psi_eq;
            floor(psi)   = -sqrt(sq_plus)   при psi > -psi_eq,
                             +sqrt(sq_plus)   при psi < -psi_eq.

        Смена знака -- это не косметика: она добавляет «языки» за +-psi_eq.
        Корпус, наклонённый почти горизонтально, ВСЁ ЕЩЁ восстановим, если
        качается назад достаточно быстро -- но не слишком быстро, иначе
        перелетит в другую сторону. Взять просто линию уровня значит потерять
        эти области и объявить невосстановимым то, что восстановимо.

        Работает и со скаляром, и с массивом psi -- соглашение A8.

        Источник: `docs/Key_Formulas.md` §3.2 и приложение C (ветка master);
        Khalil, *Nonlinear Systems*, 3-е изд., гл. 4 (области притяжения) и
        гл. 8 (седловые многообразия). Численно это же считает
        `src/recoverable.py` со второго семестра -- совпадение проверено
        тестом.
        """
        p = self.p
        psi = np.asarray(psi, dtype=float)
        K = self.separatrix_level(u_max)
        th_eq = self.saddle_angle(u_max)

        work = u_max * (p.gamma * psi + p.beta * np.sin(psi))
        gap = K - p.gamma * p.D * np.cos(psi)
        Delta = self.Delta(psi)
        sq_minus = 2.0 * (gap - work) / Delta      # H(-u_max) = K
        sq_plus = 2.0 * (gap + work) / Delta       # H(+u_max) = K

        sm = np.sqrt(np.clip(sq_minus, 0.0, None))
        sp = np.sqrt(np.clip(sq_plus, 0.0, None))
        ceiling = np.where(psi < th_eq, sm, -sm)
        floor = np.where(psi > -th_eq, -sp, sp)
        return floor, ceiling

    def is_recoverable(self, u_max: float, psi, dpsi):
        """Булев ответ «отсюда можно вернуться при |u| <= u_max»."""
        floor, ceiling = self.recoverable_bounds(u_max, psi)
        return (np.asarray(dpsi) > floor) & (np.asarray(dpsi) < ceiling)

    def linearize_tilt(self):
        """(A_t, B_t) приведённой подсистемы наклона z = (psi, dpsi).

            A_t = [[0, 1], [gamma D / Delta_0, 0]],   B_t = [0, (gamma+beta)/Delta_0]^T

        Почему такая подсистема вообще существует. В правые части НЕ входят ни
        theta, ни dtheta (§1.4): наклон живёт своей жизнью, а колесо -- это цепочка
        интеграторов, ведомая наклоном и моментом. Система треугольная, и
        верхнее звено этой цепочки замкнуто само на себя.

        Это ровно строки и столбцы (psi, dpsi) из `linearize_upright` --
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
