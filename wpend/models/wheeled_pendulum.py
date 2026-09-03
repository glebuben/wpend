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
        theta, _phi, dtheta, _dphi = x
        p = self.p
        s, c = np.sin(theta), np.cos(theta)
        Delta = p.alpha * p.gamma - p.beta ** 2 * c ** 2
        torque = u[0]

        ddtheta = ((p.gamma + p.beta * c) * torque
                   + p.gamma * p.D * s
                   - p.beta ** 2 * s * c * dtheta ** 2) / Delta

        ddphi = (p.alpha * p.beta * s * dtheta ** 2
                 - p.beta * p.D * s * c
                 - (p.alpha + p.beta * c) * torque) / Delta

        return np.array([dtheta, _dphi, ddtheta, ddphi])

    # --- оракулы для тестов и анализа --------------------------------------

    def first_integral(self, x, u_const: float) -> float:
        """H(theta, dtheta; u) = 1/2 Delta dtheta^2 - u (gamma theta + beta sin theta)
        + gamma D cos(theta).

        При ПОСТОЯННОМ u выполняется dH/dt = 0 (Key_Formulas §3, вывод в
        приложении B): приведённая динамика наклона одномерна и потому
        интегрируема.  Именно на уровнях H строится множество восстановимых
        состояний.  Для тестов это точный оракул: RK4 обязан сохранять H с
        точностью O(dt^4).
        """
        theta, _phi, dtheta, _dphi = x
        p = self.p
        return (0.5 * self.Delta(theta) * dtheta ** 2
                - u_const * (p.gamma * theta + p.beta * np.sin(theta))
                + p.gamma * p.D * np.cos(theta))

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
