"""УПРАЖНЕНИЕ 3 (~30 минут). Написать свою System.

Маятник с квадратичным сопротивлением воздуха (в отличие от вязкого трения
из библиотеки, сила растёт как квадрат скорости):

    I * ddtheta = m g l sin(theta) - c_q * dtheta * |dtheta| + u,      I = m l^2

Модуль в дописке не случаен: сопротивление всегда направлено ПРОТИВ движения,
поэтому член нечётен по dtheta.  Напишешь dtheta**2 -- получишь силу, которая
при движении назад разгоняет, и проверка на диссипацию это поймает.

Обязательное требование -- соглашение о пачках: f должна работать и с одним
состоянием (2,), и с пачкой (M, 2).

    python exercises/03_system.py
"""

from _check import Checks   # noqa: E402

from dataclasses import dataclass

import numpy as np

from wpend import RK4Integrator, System, ZeroController, rollout, rollout_many
from wpend.models import Pendulum


@dataclass
class DragParams:
    m: float = 1.0
    l: float = 1.0
    g: float = 9.81
    cq: float = 0.2                # коэффициент квадратичного сопротивления
    u_max: float | None = None     # предел момента мотора, None = без предела

    @property
    def I(self) -> float:
        return self.m * self.l ** 2


class DragPendulum(System):
    """x = (theta, dtheta), u = (torque,); theta отсчитывается от ВЕРХНЕЙ вертикали."""

    def __init__(self, params: DragParams | None = None, **kwargs):
        self.p = params if params is not None else DragParams(**kwargs)

    @property
    def n_state(self) -> int:
        return 2

    @property
    def n_action(self) -> int:
        return 1

    @property
    def state_names(self):
        return ("theta", "dtheta")

    @property
    def u_bounds(self):
        # ---- TODO 3.2: вернуть (u_min, u_max) как массивы формы (1,),
        #      либо None, если p.u_max is None ---------------------------------
        raise NotImplementedError("напиши u_bounds")
        # ----------------------------------------------------------------------

    def f(self, t, x, u):
        # ---- TODO 3.1: правая часть; индексы x[..., i] и np.stack(axis=-1) ----
        raise NotImplementedError("напиши f")
        # ----------------------------------------------------------------------

    # оракул для проверок -- дан готовым
    def energy(self, x) -> float:
        x = np.asarray(x, dtype=float)
        theta, dtheta = x[..., 0], x[..., 1]
        p = self.p
        return 0.5 * p.I * dtheta ** 2 + p.m * p.g * p.l * np.cos(theta)


def main():
    c = Checks("Упражнение 3 -- своя система")
    system = DragPendulum(cq=0.2)

    one = system.f(0.0, np.array([0.2, -1.0]), np.array([0.3]))
    c.expect("f(одно состояние) возвращает форму (2,)", np.shape(one) == (2,))

    c.expect("первая компонента f -- это dtheta",
             np.isclose(one[0], -1.0),
             "dx/dt = (dtheta, ddtheta): скорость изменения угла есть сама "
             "угловая скорость, это кинематика, а не динамика.")

    X = np.array([[0.2, -1.0], [-0.4, 2.0], [0.0, 0.0], [1.2, -3.0]])
    U = np.array([[0.3], [-0.5], [0.0], [1.0]])
    batched = system.f(0.0, X, U)
    rowwise = np.array([system.f(0.0, X[i], U[i]) for i in range(len(X))])
    c.expect("соглашение о пачках: f((M,2)) -> (M,2) и совпадает построчно",
             np.shape(batched) == (4, 2) and np.allclose(batched, rowwise, atol=1e-12),
             "векторная формула обязана быть ТОЙ ЖЕ формулой.",
             "распаковка theta, dtheta = x работает только для одного состояния")

    plain = DragPendulum(cq=0.0)
    lib = Pendulum(c=0.0)
    agree = all(np.allclose(plain.f(0.0, X[i], U[i]), lib.f(0.0, X[i], U[i]), atol=1e-12)
                for i in range(len(X)))
    c.expect("при cq = 0 совпадает с библиотечным Pendulum",
             agree,
             "твоя модель -- надстройка над известной: убрав сопротивление, "
             "обязана вернуться к ней в точности.",
             "проверь знак у sin: theta от ВЕРХНЕЙ вертикали, гравитация "
             "уводит от равновесия, поэтому +m g l sin(theta)")

    a = plain.f(0.0, np.array([0.3, 1.1]), np.array([0.0]))
    b = plain.f(0.0, np.array([-0.3, -1.1]), np.array([0.0]))
    c.expect("нечётность: f(-x) = -f(x) при u = 0",
             np.allclose(a, -b, atol=1e-12),
             "задача симметрична относительно переворота: наклон вправо со "
             "скоростью вправо -- зеркало наклона влево со скоростью влево.")

    drag = DragPendulum(cq=0.5)
    traj = rollout(drag, ZeroController(1), RK4Integrator(), [0.9, 0.0], 1e-3, 4000)
    E = drag.energy(traj.x)
    c.expect("сопротивление ТОЛЬКО отбирает энергию",
             np.all(np.diff(E) <= 1e-9),
             f"E: {E[0]:.4f} -> {E[-1]:.4f} Дж, монотонно.",
             "если энергия где-то растёт -- почти наверняка написано dtheta**2 "
             "вместо dtheta*|dtheta|: такой член разгоняет при движении назад")

    limited = DragPendulum(cq=0.2, u_max=2.0)
    from wpend import ConstantController
    lt = rollout(limited, ConstantController([50.0]), RK4Integrator(),
                 [0.0, 0.0], 1e-3, 100)
    c.expect("предел момента соблюдён и записан в траекторию",
             np.allclose(lt.u, 2.0),
             "регулятор просил 50, мотор может 2: rollout клиппует по u_bounds "
             "и пишет в траекторию РЕАЛЬНО приложенное управление.")

    batch = rollout_many(drag, ZeroController(1), RK4Integrator(),
                         np.array([[0.2, 0.0], [0.5, 0.0], [-0.8, 0.0]]),
                         dt=1e-3, n_steps=2000, stride=10)
    seq = [rollout(drag, ZeroController(1), RK4Integrator(), x0, 1e-3, 2000)
           for x0 in [[0.2, 0.0], [0.5, 0.0], [-0.8, 0.0]]]
    c.expect("твоя модель работает в rollout_many (векторный прогон сетки)",
             all(np.allclose(batch[i].x, seq[i].x[::10], atol=1e-12) for i in range(3)),
             "это и есть выигрыш от соглашения о пачках: сетка НУ считается "
             "одним прогоном, а не M отдельными.")

    c.done()


if __name__ == "__main__":
    main()
