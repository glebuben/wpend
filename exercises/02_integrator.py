"""УПРАЖНЕНИЕ 2 (~25 минут). Написать свой Integrator и измерить его порядок.

Метод средней точки (RK2):

    k1 = f(t,        x,             u)
    k2 = f(t + dt/2, x + dt/2 * k1, u)
    x_next = x + dt * k2

Главная проверка -- не "результат похож на правду", а ИЗМЕРЕННЫЙ порядок
сходимости.  Считаем глобальную ошибку на трёх сетках, строим наклон
log(ошибка) от log(dt) и требуем 2.0.  Если ты случайно напишешь Эйлера,
наклон выйдет 1 и это будет видно; если случайно RK4 -- выйдет 4.

    python exercises/02_integrator.py
"""

from _check import Checks   # noqa: E402

import numpy as np

from wpend import EulerIntegrator, Integrator, RK4Integrator, rollout, ZeroController
from wpend.models import Pendulum


class Midpoint(Integrator):
    """Явный метод средней точки, порядок 2."""

    # ---- TODO 2.1: объяви порядок схемы -------------------------------------
    order = None

    def step(self, f, t, x, u, dt):
        x = np.asarray(x, dtype=float)
        # ---- TODO 2.2: один шаг метода средней точки -------------------------
        raise NotImplementedError("напиши step()")
        # ----------------------------------------------------------------------


# --- эталонная задача с известным решением: dx/dt = -2x + u -------------------

def f_linear(t, x, u):
    return -2.0 * x + u


def exact(t, x0=1.0, u=0.5):
    return (x0 - u / 2.0) * np.exp(-2.0 * t) + u / 2.0


def global_error(integrator, dt, T=1.0):
    n = int(round(T / dt))
    x, t = np.array([1.0]), 0.0
    for k in range(n):
        x = integrator.step(f_linear, t, x, np.array([0.5]), dt)
        t = (k + 1) * dt
    return abs(x[0] - exact(T))


def observed_order(integrator):
    dts = [1e-2, 5e-3, 2.5e-3]
    errs = [global_error(integrator, dt) for dt in dts]
    slopes = [np.log(errs[i] / errs[i + 1]) / np.log(dts[i] / dts[i + 1])
              for i in range(len(dts) - 1)]
    return float(np.mean(slopes))


def main():
    c = Checks("Упражнение 2 -- свой интегратор")
    mid = Midpoint()

    c.expect("порядок объявлен как 2", mid.order == 2,
             "это число -- не украшение: следующая проверка сверяет с ним "
             "измеренный наклон.")

    x_in = np.array([1.0])
    x_copy = x_in.copy()
    mid.step(f_linear, 0.0, x_in, np.array([0.0]), 0.01)
    c.expect("step не портит входное состояние", np.array_equal(x_in, x_copy),
             "x += ... меняет массив на месте и ломает вызывающий код; "
             "нужно возвращать новый массив.",
             "используй x + dt * k2, а не x += dt * k2")

    order = observed_order(mid)
    c.expect(f"измеренный порядок = {order:.2f} (ожидается ~2)",
             abs(order - 2.0) < 0.15,
             "наклон log(ошибка) по log(dt) на трёх сетках.",
             "если вышло ~1 -- это Эйлер: k2 считается в СЕРЕДИНЕ шага, "
             "но результат равен x + dt*k2, а не x + dt/2*(k1+k2)")

    e_mid = global_error(mid, 1e-2)
    e_eul = global_error(EulerIntegrator(), 1e-2)
    e_rk4 = global_error(RK4Integrator(), 1e-2)
    c.expect("точнее Эйлера, но грубее RK4",
             e_rk4 < e_mid < e_eul,
             f"ошибка при dt=1e-2: RK4 {e_rk4:.2e} < Midpoint {e_mid:.2e} "
             f"< Euler {e_eul:.2e}.")

    system = Pendulum(c=0.0)
    traj = rollout(system, ZeroController(1), Midpoint(), [0.5, 0.0], 1e-4, 20000)
    E = np.array([system.energy(x) for x in traj.x])
    drift = np.max(np.abs(E - E[0]))
    c.expect(f"энергия почти сохраняется (дрейф {drift:.2e})",
             drift < 1e-6,
             "физика не зависит от схемы: при u=0 и без трения энергия -- "
             "первый интеграл, и любая сходящаяся схема обязана его почти держать.")

    c.done()


if __name__ == "__main__":
    main()
