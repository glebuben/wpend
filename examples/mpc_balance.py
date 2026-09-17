"""MPC против ЛКР и против заранее рассчитанного плана iLQR.

Три опыта на колёсном маятнике (мир: dt = 1 мс, RK4, 3 с):

  1. Номинал: модель регулятора совпадает с миром. Цена J, момент, время
     одного перепланирования.
  2. Неточная модель: мир с корпусом 12 кг, регуляторы считают, что 10 кг.
     План iLQR, посчитанный заранее, исполняется со слежением K_k, MPC
     перепланирует по неверной модели из измеренного состояния.
  3. Шум датчика: GaussianNoiseSensor на наклоне и скорости наклона.
  4. Большой наклон 1.3 рад при точной модели -- там, где ЛКР не держит.

MPC: такт 20 мс (zero-order hold), горизонт 50 шагов = 1 с, Q_f = P,
2 итерации iLQR на такт. Предела момента нет -- iLQR его не учитывает.

Цена J считается ОДНОЙ функцией (`trajectory_cost`) по истинной траектории,
поэтому числа сравнимы между строками.

Запуск:  uv run python examples/mpc_balance.py        (~3 мин)
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    GaussianNoiseSensor,
    LinearFeedbackController,
    MPCController,
    RK4Integrator,
    TrajectoryTrackingController,
    rollout,
)
from wpend.ddp import ilqr, trajectory_cost
from wpend.lqr import lqr
from wpend.models import WheeledPendulum

Q = np.diag([100.0, 1.0, 10.0, 1.0])
R = np.array([[1.0]])
DT, T = 1e-3, 3.0
N_SIM = int(round(T / DT))
DT_PLAN, HORIZON = 0.02, 50


def planned_tracking(model, K, P, x0):
    """План iLQR на весь прогон по МОДЕЛИ, с шагом мира, и его слежение."""
    rk4 = RK4Integrator()
    u0 = rollout(model, LinearFeedbackController(K), rk4, x0, DT, N_SIM).u
    x_bar, u_bar, K_seq, _ = ilqr(model, rk4, x0, DT, N_SIM, Q, R, P, u_init=u0)
    return TrajectoryTrackingController(x_bar, u_bar, K_seq, DT)


def run(world, controller, x0, sensor=None):
    t_start = time.perf_counter()
    traj = rollout(world, controller, RK4Integrator(), x0, DT, N_SIM, sensor=sensor)
    elapsed = time.perf_counter() - t_start
    J = trajectory_cost(traj.x, traj.u, Q, R, P_GLOBAL, DT)
    return traj, J, elapsed


def row(name, traj, J, extra=""):
    held = np.all(np.abs(traj.x[:, 0]) < np.pi / 2)
    print(f"  {name:<22} J = {J:9.2f}   max|u| = {np.abs(traj.u).max():6.2f}   "
          f"|theta(T)| = {abs(traj.x[-1, 0]):8.2e}   "
          f"{'удержан' if held else 'УПАЛ'}  {extra}")


def main():
    global P_GLOBAL
    model = WheeledPendulum()                     # то, что знает регулятор
    A, B = model.linearize_upright()
    K, P = lqr(A, B, Q, R)
    P_GLOBAL = P
    rk4 = RK4Integrator()

    def mpc():
        return MPCController(model, rk4, DT_PLAN, HORIZON, Q, R, P, K_init=K)

    print(f"мир dt = {DT}, {T} с; MPC такт {DT_PLAN} с, горизонт "
          f"{HORIZON} шагов = {HORIZON * DT_PLAN:g} с\n")

    print("1. Номинал: модель = мир")
    for theta0 in (0.3, 0.9):
        x0 = [theta0, 0.0, 0.0, 0.0]
        print(f" theta0 = {theta0}")
        row("ЛКР", *run(model, LinearFeedbackController(K), x0)[:2])
        row("план iLQR + слежение", *run(model, planned_tracking(model, K, P, x0), x0)[:2])
        ctrl = mpc()
        traj, J, el = run(model, ctrl, x0)
        row("MPC", traj, J, f"({1e3 * el / ctrl.n_replans:.0f} мс на такт при "
                            f"такте {1e3 * DT_PLAN:.0f} мс)")

    print("\n2. Неточная модель: мир mb = 12 кг, регуляторы думают 10 кг")
    world = WheeledPendulum(mb=12.0)
    for theta0 in (0.3, 0.9):
        x0 = [theta0, 0.0, 0.0, 0.0]
        print(f" theta0 = {theta0}")
        row("ЛКР", *run(world, LinearFeedbackController(K), x0)[:2])
        row("план iLQR + слежение", *run(world, planned_tracking(model, K, P, x0), x0)[:2])
        row("MPC", *run(world, mpc(), x0)[:2])

    print("\n3. Шум датчика: sigma(theta) = 0.01 рад, sigma(dtheta) = 0.05 рад/с")
    x0 = [0.3, 0.0, 0.0, 0.0]

    def noise():
        return GaussianNoiseSensor(sigma=[0.01, 0.0, 0.05, 0.0], seed=0)

    row("ЛКР", *run(model, LinearFeedbackController(K), x0, noise())[:2])
    row("план iLQR + слежение", *run(model, planned_tracking(model, K, P, x0), x0,
                                     noise())[:2])
    row("MPC", *run(model, mpc(), x0, noise())[:2])

    # Начальное приближение для плана -- прогон ЛКР, а он отсюда падает. Длинный
    # план на весь прогон (3000 шагов) начинает с упавшей траектории; короткий
    # горизонт MPC (1 с) начинает с куска той же траектории, но перепланирует
    # раньше, чем корпус успевает уйти далеко. Гипотеза, а не доказательство:
    # см. PROPOSALS.md A23.
    print("\n4. Большой наклон: theta0 = 1.3 рад, модель = мир")
    x0 = [1.3, 0.0, 0.0, 0.0]
    row("ЛКР", *run(model, LinearFeedbackController(K), x0)[:2])
    row("план iLQR + слежение", *run(model, planned_tracking(model, K, P, x0), x0)[:2])
    row("MPC", *run(model, mpc(), x0)[:2])


if __name__ == "__main__":
    main()
