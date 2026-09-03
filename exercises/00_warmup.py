"""УПРАЖНЕНИЕ 0 (разминка, ~10 минут). Собрать замкнутый контур из готовых блоков.

Ничего своего писать не надо -- надо один раз руками соединить пять слоёв и
увидеть, что во что втыкается.

Что делать: заполнить TODO ниже и запустить

    python exercises/00_warmup.py

Вопрос на подумать (ответ в конце файла, не подглядывай сразу):
почему в Trajectory управлений на одно меньше, чем состояний?
"""

from _check import Checks   # noqa: E402  (он же чинит sys.path)

import numpy as np

from wpend import LinearFeedbackController, RK4Integrator, ZeroController, rollout
from wpend.models import Pendulum

# ЛКР для линеаризации маятника в верхнем положении, Q = diag(10, 1), R = 1.
K = np.array([[20.11709, 6.421385]])


def build_and_run():
    """Собери контур и верни две траектории: без управления и с обратной связью."""
    # ---- TODO 0.1: создай маятник с параметрами по умолчанию -----------------
    system = None
    # ---- TODO 0.2: создай интегратор RK4 -------------------------------------
    integrator = None
    # ---- TODO 0.3: создай регулятор линейной обратной связи с матрицей K ------
    controller = None
    # --------------------------------------------------------------------------
    if system is None or integrator is None or controller is None:
        raise NotImplementedError("заполни TODO 0.1-0.3")

    # ---- TODO 0.4: прогони 5 секунд с шагом 1 мс из начального угла 0.3 рад ---
    #      подсказка: сигнатура -- rollout(system, controller, integrator,
    #                                      x0, dt, n_steps)
    closed = None
    if closed is None:
        raise NotImplementedError("заполни TODO 0.4")

    free = rollout(system, ZeroController(1), integrator, [0.3, 0.0], 1e-3, 5000)
    return system, free, closed


def main():
    c = Checks("Упражнение 0 -- сборка контура")
    system, free, closed = build_and_run()

    c.expect("форма траектории: t (5001,), x (5001, 2), u (5000, 1)",
             closed.t.shape == (5001,) and closed.x.shape == (5001, 2)
             and closed.u.shape == (5000, 1),
             "u короче x на единицу: u_k действует МЕЖДУ x_k и x_{k+1}.",
             "n_steps=5000 при dt=1e-3 -- это 5 секунд")

    c.expect("без управления маятник падает",
             np.linalg.norm(free.x[-1]) > 1.0,
             "верхнее положение неустойчиво: собственное число +sqrt(g/l) > 0.")

    c.expect("с обратной связью маятник удержан",
             np.linalg.norm(closed.x[-1]) < 1e-3,
             "тот же объект, тот же интегратор -- разница только в регуляторе.")

    c.expect("управление ненулевое и конечное",
             np.max(np.abs(closed.u)) > 0.1 and np.all(np.isfinite(closed.u)),
             "если тут ноль -- скорее всего подставлен ZeroController.")

    e0, e1 = system.energy(closed.x[0]), system.energy(closed.x[-1])
    e_top = system.p.m * system.p.g * system.p.l
    c.expect("регулятор ДОБАВИЛ энергию в систему",
             e1 > e0 and abs(e1 - e_top) < 1e-3,
             f"E: {e0:.4f} -> {e1:.4f} Дж, а максимум равен m*g*l = {e_top:.4f}. "
             "Верхняя вертикаль -- максимум потенциальной энергии, поэтому "
             "удержание там стоит работы: мотор не гасит движение, а поднимает "
             "маятник обратно наверх. Интуиция \"регулятор всегда диссипативен\" "
             "здесь неверна.")

    c.done()


if __name__ == "__main__":
    main()

# ОТВЕТ на вопрос из шапки:
# u_k -- это управление, которое действовало на ИНТЕРВАЛЕ [t_k, t_{k+1}), то есть
# между состоянием x_k и x_{k+1}. У последнего состояния траектории интервала
# впереди нет, поэтому управлений ровно на одно меньше. Если бы длины совпадали,
# пришлось бы либо выдумывать лишнее управление, либо терять первое.
