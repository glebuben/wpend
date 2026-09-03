"""УПРАЖНЕНИЕ 1 (~25 минут). Написать свой Controller.

Часть A: ПД-регулятор своими руками.  Проверка -- он обязан совпасть с готовым
LinearFeedbackController до 1e-12: это два независимых пути вычисления одного и
того же, и расхождение означает, что кто-то из двоих неправ.

Часть B (по желанию): релейный регулятор.  Показывает, что контракт Controller
ничем не ограничивает закон управления -- он не обязан быть линейным.

    python exercises/01_controller.py
"""

from _check import Checks   # noqa: E402

import numpy as np

from wpend import Controller, LinearFeedbackController, RK4Integrator, rollout
from wpend.models import Pendulum

KP, KD = 25.0, 8.0


class MyPD(Controller):
    """u = -kp * theta - kd * dtheta для маятника x = (theta, dtheta).

    ВАЖНО -- соглашение о пачках: act должен работать и с одним состоянием
    формы (2,), возвращая (1,), и с пачкой (M, 2), возвращая (M, 1).
    Проще всего этого добиться, обращаясь к компонентам через x_hat[..., i]
    и собирая ответ через np.stack([...], axis=-1).
    """

    def __init__(self, kp: float, kd: float):
        self.kp = float(kp)
        self.kd = float(kd)

    def act(self, t, x_hat):
        # ---- TODO 1.1 -------------------------------------------------------
        raise NotImplementedError("напиши act() для ПД-регулятора")
        # ---------------------------------------------------------------------


class BangBang(Controller):
    """u = -u_max * sign(kp*theta + kd*dtheta): мотор всегда на упоре.

    Часть B, по желанию. Если пропускаешь -- просто не трогай TODO 1.2,
    проверки для неё будут пропущены.
    """

    def __init__(self, kp: float, kd: float, u_max: float):
        self.kp, self.kd, self.u_max = float(kp), float(kd), float(u_max)

    def act(self, t, x_hat):
        # ---- TODO 1.2 (необязательно) ----------------------------------------
        raise NotImplementedError("часть B пропущена")
        # ---------------------------------------------------------------------


def main():
    c = Checks("Упражнение 1 -- свой регулятор")
    system = Pendulum()
    mine = MyPD(KP, KD)
    reference = LinearFeedbackController(np.array([[KP, KD]]))

    one = np.array([0.2, -1.3])
    c.expect("act(одно состояние) возвращает форму (1,)",
             np.shape(mine.act(0.0, one)) == (1,),
             "n_action = 1, значит управление -- вектор длины 1, а не число.",
             "np.stack([...], axis=-1) даёт нужную форму сам")

    rng = np.random.default_rng(0)
    states = rng.normal(size=(50, 2))
    same = all(np.allclose(mine.act(0.0, s), reference.act(0.0, s), atol=1e-12)
               for s in states)
    c.expect("совпадает с LinearFeedbackController на 50 случайных состояниях",
             same,
             "ПД -- частный случай u = -K(x - x_ref) при K = [kp, kd].",
             "проверь знаки: оба слагаемых входят со знаком минус")

    batch = mine.act(0.0, states)
    rowwise = np.array([mine.act(0.0, s) for s in states])
    c.expect("соглашение о пачках: act((M,2)) -> (M,1) и совпадает построчно",
             np.shape(batch) == (50, 1) and np.allclose(batch, rowwise, atol=1e-12),
             "без этого твой регулятор не поедет в rollout_many и в карту НУ.",
             "распаковка theta, dtheta = x_hat ломает пачку; нужны индексы [..., i]")

    traj = rollout(system, mine, RK4Integrator(), [0.3, 0.0], 1e-3, 5000)
    c.expect("маятник удержан в верхнем положении",
             np.linalg.norm(traj.x[-1]) < 1e-3,
             f"конечное состояние: theta={traj.x[-1, 0]:+.2e}, "
             f"dtheta={traj.x[-1, 1]:+.2e}")

    unstable = rollout(system, MyPD(5.0, 8.0), RK4Integrator(), [0.3, 0.0], 1e-3, 5000)
    c.expect("слишком малый kp НЕ удерживает",
             np.linalg.norm(unstable.x[-1]) > 1.0,
             "kp должен перебороть гравитацию: нужно kp > m*g*l = 9.81, иначе "
             "верхнее положение остаётся неустойчивым при любом kd.")

    # --- часть B ---
    try:
        relay = BangBang(KP, KD, u_max=5.0)
        relay.act(0.0, one)
    except NotImplementedError:
        c.skip("часть B: релейный регулятор", "TODO 1.2 не заполнен")
    else:
        weak = Pendulum(u_max=5.0)
        rt = rollout(weak, relay, RK4Integrator(), [0.3, 0.0], 1e-3, 5000)
        c.expect("релейный регулятор всегда на упоре",
                 np.allclose(np.abs(rt.u), 5.0),
                 "|u| = u_max в каждый момент -- отсюда и название.")
        c.expect("релейный регулятор удерживает корпус (с дребезгом)",
                 np.max(np.abs(rt.x[-500:, 0])) < 0.05,
                 "около нуля управление скачет между +-u_max: угол не уходит, "
                 "но и в точный ноль не садится.")

    c.done()


if __name__ == "__main__":
    main()
