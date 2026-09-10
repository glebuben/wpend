"""Восстановить оценку x_hat по готовой Trajectory и сравнить её с истиной.

ЗАЧЕМ ОТДЕЛЬНЫЙ СКРИПТ.  Trajectory хранит (t, x, u) -- только ИСТИНУ, и
правило 6 ARCHITECTURE.md запрещает всему, что идёт после прогона, знать
что-то ещё.  Значит увидеть оценку можно ровно одним способом, не трогая
ядро: прогнать датчик и оцениватель ВТОРОЙ РАЗ по уже записанной траектории.
Никаких новых полей, никаких новых сущностей -- чистый downstream.

ПОЧЕМУ ЭТО ТОЧНО, А НЕ ПРИБЛИЗИТЕЛЬНО.  И датчик, и оцениватель
детерминированы своим seed и порядком вызовов.  Значит повтор даёт ту же
последовательность, ЕСЛИ порядок вызовов повторён буква в букву -- включая
лишний measure(t0) перед estimator.reset(y0), который rollout делает до
цикла.  Забыть его -- сдвинуть генератор шума на один розыгрыш, и оценка
получится от другой реализации шума: правдоподобная, но не та, которую видел
регулятор.  Ошибка тихая.

Поэтому у восстановления есть ОРАКУЛ: пересчитанное управление обязано
совпасть с traj.u ПОБИТОВО.  Совпало -- значит воспроизведён ровно тот
прогон; не совпало -- значит порядок вызовов разошёлся, и смотреть на
картинку нельзя.

Запуск:

    uv run python examples/estimator_replay.py
    uv run python examples/estimator_replay.py --json out.json --seconds 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from wpend import (
    ComplementaryEstimator,
    IMUSensor,
    LinearFeedbackController,
    RK4Integrator,
    rollout,
)
from wpend.models import WheeledPendulum

# ЛКР ТОЛЬКО ПО НАКЛОНУ (A17): нули стоят в столбцах phi и dphi -- ровно в тех,
# что ненаблюдаемы при одном ИДУ.  Значит выдуманная оценка колеса физически не
# может попасть в управление, и это доказуемо, а не "надеемся".
K_TILT = np.array([[15.8887, 0.0, 3.4977, 0.0]])


def replay(system, controller, tau, *, x0, dt, n_steps, seed=0, wheel="dead_reckon",
           sensor_kw=None):
    """Прогнать контур и восстановить x_hat. Возвращает (traj, x_hat).

    Порядок вызовов ниже -- копия цикла rollout; см. докстринг модуля.
    """
    sensor_kw = dict(sensor_kw or {})
    make_sensor = lambda: IMUSensor(system, dt=dt, mode="imu", seed=seed, **sensor_kw)
    make_est = lambda: ComplementaryEstimator(system, dt=dt, tau=tau, wheel=wheel)

    traj = rollout(system, controller, RK4Integrator(), x0=list(x0), dt=dt,
                   n_steps=n_steps, sensor=make_sensor(), estimator=make_est())

    sensor, est = make_sensor(), make_est()
    sensor.reset()
    est.reset(sensor.measure(traj.t[0], traj.x[0], np.zeros(system.n_action)))
    controller.reset()

    x_hat = np.empty((n_steps, system.n_state))
    u_re = np.empty((n_steps, system.n_action))
    u_prev = np.zeros(system.n_action)
    for k in range(n_steps):
        y = sensor.measure(traj.t[k], traj.x[k], u_prev)
        x_hat[k] = est.estimate(traj.t[k], y, u_prev)
        u_re[k] = system.clip_action(controller.act(traj.t[k], x_hat[k]))
        u_prev = u_re[k]

    if not np.array_equal(u_re, traj.u):
        raise AssertionError(
            "восстановление разошлось с прогоном: пересчитанное управление не "
            "совпало с traj.u побитово.  Порядок вызовов датчика/оценивателя "
            f"отличается (max|du| = {np.max(np.abs(u_re - traj.u)):.3e})."
        )
    return traj, x_hat


TAUS = (0.05, 0.1, 0.3, 0.5, 1.0, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--dt", type=float, default=1e-3)
    ap.add_argument("--theta0", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stride", type=int, default=10, help="прореживание для вывода")
    ap.add_argument("--json", default=None, help="куда сложить ряды для графиков")
    args = ap.parse_args()

    system = WheeledPendulum(u_max=1.5)
    n = int(round(args.seconds / args.dt))
    ctrl = LinearFeedbackController(K_TILT)

    print(f"ЛКР наклона, x0 = ({args.theta0}, 0, 0, 0), {args.seconds} с, "
          f"dt = {args.dt}, seed = {args.seed}\n")
    print(f"{'tau, с':>7} {'RMSE theta':>12} {'смещение в хвосте':>18} "
          f"{'|ошибка dphi|':>14} {'max|theta|':>11}  исход")

    out = {"t": None, "runs": {}}
    for tau in TAUS:
        traj, x_hat = replay(system, ctrl, tau, x0=[args.theta0, 0, 0, 0],
                             dt=args.dt, n_steps=n, seed=args.seed)
        err = x_hat[:, 0] - traj.x[:n, 0]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        tail = float(np.mean(err[n // 2:]))
        dphi_err = float(x_hat[-1, 3] - traj.x[n - 1, 3])
        mx = float(np.max(np.abs(traj.x[:n, 0])))
        print(f"{tau:>7.2f} {rmse:>12.5f} {tail:>+18.5f} {abs(dphi_err):>14.3f} "
              f"{mx:>11.3f}  {'УПАЛ' if mx > 0.6 else 'удержан'}")

        sl = slice(0, n, args.stride)
        if out["t"] is None:
            out["t"] = np.round(traj.t[sl], 6).tolist()
        # ИСТИНА У КАЖДОГО tau СВОЯ.  Оценка входит в управление, управление --
        # в правую часть, значит мир при разных tau развивается по-разному
        # (правило 4 ARCHITECTURE.md).  Сохранить истину один раз и сравнивать
        # с ней все прогоны -- тихая ошибка: числа выглядят правдоподобно.
        out["runs"][f"{tau}"] = {
            "theta": np.round(x_hat[sl, 0], 6).tolist(),
            "dphi": np.round(x_hat[sl, 3], 4).tolist(),
            "true_theta": np.round(traj.x[sl, 0], 6).tolist(),
            "true_dphi": np.round(traj.x[sl, 3], 4).tolist(),
            "rmse": round(rmse, 6),
            "tail": round(tail, 6),
            "dphi_err": round(dphi_err, 4),
            "fell": bool(np.max(np.abs(traj.x[:n, 0])) > 0.6),
            "max_theta": round(float(np.max(np.abs(traj.x[:n, 0]))), 4),
        }

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"\nряды: {args.json}")
    print("\nОракул восстановления пройден: пересчитанное управление совпало с "
          "traj.u побитово на всех tau.")


if __name__ == "__main__":
    main()
