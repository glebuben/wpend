"""Сравнение оценивателей: что делает контур, когда меняется только оцениватель.

Два вопроса, и они РАЗНЫЕ.

1. КАК ВЕДЁТ СЕБЯ СИСТЕМА.  Опора -- идеальный оцениватель (FullStateSensor +
   PassthroughEstimator, x_hat = x побитово).  Всё остальное сравнивается с
   ним: та же модель, тот же регулятор, то же начальное условие, отличается
   ровно один слой.  Разница ИСТИННЫХ траекторий -- цена несовершенной оценки,
   и больше ничего.

2. КАК ОЦЕНКА ОТЛИЧАЕТСЯ ОТ ИСТИНЫ.  Это про сам оцениватель, и на железе
   такой график построить нельзя вовсе: там истина неизвестна.

ВАЖНО: истина у каждого оценивателя СВОЯ.  Оценка входит в управление,
управление -- в правую часть, значит мир при разных оценивателях развивается
по-разному (правило 4 ARCHITECTURE.md).  Сравнивать чужую оценку с чужой
истиной -- тихая ошибка: числа выглядят правдоподобно.

Оценка не хранится в Trajectory (правило 6, там только истина), поэтому
восстанавливается повторным прогоном датчика и оценивателя по записанной
траектории.  Оракул восстановления -- пересчитанное управление обязано
совпасть с traj.u ПОБИТОВО; иначе порядок вызовов разошёлся.

Запуск:

    uv run python examples/estimator_compare.py
    uv run python examples/estimator_compare.py --json out.json --seconds 8
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from wpend import (
    ComplementaryEstimator,
    FullStateSensor,
    IMUSensor,
    LinearFeedbackController,
    PassthroughEstimator,
    RK4Integrator,
    rollout,
)
from wpend.models import WheeledPendulum

# ЛКР ТОЛЬКО ПО НАКЛОНУ (A17): нули в столбцах phi и dphi -- ровно там, где
# состояние ненаблюдаемо при одном ИДУ.  Значит выдуманная оценка колеса
# физически не может попасть в управление, и это доказуемо.
K_TILT = np.array([[15.8887, 0.0, 3.4977, 0.0]])
DT = 1e-3
SEED = 0


def imu(system):
    return IMUSensor(system, dt=DT, mode="imu", seed=SEED)


def comp(tau):
    return lambda system: ComplementaryEstimator(system, dt=DT, tau=tau, wheel="zero")


CONFIGS = [
    ("ideal", "идеальный оцениватель",
     lambda s: FullStateSensor(), lambda s: PassthroughEstimator(), None),
    ("acc",  "только акселерометр (tau = 0)",   imu, comp(0.0),      0.0),
    ("t005", "комплементарный, tau = 0.05 с",   imu, comp(0.05),     0.05),
    ("t03",  "комплементарный, tau = 0.3 с",    imu, comp(0.30),     0.30),
    ("t20",  "комплементарный, tau = 2 с",      imu, comp(2.00),     2.00),
    ("gyro", "только гироскоп (tau -> inf)",    imu, comp(np.inf),   float("inf")),
]


def run_one(system, make_sensor, make_est, x0, n_steps):
    """Прогнать контур и восстановить x_hat. Возвращает (traj, x_hat)."""
    ctrl = LinearFeedbackController(K_TILT)
    traj = rollout(system, ctrl, RK4Integrator(), x0=list(x0), dt=DT,
                   n_steps=n_steps, sensor=make_sensor(system),
                   estimator=make_est(system))

    sensor, est = make_sensor(system), make_est(system)
    sensor.reset()
    est.reset(sensor.measure(traj.t[0], traj.x[0], np.zeros(system.n_action)))
    ctrl.reset()

    x_hat = np.empty((n_steps, system.n_state))
    u_re = np.empty((n_steps, system.n_action))
    u_prev = np.zeros(system.n_action)
    for k in range(n_steps):
        y = sensor.measure(traj.t[k], traj.x[k], u_prev)
        x_hat[k] = est.estimate(traj.t[k], y, u_prev)
        u_re[k] = system.clip_action(ctrl.act(traj.t[k], x_hat[k]))
        u_prev = u_re[k]

    if not np.array_equal(u_re, traj.u):
        raise AssertionError(
            "восстановление разошлось с прогоном: пересчитанное управление не "
            f"совпало с traj.u побитово (max|du| = {np.max(np.abs(u_re - traj.u)):.3e})"
        )
    return traj, x_hat


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--theta0", type=float, default=0.05)
    ap.add_argument("--stride", type=int, default=10)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    system = WheeledPendulum(u_max=1.5)
    n = int(round(args.seconds / DT))
    x0 = [args.theta0, 0.0, 0.0, 0.0]
    sl = slice(0, n, args.stride)

    print(f"один регулятор (ЛКР наклона), одно НУ ({args.theta0}, 0, 0, 0), "
          f"{args.seconds} с, dt = {DT}, seed = {SEED}.  Меняется ТОЛЬКО оцениватель.\n")
    print(f"{'оцениватель':<32}{'вес a':>9}{'RMSE оценки':>13}"
          f"{'откл. от идеала':>17}{'max|theta|':>12}  исход")

    out = {"t": None, "order": [k for k, *_ in CONFIGS], "runs": {}}
    ref_theta = None

    for key, label, mk_sensor, mk_est, tau in CONFIGS:
        traj, x_hat = run_one(system, mk_sensor, mk_est, x0, n)
        theta_true = traj.x[:n, 0]
        if ref_theta is None:
            ref_theta = theta_true.copy()

        err = x_hat[:, 0] - theta_true
        rmse = float(np.sqrt(np.mean(err ** 2)))
        dev = float(np.max(np.abs(theta_true - ref_theta)))
        mx = float(np.max(np.abs(theta_true)))
        a = 1.0 if tau is None or np.isinf(tau) else tau / (tau + DT)

        print(f"{label:<32}{a:>9.4f}{rmse:>13.5f}{dev:>17.5f}{mx:>12.3f}"
              f"  {'УПАЛ' if mx > 0.6 else 'удержан'}")

        if out["t"] is None:
            out["t"] = np.round(traj.t[sl], 6).tolist()
        out["runs"][key] = {
            "label": label,
            "tau": None if tau is None else ("inf" if np.isinf(tau) else tau),
            "a": None if tau is None else round(a, 6),
            "true_theta": np.round(theta_true[sl], 6).tolist(),
            "hat_theta": np.round(x_hat[sl, 0], 6).tolist(),
            "u": np.round(traj.u[sl, 0], 5).tolist(),
            "rmse": round(rmse, 6),
            "dev": round(dev, 6),
            "max_theta": round(mx, 4),
            "fell": bool(mx > 0.6),
        }

    print("\nОракул восстановления пройден на всех конфигурациях: пересчитанное "
          "управление совпало с traj.u побитово.")
    print("'откл. от идеала' -- расхождение ИСТИННЫХ траекторий, а не оценок.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        print(f"\nряды: {args.json}")


if __name__ == "__main__":
    main()
