"""ER-025: фильтр Калмана с уравнениями акселерометра -- четыре мира.

Один и тот же ЛКР, один и тот же старт, меняются мир и режим смещений:

    мир         смещение чипа             фильтр
    ---------   -----------------------   -------------------------------
    чистый      «code»: b0 из IMUSensor   bias_prior=None (P0 = b0^2)
    чистый      сырой чип (MPU-6050)      калибровка на подставке 1 с
    возмущение  «code»                    + модель возмущения (согласованная)
    возмущение  сырой чип                 калибровка + модель возмущения

и отдельно -- что будет с сырым чипом БЕЗ калибровки (фильтр думает, что
смещение порядка 0.01 рад/с, а оно 0.2).

Печатается СКО ошибки оценки на последних 2 с и ошибки смещений в конце;
для мира с возмущением -- ошибка оценки толчка.  Если установлен matplotlib,
--plot сохраняет картинку (в зависимостях проекта его нет, как и в
examples/extremal_field.py).

Запуск:

    uv run python examples/kalman_accel.py
    uv run python examples/kalman_accel.py --seconds 8 --plot figures/er025_kalman.png
"""

from __future__ import annotations

import argparse

import numpy as np

from wpend import (
    MPU6050_RAW_B0,
    EncoderSensor,
    IMUSensor,
    KalmanEstimator,
    LinearFeedbackController,
    RK4Integrator,
    StackedSensor,
    calibrate_on_stand,
    draw_turn_on_bias,
    rollout,
)
from wpend.lqr import lqr
from wpend.models import DisturbedWheeledPendulum, WheeledPendulum

DT = 1e-3
IMU = dict(dt=DT, d=0.20)            # общие параметры датчика прогона и подставки
DIST = dict(sigma_w=(0.3, 0.3), tau_w=0.2)
X0 = [0.1, 0.0, 0.0, 0.0]


def run(world, nominal, *, raw_chip, calibrate, disturbance, seconds, seed=1):
    b_init = draw_turn_on_bias(**MPU6050_RAW_B0, seed=seed) if raw_chip else None
    prior = None
    if calibrate:
        prior = calibrate_on_stand(nominal, IMU, b_init=b_init, T_c=1.0,
                                   sigma_jig=1e-2, seed=seed + 100)
    imu = IMUSensor(world, mode="imu", seed=seed, b_init=b_init, **IMU)
    sensor = StackedSensor(imu, EncoderSensor(world, dt=DT))
    est = KalmanEstimator(nominal, dt=DT, d=IMU["d"], bias_prior=prior,
                          disturbance=disturbance)
    A, B = nominal.linearize_upright()
    K = np.atleast_2d(lqr(A, B, np.diag([100, 1, 10, 1.0]), np.eye(1))[0])
    traj = rollout(world, LinearFeedbackController(K), RK4Integrator(), x0=X0, dt=DT,
                   n_steps=int(round(seconds / DT)), sensor=sensor, estimator=est)
    return traj, est, imu


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--plot", default=None, help="куда сохранить картинку (нужен matplotlib)")
    args = ap.parse_args()

    nominal = WheeledPendulum()
    windy = DisturbedWheeledPendulum(dt=DT, seed=7, **DIST)
    cases = [
        ("чистый, code", nominal, False, False, None),
        ("чистый, сырой чип + калибровка", nominal, True, True, None),
        ("чистый, сырой чип БЕЗ калибровки", nominal, True, False, None),
        ("возмущение, code", windy, False, False, DIST),
        ("возмущение, сырой чип + калибровка", windy, True, True, DIST),
        ("возмущение, сырой чип БЕЗ калибровки", windy, True, False, DIST),
    ]
    tail = int(round(2.0 / DT))
    head = int(round(0.5 / DT))
    print(f"{'сценарий':38s} {'упал':>5s} {'psi<0.5с':>9s} {'psi':>9s} {'dpsi':>9s} {'theta':>9s} "
          f"{'|b_ax|':>8s} {'|b_g|':>8s} {'w_psi':>6s} {'w_th':>6s}")
    results = []
    for name, world, raw, cal, dist in cases:
        traj, est, imu = run(world, nominal, raw_chip=raw, calibrate=cal,
                             disturbance=dist, seconds=args.seconds)
        fell = bool(np.any(np.abs(traj.x[:, 0]) >= np.pi / 2))
        e = traj.x_hat[-tail:] - traj.x[:-1][-tail:]
        rms = np.sqrt(np.mean(e ** 2, axis=0))
        db = np.abs(est.aux["b_hat"] - imu._b)
        w_err = ["", ""]
        if traj.w_hat is not None:
            w_true = world.disturbance(traj.t[:-1])
            w_rms = np.sqrt(np.mean((traj.w_hat[-tail:] - w_true[-tail:]) ** 2, axis=0))
            w_err = [f"{w_rms[0]:.3f}", f"{w_rms[1]:.3f}"]
        e_head = np.max(np.abs(traj.x_hat[:head, 0] - traj.x[:head, 0]))
        print(f"{name:38s} {'да' if fell else 'нет':>5s} {e_head:9.2e} {rms[0]:9.2e} {rms[2]:9.2e} {rms[1]:9.2e} "
              f"{db[0]:8.1e} {db[2]:8.1e} {w_err[0]:>6s} {w_err[1]:>6s}")
        results.append((name, traj, world))
    print("\npsi<0.5с -- наибольшая ошибка наклона в первые 0.5 с; СКО ошибок -- на последних 2 с; |b| -- ошибка смещения в конце; "
          "w -- СКО ошибки оценки толчка, Н·м (у возмущения СКО 0.3).")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib не установлен -- картинка пропущена")
            return
        fig, ax = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
        for name, traj, world in results:
            t = traj.t[:-1]
            ax[0].plot(t, traj.x_hat[:, 0] - traj.x[:-1, 0], lw=0.8, label=name)
            ax[1].plot(t, traj.x_hat[:, 2] - traj.x[:-1, 2], lw=0.8, label=name)
            if traj.w_hat is not None:
                ax[2].plot(t, world.disturbance(t)[:, 1], "k", lw=0.8)
                ax[2].plot(t, traj.w_hat[:, 1], lw=0.8, label=f"{name}: оценка")
        ax[0].set_ylabel("ошибка psi, рад")
        ax[1].set_ylabel("ошибка dpsi, рад/с")
        ax[2].set_ylabel("w_theta, Н·м (чёрн. -- истина)")
        ax[2].set_xlabel("t, с")
        for a in ax:
            a.legend(fontsize=7)
            a.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"картинка: {args.plot}")


if __name__ == "__main__":
    main()
