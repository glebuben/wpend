"""Устоит ли MPC, если планировщик предела не знает, а мотор его обрезает.

Постановка (решение Глеба 15.09): iLQR внутри MPC планирует БЕЗ предела
момента, а мир, как всегда, обрезает запрошенное управление в
`System.clip_action`. Вопрос: на какой части карты начальных условий корпус
удерживается, по сравнению с ЛКР (тоже обрезанным) и с реле.

Опорные линии, которые от регулятора не зависят:
  * `is_recoverable` -- аналитическое множество восстановимости: снаружи не
    спасёт никакой закон с |u| <= u_max;
  * чистое реле по сепаратрисе -- совпадает с этим множеством (A16), то есть
    это «лучшее из возможного».

Настройки те же, что в окне: Q = diag(100, 1, 10, 1), R = 1, горизонт 5 с,
dt = 1 мс, упал = |theta| >= 1.3, карта в 1.3 раза шире множества. MPC:
такт 20 мс, горизонт 1 с, Q_f = P.

Карта печатается символами (matplotlib в dev не утверждён, §B11):
    B -- держат оба (ЛКР и MPC)     M -- держит только MPC
    L -- держит только ЛКР           . -- не держит никто, но восстановимо
    пробел -- невосстановимо и никто не держит
    ! -- невосстановимо, но кто-то «держит» (быть не должно: проверка)
Строки -- dtheta0 сверху вниз по убыванию, столбцы -- theta0 слева направо.

Запуск:  uv run python examples/mpc_clip_map.py --u-max 3 --grid 11
(MPC -- отдельный прогон на клетку, ~8 с; 121 клетка на 2 ядрах ~8 мин)
"""

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    BangBangLQRController,
    LinearFeedbackController,
    MPCController,
    RK4Integrator,
    rollout,
    rollout_many,
)
from wpend.lqr import lqr
from wpend.models import WheeledPendulum

Q = np.diag([100.0, 1.0, 10.0, 1.0])
R = np.array([[1.0]])
DT, HORIZON_S, THETA_FALL, MAP_FIT = 1e-3, 5.0, 1.3, 1.3
N_SIM = int(round(HORIZON_S / DT))
DT_PLAN, PLAN_STEPS = 0.02, 50


def _never(x):
    return np.zeros(np.shape(x)[:-1], dtype=bool)


def fell_mask(x_theta):
    """(M, T) -> (M,): первое пересечение |theta| = theta_fall после старта."""
    return (np.abs(x_theta[:, 1:]) >= THETA_FALL).any(axis=1)


def run_mpc_cell(args):
    """Один прогон MPC. Функция верхнего уровня -- её можно отдать процессу."""
    u_max, x0 = args
    model = WheeledPendulum()                  # модель регулятора: без предела
    world = WheeledPendulum(u_max=u_max)       # мир: обрезает
    A, B = model.linearize_upright()
    K, P = lqr(A, B, Q, R)
    ctrl = MPCController(model, RK4Integrator(), DT_PLAN, PLAN_STEPS, Q, R, P, K_init=K)
    traj = rollout(world, ctrl, RK4Integrator(), x0, DT, N_SIM)
    theta = traj.x[:, 0]
    saturated = float(np.mean(np.abs(traj.u[:, 0]) >= u_max - 1e-9))
    return bool(fell_mask(theta[None])[0]), abs(theta[-1]), saturated


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--u-max", type=float, default=3.0)
    ap.add_argument("--grid", type=int, default=11)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--save", type=str, default=None, help="куда сохранить .npz")
    args = ap.parse_args()
    u_max, n = args.u_max, args.grid

    model = WheeledPendulum()
    world = WheeledPendulum(u_max=u_max)
    A, B = model.linearize_upright()
    K, P = lqr(A, B, Q, R)

    theta_max = min(MAP_FIT * world.saddle_angle(u_max), 0.98 * THETA_FALL)
    _, ceiling = world.recoverable_bounds(u_max, np.zeros(1))
    dtheta_max = MAP_FIT * float(ceiling[0])
    thetas = np.linspace(-theta_max, theta_max, n)
    dthetas = np.linspace(-dtheta_max, dtheta_max, n)
    TH, DTH = np.meshgrid(thetas, dthetas, indexing="xy")
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = TH.ravel(), DTH.ravel()
    recoverable = world.is_recoverable(u_max, X0[:, 0], X0[:, 2])

    lqr_batch = rollout_many(world, LinearFeedbackController(K), RK4Integrator(),
                             X0, DT, N_SIM, 10)
    held_lqr = ~fell_mask(lqr_batch.x[:, :, 0])
    bang = BangBangLQRController(K, u_max, _never, world)
    bang_batch = rollout_many(world, bang, RK4Integrator(), X0, DT, N_SIM, 10)
    held_bang = ~fell_mask(bang_batch.x[:, :, 0])

    t0 = time.perf_counter()
    with ProcessPoolExecutor(args.workers) as pool:
        out = list(pool.map(run_mpc_cell, [(u_max, x) for x in X0]))
    held_mpc = ~np.array([o[0] for o in out])
    theta_end = np.array([o[1] for o in out])
    sat = np.array([o[2] for o in out])
    elapsed = time.perf_counter() - t0

    print(f"u_max = {u_max} Н·м, сетка {n}x{n}, theta0 до ±{theta_max:.3f}, "
          f"dtheta0 до ±{dtheta_max:.2f}; MPC считался {elapsed / 60:.1f} мин")
    print(f"  восстановимо: {recoverable.sum():4d}")
    print(f"  реле:         {held_bang.sum():4d}   (совпадает с множеством в "
          f"{(held_bang == recoverable).sum()} клетках из {n * n})")
    print(f"  ЛКР + clip:   {held_lqr.sum():4d}")
    print(f"  MPC + clip:   {held_mpc.sum():4d}   (только MPC: "
          f"{(held_mpc & ~held_lqr).sum()}, только ЛКР: {(held_lqr & ~held_mpc).sum()})")
    kept = held_mpc
    if kept.any():
        print(f"  MPC, удержанные: max |theta(5 с)| = {theta_end[kept].max():.2e}, "
              f"доля шагов в насыщении до {sat[kept].max():.2f}")

    print("\n  dtheta0 \\ theta0 ->")
    for iy in range(n - 1, -1, -1):
        line = []
        for ix in range(n):
            m = iy * n + ix
            anyone = held_lqr[m] or held_mpc[m]
            if not recoverable[m]:
                line.append("!" if anyone else " ")
            elif held_lqr[m] and held_mpc[m]:
                line.append("B")
            elif held_mpc[m]:
                line.append("M")
            elif held_lqr[m]:
                line.append("L")
            else:
                line.append(".")
        print(f"  {dthetas[iy]:+6.2f} |" + " ".join(line) + "|")

    if args.save:
        np.savez(args.save, thetas=thetas, dthetas=dthetas, recoverable=recoverable,
                 held_lqr=held_lqr, held_bang=held_bang, held_mpc=held_mpc,
                 theta_end=theta_end, sat=sat)


if __name__ == "__main__":
    main()
