"""Реле сразу, передача ЛКР по |theta| < eps: какой ЛКР и какой eps.

Запуск:  uv run --extra design python examples/bang_eps_map.py [--grid 31] [--horizon 30]

Карта по соглашению проекта (CLAUDE.md, Глеб 17.09): интересен theta в
[-pi/2, pi/2], сетка берётся с запасом 1.3 по обеим осям -- theta0 до
±1.3·pi/2, dtheta0 до 1.3 от максимума множества восстановимости на
[-pi/2, pi/2]; phi0 = dphi0 = 0. «Удержал» -- |theta|
ни разу не достиг pi, «сошёлся» -- удержал и в конце |theta| < 0.05,
|dtheta| < 0.1, |dphi| < 0.1. Потолок -- `is_recoverable` (A16).

Сравниваются три цены ЛКР, каждая сама по себе и после реле:
  base -- Q = diag(100, 1, 10, 1): ЛКР окна;
  soft -- Q = diag(100, 1e-2, 10, 1e-2): гейны по phi, dphi снижены (bang-eps);
  tilt -- `lqr_tilt`: колесо не в цене вовсе.
Третья фаза (A27): реле -> |theta| < eps -> soft -> base, передача на base по
сертифицированному эллипсоиду x^T P_base x <= c* (c* -- `certified_level` при
том же u_max). Колонка «успокоился» -- медиана и 90-й процентиль момента, после
которого состояние навсегда в |theta| < 0.05, |dtheta| < 0.1, |phi| < 0.5,
|dphi| < 0.1 (по сошедшимся клеткам).
Колонка dth@handover -- медиана и максимум |dtheta| в момент первого входа в
полосу по восстановимым клеткам: полоса по одному theta скорость не проверяет.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    BangBangLQRController,
    LinearFeedbackController,
    RK4Integrator,
    rollout_many,
    theta_band_region,
)
from wpend import ellipsoid_region
from wpend.lqr import certified_level, lqr, lqr_tilt
from wpend.models import WheeledPendulum

R = np.array([[1.0]])
COSTS = {
    "base": np.diag([100.0, 1.0, 10.0, 1.0]),
    "soft": np.diag([100.0, 1e-2, 10.0, 1e-2]),
    "tilt": None,
}
DT = 1e-3


MAP_FIT = 1.3
THETA_MAP = np.pi / 2


def gains(wp):
    """K для каждой цены и (K, P) базы -- P нужна третьей фазе."""
    A, B = wp.linearize_upright()
    out = {}
    for name, Q in COSTS.items():
        if Q is None:
            out[name], _ = lqr_tilt(wp, np.diag([100.0, 10.0]), R)
        else:
            out[name], _ = lqr(A, B, Q, R)
    _, P_base = lqr(A, B, COSTS["base"], R)
    return out, P_base


def grid(wp, u_max, n):
    th = np.linspace(-THETA_MAP, THETA_MAP, 401)
    floor, ceiling = wp.recoverable_bounds(u_max, th)
    reach = np.abs(np.concatenate([floor, ceiling]))
    dth_max = MAP_FIT * float(reach[np.isfinite(reach)].max())
    TH, DTH = np.meshgrid(np.linspace(-MAP_FIT * THETA_MAP, MAP_FIT * THETA_MAP, n),
                          np.linspace(-dth_max, dth_max, n))
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = TH.ravel(), DTH.ravel()
    return X0


def score(wp, ctrl, X0, horizon):
    b = rollout_many(wp, ctrl, RK4Integrator(), X0, dt=DT,
                     n_steps=int(round(horizon / DT)), stride=20)
    held = (np.abs(b.x[:, :, 0]) < np.pi).all(axis=1)
    x_end = b.x[:, -1]
    conv = (held & (np.abs(x_end[:, 0]) < 0.05) & (np.abs(x_end[:, 2]) < 0.1)
            & (np.abs(x_end[:, 3]) < 0.1))
    calm = ((np.abs(b.x[:, :, 0]) < 0.05) & (np.abs(b.x[:, :, 2]) < 0.1)
            & (np.abs(b.x[:, :, 1]) < 0.5) & (np.abs(b.x[:, :, 3]) < 0.1))
    # Последний «неспокойный» отсчёт + 1: с этого момента состояние спокойно навсегда.
    last_bad = calm.shape[1] - np.argmax(~calm[:, ::-1], axis=1)
    last_bad[calm.all(axis=1)] = 0
    t_calm = b.t[np.minimum(last_bad, calm.shape[1] - 1)]
    t_calm[~(conv & calm[:, -1])] = np.nan
    return b, held, conv, t_calm


def calm_str(t_calm):
    t = t_calm[np.isfinite(t_calm)]
    return "   —" if t.size == 0 else f"{np.median(t):5.1f} / {np.percentile(t, 90):5.1f} с"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--u-max", type=float, nargs="+", default=[1.5, 3.0, 10.0])
    ap.add_argument("--eps", type=float, nargs="+", default=[0.02, 0.05, 0.1])
    args = ap.parse_args()

    for u_max in args.u_max:
        wp = WheeledPendulum(u_max=u_max)
        K, P_base = gains(wp)
        c_base, _ = certified_level(wp, K["base"], P_base, u_max=u_max, n_dirs=1500,
                                    ds=4e-3, s_max=8.0, theta_max=THETA_MAP, seed=0)
        X0 = grid(wp, u_max, args.grid)
        rec = wp.is_recoverable(u_max, X0[:, 0], X0[:, 2])
        print(f"\nu_max = {u_max:g} Н·м: восстановимо {rec.sum()} из {len(X0)}, "
              f"c*_base = {c_base:.3g}")
        print(f"  {'регулятор':34s} {'удержал':>8s} {'сошёлся':>8s}  "
              f"{'успокоился med / p90':>20s}   dth@handover (med / max)")

        def line(label, ctrl, eps=None):
            b, held, conv, t_calm = score(wp, ctrl, X0, args.horizon)
            tail = ""
            if eps is not None:
                # stride прореживает запись, поэтому момент передачи -- с точностью
                # до 20 шагов; для порядка величины |dtheta| этого хватает.
                i = np.argmax(np.abs(b.x[:, :, 0]) < eps, axis=1)
                dth = np.abs(b.x[np.arange(len(X0)), i, 2])[rec]
                tail = f"   {np.median(dth):.2f} / {dth.max():.2f}"
            print(f"  {label:34s} {held.sum():8d} {conv.sum():8d}  {calm_str(t_calm):>20s}{tail}")

        for name in COSTS:
            line(f"LQR {name}", LinearFeedbackController(K[name]))
        for eps in args.eps:
            for name in COSTS:
                line(f"relay -> |th|<{eps:g} -> {name}",
                     BangBangLQRController(K[name], u_max, theta_band_region(eps), wp), eps)
            line(f"relay -> |th|<{eps:g} -> soft -> base",
                 BangBangLQRController(K["soft"], u_max, theta_band_region(eps), wp,
                                       K_final=K["base"],
                                       final_region=ellipsoid_region(P_base, c_base)), eps)

if __name__ == "__main__":
    main()
