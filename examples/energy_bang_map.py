"""ER-018: накачка ПОЛНОЙ энергии против реле по сепаратрисе.

Запуск:  uv run --extra design python examples/energy_bang_map.py [--grid 31]

Сравниваются две первые фазы при одинаковом всём остальном:

  bang    -- u = -u_max sign(sigma), sigma -- уровень первого интеграла H,
             приходящий в верхнее положение (Key_Formulas §3);
  energy  -- u = u_max sign(E* - E) sign(dpsi - dtheta), E -- ПОЛНАЯ энергия
             машины, E* = D (Key_Formulas §4, решение Глеба 22.09).

РАЗНЫЕ У НИХ НЕ ТОЛЬКО ЗАКОНЫ ПЕРВОЙ ФАЗЫ, НО И КРИТЕРИЙ ПЕРЕДАЧИ (решение
Глеба 22.09). У сепаратрисы это |psi| < eps, у энергии -- КОНЪЮНКЦИЯ
|E - E*| < eps_E И |psi| < eps: качаем, пока энергии не станет ровно столько,
сколько наверху, и отдаём управление в тот момент, когда корпус проходит
вертикаль С нужной энергией. Одного уровня энергии мало -- множество E = E*
трёхмерно и почти всё лежит далеко от вертикали (§4.2); одной полосы мало --
корпус может пролетать ноль без нужной энергии.

Вторая фаза (|psi| < eps -> мягкий ЛКР) и третья (сертифицированный эллипсоид
базового ЛКР, A27) у обоих ОДНИ И ТЕ ЖЕ: `EnergyBangBangController` наследует
их у `BangBangLQRController`, поэтому строка таблицы меняет ровно один закон.

Карта -- по соглашению проекта (CLAUDE.md, Глеб 17.09): psi0 до ±1.3·pi/2,
dpsi0 до 1.3 от максимума множества восстановимости на [-pi/2, pi/2],
theta0 = dtheta0 = 0. «Удержал» -- |psi| ни разу не достиг pi; «сошёлся» --
удержал и в конце |psi| < 0.05, |dpsi| < 0.1, |dtheta| < 0.1. Потолок --
`is_recoverable` (A16).

Вторая таблица отвечает на вопрос «почему»: она печатает, сколько энергии в
момент, когда накачка вышла на уровень, сидит в КОЛЕСЕ. Разложение §4.2

    E = H(psi, dpsi; 0)/gamma + p_theta^2/(2 gamma)

точное, и всё, что ушло в p_theta, наклону не досталось: E = E* при
p_theta != 0 означает H/gamma < D, то есть корпус до вертикали не доходит
принципиально, а не из-за настройки eps.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    BangBangLQRController,
    EnergyBangBangController,
    LinearFeedbackController,
    RK4Integrator,
    all_regions,
    ellipsoid_region,
    energy_band_region,
    psi_band_region,
    rollout_many,
)
from wpend.estimator import ComplementaryEstimator
from wpend.lqr import certified_level, lqr
from wpend.models import WheeledPendulum
from wpend.sensor import EncoderSensor, IMUSensor, StackedSensor

R = np.array([[1.0]])
Q_BASE = np.diag([100.0, 1.0, 10.0, 1.0])
Q_SOFT = np.diag([100.0, 1e-2, 10.0, 1e-2])
DT = 1e-3
MAP_FIT = 1.3
PSI_MAP = np.pi / 2

#: Вынос ИДУ от оси колеса вдоль корпуса [м] -- одно число у датчика и у
#: оценивателя (A31): разъехавшись, они молча вычитали бы не ту инерционную
#: часть. Значения шума -- умолчания окна (`wpend/viz/explorer.py`).
IMU_D = 0.20


def imu_pair(wp, seed):
    """Датчик и оцениватель как в окне: ИДУ + энкодер, комплементарный фильтр.

    Без энкодера колесо ненаблюдаемо (решение с куратором 17.09), поэтому
    вариант ровно один. `accel_offset` компенсирует кажущуюся вертикаль -- без
    неё контур раскачивается при любом tau после смены l, r модели (A31).
    """
    sensor = StackedSensor(
        IMUSensor(wp, dt=DT, mode="imu", seed=seed, d=IMU_D,
                  sigma_g=1e-4, sigma_a=1e-3, b0_g=1e-2, b0_a=1e-2 * 9.8),
        EncoderSensor(wp, dt=DT, counts_per_rev=2048))
    est = ComplementaryEstimator(wp, dt=DT, tau=0.30, wheel="encoder",
                                 accel_offset=IMU_D)
    return sensor, est


def gains(wp):
    A, B = wp.linearize_upright()
    K_base, P_base = lqr(A, B, Q_BASE, R)
    K_soft, _ = lqr(A, B, Q_SOFT, R)
    return K_base, P_base, K_soft


def grid(wp, u_max, n):
    psi = np.linspace(-PSI_MAP, PSI_MAP, 401)
    floor, ceiling = wp.recoverable_bounds(u_max, psi)
    reach = np.abs(np.concatenate([floor, ceiling]))
    dpsi_max = MAP_FIT * float(reach[np.isfinite(reach)].max())
    PSI, DPSI = np.meshgrid(
        np.linspace(-MAP_FIT * PSI_MAP, MAP_FIT * PSI_MAP, n),
        np.linspace(-dpsi_max, dpsi_max, n))
    X0 = np.zeros((n * n, 4))
    X0[:, 0], X0[:, 2] = PSI.ravel(), DPSI.ravel()
    return X0


def score(wp, ctrl, X0, horizon, stride=20, pair=None):
    sensor, est = pair if pair is not None else (None, None)
    b = rollout_many(wp, ctrl, RK4Integrator(), X0, dt=DT,
                     n_steps=int(round(horizon / DT)), stride=stride,
                     sensor=sensor, estimator=est)
    held = (np.abs(b.x[:, :, 0]) < np.pi).all(axis=1)
    x_end = b.x[:, -1]
    conv = (held & (np.abs(x_end[:, 0]) < 0.05) & (np.abs(x_end[:, 2]) < 0.1)
            & (np.abs(x_end[:, 3]) < 0.1))
    calm = ((np.abs(b.x[:, :, 0]) < 0.05) & (np.abs(b.x[:, :, 2]) < 0.1)
            & (np.abs(b.x[:, :, 1]) < 0.5) & (np.abs(b.x[:, :, 3]) < 0.1))
    last_bad = calm.shape[1] - np.argmax(~calm[:, ::-1], axis=1)
    last_bad[calm.all(axis=1)] = 0
    t_calm = b.t[np.minimum(last_bad, calm.shape[1] - 1)]
    t_calm[~(conv & calm[:, -1])] = np.nan
    return b, held, conv, t_calm


def calm_str(t_calm):
    t = t_calm[np.isfinite(t_calm)]
    return "   —" if t.size == 0 else f"{np.median(t):5.1f} / {np.percentile(t, 90):5.1f} с"


def switches(b, region):
    """(число смен знака u до передачи, момент передачи) по каждой клетке.

    Передача ищется как первый отсчёт, попавший в РЕГИОН — тот же предикат,
    что у регулятора, а не его копия: у сепаратрисы это полоса по psi, у
    энергии — полоса И уровень энергии. Запись прорежена, поэтому число
    переключений — ОЦЕНКА СНИЗУ: смена знака между сохранёнными кадрами не
    видна. Для вопроса «0-1 или десятки» этого достаточно, точное число даёт
    stride = 1.
    """
    inside = np.asarray(region(b.x), dtype=bool)
    ever = inside.any(axis=1)
    i = np.where(ever, np.argmax(inside, axis=1), inside.shape[1] - 1)
    u = np.sign(b.u[:, :, 0])
    n = np.array([np.count_nonzero(np.diff(u[m, :max(i[m], 1)]) != 0)
                  for m in range(len(i))])
    t = np.where(ever, b.t[np.minimum(i, len(b.t) - 1)], np.nan)
    return n, t, ever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--horizon", type=float, default=30.0)
    ap.add_argument("--u-max", type=float, nargs="+", default=[1.5, 3.0, 10.0])
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--energy", nargs="+", choices=["total", "tilt"],
                    default=["total", "tilt"],
                    help="какую энергию качать: total -- полную энергию машины "
                         "(dE/dt = u(dpsi - dtheta)); tilt -- энергию приведённой "
                         "динамики наклона (dE_psi/dt = u b(psi) dpsi / gamma, "
                         "скорости колеса в ней нет вовсе)")
    ap.add_argument("--eps-e", type=float, default=0.5,
                    help="ширина полосы по энергии для energy-строк. Масштаб: "
                         "D(1 - cos d) -- энергия, которой не хватает корпусу, "
                         "отклонённому на d; при D = 98 и d = 0.1 рад это 0.49")
    ap.add_argument("--estimator", choices=["ideal", "comp"], default="ideal",
                    help="ideal -- регулятор видит истину; comp -- ИДУ + энкодер "
                         "и комплементарный фильтр (шаг 5 карточки ER-018)")
    ap.add_argument("--stride", type=int, default=20,
                    help="прореживание записи; число переключений точно только при 1")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    for u_max in args.u_max:
        wp = WheeledPendulum(u_max=u_max)
        K_base, P_base, K_soft = gains(wp)
        c_base, _ = certified_level(wp, K_base, P_base, u_max=u_max, n_dirs=1500,
                                    ds=4e-3, s_max=8.0, psi_max=PSI_MAP, seed=0)
        X0 = grid(wp, u_max, args.grid)
        pair = None if args.estimator == "ideal" else imu_pair(wp, args.seed)
        rec = wp.is_recoverable(u_max, X0[:, 0], X0[:, 2])
        print(f"\nu_max = {u_max:g} Н·м: восстановимо {rec.sum()} из {len(X0)}, "
              f"c*_base = {c_base:.3g}, eps = {args.eps:g}, eps_E = {args.eps_e:g}, "
              f"оценка = {args.estimator}, stride = {args.stride}")
        print(f"  {'первая фаза -> вторая -> третья':50s} {'удержал':>8s} "
              f"{'сошёлся':>8s}  {'успокоился med / p90':>20s}  "
              f"{'передач':>8s}  {'переключений med / max':>22s}")

        band = psi_band_region(args.eps)
        rows = [
            ("—  -> soft (без первой фазы)",
             LinearFeedbackController(K_soft), band),
            ("bang    -> |psi|<eps -> soft",
             BangBangLQRController(K_soft, u_max, band, wp), band),
            ("bang    -> |psi|<eps -> soft -> base",
             BangBangLQRController(K_soft, u_max, band, wp,
                                   K_final=K_base,
                                   final_region=ellipsoid_region(P_base, c_base)), band),
        ]
        for mode in args.energy:
            e_band = all_regions(
                energy_band_region(wp, args.eps_e, energy=mode), band)
            tag = "E" if mode == "total" else "E_psi"
            rows += [
                (f"{tag:5s} -> |dE|<eps_E & |psi|<eps -> soft",
                 EnergyBangBangController(K_soft, u_max, e_band, wp,
                                          energy=mode), e_band),
                (f"{tag:5s} -> |dE|<eps_E & |psi|<eps -> soft -> base",
                 EnergyBangBangController(K_soft, u_max, e_band, wp,
                                          energy=mode, K_final=K_base,
                                          final_region=ellipsoid_region(P_base, c_base)),
                 e_band),
            ]

        last = {}
        for label, ctrl, region in rows:
            b, held, conv, t_calm = score(wp, ctrl, X0, args.horizon,
                                          stride=args.stride, pair=pair)
            n_sw, t_h, ever = switches(b, region)
            r = rec & ever
            sw = (f"{np.median(n_sw[r]):6.0f} / {n_sw[r].max():5d}"
                  if r.any() else "        —")
            print(f"  {label:50s} {held.sum():8d} {conv.sum():8d}  "
                  f"{calm_str(t_calm):>20s}  {int((rec & ever).sum()):8d}  {sw:>22s}")
            last[label.split()[0]] = b

        # --- почему: куда уходит накачанная энергия -------------------------
        b = last[("E" if args.energy[-1] == "total" else "E_psi")]
        E = wp.energy(b.x)
        p_theta = wp.wheel_momentum(b.x)
        gam = wp.p.gamma
        # первый отсчёт, где промах по энергии впервые стал меньше 1 % от D
        near = np.abs(E - wp.E_upright) < 0.01 * wp.p.D
        got = near.any(axis=1) & rec
        if got.any():
            i = np.argmax(near, axis=1)
            rows_ = np.flatnonzero(got)
            wheel = p_theta[rows_, i[rows_]] ** 2 / (2.0 * gam)
            tilt = wp.first_integral(b.x[rows_, i[rows_]], 0.0) / gam
            print(f"  накачка вышла на уровень E* в {got.sum()} восстановимых "
                  f"клетках из {rec.sum()}:")
            print(f"    в колесе p^2/2gamma  med {np.median(wheel):7.2f}  "
                  f"max {wheel.max():7.2f}   (E* = {wp.E_upright:.1f})")
            print(f"    наклону H/gamma      med {np.median(tilt):7.2f}  "
                  f"min {tilt.min():7.2f}   (нужно {wp.p.D:.1f})")
        else:
            print("  накачка ни в одной восстановимой клетке не вышла на E*")


if __name__ == "__main__":
    main()
