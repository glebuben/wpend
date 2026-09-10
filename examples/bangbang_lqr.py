"""Реле по сепаратрисе + ЛКР: два ответа на вопрос «где ЛКР можно доверять».

Запуск:  uv run --extra design python examples/bangbang_lqr.py

Считает для колёсного маятника со слабым мотором (u_max = 1.5 Н·м, при
котором засатурированный ЛКР заведомо не справляется с большими наклонами):

  1. ЛКР из уравнения Риккати и сертифицированный уровень c* -- с учётом
     насыщения и без него, чтобы увидеть цену этого «с учётом»;
  2. карту исходов чистого ЛКР на сетке НУ -- измеренный бассейн;
  3. четыре прогона по одной и той же сетке: чистый ЛКР, чистое реле, и реле
     с передачей управления по каждому из двух регионов.

Считаем два разных «получилось»:
  * УДЕРЖАЛ  -- корпус не упал (критерий карты, |theta| < pi/2);
  * НАКЛОН   -- корпус пришёл в вертикаль и не качается (|theta|, |dtheta| ~ 0);
  * +КОЛЕСО  -- вдобавок остановлено колесо (|dphi| ~ 0).

Горизонт выбран с запасом (20 с). Колонка +КОЛЕСО очень чувствительна к нему:
колесо успокаивается заметно позже наклона, и на коротком прогоне регулятор
выглядит неспособным вернуть колесо, хотя он просто не успел.
Разница между этими двумя колонками -- и есть содержание примера.

Колонка ПЕРЕДАЛ -- в скольких клетках реле вообще дошло до региона и отдало
управление ЛКР. Именно она объясняет две предыдущие: у варианта с эллипсоидом
передачи почти нет, у варианта с картой она происходит слишком рано.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from wpend import (
    BangBangLQRController,
    LinearFeedbackController,
    RK4Integrator,
    ellipsoid_region,
    grid_region,
    rollout_many,
)
from wpend.lqr import certified_level, lqr, lqr_tilt
from wpend.models import WheeledPendulum
from wpend.viz.grid import HELD, GridSpec, classify

U_MAX = 1.5
Q = np.diag([100.0, 1.0, 10.0, 1.0])
R = np.array([[1.0]])
# Горизонт 20 с, а не 6: на шести секундах колонка +КОЛЕСО мерила СКОРОСТЬ
# успокоения, а не способность успокоить. Чистый ЛКР на 6 с давал 315 из 639
# удержанных, на 20 с -- все 639. Вывод «ЛКР не возвращает колесо» был
# артефактом горизонта, а не свойством регулятора.
DT, N_STEPS = 1e-3, 20000


def never(x):
    """Регион, в который нельзя войти -- чистое реле, без передачи."""
    return np.zeros(np.shape(x)[:-1], dtype=bool)


def score(system, controller, X0):
    batch = rollout_many(system, controller, RK4Integrator(), X0,
                         dt=DT, n_steps=N_STEPS, stride=50)
    outcome, _ = classify(batch, i_theta=0, theta_fall=np.pi / 2)
    held = outcome == HELD
    x_end = batch.x[:, -1, :]
    tilt_home = held & (np.abs(x_end[:, 0]) < 1e-3) & (np.abs(x_end[:, 2]) < 1e-3)
    home = tilt_home & (np.abs(x_end[:, 3]) < 1e-2)
    # Флаг защёлки после прогона -- сколько клеток вообще дошло до передачи.
    engaged = getattr(controller, "_engaged", None)
    n_switch = None if engaged is None else int(np.asarray(engaged).sum())
    return held, tilt_home, home, n_switch


def main():
    wp = WheeledPendulum(u_max=U_MAX)
    A, B = wp.linearize_upright()
    K, P = lqr(A, B, Q, R)
    print(f"K = {np.array2string(K[0], precision=6)}   (Q = diag{np.diag(Q)}, R = 1)")

    c_free, _ = certified_level(wp, K, P, u_max=None, n_dirs=4000, ds=2e-3,
                                s_max=8.0, theta_max=np.pi / 2)
    c_sat, _ = certified_level(wp, K, P, u_max=U_MAX, n_dirs=4000, ds=2e-3,
                               s_max=8.0, theta_max=np.pi / 2)
    print(f"сертифицированный уровень c*: {c_free:.4g} без насыщения, "
          f"{c_sat:.4g} с ним -- в {c_free / c_sat:.0f} раз меньше")
    print(f"  досягаемость эллипсоида: |theta| <= {np.sqrt(c_sat / P[0, 0]):.4f}, "
          f"|dtheta| <= {np.sqrt(c_sat / P[2, 2]):.4f}, "
          f"|dphi| <= {np.sqrt(c_sat / P[3, 3]):.3f}")

    # Проект по ОДНОЙ подсистеме наклона: она замкнута сама на себя (§1.4),
    # поэтому её можно стабилизировать отдельно -- и регион получается на
    # порядок шире, потому что регулятору не нужно возвращать колесо.
    K_t, P_t = lqr_tilt(wp, np.diag([100.0, 10.0]), np.array([[1.0]]))
    c_t, _ = certified_level(wp, K_t, P_t, u_max=U_MAX, coords=(0, 2),
                             n_dirs=4000, ds=2e-3, s_max=8.0,
                             theta_max=np.pi / 2)
    print(f"\nЛКР только по наклону: K = {np.array2string(K_t[0], precision=4)}, "
          f"c*_t = {c_t:.4g}")
    print(f"  досягаемость: |theta| <= {np.sqrt(c_t / P_t[0, 0]):.4f}, "
          f"|dtheta| <= {np.sqrt(c_t / P_t[2, 2]):.4f}")

    # Срез 4-мерного эллипсоида по плоскости наклона: обнуляем строки и
    # столбцы колеса. Тогда x^T P_slice x -- та же форма, посчитанная от
    # состояния с фиктивно обнулённым колесом.
    P_slice = np.zeros_like(P)
    P_slice[np.ix_([0, 2], [0, 2])] = P[np.ix_([0, 2], [0, 2])]

    spec = GridSpec(n=41, theta_max=0.6, dtheta_max=3.0)
    X0 = spec.initial_states(wp.n_state, i_theta=0, i_dtheta=2)
    total = X0.shape[0]

    held_lqr, _, home_lqr, _ = score(wp, LinearFeedbackController(K), X0)
    mask = held_lqr.reshape(spec.n, spec.n)      # измеренный бассейн -- карта

    rows = [
        ("чистый ЛКР", LinearFeedbackController(K)),
        ("чистое реле", BangBangLQRController(K, U_MAX, never, wp)),
        ("реле -> ЛКР по эллипсоиду",
         BangBangLQRController(K, U_MAX, ellipsoid_region(P, c_sat), wp)),
        ("  то же, с привязкой колеса",
         BangBangLQRController(K, U_MAX, ellipsoid_region(P, c_sat), wp,
                               wheel_ref=True)),
        ("реле -> ЛКР наклона (цилиндр)",
         BangBangLQRController(K_t, U_MAX, ellipsoid_region(P_t, c_t), wp)),
        # Проверка идеи «пусть реле доведёт наклон, а дальше полный ЛКР сам
        # вернёт колесо»: зона -- срез 4-мерного эллипсоида по (theta, dtheta),
        # то есть колесо в проверке не участвует. Ответ отрицательный, см. §B8.
        ("реле -> ЛКР 4D по срезу наклона",
         BangBangLQRController(K, U_MAX, ellipsoid_region(P_slice, c_sat), wp)),
        ("реле -> ЛКР по карте",
         BangBangLQRController(K, U_MAX, grid_region(mask, spec.thetas,
                                                     spec.dthetas), wp)),
    ]

    print(f"\nсетка {spec.n}x{spec.n} = {total} НУ, горизонт "
          f"{DT * N_STEPS:.0f} с, u_max = {U_MAX}\n")
    print(f"{'':30s}  {'УДЕРЖАЛ':>9s}  {'НАКЛОН':>9s}  {'+КОЛЕСО':>9s}  {'ПЕРЕДАЛ':>9s}")
    for name, ctrl in rows:
        held, tilt_home, home, n_switch = score(wp, ctrl, X0)
        sw = "     --  " if n_switch is None else f"{n_switch:5d}/{total:<4d}"
        print(f"{name:30s}  {held.sum():5d}/{total:<4d}  {tilt_home.sum():5d}/{total:<4d}  "
              f"{home.sum():5d}/{total:<4d}  {sw}")


if __name__ == "__main__":
    main()
