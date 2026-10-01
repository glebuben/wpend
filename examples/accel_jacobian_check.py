"""ER-025, шаг 3: проверка выкладок docs/accel_kalman.md §3 и §5.1.

Это не оцениватель и не прогон контура -- только числа, которые документ
утверждает, против независимого способа их получить:

1. Тождество §3.2: a_x = (u + w_psi)/(m_b l) - (d* - d) ddpsi -- ТОЧНОЕ, при
   любых psi, dpsi, u, w и выносе d.  Сверка с кинематикой
   WheeledPendulum.specific_force (до машинной точности).
2. Якобианы WheeledPendulum.jacobian и specific_force_jacobian против
   центральных разностей -- ошибка обязана убывать как eps^2 (порядок 2).
3. Таблица §3.6 «куда ставить датчик»: чувствительности a_x к наклону и к
   моменту в вертикали при разных d, kappa_psi, kappa_u (для d = 0.2
   обязаны выйти -1.368 и -0.0828 из docs/accel_compensation.md §2).
4. Ранги матрицы наблюдаемости §5.1 для наборов измерений.

Запуск:

    uv run python examples/accel_jacobian_check.py
"""

from __future__ import annotations

import numpy as np

from wpend.models import WheeledPendulum

D_SENSOR = 0.2


def accel_with_force(m, x, u, w):
    """Ускорения (ddpsi, ddtheta) с внешней силой -- независимо от jacobian:
    правая часть f плюс M^-1 w."""
    dx = m.f(0.0, x, u)
    q = np.einsum("...ij,...j->...i", m.mass_matrix_inverse(x[..., 0]), w)
    return dx[..., 2] + q[..., 0], dx[..., 3] + q[..., 1]


def check_identity(m, rng, n=1000):
    X = np.column_stack([rng.uniform(-3, 3, n), rng.uniform(-5, 5, n),
                         rng.uniform(-8, 8, n), rng.uniform(-8, 8, n)])
    U = rng.uniform(-20, 20, (n, 1))
    W = rng.uniform(-5, 5, (n, 2))
    worst = 0.0
    for d in (0.0, D_SENSOR, 1.0, m.center_of_percussion, 1.8):
        a_x, _ = m.specific_force(0.0, X, U, d, w=W)
        ddpsi, _ = accel_with_force(m, X, U, W)
        rhs = (U[:, 0] + W[:, 0]) / (m.p.mb * m.p.l) - (m.center_of_percussion - d) * ddpsi
        worst = max(worst, float(np.max(np.abs(a_x - rhs))))
    print(f"1. тождество §3.2, 5 выносов x {n} точек: max |ошибка| = {worst:.1e} м/с^2")
    return worst


def fd_errors(m, rng, eps, n=300):
    X = np.column_stack([rng.uniform(-1.5, 1.5, n), rng.uniform(-5, 5, n),
                         rng.uniform(-5, 5, n), rng.uniform(-5, 5, n)])
    U = rng.uniform(-10, 10, (n, 1))
    W = rng.uniform(-3, 3, (n, 2))

    def field(x, u, w):
        dx = m.f(0.0, x, u).copy()
        dx[..., 2:] += np.einsum("...ij,...j->...i", m.mass_matrix_inverse(x[..., 0]), w)
        return dx

    def sens(x, u, w):
        return np.stack(m.specific_force(0.0, x, u, D_SENSOR, w=w), axis=-1)

    def central(fun, x, u, w, which, k):
        e = np.zeros({"x": 4, "u": 1, "w": 2}[which])
        e[k] = eps
        args_p = {"x": x, "u": u, "w": w}
        args_m = dict(args_p)
        args_p[which] = args_p[which] + e
        args_m[which] = args_m[which] - e
        return (fun(**args_p) - fun(**args_m)) / (2 * eps)

    A, B, G = m.jacobian(0.0, X, U, W)
    C, Du, Cw = m.specific_force_jacobian(0.0, X, U, D_SENSOR, W)
    err = 0.0
    for mat, fun, which, n_in in ((A, field, "x", 4), (B, field, "u", 1), (G, field, "w", 2),
                                  (C, sens, "x", 4), (Du, sens, "u", 1), (Cw, sens, "w", 2)):
        num = np.stack([central(fun, X, U, W, which, k) for k in range(n_in)], axis=-1)
        err = max(err, float(np.max(np.abs(mat - num))))
    return err


def check_jacobians(m):
    print("2. якобианы против центральных разностей (ошибка ~ eps^2):")
    prev = None
    for eps in (1e-2, 5e-3, 2.5e-3):
        err = fd_errors(m, np.random.default_rng(1), eps)
        order = "" if prev is None else f"   порядок {np.log2(prev / err):.2f}"
        print(f"   eps = {eps:.4f}: max |ошибка| = {err:.3e}{order}")
        prev = err


def mounting_table(m):
    A, B = m.linearize_upright()
    A_psi, B_psi = A[2, 0], B[2, 0]
    g, ml = m.p.g, m.p.mb * m.p.l
    ds = m.center_of_percussion
    du = ds - 1.0 / (ml * B_psi)
    print(f"3. d* = {ds:.4f} м (центр качания), d_u = {du:.4f} м (слеп к моменту)")
    print("     d, м   da_x/dpsi   da_x/du   kappa_psi   kappa_u")
    for d in (0.0, D_SENSOR, 0.5, 1.0, du, ds, 1.8):
        C, Du, _ = m.specific_force_jacobian(0.0, np.zeros(4), np.zeros(1), d)
        cx, kx = C[0, 0], Du[0, 0]
        print(f"   {d:6.3f}  {cx:10.3f}  {kx:8.4f}  {1 + cx / g:10.3f}  {kx / g:8.4f}")


def observability_rank(A4, rows, with_bias):
    """Ранг [C; CA; ...; CA^(n-1)].  rows -- пары (строка по x, канал).
    Со смещениями состояние дописано (b_ax, b_az, b_g): смещения постоянны
    (нулевой блок A) и входят в свой канал единицей; энкодер и atan2
    смещения не несут."""
    if not with_bias:
        A = A4
        C = np.array([r for r, _ in rows])
    else:
        A = np.zeros((7, 7))
        A[:4, :4] = A4
        C = []
        for r, channel in rows:
            e = np.zeros(3)
            if channel in ("ax", "az", "gyro"):
                e[("ax", "az", "gyro").index(channel)] = 1.0
            C.append(np.concatenate([r, e]))
        C = np.array(C)
    n = A.shape[0]
    O = np.vstack([C @ np.linalg.matrix_power(A, k) for k in range(n)])
    return np.linalg.matrix_rank(O, tol=1e-9), n


def observability_ranks(m):
    """§5.1: ранги в вертикали.  Порог 1e-9 -- как в tests/test_encoder.py."""
    A4, _ = m.linearize_upright()
    C, _, _ = m.specific_force_jacobian(0.0, np.zeros(4), np.zeros(1), D_SENSOR)
    ax, az = (C[0], "ax"), (C[1], "az")
    gyro = (np.array([0, 0, 1, 0.0]), "gyro")
    enc = (np.array([-1, 1, 0, 0.0]), "enc")        # энкодер меряет theta - psi
    atan = (np.array([1, 0, 0, 0.0]), "atan")
    sets = [("atan2 + гироскоп", [atan, gyro], False),
            ("a_x, a_z + гироскоп", [ax, az, gyro], False),
            ("a_x, a_z + гироскоп + энкодер", [ax, az, gyro, enc], False),
            ("a_x, a_z + гироскоп, со смещениями", [ax, az, gyro], True),
            ("то же + энкодер", [ax, az, gyro, enc], True)]
    print("4. ранги наблюдаемости в вертикали:")
    for title, rows, with_bias in sets:
        r, n = observability_rank(A4, rows, with_bias)
        print(f"   {title:38s} {r} из {n}")


def main():
    m = WheeledPendulum()
    check_identity(m, np.random.default_rng(0))
    check_jacobians(m)
    mounting_table(m)
    observability_ranks(m)


if __name__ == "__main__":
    main()
