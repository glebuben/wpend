"""Шаг фильтра Калмана ER-025 в виде, который компилирует numba (S3).

docs/kalman_speed.md §7.  Это ТА ЖЕ математика, что KalmanEstimator._predict1
и _update1 (скалярный путь, S1): прогноз RK4 по номинальной WheeledPendulum,
P <- F P F^T + Q с F = I + J dt + J^2 dt^2/2, затем последовательная
скалярная коррекция по «обелённым» каналам ИДУ и энкодеру.  Только записана
циклами по маленьким массивам и на math -- так её понимает numba.

numba НЕОБЯЗАТЕЛЬНА (решение Глеба 01.10).  Без неё модуль импортируется,
HAVE_NUMBA = False, и KalmanEstimator идёт по numpy-пути: тот же результат,
только медленнее.  В чистом Python эти циклы медленнее S1 -- поэтому без
numba они не вызываются вовсе.

Оракул -- совпадение с numpy-путём на одной последовательности измерений
(tests/test_accel_kalman.py), для одной клетки и для пачки, и независимость
результата пачки от числа потоков.
"""

from __future__ import annotations

import math

import numpy as np

try:                                      # необязательная зависимость
    import numba as _nb
    HAVE_NUMBA = True
except ImportError:                       # pragma: no cover -- окружение без fast
    _nb = None
    HAVE_NUMBA = False


def _terms(pp, psi, dpsi, u, w_psi, w_th):
    """Ускорения и их производные в точке (Key_Formulas §2).

    pp = (alpha, beta, gamma, D, g, r).  Возвращает (ddpsi, ddth, A20, A22,
    A30, A32, Mi00, Mi01, Mi11) -- строки 3-4 якобиана поля и M^-1."""
    al, be, ga, D = pp[0], pp[1], pp[2], pp[3]
    s, c = math.sin(psi), math.cos(psi)
    s2, c2 = 2.0 * s * c, c * c - s * s
    dp2 = dpsi * dpsi
    N = (ga + be * c) * u + ga * D * s - be * be * s * c * dp2 + ga * w_psi - be * c * w_th
    R = al * be * s * dp2 - be * D * s * c - (al + be * c) * u - be * c * w_psi + al * w_th
    N_psi = -be * s * u + ga * D * c - be * be * c2 * dp2 + be * s * w_th
    R_psi = al * be * c * dp2 - be * D * c2 + be * s * u + be * s * w_psi
    De = al * ga - be * be * c * c
    dDe = be * be * s2
    De2 = De * De
    return (N / De, R / De,
            (N_psi * De - N * dDe) / De2, -2.0 * be * be * s * c * dpsi / De,
            (R_psi * De - R * dDe) / De2, 2.0 * al * be * s * dpsi / De,
            ga / De, -be * c / De, al / De)


def _field(pp, x0, x2, u, w_psi, w_th):
    """(ddpsi, ddtheta) номинальной модели с силой w."""
    al, be, ga, D = pp[0], pp[1], pp[2], pp[3]
    s, c = math.sin(x0), math.cos(x0)
    De = al * ga - be * be * c * c
    dp2 = x2 * x2
    ddpsi = ((ga + be * c) * u + ga * D * s - be * be * s * c * dp2
             + ga * w_psi - be * c * w_th) / De
    ddth = (al * be * s * dp2 - be * D * s * c - (al + be * c) * u
            + al * w_th - be * c * w_psi) / De
    return ddpsi, ddth


def _predict(z, P, u, pp, dt, has_w, phi_w, inv_tau, Qfull):
    n = z.shape[0]
    w_psi = 0.0
    w_th = 0.0
    if has_w:
        w_psi = phi_w[0] * z[4]
        w_th = phi_w[1] * z[5]
    x0, x1, x2, x3 = z[0], z[1], z[2], z[3]
    h2 = dt / 2.0
    a1, b1 = _field(pp, x0, x2, u, w_psi, w_th)
    a2, b2 = _field(pp, x0 + h2 * x2, x2 + h2 * a1, u, w_psi, w_th)
    a3, b3 = _field(pp, x0 + h2 * (x2 + h2 * a1), x2 + h2 * a2, u, w_psi, w_th)
    a4, b4 = _field(pp, x0 + dt * (x2 + h2 * a2), x2 + dt * a3, u, w_psi, w_th)
    # производные угла -- это скорости в промежуточных точках RK4
    v1, v2, v3, v4 = x2, x2 + h2 * a1, x2 + h2 * a2, x2 + dt * a3
    o1, o2, o3, o4 = x3, x3 + h2 * b1, x3 + h2 * b2, x3 + dt * b3
    k6 = dt / 6.0

    # Якобиан в начале шага, как у numpy-пути.
    _, _, A20, A22, A30, A32, Mi00, Mi01, Mi11 = _terms(pp, x0, x2, u, w_psi, w_th)
    J = np.zeros((n, n))
    J[0, 2] = 1.0
    J[1, 3] = 1.0
    J[2, 0] = A20
    J[2, 2] = A22
    J[3, 0] = A30
    J[3, 2] = A32
    if has_w:
        J[2, 4] = Mi00
        J[2, 5] = Mi01
        J[3, 4] = Mi01
        J[3, 5] = Mi11
        J[4, 4] = -inv_tau[0]
        J[5, 5] = -inv_tau[1]
    F = np.empty((n, n))
    half = dt * dt / 2.0
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += J[i, k] * J[k, j]
            F[i, j] = J[i, j] * dt + acc * half
        F[i, i] += 1.0
    FP = np.zeros((n, n))
    for i in range(n):
        for k in range(n):
            f = F[i, k]
            if f != 0.0:
                for j in range(n):
                    FP[i, j] += f * P[k, j]
    for i in range(n):
        for j in range(n):
            acc = 0.0
            for k in range(n):
                acc += FP[i, k] * F[j, k]
            P[i, j] = acc
        P[i, i] += Qfull[i]

    z[0] = x0 + k6 * (v1 + 2.0 * v2 + 2.0 * v3 + v4)
    z[1] = x1 + k6 * (o1 + 2.0 * o2 + 2.0 * o3 + o4)
    z[2] = x2 + k6 * (a1 + 2.0 * a2 + 2.0 * a3 + a4)
    z[3] = x3 + k6 * (b1 + 2.0 * b2 + 2.0 * b3 + b4)
    if has_w:
        z[4] = w_psi
        z[5] = w_th


def _update(z, P, y, u, pp, d, has_w, ib0, enc, Linv, R_enc, Rdiag, diag, out_h, out_S):
    n = z.shape[0]
    m = 4 if enc else 3
    g, r = pp[4], pp[5]
    x0, x1, x2 = z[0], z[1], z[2]
    w_psi = 0.0
    w_th = 0.0
    if has_w:
        w_psi = z[4]
        w_th = z[5]
    ddpsi, ddth, A20, A22, A30, A32, Mi00, Mi01, Mi11 = _terms(pp, x0, x2, u, w_psi, w_th)
    s, c = math.sin(x0), math.cos(x0)
    h = np.empty(m)
    h[0] = -g * s + d * ddpsi + r * c * ddth + z[ib0]
    h[1] = g * c - d * x2 * x2 + r * s * ddth + z[ib0 + 1]
    h[2] = x2 + z[ib0 + 2]
    C = np.zeros((m, n))
    C[0, 0] = -g * c + d * A20 - r * s * ddth + r * c * A30
    C[0, 2] = d * A22 + r * c * A32
    C[1, 0] = -g * s + r * c * ddth + r * s * A30
    C[1, 2] = -2.0 * d * x2 + r * s * A32
    if has_w:
        C[0, 4] = d * Mi00 + r * c * Mi01
        C[0, 5] = d * Mi01 + r * c * Mi11
        C[1, 4] = r * s * Mi01
        C[1, 5] = r * s * Mi11
    C[0, ib0] = 1.0
    C[1, ib0 + 1] = 1.0
    C[2, ib0 + 2] = 1.0
    C[2, 2] = 1.0
    if enc:
        h[3] = x1 - x0
        C[3, 0] = -1.0
        C[3, 1] = 1.0
    if diag:
        for i in range(m):
            acc = 0.0
            for k in range(n):
                ck = C[i, k]
                if ck != 0.0:
                    for l in range(n):
                        acc += ck * P[k, l] * C[i, l]
            out_h[i] = h[i]
            out_S[i] = acc + Rdiag[i]
    # «обеление» каналов ИДУ: c' = L^-1 c, nu' = L^-1 nu
    Cw = np.zeros((m, n))
    nw = np.empty(m)
    rr = np.ones(m)
    for i in range(3):
        acc = 0.0
        for j in range(i + 1):
            lij = Linv[i, j]
            acc += lij * (y[j] - h[j])
            for k in range(n):
                Cw[i, k] += lij * C[j, k]
        nw[i] = acc
    if enc:
        for k in range(n):
            Cw[3, k] = C[3, k]
        nw[3] = y[3] - h[3]
        rr[3] = R_enc
    z_prior = z.copy()
    Pc = np.empty(n)
    for i in range(m):
        sk = rr[i]
        for a in range(n):
            acc = 0.0
            for b in range(n):
                acc += P[a, b] * Cw[i, b]
            Pc[a] = acc
        resid = nw[i]
        for a in range(n):
            sk += Cw[i, a] * Pc[a]
            resid -= Cw[i, a] * (z[a] - z_prior[a])
        for a in range(n):
            ka = Pc[a] / sk
            z[a] += ka * resid
            for b in range(n):
                P[a, b] -= ka * Pc[b]
    for a in range(n):
        for b in range(a + 1, n):
            v = 0.5 * (P[a, b] + P[b, a])
            P[a, b] = v
            P[b, a] = v


step_serial = step_parallel = None

if HAVE_NUMBA:
    _jit = _nb.njit(cache=True)
    _terms = _jit(_terms)
    _field = _jit(_field)
    _predict = _jit(_predict)
    _update = _jit(_update)

    # Шаг по пачке: клетки независимы, каждая -- на своём месте в Z, P.
    # Две сборки: последовательная (одна клетка, малые пачки -- запуск
    # потоков дороже самого шага) и с prange.  Сумм между клетками нет,
    # поэтому результат не зависит от числа потоков.

    @_nb.njit(cache=True)
    def step_serial(Z, P, Y, U, predict, pp, dt, d, has_w, phi_w, inv_tau, Qfull, ib0,
                    enc, Linv, R_enc, Rdiag, diag, out_h, out_S):
        for i in range(Z.shape[0]):
            if predict:
                _predict(Z[i], P[i], U[i], pp, dt, has_w, phi_w, inv_tau, Qfull)
            _update(Z[i], P[i], Y[i], U[i], pp, d, has_w, ib0, enc, Linv, R_enc, Rdiag,
                    diag, out_h[i], out_S[i])

    @_nb.njit(cache=True, parallel=True)
    def step_parallel(Z, P, Y, U, predict, pp, dt, d, has_w, phi_w, inv_tau, Qfull, ib0,
                      enc, Linv, R_enc, Rdiag, diag, out_h, out_S):
        for i in _nb.prange(Z.shape[0]):
            if predict:
                _predict(Z[i], P[i], U[i], pp, dt, has_w, phi_w, inv_tau, Qfull)
            _update(Z[i], P[i], Y[i], U[i], pp, d, has_w, ib0, enc, Linv, R_enc, Rdiag,
                    diag, out_h[i], out_S[i])


#: С какого размера пачки выгоднее потоки (замер, docs/kalman_speed.md §9).
PARALLEL_FROM = 32
