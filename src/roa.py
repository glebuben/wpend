"""
LQR region-of-attraction / Vdot certificate helpers (CLAUDE.md sec. 9).

Single source of truth for the machinery previously duplicated across
scripts/figures/roa_lqr.py, scripts/figures/roa_lqr_phys.py,
scripts/maps/lqr_map.py and scripts/verify/verify_lqr_roa.py.

Every function takes a controller exposing `.p` (SegwayParams), `.P`
(cost-to-go), `.K` (gain) and `.f_batch(X, u_max)` (vectorized nonlinear
closed-loop field) -- both LQRController (psi coords) and LQRControllerPhys
(physical coords) qualify, so the same code serves both coordinate systems.
"""

from __future__ import annotations

import numpy as np

THETA_MAX = np.pi / 2.0     # model is meaningless once the body is horizontal
THETA_FALL = 1.45           # latch divergence once the body passes this tilt


def closed_loop_accel(ctrl, th, ps, dth, dps, u_max: float = 0.0):
    """Vectorized (ddtheta, ddpsi) of the nonlinear closed loop u = -K x on
    arrays of any (common) shape, with optional saturation |u| <= u_max."""
    p = ctrl.p
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        ct, st = np.cos(th), np.sin(th)
        a = p.alpha + 2.0 * p.beta * ct + p.gamma
        b = p.gamma + p.beta * ct
        c = p.gamma
        det2 = a * c - b * b
        K = np.asarray(ctrl.K).ravel()
        u = -(K[0] * th + K[1] * ps + K[2] * dth + K[3] * dps)
        if u_max and u_max > 0.0:
            u = np.clip(u, -u_max, u_max)
        r1 = p.beta * st * dth ** 2 + p.D * st
        r2 = p.beta * st * dth ** 2 - u
        ddth = (c * r1 - b * r2) / det2
        ddps = (-b * r1 + a * r2) / det2
    return ddth, ddps


def vdot_batch(ctrl, X, u_max: float = 0.0):
    """Vdot = 2 x^T P f_nl(x, -K x) for a stack of states X of shape (N, 4)."""
    X = np.atleast_2d(np.asarray(X, float))
    F = ctrl.f_batch(X, u_max)
    return 2.0 * np.einsum("ij,jk,ik->i", X, ctrl.P, F)


def certified_level(ctrl, *, n_dirs=20000, s_max=8.0, ds=5e-3, seed=0,
                    u_max: float = 0.0, theta_max: float = THETA_MAX):
    """Largest c with {x^T P x <= c} subset {Vdot < 0} (minus the origin).

    Line search: along many random directions, march out from the origin to
    the first Vdot >= 0 crossing and record V there; c* is the minimum.
    Returns (c_star, binding_direction).  NOTE: s_max must exceed the
    ellipsoid's reach in every direction, otherwise near-tangent directions
    terminate before the boundary (see CLAUDE.md sec. 9.5)."""
    rng = np.random.default_rng(seed)
    D = rng.standard_normal((n_dirs, ctrl.P.shape[0]))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    dPd = np.einsum("ij,jk,ik->i", D, ctrl.P, D)

    c_dir = np.full(n_dirs, np.inf)
    active = np.ones(n_dirs, bool)

    s = ds
    while s <= s_max and active.any():
        idx = np.where(active)[0]
        X = s * D[idx]
        left = np.abs(X[:, 0]) >= theta_max
        vd = vdot_batch(ctrl, X, u_max)
        cross = (vd >= 0.0) & ~left
        hit = idx[cross]
        c_dir[hit] = s * s * dPd[hit]
        active[idx[cross | left]] = False
        s += ds

    c_star = float(c_dir.min())
    best_dir = D[int(np.argmin(c_dir))].copy()
    return c_star, best_dir


def slice_fields(ctrl, th_grid, dth_grid):
    """Vdot sign on the (theta, dtheta) slice with the other coords = 0."""
    TH, DTH = np.meshgrid(th_grid, dth_grid)
    X = np.stack([TH.ravel(), np.zeros(TH.size),
                  DTH.ravel(), np.zeros(TH.size)], axis=1)
    Vd = vdot_batch(ctrl, X).reshape(TH.shape)
    return TH, DTH, Vd


def vdot_slice(ctrl, i, j, Gi, Gj, base):
    """Vdot on the 2-D slice where coords i,j vary and the rest equal base."""
    X = np.tile(np.asarray(base, float), (Gi.size, 1))
    X[:, i] = Gi.ravel()
    X[:, j] = Gj.ravel()
    return vdot_batch(ctrl, X).reshape(Gi.shape)


def ellipse_ij(P, c, i, j, xi):
    """xj(xi) on the ellipse cross-section P_ii xi^2 + 2P_ij xi xj + P_jj xj^2 = c."""
    a = P[j, j]
    b = 2.0 * P[i, j] * xi
    cc = P[i, i] * xi ** 2 - c
    disc = b ** 2 - 4.0 * a * cc
    m = disc >= 0
    sq = np.sqrt(np.where(m, disc, 0.0))
    up = np.full_like(xi, np.nan)
    lo = np.full_like(xi, np.nan)
    up[m] = (-b[m] + sq[m]) / (2.0 * a)
    lo[m] = (-b[m] - sq[m]) / (2.0 * a)
    return up, lo, m


def roa_ellipse(ctrl, c, th_fine):
    """dtheta(theta) on the (theta, dtheta) slice of the ellipsoid x^T P x = c."""
    return ellipse_ij(ctrl.P, c, 0, 2, th_fine)


def simulate_grid(ctrl, TH0, DTH0, *, u_max: float = 0.0, dphi0=None,
                  T=6.0, dt=2.5e-3, theta_fall: float = THETA_FALL):
    """RK4 the full 4-D nonlinear closed loop from every grid IC.

    Initial wheel state: psi0 = 0 always; if `dphi0` is None the relative
    wheel rate starts at rest (dpsi0 = 0, the verify-script convention),
    otherwise the *wheel* starts at dphi0 so dpsi0 = dphi0 - dtheta0 (the
    interactive-map convention -- for the LQR this matters since u = -K x
    depends on dpsi).

    Returns (code, th, dth): an int code per cell (0 = stayed up, 1 = fell
    forward, 2 = fell backward) and the final tilt state arrays.
    """
    th = TH0.astype(float).copy()
    ps = np.zeros_like(th)
    dth = DTH0.astype(float).copy()
    dps = np.zeros_like(th) if dphi0 is None else dphi0 - dth
    code = np.zeros_like(th, dtype=int)           # 0 = still up / recovered
    n_steps = int(T / dt)

    def deriv(th, ps, dth, dps):
        ddth, ddps = closed_loop_accel(ctrl, th, ps, dth, dps, u_max)
        return dth, dps, ddth, ddps

    # Extreme gains can blow a cell up within one RK4 step before it latches;
    # ignore the transient overflow and sanitize below.
    for _ in range(n_steps):
      with np.errstate(over="ignore", invalid="ignore"):
        new_fwd = (code == 0) & ~(np.abs(th) < theta_fall) & (th > 0)
        new_bwd = (code == 0) & ~(np.abs(th) < theta_fall) & (th <= 0)
        code[new_fwd] = 1
        code[new_bwd] = 2
        active = (code == 0)
        if not active.any():
            break
        # Neutralize fallen cells so their (possibly huge) state never feeds
        # back into the vectorized derivative and overflows.
        th = np.where(active, th, np.sign(th) * theta_fall)
        dth = np.where(active, dth, 0.0)
        ps = np.where(active, ps, 0.0)
        dps = np.where(active, dps, 0.0)
        k1 = deriv(th, ps, dth, dps)
        k2 = deriv(th + 0.5 * dt * k1[0], ps + 0.5 * dt * k1[1],
                   dth + 0.5 * dt * k1[2], dps + 0.5 * dt * k1[3])
        k3 = deriv(th + 0.5 * dt * k2[0], ps + 0.5 * dt * k2[1],
                   dth + 0.5 * dt * k2[2], dps + 0.5 * dt * k2[3])
        k4 = deriv(th + dt * k3[0], ps + dt * k3[1],
                   dth + dt * k3[2], dps + dt * k3[3])
        th_n = th + dt / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        ps_n = ps + dt / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        dth_n = dth + dt / 6.0 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        dps_n = dps + dt / 6.0 * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])
        # Sanitize/bound blow-ups: map non-finite to a value past theta_fall and
        # clip to a large finite range so squares can never overflow.  Any cell
        # this affects has |theta| well past theta_fall and latches as "fallen".
        th_n = np.clip(np.nan_to_num(th_n, nan=2.0, posinf=2.0, neginf=-2.0),
                       -1.0e4, 1.0e4)
        dth_n = np.clip(np.nan_to_num(dth_n, nan=0.0, posinf=0.0, neginf=0.0),
                        -1.0e4, 1.0e4)
        ps_n = np.clip(np.nan_to_num(ps_n, nan=0.0), -1.0e6, 1.0e6)
        dps_n = np.clip(np.nan_to_num(dps_n, nan=0.0), -1.0e6, 1.0e6)
        th = np.where(active, th_n, th)
        ps = np.where(active, ps_n, ps)
        dth = np.where(active, dth_n, dth)
        dps = np.where(active, dps_n, dps)
    return code, th, dth
