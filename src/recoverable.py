"""
Bang-bang recoverable set of the reduced tilt dynamics (CLAUDE.md sec. 6).

Single source of truth for the math previously duplicated across
scripts/maps/recoverable_map.py, scripts/maps/lqr_map.py (recov_*),
scripts/verify/verify_recoverable_set.py and scripts/figures/*.

All functions accept an optional SegwayParams `p` (default: the standard
plant), so existing call sites that used module-level constants stay simple.
"""

from __future__ import annotations

import numpy as np

from .system import SegwayParams

_P = SegwayParams()


def Delta(theta, p: SegwayParams = _P):
    """Mass-matrix determinant of the reduced tilt dynamics (> 0)."""
    return p.alpha * p.gamma - p.beta ** 2 * np.cos(theta) ** 2


def b_(theta, p: SegwayParams = _P):
    """b(theta) = gamma + beta cos(theta)."""
    return p.gamma + p.beta * np.cos(theta)


def theta_ddot(theta, dtheta, u, p: SegwayParams = _P):
    """Reduced tilt acceleration (CLAUDE.md sec. 2.3); phi-free."""
    return (b_(theta, p) * u
            + p.gamma * p.D * np.sin(theta)
            - p.beta ** 2 * np.sin(theta) * np.cos(theta) * dtheta ** 2) / Delta(theta, p)


def H_(theta, dtheta, u, p: SegwayParams = _P):
    """First integral; conserved while u is constant (sec. 6.2 / 6.9)."""
    return (0.5 * Delta(theta, p) * dtheta ** 2
            - u * (p.gamma * theta + p.beta * np.sin(theta))
            + p.gamma * p.D * np.cos(theta))


def find_theta_eq(u_max: float, p: SegwayParams = _P) -> float:
    """Saddle angle for u = -u_max; 0 for u_max <= 0, pi/2 once u_max >= D."""
    if u_max <= 0.0:
        return 0.0
    if u_max >= p.D:
        return np.pi / 2.0
    f = lambda th: u_max * (p.gamma + p.beta * np.cos(th)) - p.gamma * p.D * np.sin(th)
    lo, hi = 0.0, np.pi / 2.0 - 1e-7
    if f(lo) * f(hi) > 0:
        return np.pi / 2.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def K_level(u_max: float, p: SegwayParams = _P) -> float:
    """Saddle (separatrix) level K(u_max) = H_-(theta_eq, 0)."""
    th_eq = find_theta_eq(u_max, p)
    return (u_max * (p.gamma * th_eq + p.beta * np.sin(th_eq))
            + p.gamma * p.D * np.cos(th_eq))


def boundary_curves(u_max: float, th_fine, p: SegwayParams = _P):
    """theta_dot^2(theta) on the two boundary level sets H_-+ = K.

    Returns (sq_minus, sq_plus, theta_eq, K)."""
    K = K_level(u_max, p)
    base = (K - p.gamma * p.D * np.cos(th_fine)) / Delta(th_fine, p)
    work = u_max * (p.gamma * th_fine + p.beta * np.sin(th_fine)) / Delta(th_fine, p)
    sq_minus = 2.0 * (base - work)      # H_- = K  (u = -u_max brake)
    sq_plus = 2.0 * (base + work)       # H_+ = K  (u = +u_max brake)
    return sq_minus, sq_plus, find_theta_eq(u_max, p), K


def ceiling_floor(u_max, TH, p: SegwayParams = _P):
    """Upper/lower theta_dot bounds of the recoverable set at each theta.

    The viability-kernel boundary is the union of the two saddle *stable
    manifolds*.  Each manifold is an H = K level curve, but the stable branch
    flips the sign of theta_dot at the saddle (the stable eigenvector has slope
    -sqrt(f') so theta_dot > 0 for theta < theta_eq and theta_dot < 0 for
    theta > theta_eq).  Hence:

      * upper bound  (H_- manifold, forward-brake saddle at +theta_eq):
            ceiling(theta) = +sqrt(sq_minus)  for theta <  theta_eq,
                             -sqrt(sq_minus)  for theta >  theta_eq;
      * lower bound  (H_+ manifold, backward-brake saddle at -theta_eq):
            floor(theta)   = -sqrt(sq_plus)   for theta > -theta_eq,
                             +sqrt(sq_plus)   for theta < -theta_eq.

    The "sign flip after the saddle" adds the backward-moving slivers past
    +-theta_eq (e.g. a near-horizontal forward tilt is still recoverable if it
    is swinging back fast enough but not too fast)."""
    K = K_level(u_max, p)
    th_eq = find_theta_eq(u_max, p)
    Dl = Delta(TH, p)
    s = p.gamma * TH + p.beta * np.sin(TH)
    g = K - p.gamma * p.D * np.cos(TH)
    sq_minus = 2.0 * (g - u_max * s) / Dl
    sq_plus = 2.0 * (g + u_max * s) / Dl
    sm = np.sqrt(np.clip(sq_minus, 0.0, None))
    sp = np.sqrt(np.clip(sq_plus, 0.0, None))
    ceiling = np.where(TH < th_eq, sm, -sm)
    floor = np.where(TH > -th_eq, -sp, sp)
    return ceiling, floor, sq_minus, sq_plus, th_eq


def recoverable_mask(u_max: float, TH, DTH, p: SegwayParams = _P):
    """Boolean grid: True where the state is recoverable under |u| <= u_max,
    i.e. floor(theta) < theta_dot < ceiling(theta)."""
    ceiling, floor, _, _, _ = ceiling_floor(u_max, TH, p)
    return (DTH < ceiling) & (DTH > floor)


def vectorized_sim(th0, dth0, u, *, T=4.5, dt=2e-3,
                   theta_bound=np.pi / 2, p: SegwayParams = _P):
    """
    Simulate the planar tilt ODE from every (th0[i,j], dth0[i,j]) initial
    condition under constant input u.  Returns an integer array with the same
    shape as th0:
         0 = stayed inside |theta| < theta_bound for the entire horizon,
         1 = reached theta >  theta_bound  (forward fall),
        -1 = reached theta < -theta_bound  (backward fall).
    """
    th  = th0.astype(float).copy()
    dth = dth0.astype(float).copy()
    out = np.zeros_like(th, dtype=int)
    n_steps = int(T / dt)
    for _ in range(n_steps):
        # Record fresh failures before stepping.
        new_fwd = (out == 0) & (th >=  theta_bound)
        new_bwd = (out == 0) & (th <= -theta_bound)
        out[new_fwd] =  1
        out[new_bwd] = -1
        active = (out == 0)
        if not active.any():
            break
        # RK4 (compute everywhere -- masking arrays is more overhead than gain).
        k1th  = dth
        k1dth = theta_ddot(th, dth, u, p)
        k2th  = dth + 0.5 * dt * k1dth
        k2dth = theta_ddot(th + 0.5 * dt * k1th,  dth + 0.5 * dt * k1dth, u, p)
        k3th  = dth + 0.5 * dt * k2dth
        k3dth = theta_ddot(th + 0.5 * dt * k2th,  dth + 0.5 * dt * k2dth, u, p)
        k4th  = dth + dt * k3dth
        k4dth = theta_ddot(th + dt * k3th,  dth + dt * k3dth, u, p)
        th_new  = th  + dt / 6.0 * (k1th  + 2 * k2th  + 2 * k3th  + k4th)
        dth_new = dth + dt / 6.0 * (k1dth + 2 * k2dth + 2 * k3dth + k4dth)
        # Only update grid points that haven't failed yet.
        th  = np.where(active, th_new,  th)
        dth = np.where(active, dth_new, dth)
    return out
