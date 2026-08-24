"""
Analytical (sympy) derivation of the recoverable-set boundary in (theta, theta_dot)
for the wheeled pendulum under torque limit |u| <= u_max.

The reduced tilt ODE is

    ddtheta = (b(theta) u + gamma D sin theta - beta^2 sin theta cos theta dtheta^2)
              / Delta(theta)

where b(theta) = gamma + beta cos theta and Delta(theta) = alpha gamma - beta^2 cos^2 theta.
Substituting w(theta) = dtheta^2 / 2 makes the equation FIRST-ORDER LINEAR in w:

    dw/dtheta + P(theta) w = Q(theta),
        P(theta) = 2 beta^2 sin theta cos theta / Delta(theta)
                 = Delta'(theta) / Delta(theta),   so  IF = Delta(theta).

Therefore
    d/dtheta [ Delta(theta) w(theta) ] = b(theta) u + gamma D sin theta,
and the integration is elementary:
    Delta(theta) w(theta) = u (gamma theta + beta sin theta) - gamma D cos theta + C.

Boundary condition: the recoverable-set boundary is the stable manifold of the
SADDLE EQUILIBRIUM of the brake-saturated dynamics (u = -u_max).  The saddle
sits at (theta_eq(u_max), 0) where theta_eq solves

    u_max (gamma + beta cos theta_eq) = gamma D sin theta_eq                  (*)

For u_max < D this has a solution theta_eq in (0, pi/2); for u_max >= D the
formal root collapses to theta_eq = pi/2 (the brake wins everywhere in [0, pi/2]).
We pin C by requiring w(theta_eq) = 0, giving

    C = u_max (gamma theta_eq + beta sin theta_eq) + gamma D cos theta_eq.

A naive "pin at pi/2" derivation only works for u_max >= D; for u_max < D it
yields an unphysical curve (theta_dot^2 < 0) between theta_eq and pi/2.

Run:
    python recoverable_set_symbolic.py
prints the closed-form expression and writes
    figures/fig_recoverable_set_symbolic.png
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


# ============================================================================
#  Symbolic derivation
# ============================================================================
theta, u_max, theta_eq = sp.symbols("theta u_max theta_eq", real=True)
alpha, beta_, gamma_, D = sp.symbols("alpha beta gamma D", positive=True)

b     = gamma_ + beta_ * sp.cos(theta)
Delta = alpha * gamma_ - beta_**2 * sp.cos(theta)**2

# Integrating-factor identity:  d/dtheta(Delta) == 2 beta^2 sin*cos
assert sp.simplify(sp.diff(Delta, theta)
                   - 2*beta_**2 * sp.sin(theta) * sp.cos(theta)) == 0

# d/dtheta(Delta * w) = b u + gamma D sin theta   with u = -u_max
RHS_brake = (-u_max) * b + gamma_ * D * sp.sin(theta)
integral_RHS = sp.integrate(RHS_brake, theta)

# Pin C so that w(theta_eq) = 0:
integral_at_saddle = integral_RHS.subs(theta, theta_eq)
C = -integral_at_saddle                                # because Delta * 0 = integral + C

# Closed form for dtheta^2
Delta_w = sp.simplify(integral_RHS + C)
dtheta_sq_expr = sp.simplify(2 * Delta_w / Delta)

# Saddle equation (used to compute theta_eq numerically for a given u_max)
saddle_eq_expr = u_max * (gamma_ + beta_ * sp.cos(theta_eq)) - gamma_ * D * sp.sin(theta_eq)

print("=" * 78)
print("Closed-form boundary curve  dtheta^2 = f(theta;  u_max, theta_eq(u_max))")
print("=" * 78)
print("Saddle equation: u_max (gamma + beta cos theta_eq) = gamma D sin theta_eq")
print(f"    -> {sp.simplify(saddle_eq_expr)} = 0")
print()
print("dtheta^2 numerator   :",
      sp.simplify(sp.together(dtheta_sq_expr).as_numer_denom()[0]))
print("dtheta^2 denominator :",
      sp.simplify(sp.together(dtheta_sq_expr).as_numer_denom()[1]))
print()
print("LaTeX:")
print(sp.latex(dtheta_sq_expr))
print()


# ============================================================================
#  Numerical evaluation + plot
# ============================================================================
# Plant parameters (cell 49 of the notebook)
mb, mw, l_, r_, g_ = 10.0, 1.0, 0.30, 0.05, 9.8
Ib = mb * l_**2 / 3.0
Iw = 0.7 * mw * r_**2
NUM = {
    alpha: Ib + mb * l_**2,
    beta_: mb * r_ * l_,
    gamma_: Iw + (mw + mb) * r_**2,
    D    : mb * g_ * l_,
}
print("Plant parameters:")
for s in (alpha, beta_, gamma_, D):
    print(f"    {s} = {float(NUM[s]):.4f}")
print(f"    Gravity torque at theta=pi/2:  D = {float(NUM[D]):.2f} N m\n")

# Build numpy callables
saddle_f = sp.lambdify((theta_eq, u_max),
                       saddle_eq_expr.subs(NUM), modules="numpy")
dtheta_sq_f = sp.lambdify((theta, u_max, theta_eq),
                          dtheta_sq_expr.subs(NUM), modules="numpy")

D_val = float(NUM[D])


def find_theta_eq(u_max_val: float) -> float:
    """Solve the saddle equation for theta_eq in [0, pi/2]."""
    if u_max_val >= D_val:
        return np.pi / 2
    # f(0) > 0 (LHS = u_max gamma > 0, RHS = 0), f(pi/2) < 0
    th_grid = np.linspace(1e-6, np.pi/2 - 1e-6, 2000)
    vals = saddle_f(th_grid, u_max_val)
    sign_change = np.where(np.diff(np.sign(vals)) != 0)[0]
    if len(sign_change) == 0:
        return np.pi / 2
    i = sign_change[0]
    # Refine with bisection
    lo, hi = th_grid[i], th_grid[i + 1]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if saddle_f(mid, u_max_val) * saddle_f(lo, u_max_val) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


u_maxes = [2.0, 5.0, 10.0, 20.0, D_val, 50.0]
colors  = plt.cm.viridis(np.linspace(0.05, 0.85, len(u_maxes)))

fig, ax = plt.subplots(figsize=(11, 7))
th_grid = np.linspace(-np.pi/2 + 1e-4, np.pi/2 - 1e-4, 4000)

print(f"{'u_max [N m]':>12}   {'theta_eq [deg]':>16}   "
      f"{'theta_eq [rad]':>16}   {'regime':>12}")
print("-" * 70)
for um, col in zip(u_maxes, colors):
    th_eq = find_theta_eq(um)
    sq = dtheta_sq_f(th_grid, um, th_eq)
    sq = np.where(np.isfinite(sq), sq, -1.0)

    label = f"u_max = {um:5.2f} N m"
    regime = "u_max >= D" if um >= D_val else "u_max <  D"
    print(f"{um:>12.3f}   {np.degrees(th_eq):>16.3f}   "
          f"{th_eq:>16.4f}   {regime:>12}")

    # Forward-fall stable manifold: upper branch for theta <= theta_eq, lower for >=.
    mask_up = (th_grid <= th_eq) & (sq >= 0)
    mask_lo = (th_grid >= th_eq) & (sq >= 0)
    if mask_up.any():
        ax.plot(th_grid[mask_up],  np.sqrt(sq[mask_up]),
                color=col, lw=2, label=label)
    if mask_lo.any():
        ax.plot(th_grid[mask_lo], -np.sqrt(sq[mask_lo]),
                color=col, lw=2)
    # Backward-fall stable manifold by symmetry (theta -> -theta, dtheta -> -dtheta)
    if mask_up.any():
        ax.plot(-th_grid[mask_up], -np.sqrt(sq[mask_up]),
                color=col, lw=2, ls="--", alpha=0.7)
    if mask_lo.any():
        ax.plot(-th_grid[mask_lo],  np.sqrt(sq[mask_lo]),
                color=col, lw=2, ls="--", alpha=0.7)

    # Saddle points
    ax.scatter([th_eq, -th_eq], [0.0, 0.0],
               color=col, s=42, zorder=5, edgecolors="k", linewidths=0.6)

    # ----- ddtheta = 0 nullcline (informational, NOT the recovery boundary) -----
    # ddtheta = 0  <=>  dtheta^2 = (gamma D sin - u_max b) / (beta^2 sin cos)
    # Plot only the positive root, and only where the RHS is real & positive.
    p = NUM
    a_v, b_v, g_v, D_v = float(p[alpha]), float(p[beta_]), float(p[gamma_]), float(p[D])
    ct, st = np.cos(th_grid), np.sin(th_grid)
    denom = b_v**2 * st * ct
    null_sq = np.where(np.abs(denom) > 1e-9,
                       (g_v * D_v * st - um * (g_v + b_v * ct)) / denom, np.nan)
    null_mask = (null_sq >= 0) & (th_grid > 0)
    if null_mask.any():
        ax.plot( th_grid[null_mask],  np.sqrt(null_sq[null_mask]),
                color=col, lw=1.0, ls=":", alpha=0.55)
        ax.plot(-th_grid[null_mask], -np.sqrt(null_sq[null_mask]),
                color=col, lw=1.0, ls=":", alpha=0.55)

# Annotation
ax.axhline(0, color="k", lw=0.4)
ax.axvline(0, color="k", lw=0.4)
ax.axvline( np.pi/2, color="red", lw=0.8, alpha=0.5,
           label=r"$|\theta| = \pi/2$  (body horizontal)")
ax.axvline(-np.pi/2, color="red", lw=0.8, alpha=0.5)

ax.set_xlabel(r"$\theta$  [rad]")
ax.set_ylabel(r"$\dot\theta$  [rad/s]")
ax.set_title("Recoverable-set boundary  vs.  ddtheta = 0 nullcline\n"
             "solid: forward-fall stable manifold (boundary)   "
             "dashed: backward-fall mirror   "
             "dotted: ddtheta = 0 nullcline (NOT the boundary)")
ax.grid(alpha=0.3)
ax.set_xlim(-np.pi/2 - 0.05, np.pi/2 + 0.05)
ax.set_ylim(-15, 15)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
fig.tight_layout()

out = ROOT / "figures" / "fig_recoverable_set_symbolic.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(f"\nwrote {out}")
