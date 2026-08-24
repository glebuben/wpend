"""
Boundary of the recoverable set in (theta, theta_dot) under a torque limit
|u| <= u_max for the wheeled-pendulum.

Key observation
---------------
After eliminating phi_ddot from the two Euler-Lagrange equations, the tilt
dynamics reduce to a single second-order ODE in theta:

    theta_ddot = [b(theta) u  +  gamma D sin(theta)
                  -  beta^2 sin(theta) cos(theta) theta_dot^2]  /  Delta(theta)

with
    b(theta)     = gamma + beta cos(theta)
    Delta(theta) = alpha gamma - beta^2 cos^2(theta).

Crucially, the right-hand side does NOT involve phi or phi_dot.  The cross
terms involving phi_dot * theta_dot in the Lagrangian cancel against d/dt of
the conjugate momentum, leaving only theta_dot^2.  So:

    *  phi is cyclic (translation invariance)
    *  phi_dot is also irrelevant for the tilt subsystem (Galilean invariance:
       shifting the wheel velocity by any constant changes nothing)

The "does the body fall" question therefore lives in a TWO-DIMENSIONAL phase
plane (theta, theta_dot) -- the critical boundary is a 1-D CURVE, not a 2-D
surface.  The 3-D state space (theta, theta_dot, phi_dot) intuition is wrong:
phi_dot drops out of the relevant dynamics, not just phi.

How the boundary is computed
----------------------------
With input bounded by |u| <= u_max, the maximum braking torque is u = -u_max
when (theta > 0, theta_dot > 0).  The boundary trajectory in (theta, theta_dot)
is the trajectory under u = -u_max that *just barely* reaches the failure
boundary theta = pi/2 with theta_dot = 0.  We obtain it by integrating
BACKWARDS in time from the touch point (pi/2, 0):

    forward time : (theta_0, theta_dot_0) ---u = -u_max---> (pi/2, 0)
    backward     : (pi/2, 0)  ----  reverse the ODE  ---->  (theta_0, theta_dot_0)

Reflecting through the origin gives the backward-fall branch (the mirror
trajectory that just barely avoids reaching theta = -pi/2 with theta_dot = 0).

The interior region enclosed by both branches is the *viability kernel*
under the torque limit:  for any initial (theta, theta_dot) inside, a control
|u(t)| <= u_max exists that keeps theta away from +-pi/2 forever.  Outside,
the body falls regardless of control.

Run
---
    python recoverable_set.py
    # writes figures/fig_recoverable_set.png
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# scripts/ live two levels below the project root; make `src` importable
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams
from src.recoverable import theta_ddot


# ---------- Plant parameters --------------------------------------------------
p = SegwayParams()
alpha, beta_, gamma_, D = p.alpha, p.beta, p.gamma, p.D

print(f"Lumped parameters:")
print(f"    alpha = I_b + m_b l^2           = {alpha:.4f} kg m^2")
print(f"    beta  = m_b r l                 = {beta_:.4f} kg m^2")
print(f"    gamma = I_w + (m_w + m_b) r^2   = {gamma_:.4f} kg m^2")
print(f"    D     = m_b g l                 = {D:.4f} N m")
print(f"Gravity torque at theta = pi/2: D = {D:.2f} N m  "
      f"(u_max must reach this to *just* hold the body horizontal at rest).")


# ---------- Backwards-in-time RK4 -------------------------------------------
def rk4_backwards(f, x0, h, n_steps, stop=lambda x: False):
    """Integrate dx/dt = f(x) BACKWARDS in time using classical RK4.

    Returns an array of shape (k, 2) of the visited states until the stop
    predicate trips or n_steps are exhausted.
    """
    x = np.asarray(x0, dtype=float)
    pts = [x.copy()]
    for _ in range(n_steps):
        k1 = -np.asarray(f(x))
        k2 = -np.asarray(f(x + h * k1 / 2))
        k3 = -np.asarray(f(x + h * k2 / 2))
        k4 = -np.asarray(f(x + h * k3))
        x  = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        if not np.isfinite(x).all() or stop(x):
            break
        pts.append(x.copy())
    return np.asarray(pts)


# ---------- Compute boundary curves for several u_max -----------------------
u_maxes = [2.0, 5.0, 10.0, 20.0, D, 50.0]    # includes u_max = D as reference
fig, ax = plt.subplots(figsize=(10, 6.5))
colors = plt.cm.viridis(np.linspace(0.05, 0.85, len(u_maxes)))

theta_seed = np.pi / 2 - 1e-4   # just below the failure boundary

for u_max, col in zip(u_maxes, colors):
    def f(x, _u=-u_max):
        return [x[1], theta_ddot(x[0], x[1], _u)]
    stop = lambda x: x[1] > 30.0 or x[1] < -5.0 or x[0] < -np.pi/2

    pts = rk4_backwards(f, [theta_seed, 0.0], h=0.0015, n_steps=8000, stop=stop)
    th, dth = pts[:, 0], pts[:, 1]

    label = f"u_max = {u_max:5.2f} N m"
    if abs(u_max - D) < 0.5:
        label += "   (= D, body holds horizontal at rest)"
    ax.plot( th,  dth, color=col, lw=2.0, label=label)
    ax.plot(-th, -dth, color=col, lw=2.0, ls="--", alpha=0.7)

ax.axhline(0, color="k", lw=0.4)
ax.axvline(0, color="k", lw=0.4)
ax.axvline( np.pi/2, color="red", lw=0.8, alpha=0.6,
           label=r"$|\theta| = \pi/2$  (body horizontal: failure)")
ax.axvline(-np.pi/2, color="red", lw=0.8, alpha=0.6)

ax.set_xlabel(r"$\theta$  [rad]")
ax.set_ylabel(r"$\dot\theta$  [rad/s]")
ax.set_title("Critical boundary of the recoverable set under torque limit"
             "  --  dashed = backward-fall branch")
ax.grid(alpha=0.3)
ax.set_xlim(-np.pi/2 - 0.05, np.pi/2 + 0.05)
ax.set_ylim(-12, 12)
ax.legend(loc="upper left", framealpha=0.9, fontsize=9)
fig.tight_layout()

out = ROOT / "figures" / "fig_recoverable_set.png"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, dpi=140)
print(f"\nwrote {out}")
