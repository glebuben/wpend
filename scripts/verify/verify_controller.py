"""Symbolic verification of the Lyapunov-based controller for the segway.
Checks equations (16)-(24) of Segway_Lyapunov_Controller.docx:
  - dot(Phi) = v * dot(xi)                                   (eq 18)
  - the explicit form of ddot(xi) used in deriving (20)
  - dot(V) = -dot(xi)^2 with the control law (23)            (eq 24)
  - the closed-form motor torque u in terms of v             (eq 25)
"""
import sympy as sp

# State / parameter symbols
theta, dtheta, ddtheta = sp.symbols(r'theta dtheta ddtheta')
psi, dpsi, ddpsi       = sp.symbols(r'psi dpsi ddpsi')
v                       = sp.symbols('v')
kp, ki, kl              = sp.symbols('k_p k_i k_l', positive=True)
alpha, beta_, gamma_, D = sp.symbols('alpha beta gamma D', positive=True)

# Effective coefficients (eq. 11, 13, 22)
a = alpha + 2*beta_*sp.cos(theta) + gamma_
b = gamma_ + beta_*sp.cos(theta)
c = gamma_
Delta = sp.simplify(a*c - b**2)
assert sp.simplify(Delta - (alpha*gamma_ - beta_**2*sp.cos(theta)**2)) == 0

# PFL form of theta-dynamics (eq. 15a)
ddtheta_expr = (sp.sin(theta)*(D + beta_*dtheta**2) - b*v)/a

# Auxiliary variable (eq. 16)
dxi = dpsi + kp*b*dtheta

# Drift-conserved energy (eq. 17)
Phi = kp*D*(1 - sp.cos(theta)) - sp.Rational(1,2)*kp*a*dtheta**2 + sp.Rational(1,2)*dpsi**2

# Total time derivative, treating each state as a function of t
def time_diff(expr):
    ts = sp.Symbol('t_')
    th = sp.Function('theta')(ts); ps = sp.Function('psi')(ts)
    sub = {theta: th, dtheta: sp.diff(th, ts), psi: ps, dpsi: sp.diff(ps, ts)}
    d   = sp.diff(expr.subs(sub), ts)
    inv = {sp.diff(th, ts, 2): ddtheta, sp.diff(ps, ts, 2): ddpsi,
           sp.diff(th, ts):    dtheta,  sp.diff(ps, ts):    dpsi,
           th: theta, ps: psi}
    return d.subs(inv)

# (18) check: dot(Phi) = v * dot(xi)
dPhi = sp.simplify(time_diff(Phi).subs(ddtheta, ddtheta_expr).subs(ddpsi, v))
print("dot(Phi) - v*dot(xi) :", sp.simplify(dPhi - v*dxi))

# Explicit form of ddot(xi) (used in eq. 20)
ddxi = sp.simplify(time_diff(dxi).subs(ddtheta, ddtheta_expr).subs(ddpsi, v))
delta0   = (a - kp*b**2)/a
drift_xi = (kp*sp.sin(theta)/a)*(b*D - beta_*dtheta**2*(alpha + beta_*sp.cos(theta)))
print("ddot(xi) - (drift + delta0*v) :", sp.simplify(ddxi - (drift_xi + delta0*v)))

# (22) delta(theta)
delta = kl + delta0
print("delta(theta) - (kl + 1 - kp*b^2/a) :",
      sp.simplify(delta - (kl + 1 - kp*b**2/a)))

# (23) feedback v and (24) dot(V) = -dot(xi)^2
xi_sym = sp.symbols('xi')
eta    = ki*xi_sym + drift_xi
v_ctrl = -(dxi + eta)/delta
dV     = dxi*(ki*xi_sym + ddxi.subs(v, v_ctrl) + kl*v_ctrl)
print("dot(V) + dot(xi)^2 :", sp.simplify(dV + dxi**2))

# (25) motor torque u in terms of v
u_expr     = -(b*ddtheta_expr + c*v - beta_*sp.sin(theta)*dtheta**2)
u_expected = beta_*sp.sin(theta)*dtheta**2 \
             - (b*sp.sin(theta)*(D + beta_*dtheta**2) + Delta*v)/a
print("u_expected - u_computed :", sp.simplify(u_expr - u_expected))
