"""Symbolic re-derivation of segway EOMs from the Lagrangian.
Reproduces equations (8a)-(8b) of Segway_Lyapunov_Controller.docx.
"""
import sympy as sp

t = sp.symbols('t')
theta, varphi = sp.Function(r'theta')(t), sp.Function(r'varphi')(t)
r, l, g_ = sp.symbols('r l g', positive=True)
mb, mw, Ib, Iw = sp.symbols('m_b m_w I_b I_w', positive=True)

# Lumped parameters
alpha  = Ib + mb*l**2
beta_  = mb*r*l
gamma_ = Iw + (mw + mb)*r**2
D      = mb*g_*l

# Geometry
x_body = r*varphi + l*sp.sin(theta)
y_body = r + l*sp.cos(theta)
vx, vy = sp.diff(x_body, t), sp.diff(y_body, t)

# Lagrangian
T_body  = sp.Rational(1,2)*mb*(vx**2 + vy**2) + sp.Rational(1,2)*Ib*sp.diff(theta,t)**2
T_wheel = sp.Rational(1,2)*(mw*r**2 + Iw)*sp.diff(varphi,t)**2
V_pot   = mb*g_*y_body
L       = sp.simplify(T_body + T_wheel - V_pot)

# Euler-Lagrange
q   = sp.Matrix([theta, varphi])
dq  = sp.diff(q, t)
ddq = sp.diff(dq, t)
eq_ = sp.simplify(sp.Matrix(
    [sp.diff(L.diff(dq[i]), t) - sp.diff(L, q[i]) for i in range(2)]))

# Expected closed forms (eq. 8 of the report)
exp1 = alpha*ddq[0] + beta_*sp.cos(theta)*ddq[1] - D*sp.sin(theta)
exp2 = beta_*sp.cos(theta)*ddq[0] + gamma_*ddq[1] - beta_*sp.sin(theta)*sp.diff(theta,t)**2

print("theta-equation residual :", sp.simplify(eq_[0] - exp1))
print("varphi-equation residual:", sp.simplify(eq_[1] - exp2))
