# CLAUDE.md — Wheeled-Pendulum (Segway) Project Notes

This file is the project's reference. Load it into context when continuing
work; it summarizes the dynamics, controller, code structure, and
experiments developed in this chat.

---

## 1. Project context

**Long-term goal.** Stabilize an under-actuated wheeled robot in a lab. The
full robot is more complex than a segway, but a wheel + inverted-pendulum
("wheeled-pendulum") model is the right benchmark: a single motor torque
between the wheel and the body must keep the body upright while also
controlling the wheel.

**Term 1** (https://github.com/glebuben/Wheeled-Pendulum-RL) — built the
Lagrangian model and attempted a data-driven controller (vectorized
REINFORCE).

**Term 2 (this chat)** — model-based control. Based on the Advanced Control
Methods course material, adapted the Lyapunov-based cart-pole controller of
Aguilar-Ibáñez, Gutiérrez Frias and Suárez Castañón (Nonlinear Dynamics 40,
367–374, 2005) to the wheeled-pendulum. Companion ACM projects at
https://github.com/glebuben/Control-sandbox.

---

## 2. Physical model

### 2.1 Geometry and coordinates

- `θ`: body tilt from upward vertical (0 = upright).
- `φ`: wheel rotation angle in the world frame (rolls without slipping).
- Wheel radius `r`; body CG at distance `l` from wheel axle.
- Wheel centre: `(rφ, r)`. Body CG: `(rφ + l sinθ, r + l cosθ)`.

### 2.2 Parameters

Default plant (from notebook cell 49, `src/system.py`):
`m_b = 10 kg`, `m_w = 1 kg`, `l = 0.30 m`, `r = 0.05 m`, `g = 9.8 m/s²`,
`I_b = m_b l² / 3` (slender rod), `I_w = 0.7 m_w r²`.

Lumped (used everywhere in the math):

| symbol | definition                            | value (defaults) |
|--------|---------------------------------------|------------------|
| α      | `I_b + m_b l²`                        | 1.200            |
| β      | `m_b r l`                             | 0.150            |
| γ      | `I_w + (m_w + m_b) r²`                | 0.029            |
| D      | `m_b g l`                             | 29.4 N·m         |
| b(θ)   | `γ + β cosθ`                          | —                |
| a(θ)   | `α + 2β cosθ + γ`  (used after change of coords) | —     |
| Δ(θ)   | `αγ − β² cos²θ`  (mass-matrix det, > 0) | —              |

### 2.3 Equations of motion (Euler–Lagrange)

With generalized coordinates `q = (θ, φ)ᵀ` and motor torque acting as
`τ = (u, −u)ᵀ`:

```
α θ̈ + β cosθ φ̈ − D sinθ          =  u        (body)
β cosθ θ̈ + γ φ̈ − β sinθ θ̇²       = −u        (wheel)
```

i.e. `M(θ) q̈ + C(θ, θ̇) q̇ + G(θ) = τ` with

```
M = [α, β cosθ; β cosθ, γ],
G = (−D sinθ, 0)ᵀ,
C q̇ = (0, −β sinθ θ̇²)ᵀ.
```

`φ` is a cyclic coordinate (does not appear in `L`), and after eliminating
`φ̈` the tilt dynamics reduce to a single 2-D ODE in (θ, θ̇):

```
θ̈ = ( b(θ) u + γ D sinθ − β² sinθ cosθ θ̇² ) / Δ(θ)
```

Neither `φ` nor `φ̇` appears — Galilean invariance of the idealised
no-slip / no-friction model. Bringing them back requires modelling rolling
resistance, motor back-EMF, traction limits, or aerodynamic drag.

### 2.4 Coordinate change for controller design

Define `ψ := φ − θ` (relative wheel–body angle). In `(θ, ψ)` the input
acts only on `ψ` (`τ̃ = (0, −u)ᵀ`), and the system has the cart-pole
structure of Aguilar-Ibáñez et al.:

```
a(θ) θ̈ + b(θ) ψ̈ − β sinθ θ̇² − D sinθ = 0     (unactuated)
b(θ) θ̈ +   γ ψ̈ − β sinθ θ̇²            = −u   (actuated)
```

---

## 3. Lyapunov-based controller (the main deliverable)

### 3.1 Partial feedback linearization

Introduce a virtual input `v` through

```
u = β sinθ θ̇² − [ b(θ) sinθ (D + β θ̇²) + Δ(θ) v ] / a(θ).
```

The closed-loop becomes

```
θ̈ = [ sinθ (D + β θ̇²) − b(θ) v ] / a(θ),
ψ̈ = v.
```

### 3.2 Auxiliary variable and drift-conserved Φ

```
ξ  = ψ + kₚ ( γ θ + β sinθ ),                 (eq. 16)
ξ̇ = ψ̇ + kₚ b(θ) θ̇,
Φ  = kₚ D ( 1 − cosθ ) − ½ kₚ a(θ) θ̇² + ½ ψ̇².     (eq. 17)
```

Property: along the PFL'd dynamics, `Φ̇ = v · ξ̇`.

### 3.3 Lyapunov function

```
V(x) = ½ kᵢ ξ² + ½ ξ̇² + k_L Φ.                (eq. 19)
```

Compute `V̇`:
```
V̇ = ξ̇ [ η(x) + δ(θ) v ],
η(x) = kᵢ ξ + (kₚ sinθ / a(θ)) [ D b(θ) − β θ̇² (α + β cosθ) ],
δ(θ) = k_L + 1 − kₚ b(θ)² / a(θ).
```

### 3.4 Feedback law

```
v = − ( ξ̇ + η(x) ) / δ(θ),
u =   β sinθ θ̇²
    + [ Δ(θ) ( ξ̇ + η ) / δ − b(θ) sinθ (D + β θ̇²) ] / a(θ).
```

This yields `V̇ = −ξ̇² ≤ 0`.

### 3.5 Stability and attraction region

Gain condition (eq. 27):
```
kₚ (γ + β)² > (1 + k_L) (α + 2β + γ).
```

For default parameters this requires roughly `kₚ > 47.6`. A working tuning is
`kₚ = 100, kᵢ = 1.0, k_L = 0.5` (gives 3.21 > 2.29).

Critical angle (eq. 28): smallest `|θ_S| > 0` with `δ(θ_S) = 0`. For the
default tuning, `θ_S ≈ 0.653 rad ≈ 37.4°`.

Attraction threshold (eq. 30): `K_S = k_L kₚ D (1 − cos θ_S) ≈ 302.7`.

**Proposition.** With gains satisfying eq. (27) and `V(x₀) < K_S` (and
`|θ₀| < θ_S`), the closed-loop trajectory converges to the upright
equilibrium `(0, 0, 0, 0)`. Proof by LaSalle's invariance principle on
`{ξ̇ = 0}`; the only invariant subset is the origin (details in §5 of the
full report, `docs/Segway_Lyapunov_Controller.docx`).

---

## 4. Code structure

```
Early Research/
├── main.py                  python main.py            -> writes figures/fig_*.png
│                            python main.py --menu     -> opens the launcher menu
│                            python main.py --animate  -> opens pygame window
│                            python main.py --map / --lqr-map / --hessian-map
├── CLAUDE.md                this file
├── src/                     the importable package
│   ├── __init__.py
│   ├── system.py            SegwayParams (dataclass) + SegwayDynamics.rhs(state, u)
│   ├── controller.py        ControllerGains + LyapunovController
│   │                            .diagnostics(state)   -> dict (u, v, ξ, ξ̇, η, δ, V, Φ)
│   │                            .control(state)       -> u
│   │                            .V(state)
│   │                            .critical_angle(), .K_S()
│   │                            .gain_condition_value(), .check_gain_condition()
│   ├── simulation.py        rk4_step(...), simulate(dynamics, controller, state0, T,
│   │                            n_out=1200, inner_dt=5e-4)
│   │                        Returns dict: t, theta, psi, phi, dtheta, dpsi, dphi,
│   │                            u, v, xi, dxi, V, Vdot
│   ├── recoverable.py       bang-bang recoverable-set math (sec. 6), shared by
│   │                            the maps/verify/figure scripts: Delta, b_,
│   │                            theta_ddot, H_, find_theta_eq, K_level,
│   │                            boundary_curves, ceiling_floor,
│   │                            recoverable_mask, vectorized_sim
│   ├── roa.py               LQR ROA machinery (sec. 9), shared likewise:
│   │                            closed_loop_accel, vdot_batch, certified_level,
│   │                            slice_fields, vdot_slice, ellipse_ij,
│   │                            roa_ellipse, simulate_grid (u_max/dphi0 options)
│   ├── lqr.py               linearize_upright, solve_care, LQRWeights,
│   │                            LQRController  (sec. 9.3)
│   ├── lqr_phys.py          LQR in physical (theta, phi) coordinates (sec. 9.7)
│   ├── hessian.py           Hessian-norm grids + figure helpers
│   ├── hessian_map.py       interactive Hessian-norm heatmap window
│   ├── animation.py         pygame interactive window (Slider, Button, App,
│   │                            build_animation(...))
│   └── menu.py              pygame launcher/hub (MenuApp, build_menu()); spawns
│                                every script below as an independent process.
│                                `python main.py --menu`.
├── scripts/
│   ├── figures/             one-shot figure generators -> figures/*.png
│   │   ├── recoverable_set.py            numerical (backward-RK4) boundary (sec. 6)
│   │   ├── recoverable_set_symbolic.py   sympy closed form + plot (sec. 6)
│   │   ├── roa_lqr.py                    LQR ROA, psi design (sec. 9.2)
│   │   ├── roa_lqr_tilt.py               pure 2-D tilt-subsystem ROA (sec. 9.5)
│   │   ├── roa_lqr_phys.py               ROA in physical coordinates (sec. 9.7)
│   │   ├── roa_lqr_slices.py             phi/phidot slices of the certificate (sec. 9.6)
│   │   ├── vdot_portraits.py             Vdot phase-portrait gallery (sec. 9.8)
│   │   ├── vdot_heatmap.py               LQR Vdot heatmap (also --interactive)
│   │   └── hessian_norms.py              Hessian-norm heatmaps
│   ├── maps/                clickable interactive maps (spawn animations)
│   │   ├── recoverable_map.py            bang-bang recoverable-set map (sec. 6)
│   │   └── lqr_map.py                    LQR basin map, lambda1/lambda2 knobs (sec. 9.9)
│   └── verify/              symbolic + numerical regression tests
│       ├── verify_segway.py              symbolic EOM check (residuals = 0)
│       ├── verify_controller.py          symbolic V̇ = −ξ̇² + closed-form u check
│       ├── verify_lqr.py                 LQR self-tests, residuals ~1e-12 (sec. 9.3)
│       ├── verify_lqr_roa.py             true-basin 4-D grid sim (sec. 9.3)
│       └── verify_recoverable_set.py     vectorized grid-search boundary check (sec. 6.6)
├── docs/
│   ├── Segway_Lyapunov_Controller.docx   full report (derivation + proof)
│   ├── Early_Research_Report.docx        term report
│   ├── AIDA_project.ipynb                term-1 sympy notebook (model derivation)
│   └── Cart-pole controller.pdf          Aguilar-Ibáñez et al. (2005), the reference paper
├── figures/                 fig_states.png, fig_phase.png, fig_lyapunov.png,
│                            fig_recoverable_set.png, fig_grid_verification.png, …
└── animations/              segway_<timestamp>.gif written by Save-GIF button
```

State convention used throughout: `x = (θ, ψ, θ̇, ψ̇)`. Convert back to the
physical wheel angle with `φ = ψ + θ`.

### 4.1 Coding conventions / gotchas

- Scripts under `scripts/` prepend the project root to `sys.path`
  (`ROOT = Path(__file__).resolve().parents[2]`) so `from src...` imports work
  from any cwd, and write figures to the root-anchored `ROOT / "figures"`.
- All modules begin with `from __future__ import annotations` (the codebase
  targets Python 3.9+; `X | None` type hints would fail).
- The integrator is a fixed-step RK4 with `inner_dt = 5e-4`. Coarser than
  the initial `5e-5`; smooth dynamics tolerate this and the interactive
  window stays responsive.
- The pygame App runs `simulate(...)` on a background daemon thread
  (`threading`) so the GUI never blocks. Polling happens in the main loop.
- `figures/` and `animations/` are auto-created on first run.

---

## 5. Experiments

### 5.1 Static plots — `python main.py`

Three figures dropped into `figures/`:

1. **`fig_states.png`** — θ, ψ, θ̇, ψ̇, u, φ vs time. Default IC θ₀ = 0.30 rad,
   T = 8 s. The body settles to |θ| < 0.03° within 8 s.
2. **`fig_phase.png`** — phase portraits (θ, θ̇) and (ψ, ψ̇), trajectory
   colored by time with a `viridis` colormap. Two spirals converging to the
   origin.
3. **`fig_lyapunov.png`** — V(t) decaying from ≈ 79.8 → 4·10⁻², well below
   K_S ≈ 302.7; and V̇ (analytic −ξ̇² vs numerical `dV/dt`) overlaid.

CLI overrides: `--theta0`, `--psi0`, `--dtheta0`, `--dpsi0`, `--kp`,
`--ki`, `--kl`, `--T`.

### 5.0 Interactive menu / launcher — `python main.py --menu`

A pygame hub window (`src/menu.py`) that sits in front of every other entry
point. Left column: run-parameter sliders (θ₀, ψ₀, θ̇₀, ψ̇₀, kₚ, kᵢ, k_L,
λ₁, λ₂, u_max, T). Right column: buttons grouped into

- **Interactive windows** — *Open animation* / *Open recoverable map*.
- **Generate figures** — *Static plots* (honours the sliders) plus one button
  per figure script (`recoverable_set[_symbolic]`, `roa_lqr`, `roa_lqr_tilt`,
  `roa_lqr_phys`, `roa_lqr_slices`, `vdot_portraits`) and *Open figures folder*.
- **Options** — controller toggle (Lyapunov/LQR, passed to the animation),
  animation autostart toggle, "open PNG when done" toggle, Quit.

Like `recoverable_map.py`, every action is launched as an *independent OS
process* (`python main.py --animate ...`, `python scripts/figures/roa_lqr.py`, …), so the menu
never blocks and multiple windows can run at once. The sliders feed the
animation and the `main.py` static plots on the command line; the standalone
figure scripts use their own defaults. A status panel reports each job
(generating / done+time / closed / failed) and auto-opens the resulting PNG.

### 5.2 Interactive animation — `python main.py --animate`

Requires `pip install pygame pillow`. Layout: segway view on top, states
plot in the middle, V(t)/V̇(t) on the bottom; sliders + four buttons on
the right.

Sliders: θ₀, ψ₀, θ̇₀, ψ̇₀, kₚ, kᵢ, k_L, T.

Buttons:
- **Start** — re-simulate with current slider values (rebuilds the
  controller, re-checks gain condition (27)). Runs on a background thread so
  the window stays responsive.
- **Replay** — restart playback of the most recent simulation.
- **Pause / Resume** — toggle.
- **Save GIF** — render the current trajectory into
  `animations/segway_<timestamp>.gif` (uses Pillow).

Hotkeys: `R` = Start, Space = Pause/Resume, `G` = Save GIF, Esc = Quit.

### 5.3 Symbolic regression tests

```
python scripts/verify/verify_segway.py        # prints residuals 0 0 for the two EOMs
python scripts/verify/verify_controller.py    # prints 0 for Φ̇ − v·ξ̇,  V̇ + ξ̇²,  u_closed_form − u_PFL
```

---

## 6. Recoverable set under torque limit  *(UNDER DEVELOPMENT)*

> **Status:** the derivation works numerically and analytically, but I'm
> not yet fully convinced of the intuition behind every step. Treat this
> section as work-in-progress.

### 6.1 Setup

Question: under `|u| ≤ u_max`, from which states can the controller keep
the body away from `|θ| = π/2`?

Because `θ̈` depends only on `(θ, θ̇, u)` (φ̇ also drops out, see §2.3),
the "does it fall" question is **2-D in (θ, θ̇)**, not 3-D. The boundary
is a 1-D *curve*, not a surface.

### 6.2 First integral under constant brake

Multiply `Δ(θ) θ̈ = b u + γ D sinθ − β² sinθ cosθ θ̇²` by `θ̇`, recognise
total time derivatives, find that the centripetal piece cancels exactly
with `½ Δ'(θ) θ̇³`. Conserved quantity (for constant `u`):

```
H(θ, θ̇; u) = ½ Δ(θ) θ̇²  −  u (γθ + β sinθ)  +  γ D cosθ.
```

For maximum forward brake `u = −u_max`:
```
H_−(θ, θ̇; u_max) = ½ Δ(θ) θ̇²  +  u_max (γθ + β sinθ)  +  γ D cosθ.
```

For maximum backward brake `u = +u_max`: `H_+` with the sign on the
control-work piece flipped.

### 6.3 Saddle equation and saddle level

Under `u = −u_max`, the autonomous (θ, θ̇)-system has a saddle equilibrium
at `(θ_eq, 0)` where `θ_eq ∈ [0, π/2]` solves

```
u_max ( γ + β cos θ_eq )  =  γ D sin θ_eq.
```

For `u_max < D` there is a unique root in `(0, π/2)`; for `u_max ≥ D` take
`θ_eq = π/2` (the body can be statically held horizontal). Saddle level:

```
K(u_max) = u_max ( γ θ_eq + β sin θ_eq ) + γ D cos θ_eq.
```

### 6.4 Recoverable-set boundary (closed form)

Recoverable iff *some* admissible control keeps the body bounded. Because
the system has one input, the worst-case alternatives are the two
extremes, and recoverability is equivalent to `H_+ < K` AND `H_− < K`.
Using `max(a+b, a−b) = a + |b|`, this is one inequality:

```
½ Δ(θ) θ̇²  +  u_max |γθ + β sinθ|  +  γ D cosθ  <  K(u_max).
```

The boundary is the equality; solving for θ̇:

```
θ̇²(θ; u_max) =
    2 [ K(u_max) − u_max |γθ + β sinθ| − γ D cosθ ] / Δ(θ).
```

Sanity checks: at `θ = ±θ_eq` the RHS vanishes (boundary touches the
θ̇ = 0 axis exactly at the saddles); at the origin the RHS is positive
(upright is strictly inside the set); as `u_max → 0` the inequality
reduces to the undamped-pendulum energy bound `½ α θ̇² < γ D (1 − cosθ)`;
as `u_max → ∞` the bound explodes (the whole strip `|θ| < π/2` becomes
recoverable).

### 6.5 Why the boundary is `H = K`, not "Ḣ = 0"

A natural-sounding wrong intuition is "the boundary is where we can't
change the energy any more, so Ḣ = 0." But `Ḣ = 0` is the *defining
property* of `H` — true everywhere on every trajectory under fixed `u`.
What singles out the boundary is *which level set* we're on. Among all
the level sets, exactly one passes through the saddle: `H = K`. That level
set is the saddle's stable manifold, and it separates the bounded
("recoverable") orbits from the open ("doomed") ones.

The "can't change energy" idea is partially right, but it needs *variable*
`u`. Under `u ∈ [−u_max, +u_max]` we have

```
Ḣ_−  =  b(θ) θ̇ (u + u_max).
```

When `θ̇ > 0` (body falling forward), the minimum over `u` is achieved at
`u = −u_max` and equals **zero**: maximum brake just *holds* `H_−`
constant, no control choice can reduce it further. The boundary `H_− = K`
is therefore the level at which "barely-maintenance" is the best the
controller can do — start above and you stay above (doomed); start below
and you stay below (safe).

### 6.6 Numerical verification

`scripts/verify/verify_recoverable_set.py` does a vectorized-RK4 grid sweep:

- Pick `u_max`. Compute `θ_eq` and `K`.
- Lay out an N×N grid in (θ₀, θ̇₀); θ̇ range scales with the analytic
  boundary at θ = 0 so the green region is always fully framed.
- For every grid point, simulate under `u = −u_max` and under `u = +u_max`
  in parallel (one vectorized loop over time).
- Classify each point by outcome: survives both / falls forward / falls
  backward / falls under both.
- Overlay the closed-form curves `H_± = K` and the saddles `(±θ_eq, 0)`.
- Print the simulation/analytic agreement percentage.

Produces `figures/fig_grid_verification.png` with one subplot per
`u_max ∈ {2, 5, 15, 50}` N·m. Solid curve cleanly separates green
("survives") from red ("falls forward"); dashed cleanly separates green
from blue.

### 6.7 Things still unclear / TODO

- A more intuitive physical interpretation of the "control-work" term
  `−u (γθ + β sinθ)` in `H`. It's the integral of `b(θ) u` along θ, but I
  don't yet have a clean mechanical-energy story for it.
- The exact relationship between the 4-D viability kernel (with optimal
  switching) and the 2-D bang-bang boundary. For a single-input system I
  believe they coincide, but a clean proof is missing.
- Extension to non-idealised plant (rolling resistance, motor back-EMF,
  traction limit). Each of these reintroduces a `φ̇` term and turns the
  boundary into an honest 2-D surface in `(θ, θ̇, φ̇)`. Plan: redo the
  conservation-law derivation with the extra terms; the integrating factor
  trick should generalise.
- Why a "pin at π/2" derivation gives an unphysical (`θ̇² < 0`) curve when
  `u_max < D`: mathematically clear (the level set through `(π/2, 0)`
  isn't the separatrix for low `u_max`), but the physical reading is still
  fuzzy.

### 6.8 Cross-references

- Numerical version of the boundary: `scripts/figures/recoverable_set.py`
  (backward RK4 from the saddle).
- Symbolic version: `scripts/figures/recoverable_set_symbolic.py`
  (sympy derivation + plot, prints the closed form in LaTeX).
- Grid verification: `scripts/verify/verify_recoverable_set.py`.
- Shared implementation of all closed-form pieces: `src/recoverable.py`
  (the scripts above import it instead of redefining the math).

---

## 7. Reference papers and external code

- **Aguilar-Ibáñez, Gutiérrez Frias, Suárez Castañón** — "Lyapunov-Based
  Controller for the Inverted Pendulum Cart System," Nonlinear Dynamics 40,
  367–374 (2005). The cart-pole derivation we adapted. PDF in `docs/`.
- **Wheeled-Pendulum-RL** — https://github.com/glebuben/Wheeled-Pendulum-RL.
  Term-1 plant + REINFORCE.
- **Control-sandbox** — https://github.com/glebuben/Control-sandbox. ACM
  course projects (Lyapunov, adaptive, backstepping, MPC).

---

## 8. How to extend / next steps

- **Add backstepping or adaptive variants** in `src/controller.py` and a
  matching `--controller` flag in `main.py`.
- **Add rolling resistance / motor back-EMF** to `src/system.py` and watch
  the recoverable boundary become 3-D.
- **Wire the Lyapunov controller's saturation** to the recoverable-set
  bound: if `|u(x)|` would exceed `u_max`, fall back to a saturated brake,
  and visualise when this happens during a run.
- **Hardware bring-up**: parameter identification from a real wheeled robot
  using the same `SegwayParams` dataclass; sanity-check with the symbolic
  scripts before deploying the controller.

---

## 9. LQR fail-region / region of attraction  *(NEW)*

Goal: locate the states from which the **LQR** controller (designed on the
linearization about upright) actually fails on the true nonlinear plant.

### 9.1 The right way to phrase "linearization is bad"

The original intuition was "compare `f_nl` with its linearization and
threshold the error." Raw pointwise error has no natural threshold. The
principled version ties the error to the closed-loop Lyapunov decay. Using
the LQR cost-to-go `P` (from the CARE) and `V(x) = xᵀ P x`, along the
*nonlinear* closed loop

```
V̇(x) = −xᵀ Q_cl x + 2 xᵀ P g(x),   g(x) = f_nl(x, −Kx) − A_cl x,
```

with `Q_cl = Q + KᵀRK`. The linearization error `g(x)` is "too big" exactly
where it overpowers the nominal decay, i.e. where `V̇(x) ≥ 0`. So the
threshold is not hand-picked — it is set by `Q`. The `V̇ = 0` boundary IS
the linearization-error boundary.

### 9.2 Certified region of attraction

The largest sublevel set `{x : xᵀ P x ≤ c*}` contained in `{V̇ < 0}` (minus
the origin) is a certified ROA. `c*` is found by a vectorized line search
over random directions (`certified_level`). For the default plant and
`Q = diag(50, 0.1, 5, 0.1)`, `R = 1`:

- `c* ≈ 8.77`; certified upright-tilt extent `|θ| ≤ 0.70 rad ≈ 40°` on the
  `(θ, θ̇)` slice.
- Grid verification: **100%** of certified points actually converge
  (soundness), and the certified ellipse captures `≈ 9%` of the true basin
  area on the slice — quadratic-`V` ROA is sound but conservative.

The true basin (full 4-D nonlinear closed-loop simulation under `u = −Kx`)
is much larger; LQR only really fails in the far corners where large tilt and
large tilt-rate push outward together.

### 9.3 Code

```
src/lqr.py            linearize_upright(p) -> (A, B);  solve_care(A,B,Q,R)
                      (numpy-only Hamiltonian method, no scipy);
                      LQRWeights, LQRController (.control/.diagnostics/.V/
                      .Vdot/.residual), same interface as LyapunovController.
src/roa.py            shared ROA machinery: certified_level(ctrl) -> (c*, dir),
                      vdot_batch, slice_fields, roa_ellipse, vdot_slice, ellipse_ij,
                      closed_loop_accel, simulate_grid (used by every sec.-9 script;
                      LQRController/LQRControllerPhys expose f_batch for it).
scripts/figures/roa_lqr.py
                      thin driver over src/roa;  vdot_batch (vectorized
                      nonlinear V̇);  writes figures/fig_lqr_roa.png
                      (V̇-sign map + certified ellipse).
scripts/verify/verify_lqr_roa.py
                      full 4-D nonlinear grid sim -> true basin; overlays the
                      certified ellipse and V̇=0 boundary; prints soundness +
                      conservatism; writes figures/fig_lqr_roa_verification.png.
scripts/verify/verify_lqr.py
                      self-tests: (A,B) vs finite diff, CARE residual,
                      Hurwitz Acl, P SPD, closed-loop Lyapunov identity,
                      nonlinear V̇<0 near origin.  All residuals ~1e-12.
```

### 9.4 Caveats / next steps

- This is the *unsaturated* LQR basin. To get the operational fail region
  add the torque limit `|u| ≤ u_max` and intersect with the recoverable set
  of §6 (LQR will additionally fail wherever `|Kx| > u_max` drives the body
  past recoverability).
- The quadratic-`V` ROA is conservative; a tighter certificate would use an
  SOS program after recasting `sinθ, cosθ` as variables with `s²+c²=1`.
- `scripts/verify/verify_lqr_roa.py` grid sim takes ~30 s (T=8 s, 220² ICs); lower `N`/`T`
  for a quick look.

### 9.5 Pure 2-D tilt-subsystem ROA  (scripts/figures/roa_lqr_tilt.py)

Because falling is a `theta`-only phenomenon (the reduced tilt ODE of §2.3
contains no `psi`/`dpsi`), the honest, un-sliced ROA lives in `(theta, dtheta)`.
We design an LQR directly on the 2x2 tilt linearization
`A2 = [[0,1],[gamma D/Delta0, 0]]`, `B2 = [0, (gamma+beta)/Delta0]`
(`Delta0 = alpha gamma - beta^2`), with `Q = diag(50, 5)`, `R = 1`, reusing
`solve_care`.  `V(x) = x^T P x` (2x2 `P`), and `Vdot` is taken along the
*nonlinear* reduced tilt loop `u = -K2 x`.

Results: `K2 = [13.34, 2.62]`, poles `{-3.61, -33.70}`, `c* = 32.94`,
certified tilt extent `|theta| <= 1.37 rad ~ 78 deg`.  Grid sim (260^2..300^2):
**100% of certified points converge** and the certified ellipse now captures
**~82%** of the true basin area (vs ~9% for the 4-D slice in §9.2).  Dropping
the phantom `psi, dpsi` directions -- which is what bound the 4-D certificate --
makes the quadratic certificate nearly tight.

Figure `figures/fig_lqr_roa_tilt.png` overlays the simulated basin, the
certified ROA ellipse, and the explicit `Vdot = 0` curve; the latter is tangent
to the ellipse exactly at the binding direction, which is the geometric meaning
of `c*`.

Caveat on the line search: `s_max` in `certified_level_2d` must exceed the
ellipse's reach (here `dtheta` extent ~13 rad/s), otherwise near-vertical
directions terminate before the ellipse boundary and the "certificate" would be
unverified there.  `s_max = 16` covers it; `c*` is set by the diagonal binding
direction, not the vertical one.

### 9.6 phi/phidot slices of the 4-D certificate  (scripts/figures/roa_lqr_slices.py)

`Vdot` is computed in `vdot_batch` directly as `Vdot = 2 x^T P f_nl(x, -K x)`,
with `f_nl` the full nonlinear field (sin, cos, centripetal `dtheta^2`).  This
equals the decomposition `-x^T Qcl x + 2 x^T P g(x)` to ~1e-11, so the
linearization error `g(x)` is fully included -- it is baked into `f_nl`.

phi/phidot are already in the 4-D model via `psi = phi - theta`,
`dpsi = dphi - dtheta`; the 4-D LQR uses `psi, dpsi`.  The tilt physics
`ddtheta` is phi-independent (verified: same `ddtheta` for different
`psi, dpsi`); the only coupling is through `u = -K x`.  Consequences:

- The `Vdot = 0` set is a 3-D surface in 4-D; the `(theta, dtheta)` picture is
  the slice `psi = dpsi = 0`.  Panel (d) shows that as `dpsi (~ dphi)` grows
  0 -> 5 -> 10, the `Vdot = 0` curve in `(theta, dtheta)` moves outward (the
  decreasing region enlarges).  So yes, accounting for phidot changes the
  boundary.
- The certified ellipsoid is extremely elongated along `dpsi`: LQR barely
  penalizes the wheel velocity (`Q_dpsi = 0.1`), so `V` is nearly flat there.
  Axis-aligned extent `|dpsi| <= 10.1`; the tilted `(theta,dpsi)` cross-section
  reaches `|dpsi| ~ 26`, and `(dtheta,dpsi)` reaches `~ 51`.

Figure `figures/fig_lqr_roa_phi_slices.png`: panels (a) `(theta,dtheta)`,
(b) `(theta,dpsi)`, (c) `(dtheta,dpsi)` with the wheel-velocity axis widened so
the full ellipse cross-section is visible, and (d) the `Vdot=0` shift vs `dpsi`.

### 9.7 LQR ROA in physical (theta, phi) coordinates  (src/lqr_phys.py, scripts/figures/roa_lqr_phys.py)

To remove the `psi = phi - theta` substitution from the plots, the LQR is also
built natively in `x = (theta, phi, dtheta, dphi)`.  `SegwayDynamicsPhys.rhs`
solves `M_phi (ddtheta, ddphi) = (D sin th + u, beta sin th dth^2 - u)` with
`M_phi = [[alpha, beta cos th],[beta cos th, gamma]]`.  Linearization at upright:
`A[2,0]=gamma D/Delta0`, `A[3,0]=-beta D/Delta0`, `B[2]=(gamma+beta)/Delta0`,
`B[3]=-(alpha+beta)/Delta0`, `Delta0=alpha gamma-beta^2` (verified vs finite
differences ~1e-9).  Same `Q=diag(50,0.1,5,0.1)`, `R=1`.

Result: `K=[37.79, 0.316, 7.55, 0.477]`, `c*=9.02`, certified tilt extent
`|theta|<=0.708 rad ~40.6 deg` -- essentially identical tilt physics to the
psi-design (40.1 deg), confirming the substitution only redistributes the input
between equations and does not move the tilt ROA.  Figures
`fig_lqr_roa_phys.png` and `fig_lqr_roa_phys_slices.png` carry a genuine `dphi`
wheel-velocity axis (no psi anywhere).

### 6.9 Full derivation of the first integral H (step by step)

Start from the reduced tilt ODE (sec. 2.3) with the torque held CONSTANT at u:

    Delta(theta) ddtheta = b(theta) u + gamma D sin(theta)
                           - beta^2 sin(theta) cos(theta) dtheta^2,   (*)

with Delta(theta) = alpha gamma - beta^2 cos^2(theta),  b(theta) = gamma + beta cos(theta).

Idea: a single-DOF autonomous ODE x'' = f(x, x') always admits an
energy-type first integral; we expose it with the integrating factor dtheta
(multiply by velocity and recognise total time derivatives).

Step 1 -- multiply (*) by dtheta:

    Delta dtheta ddtheta = b u dtheta + gamma D sin(theta) dtheta
                           - beta^2 sin cos dtheta^3.            (**)

Step 2 -- rewrite the inertial (left) term.  Since
    d/dt[ 1/2 Delta(theta) dtheta^2 ] = Delta dtheta ddtheta
                                        + 1/2 Delta'(theta) dtheta^3,
and Delta'(theta) = 2 beta^2 sin(theta) cos(theta), we get
    Delta dtheta ddtheta = d/dt[ 1/2 Delta dtheta^2 ] - beta^2 sin cos dtheta^3.

Step 3 -- substitute into (**).  The centripetal cubes cancel EXACTLY:

    d/dt[ 1/2 Delta dtheta^2 ] - beta^2 sin cos dtheta^3
        = b u dtheta + gamma D sin dtheta - beta^2 sin cos dtheta^3
    =>  d/dt[ 1/2 Delta dtheta^2 ] = b u dtheta + gamma D sin(theta) dtheta.

Step 4 -- both remaining terms are total derivatives (u constant):
    b u dtheta = u (gamma + beta cos) dtheta = d/dt[ u (gamma theta + beta sin theta) ],
    gamma D sin(theta) dtheta = - d/dt[ gamma D cos(theta) ].

Hence d/dt[ 1/2 Delta dtheta^2 - u(gamma theta + beta sin theta) + gamma D cos theta ] = 0,
so the conserved first integral is

    H(theta, dtheta ; u) = 1/2 Delta(theta) dtheta^2
                           - u (gamma theta + beta sin theta)
                           + gamma D cos(theta).

Reading: 1/2 Delta dtheta^2 is kinetic-like, gamma D cos theta is the gravity
potential, and -u(gamma theta + beta sin theta) is the work of the constant
control torque (a "control potential"); -d/dtheta of it returns b(theta) u.

For a GIVEN maximum torque we just substitute the extreme admissible input:
    forward brake  u = -u_max :  H_-(theta,dtheta) = 1/2 Delta dtheta^2
                                  + u_max(gamma theta + beta sin theta) + gamma D cos theta,
    backward brake u = +u_max :  H_+(theta,dtheta) = 1/2 Delta dtheta^2
                                  - u_max(gamma theta + beta sin theta) + gamma D cos theta.

Saddle of the u=-u_max flow: an equilibrium needs dtheta=0 and ddtheta=0, i.e.
    b(theta_eq) u_max = gamma D sin(theta_eq)   ->   u_max(gamma+beta cos) = gamma D sin,
the unique root in (0, pi/2) for u_max < D (else theta_eq = pi/2).  Its energy
level K(u_max) = H_-(theta_eq, 0) = u_max(gamma theta_eq + beta sin theta_eq)
+ gamma D cos theta_eq is the separatrix level (the saddle's stable manifold),
which divides bounded (recoverable) from escaping orbits.

Boundary of the recoverable set: a state is recoverable iff SOME admissible
torque keeps it bounded; with one input the worst cases are the two extremes,
so recoverability <=> H_+ < K AND H_- < K.  Using max(a+b, a-b) = a + |b| with
a = 1/2 Delta dtheta^2 + gamma D cos theta, b = u_max(gamma theta + beta sin theta),
the two conditions collapse to one:

    1/2 Delta(theta) dtheta^2 + u_max |gamma theta + beta sin theta|
        + gamma D cos(theta)  <  K(u_max),

and solving the equality for dtheta gives the closed-form boundary curve of
sec. 6.4.

### 9.8 Vdot phase-portrait gallery  (scripts/figures/vdot_portraits.py)

Visualisations of the Lyapunov rate `Vdot = 2 x^T P f_nl(x,-Kx)` in physical
coords `(theta, phi, dtheta, dphi)`:

- `fig_vdot_2d_portraits.png` -- `Vdot` as a diverging colour field on all six
  coordinate-pair slices (others = 0), with the `Vdot=0` curve (black) and the
  certified-ROA ellipse cross-section (green).  `Vdot` is almost flat in `phi`
  (the wheel angle barely enters), strongly shaped by `theta`.
- `fig_vdot_3d_surfaces.png` -- the same `Vdot` as a height surface over each
  pair, with the `z=0` plane; the theta-pairs show the saddle/upward growth,
  the velocity-only pairs a downward bowl.
- `fig_vdot_isosurface.png` -- the `Vdot=0` surface in the 3-D slice
  `(theta,dtheta,dphi)` at `phi=0`.  Because `Vdot` is quadratic in `dphi` at
  fixed `(theta,dtheta)`, the surface is found in closed form (two sheets); the
  certified ellipsoid (also quadratic in `dphi`) sits between the sheets,
  i.e. inside `{Vdot<0}`.

No marching-cubes/skimage dependency: every surface is an explicit
solve-for-one-coordinate, so it runs with numpy + matplotlib only.

### 9.9 Interactive LQR recoverable-set map  (scripts/maps/lqr_map.py)

A clickable counterpart to `scripts/maps/recoverable_map.py`, but for the *LQR controller's
own* recoverable set -- i.e. its true basin of attraction on the nonlinear
plant -- parametrized by the two LQR weights

    Q = lambda1 * I_4 ,    R = lambda2 * I_1     (same convention as the animation).

Unlike the bang-bang map of sec. 6 (which is controller-independent, drawn from
`H_+- = K`), the region here *depends on the controller*: each grid IC
`(theta0, 0, theta_dot0, 0)` is integrated under the full 4-D nonlinear closed
loop `x' = f(x, -K x)` (`src/roa.py`'s `closed_loop_accel` /
`simulate_grid`, shared with `scripts/verify/verify_lqr_roa.py`) and classified

    0 RECOVERABLE (green) -- returns upright,
    1 falls FORWARD (red) / 2 falls BACKWARD (blue) -- body topples |theta|>=1.45.

The two lambdas are the live knobs: typing lambda1 / lambda2 in the text boxes
re-solves the CARE (`solve_care`), rebuilds `K, P`, and recomputes the whole
basin (~2-3 s for a 65x65 grid, T=6 s). Overlaid for context:

* the certified quadratic ROA `x^T P x = c*` (blue; `certified_level` 4-D line
  search, the conservative inner estimate of sec. 9.2), and
* the `Vdot = 0` curve on the `(theta, dtheta)` slice (`vdot_slice`; where the
  linearization error overtakes the nominal LQR decay).

An `u_max` radio adds optional torque saturation `|u| <= u_max` (0 = none): the
basin is simulated with that same limit and it is passed to the spawned
animation, so the picture matches what clicking actually launches.

A `recoverable-set overlay` radio (off / `H± lines` / `mask`) superimposes the
controller-independent bang-bang recoverable set of sec. 6 (`recov_ceiling_floor`
/ `recov_mask`) at the selected `u_max`.  The recoverable set is the viability
kernel, whose boundary is the union of the two saddle *stable manifolds* -- and
each manifold is an `H = K` level curve that **flips the sign of theta_dot at
its saddle** (the stable eigenvector has slope `-sqrt(f')`).  So the boundary
has four pieces:

  * UPPER bound = `H_-` stable manifold of the forward-brake saddle `(+theta_eq,
    0)`:  `theta_dot = +sqrt(sq_minus)` for `theta < theta_eq`, then
    `-sqrt(sq_minus)` for `theta > theta_eq`;
  * LOWER bound = `H_+` stable manifold of the backward-brake saddle
    `(-theta_eq, 0)`:  `theta_dot = -sqrt(sq_plus)` for `theta > -theta_eq`,
    then `+sqrt(sq_plus)` for `theta < -theta_eq`.

`recov_ceiling_floor` returns these `ceiling`/`floor` arrays and `recov_mask`
is just `floor < theta_dot < ceiling`.  The post-saddle sign flip adds the
backward-moving *slivers* past `+-theta_eq` (e.g. a near-horizontal forward tilt
`theta ~ 1.45` is still recoverable if it is swinging *back* fast enough but not
too fast: `theta_dot in (-sqrt(sq_plus), -sqrt(sq_minus))`).  This is the
correction over the earlier "closed leaf" version, which wrongly truncated the
boundary at the saddles.  `_draw_overlay` plots the two manifolds (magenta solid
`H_-`, dashed `H_+`); `mask` dims every state outside the kernel.  The overlay is
purely visual (`_redraw_from_cache` reuses the basin); use a finite `u_max > 0`
for a meaningful comparison (`u_max = 0` -> only the origin is recoverable).

Initial wheel velocity: a `phi_dot0` slider (`s_dphi`, range +-15 rad/s) sets
the wheel's starting angular velocity.  Each grid IC becomes
`(theta0, psi0=0, theta_dot0, dpsi0 = phi_dot0 - theta_dot0)`, threaded through
`simulate_grid` / `classify_grid` / `estimate_dth_bound` and the `Vdot=0` /
certified-ellipse slice (`vdot_slice`, `roa_ellipse` now take `dphi0` and
substitute `dpsi = phi_dot0 - theta_dot`).  Because the LQR uses `u = -K x` with
a nonzero `K_dpsi`, the basin genuinely shifts with `phi_dot0` -- whereas the
bang-bang recoverable set does *not* move (the reduced tilt ODE is phi-free,
sec. 2.3), which the overlay makes visually obvious.  The basin recomputes on
slider *release* (`_on_release` + `_dphi_dirty`), not during the drag, since
each re-simulation is ~2 s.  A click launches the animation seeded with the
matching `--dpsi0 = phi_dot0 - theta_dot0`.

Menu access: the launcher (`python main.py --menu`, `src/menu.py`) has an
"Open LQR recoverable map" button under *Interactive windows* that spawns
`python main.py --lqr-map --lam1 .. --lam2 .. --u-max ..` seeded from the
lambda1/lambda2/u_max sliders.

Two-window design identical to `recoverable_map.py` (same folder): hover highlights the cell
(blitted rectangle), left-click spawns an *independent* process
`python main.py --animate ... --controller lqr --lam1 .. --lam2 .. --u-max ..`,
so several LQR animations can run at once while the map stays responsive.

Entry points: `python scripts/maps/lqr_map.py` or `python main.py --lqr-map`
(`--lam1/--lam2/--u-max` seed the initial view).

Implementation notes / gotchas:
* Clicking a *non-recoverable* (red/blue) cell launches an animation whose
  trajectory diverges to inf/NaN.  The renderer maps state to pixels with
  `int()`, and `int(NaN)`/`int(inf)` raises -- on the unguarded pygame main
  loop that closed the window the instant such a frame drew (symptom: "window
  appears then disappears").  Fixed in `src/animation.py`: `_sim_worker` now
  truncates the trajectory at the fall (`|theta| > 1.7`) and `nan_to_num`s the
  arrays, and `draw_segway.to_px` clamps non-finite/out-of-range coordinates.
  The body now visibly topples and freezes instead of crashing.
* `closed_loop_accel` wraps its math in `np.errstate(over/invalid/divide=
  "ignore")` and `simulate_grid` sanitizes/clips blown-up cells (`nan_to_num`
  + clip to +-1e4) so the vectorized integrator never spams overflow warnings
  when a cell diverges before it latches as "fallen".
* The map's vertical extent auto-frames to the basin (`estimate_dth_bound`:
  a quick 1-D sweep of recoverable `|theta_dot|` at `theta = 0`, clamped to
  [4, 16] rad/s).
* Edge artifact: very stiff weights (e.g. `lambda1 = 200, lambda2 = 0.5` ->
  `K ~ [1848, 20, 421, 29]`) drive closed-loop poles faster than the fixed RK4
  step `dt = 2.5e-3` can resolve, so the basin there collapses (mostly "falls")
  -- a numerical, not physical, limit shared with the other grid-sim scripts.
  Stay within roughly the animation's slider ranges (`lambda1 in [1, 200]`,
  `lambda2 in [0.1, 20]`) and prefer moderate gains for trustworthy basins.
