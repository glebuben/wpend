#!/usr/bin/env python3
"""
Main experiment runner for the segway Lyapunov controller.

Usage
-----
  python main.py                  # static plots (fig_states.png, fig_phase.png,
                                  #               fig_lyapunov.png)
  python main.py --menu           # open the interactive launcher / menu
  python main.py --animate        # open the interactive animation
  python main.py --theta0 0.4 --kp 150 --ki 0.5 --T 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# The interactive maps live in scripts/maps/; make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts" / "maps"))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from src.system     import SegwayParams, SegwayDynamics
from src.controller import LyapunovController, ControllerGains
from src.simulation import simulate


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--menu", action="store_true",
                   help="Open the interactive launcher/menu (hub for the "
                        "animation, the map and the figure generators)")
    p.add_argument("--animate", action="store_true",
                   help="Open the interactive animation instead of producing static plots")
    p.add_argument("--map", action="store_true",
                   help="Open the clickable recoverable-set map (clicking a cell "
                        "launches an independent animation window)")
    p.add_argument("--lqr-map", dest="lqr_map", action="store_true",
                   help="Open the clickable LQR recoverable-set map (the LQR's own "
                        "basin, parametrized by lambda1/lambda2; clicking a cell "
                        "launches an independent LQR animation)")
    p.add_argument("--hessian-map", dest="hessian_map", action="store_true",
                   help="Open the interactive Hessian-norm heatmaps (a "
                        "nonlinearity map: |d^2 f/dx^2| over the (theta, dtheta) "
                        "slice, one panel per tensor norm)")
    p.add_argument("--u-max", dest="u_max", type=float, default=0.0,
                   help="torque limit [Nm] for the animation; 0 = unbounded "
                        "control law (default)")
    p.add_argument("--autostart", action="store_true",
                   help="immediately simulate the given IC when the animation opens")
    p.add_argument("--controller", choices=("lyapunov", "lqr"),
                   default="lyapunov",
                   help="controller for the animation (default: lyapunov)")
    p.add_argument("--lam1", type=float, default=50.0,
                   help="LQR weight: Q = lam1 * I (state)")
    p.add_argument("--lam2", type=float, default=1.0,
                   help="LQR weight: R = lam2 * I (input)")
    p.add_argument("--theta0",  type=float, default=0.30, help="initial body tilt [rad]")
    p.add_argument("--psi0",    type=float, default=0.00, help="initial relative wheel angle [rad]")
    p.add_argument("--dtheta0", type=float, default=0.00, help="initial tilt rate [rad/s]")
    p.add_argument("--dpsi0",   type=float, default=0.00, help="initial wheel rate [rad/s]")
    p.add_argument("--kp", type=float, default=100.0)
    p.add_argument("--ki", type=float, default=1.0)
    p.add_argument("--kl", type=float, default=0.5)
    p.add_argument("--T",  type=float, default=8.0, help="simulation horizon [s]")
    p.add_argument("--out-dir", type=Path,
                   default=Path(__file__).parent / "figures",
                   help="directory in which to drop fig_*.png "
                        "(default: ./figures)")
    return p.parse_args()


# ---------------------------------------------------------------------------
def make_static_plots(sim: dict, ctrl: LyapunovController,
                      params: SegwayParams, out_dir: Path):
    """Generate the three classic figures (states, phase, Lyapunov)."""
    plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})

    t = sim["t"]
    th, ps   = sim["theta"], sim["psi"]
    dth, dps = sim["dtheta"], sim["dpsi"]
    phi      = sim["phi"]
    u, V     = sim["u"],  sim["V"]
    Vdot     = sim["Vdot"]
    KS       = ctrl.K_S()

    # ---- 1. states vs time ---------------------------------------------------
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    axes[0, 0].plot(t, np.degrees(th),  "C0", lw=2)
    axes[0, 0].set_title("Body tilt angle")
    axes[0, 0].set_ylabel(r"$\theta$ [deg]")
    axes[0, 1].plot(t, np.degrees(dth), "C1", lw=2)
    axes[0, 1].set_title("Body tilt rate")
    axes[0, 1].set_ylabel(r"$\dot\theta$ [deg/s]")
    axes[1, 0].plot(t, np.degrees(ps),  "C2", lw=2)
    axes[1, 0].set_title("Relative wheel angle")
    axes[1, 0].set_ylabel(r"$\psi = \varphi - \theta$ [deg]")
    axes[1, 1].plot(t, np.degrees(dps), "C3", lw=2)
    axes[1, 1].set_title("Relative wheel rate")
    axes[1, 1].set_ylabel(r"$\dot\psi$ [deg/s]")
    axes[2, 0].plot(t, u,                "C4", lw=2)
    axes[2, 0].set_title("Control input")
    axes[2, 0].set_ylabel(r"$u$ [N$\cdot$m]")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 1].plot(t, np.degrees(phi), "C5", lw=2)
    axes[2, 1].set_title("Wheel rotation (cyclic coord.)")
    axes[2, 1].set_ylabel(r"$\varphi$ [deg]")
    axes[2, 1].set_xlabel("time [s]")
    for ax in axes.flat:
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Closed-loop response under Lyapunov controller", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out1 = out_dir / "fig_states.png"
    fig.savefig(out1, dpi=140)
    print(f"  wrote {out1.name}")

    # ---- 2. phase portraits with time colour --------------------------------
    def time_colored(ax, x, y, t, cmap="viridis", lw=2):
        pts  = np.array([x, y]).T.reshape(-1, 1, 2)
        segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap, array=t, linewidth=lw)
        ax.add_collection(lc)
        return lc

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    lc1 = time_colored(axes[0], np.degrees(th), np.degrees(dth), t)
    axes[0].scatter([np.degrees(th[0])], [np.degrees(dth[0])], c="C3", s=80,
                    marker="o", zorder=5, label="start")
    axes[0].scatter([0], [0], c="C0", s=140, marker="*", zorder=5, label="equilibrium")
    axes[0].set_xlabel(r"$\theta$ [deg]")
    axes[0].set_ylabel(r"$\dot\theta$ [deg/s]")
    axes[0].set_title(r"Body phase portrait  $(\theta,\dot\theta)$")
    axes[0].set_xlim(min(np.degrees(th)) * 1.1 - 2, max(np.degrees(th)) * 1.1 + 2)
    axes[0].set_ylim(min(np.degrees(dth)) * 1.1 - 2, max(np.degrees(dth)) * 1.1 + 2)
    axes[0].legend(loc="best", framealpha=0.9)
    plt.colorbar(lc1, ax=axes[0], label="time [s]")

    lc2 = time_colored(axes[1], np.degrees(ps), np.degrees(dps), t)
    axes[1].scatter([np.degrees(ps[0])], [np.degrees(dps[0])], c="C3", s=80,
                    marker="o", zorder=5, label="start")
    axes[1].scatter([0], [0], c="C0", s=140, marker="*", zorder=5, label="equilibrium")
    axes[1].set_xlabel(r"$\psi$ [deg]")
    axes[1].set_ylabel(r"$\dot\psi$ [deg/s]")
    axes[1].set_title(r"Wheel phase portrait  $(\psi,\dot\psi)$")
    axes[1].set_xlim(min(np.degrees(ps)) * 1.1 - 2, max(np.degrees(ps)) * 1.1 + 2)
    axes[1].set_ylim(min(np.degrees(dps)) * 1.1 - 2, max(np.degrees(dps)) * 1.1 + 2)
    axes[1].legend(loc="best", framealpha=0.9)
    plt.colorbar(lc2, ax=axes[1], label="time [s]")

    fig.suptitle("Phase portraits — trajectory colored by time, converging to the origin",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out2 = out_dir / "fig_phase.png"
    fig.savefig(out2, dpi=140)
    print(f"  wrote {out2.name}")

    # ---- 3. Lyapunov function -----------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    axes[0].plot(t, V, "C0", lw=2, label=r"$V(x(t))$")
    axes[0].axhline(KS, color="C3", ls="--", lw=1.2,
                    label="K_S = %.2f  (attraction-region threshold)" % KS)
    axes[0].axhline(0, color="k", lw=0.4)
    axes[0].set_ylabel(r"$V(x(t))$")
    axes[0].set_title("Lyapunov function")
    axes[0].legend(loc="upper right", framealpha=0.9)
    dVnum = np.gradient(V, t)
    axes[1].plot(t, Vdot,  "C0",  lw=2,
                 label=r"$\dot V = -\dot\xi^2$  (analytic, eq. 24)")
    axes[1].plot(t, dVnum, "C3--", lw=1.2, label=r"$dV/dt$  (numerical)")
    axes[1].axhline(0, color="k", lw=0.4)
    axes[1].set_xlabel("time [s]")
    axes[1].set_ylabel(r"$\dot V$")
    axes[1].set_title(r"Time derivative of $V$  (semi-negative; vanishes only at equilibrium)")
    axes[1].legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    out3 = out_dir / "fig_lyapunov.png"
    fig.savefig(out3, dpi=140)
    print(f"  wrote {out3.name}")


# ---------------------------------------------------------------------------
def main():
    args = parse_args()

    if args.menu:
        # The interactive launcher (defer pygame import to here).
        from src.menu import build_menu
        print("Launching the interactive menu.  Close the window to exit.")
        build_menu()
        return

    if args.map:
        # The clickable recoverable-set map (spawns animations on click).
        from recoverable_map import main as map_main
        print("Launching recoverable-set map.  Click a cell to open an animation.")
        map_main()
        return

    if args.lqr_map:
        # The clickable LQR recoverable-set map (spawns LQR animations on click).
        from lqr_map import main as lqr_map_main
        print("Launching LQR recoverable-set map.  Set lambda1/lambda2 and click "
              "a cell to open an animation.")
        lqr_map_main(lam1=args.lam1, lam2=args.lam2, u_max=args.u_max)
        return

    if args.hessian_map:
        # The interactive Hessian-norm heatmaps (defer pygame import).
        from src.hessian_map import build_hessian_map
        print("Launching Hessian-norm heatmap window.  Close the window to exit.")
        build_hessian_map(u=0.0)
        return

    if args.animate:
        # Defer pygame import so the static path doesn't need it installed.
        from src.animation import build_animation
        print("Launching interactive pygame animation.  Close the window to exit.")
        build_animation(T_sim=args.T,
                        state0=(args.theta0, args.psi0,
                                args.dtheta0, args.dpsi0),
                        u_max=args.u_max,
                        autostart=args.autostart,
                        controller=args.controller,
                        lam1=args.lam1, lam2=args.lam2)
        return

    params = SegwayParams()
    gains  = ControllerGains(kp=args.kp, ki=args.ki, kl=args.kl)
    ctrl   = LyapunovController(params, gains)
    dyn    = SegwayDynamics(params)

    lhs, rhs_ = ctrl.gain_condition_value()
    print(f"Gain condition (27):  {lhs:.4f}  >  {rhs_:.4f}   OK")
    print(f"Critical angle θ_S = {np.degrees(ctrl.critical_angle()):.2f}°")
    print(f"K_S = {ctrl.K_S():.4f}")

    state0 = [args.theta0, args.psi0, args.dtheta0, args.dpsi0]
    V0 = ctrl.V(state0)
    print(f"V(x₀) = {V0:.4f}   (attraction region requires V < K_S = {ctrl.K_S():.4f})")
    if V0 >= ctrl.K_S():
        print("WARNING: initial state outside the guaranteed attraction region.")

    print(f"Simulating for T = {args.T} s ...")
    sim = simulate(dyn, ctrl, state0, args.T)
    print(f"Final V = {sim['V'][-1]:.3e}  ({100 * sim['V'][-1] / sim['V'][0]:.2e}% of initial)")
    print(f"Final θ = {np.degrees(sim['theta'][-1]):+.4f}°,"
          f"  ψ = {np.degrees(sim['psi'][-1]):+.4f}°")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Writing figures to {args.out_dir} ...")

    make_static_plots(sim, ctrl, params, args.out_dir)


if __name__ == "__main__":
    main()
