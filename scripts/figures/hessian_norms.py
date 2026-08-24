#!/usr/bin/env python3
"""
Static Hessian-norm heatmaps (nonlinearity map of the open-loop plant).

Computes the second-derivative tensor H(x) = d^2 f / dx^2 of the open-loop
field f(x, u) over a (theta, dtheta) grid and draws one heatmap per tensor
norm (Frobenius, spectral, nuclear, max-abs).  See src/hessian.py for the math.

    python hessian_norms.py
    python hessian_norms.py --u 0 --N 200 --theta-max 1.45 --dtheta-max 12 --linear

Writes figures/fig_hessian_norms.png.  The interactive counterpart is
    python main.py --hessian-map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

# scripts/ live two levels below the project root; make `src` importable
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.system import SegwayParams
from src.hessian import norm_grids, make_figure, DEFAULT_NORMS


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--u", type=float, default=0.0,
                   help="frozen torque at which the field is evaluated [Nm]")
    p.add_argument("--N", type=int, default=200, help="grid resolution (N x N)")
    p.add_argument("--theta-max", dest="theta_max", type=float, default=1.45)
    p.add_argument("--dtheta-max", dest="dtheta_max", type=float, default=12.0)
    p.add_argument("--h", type=float, default=1.0e-3,
                   help="finite-difference step for the Hessian")
    p.add_argument("--linear", action="store_true",
                   help="use a linear color scale (default: logarithmic)")
    p.add_argument("--norms", nargs="+", default=list(DEFAULT_NORMS),
                   help=f"which norms to plot (subset of {list(DEFAULT_NORMS)})")
    p.add_argument("--out-dir", type=Path,
                   default=ROOT / "figures")
    p.add_argument("--open", action="store_true",
                   help="open the PNG when done (best-effort, platform-dependent)")
    return p.parse_args()


def main():
    a = parse_args()
    p = SegwayParams()
    print(f"Computing {a.N}x{a.N} Hessian-norm grid at u = {a.u:.1f} Nm ...")
    grids = norm_grids(p, theta_max=a.theta_max, dtheta_max=a.dtheta_max,
                       N=a.N, u=a.u, h=a.h, norms=a.norms)
    for k in grids["keys"]:
        Z = grids[k]
        print(f"  {k:10s}  min={Z.min():.4g}  max={Z.max():.4g}")

    fig = make_figure(grids, log_scale=not a.linear)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    out = a.out_dir / "fig_hessian_norms.png"
    fig.savefig(out, dpi=150)
    print(f"Wrote {out}")

    if a.open:
        import sys, subprocess, os
        try:
            if sys.platform.startswith("win"):
                os.startfile(out)                                   # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(out)])
            else:
                subprocess.Popen(["xdg-open", str(out)])
        except Exception:
            pass


if __name__ == "__main__":
    main()
