"""
Hessian-norm heatmaps -- a curvature / nonlinearity indicator for the plant.

Idea
----
The plant is a vector field  f : R^4 -> R^4,  x = (theta, psi, dtheta, dpsi),
    f(x, u) = (dtheta, dpsi, ddtheta(x,u), ddpsi(x,u)).
Its *Jacobian* J(x) = df/dx is the local linearization; what is thrown away
by linearizing is the *curvature*, captured by the second derivative -- a
rank-3 tensor

    H[i, j, k](x) = d^2 f_i / (d x_j d x_k).

|H(x)| (some tensor norm) is therefore a pointwise measure of how nonlinear
the dynamics are at x: it is exactly zero wherever f is affine and grows where
the linearization degrades.  Mapping |H| over a 2-D slice of state space gives
a "nonlinearity heatmap".

For the *open-loop* field at fixed torque u (default u = 0, the choice made for
this study) f depends only on theta and dtheta -- psi is cyclic and dpsi enters
linearly (f_1 = dpsi) -- so the honest slice is (theta, dtheta) and everything
below is plotted there.  (The code keeps the full 4-D Hessian; the psi/dpsi
rows and columns simply come out ~0, which is itself a useful sanity check.)

Tensor norms
------------
There is no single "the" norm of a 3-tensor, so several are offered and shown
side by side.  Let T = H(x) (shape 4x4x4) and let T_(m) be its mode-m
unfolding (the 4x16 matrix whose rows are indexed by axis m and whose columns
flatten the other two axes; column order does not affect singular values):

    'frobenius' : ||T||_F   = sqrt( sum_ijk T_ijk^2 )                 (entrywise 2-norm)
    'spectral'  : max_m  sigma_max( T_(m) )                           (largest mode-wise gain)
    'nuclear'   : sum_m  ||T_(m)||_*  = sum_m sum_r sigma_r(T_(m))    (sum of nuclear norms, "SNN")
    'maxabs'    : max_ijk |T_ijk|                                     (largest single curvature)

The true tensor spectral / nuclear norms are NP-hard; the unfolding-based
quantities above are the standard tractable surrogates and are what we plot.

This module is pure numpy (+ matplotlib only for the figure builder), vectorized
over a grid, and is shared by the static figure script (hessian_norms.py) and
the interactive window (src/hessian_map.py).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .system import SegwayParams


# ---------------------------------------------------------------------------
# Vectorized open-loop field  f(x, u)
# ---------------------------------------------------------------------------
def f_batch(p: SegwayParams, X: np.ndarray, u: float = 0.0) -> np.ndarray:
    """Vectorized RHS of the plant ODE, x' = f(x, u).

    X has shape (..., 4) with the last axis = (theta, psi, dtheta, dpsi).
    Returns an array of the same shape.  This reproduces SegwayDynamics.rhs
    (closed-form 2x2 inverse instead of np.linalg.solve so it batches).
    """
    X = np.asarray(X, float)
    theta = X[..., 0]
    dth = X[..., 2]
    dps = X[..., 3]

    cos = np.cos(theta)
    sin = np.sin(theta)

    a = p.alpha + 2.0 * p.beta * cos + p.gamma   # M_tilde[0,0]
    b = p.gamma + p.beta * cos                   # M_tilde[0,1] = M_tilde[1,0]
    c = p.gamma                                  # M_tilde[1,1]
    det = a * c - b * b                          # > 0

    r1 = p.beta * sin * dth * dth + p.D * sin    # forcing on the theta equation
    r2 = p.beta * sin * dth * dth - u            # forcing on the psi  equation

    ddth = (c * r1 - b * r2) / det
    ddps = (-b * r1 + a * r2) / det

    out = np.empty_like(X)
    out[..., 0] = dth
    out[..., 1] = dps
    out[..., 2] = ddth
    out[..., 3] = ddps
    return out


# ---------------------------------------------------------------------------
# Finite-difference Hessian tensor  H[..., i, j, k] = d^2 f_i / d x_j d x_k
# ---------------------------------------------------------------------------
# The 10 unique (j, k) index pairs of a symmetric 4x4 second-derivative block.
_PAIRS = [(j, k) for j in range(4) for k in range(j, 4)]


def hessian_field(p: SegwayParams, X: np.ndarray, u: float = 0.0,
                  h: float = 1.0e-3) -> np.ndarray:
    """Second-derivative tensor of f over a batch of states, by finite diffs.

    X        : (..., 4) states.
    returns  : (..., 4, 4, 4) with H[..., i, j, k] = d^2 f_i / d x_j d x_k,
               symmetric in (j, k).

    Central differences: diagonal terms use the 3-point stencil, mixed terms
    the 4-point stencil (both O(h^2) accurate).
    """
    X = np.asarray(X, float)
    batch = X.shape[:-1]
    H = np.zeros(batch + (4, 4, 4), float)

    e = np.eye(4)

    def f(shift):
        return f_batch(p, X + shift, u)

    f0 = f_batch(p, X, u)                              # (..., 4)

    for (j, k) in _PAIRS:
        ej = h * e[j]
        ek = h * e[k]
        if j == k:
            d = (f(ej) - 2.0 * f0 + f(-ej)) / (h * h)
        else:
            d = (f(ej + ek) - f(ej - ek) - f(-ej + ek) + f(-ej - ek)) / (4.0 * h * h)
        # d has shape (..., 4) = derivative of every output f_i
        H[..., :, j, k] = d
        if j != k:
            H[..., :, k, j] = d
    return H


# ---------------------------------------------------------------------------
# Tensor norms (vectorized over the leading batch axes)
# ---------------------------------------------------------------------------
def _unfoldings_svals(T: np.ndarray) -> np.ndarray:
    """Singular values of the three mode-m unfoldings of T.

    T : (..., 4, 4, 4).  Returns (..., 3, 4): for each batch point, the
    singular values (4 of them) of each of the 3 mode-m unfoldings.
    """
    # Mode-m unfolding = move axis m to the front of the (i,j,k) block, then
    # flatten the remaining two into the columns.  Column order is irrelevant
    # for singular values, so a plain reshape after the swap is fine.
    A0 = T.reshape(T.shape[:-3] + (4, 16))                       # rows = i
    A1 = np.swapaxes(T, -3, -2).reshape(T.shape[:-3] + (4, 16))  # rows = j
    A2 = np.swapaxes(T, -3, -1).reshape(T.shape[:-3] + (4, 16))  # rows = k
    s0 = np.linalg.svd(A0, compute_uv=False)
    s1 = np.linalg.svd(A1, compute_uv=False)
    s2 = np.linalg.svd(A2, compute_uv=False)
    return np.stack([s0, s1, s2], axis=-2)                        # (..., 3, 4)


def norm_frobenius(T: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum(T * T, axis=(-3, -2, -1)))


def norm_maxabs(T: np.ndarray) -> np.ndarray:
    return np.max(np.abs(T), axis=(-3, -2, -1))


def norm_spectral(T: np.ndarray) -> np.ndarray:
    s = _unfoldings_svals(T)            # (..., 3, 4)
    return np.max(s[..., 0], axis=-1)   # max over modes of largest singular value


def norm_nuclear(T: np.ndarray) -> np.ndarray:
    s = _unfoldings_svals(T)            # (..., 3, 4)
    return np.sum(s, axis=(-2, -1))     # sum of nuclear norms over the 3 modes


# Registry: key -> (human label, function).  Order defines panel order.
NORMS = {
    "frobenius": (r"Frobenius  $\|H\|_F$", norm_frobenius),
    "spectral":  (r"Spectral  $\max_m\,\sigma_{\max}(H_{(m)})$", norm_spectral),
    "nuclear":   (r"Nuclear  $\sum_m\|H_{(m)}\|_*$", norm_nuclear),
    "maxabs":    (r"Max-abs  $\max_{ijk}|H_{ijk}|$", norm_maxabs),
}
DEFAULT_NORMS = ["frobenius", "spectral", "nuclear", "maxabs"]


# ---------------------------------------------------------------------------
# Norm fields over a (theta, dtheta) grid
# ---------------------------------------------------------------------------
def norm_grids(p: SegwayParams,
               theta_max: float = 1.45,
               dtheta_max: float = 12.0,
               N: int = 160,
               u: float = 0.0,
               h: float = 1.0e-3,
               psi: float = 0.0,
               dpsi: float = 0.0,
               norms: Optional[Sequence[str]] = None) -> dict:
    """Compute |H| over a grid in (theta, dtheta).

    Returns a dict with the axes ('theta', 'dtheta'), the Hessian tensor field
    'H' (N, N, 4, 4, 4), and one 2-D array per requested norm key.
    """
    keys = list(norms) if norms is not None else list(DEFAULT_NORMS)
    th = np.linspace(-theta_max, theta_max, N)
    dth = np.linspace(-dtheta_max, dtheta_max, N)
    TH, DTH = np.meshgrid(th, dth)                       # (N, N), row = dtheta

    X = np.empty((N, N, 4))
    X[..., 0] = TH
    X[..., 1] = psi
    X[..., 2] = DTH
    X[..., 3] = dpsi

    H = hessian_field(p, X, u=u, h=h)                    # (N, N, 4, 4, 4)

    out = dict(theta=th, dtheta=dth, TH=TH, DTH=DTH, H=H, u=u)
    for k in keys:
        if k not in NORMS:
            raise KeyError(f"unknown norm '{k}'; choices: {list(NORMS)}")
        out[k] = NORMS[k][1](H)
    out["keys"] = keys
    return out


# ---------------------------------------------------------------------------
# Matplotlib figure builder (shared by the static script and the pygame window)
# ---------------------------------------------------------------------------
def make_figure(grids: dict, *, log_scale: bool = False, fig=None,
                figsize=(11.0, 8.5), dpi: int = 110, cmap: str = "magma"):
    """Build a grid of heatmaps, one per norm, from a norm_grids() result.

    Returns the matplotlib Figure (created if `fig` is None).
    """
    import matplotlib
    if fig is None:
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=figsize, dpi=dpi)
    else:
        fig.clear()

    keys: List[str] = grids["keys"]
    n = len(keys)
    ncol = 2 if n > 1 else 1
    nrow = int(np.ceil(n / ncol))

    th = grids["theta"]
    dth = grids["dtheta"]
    extent = [th[0], th[-1], dth[0], dth[-1]]

    from matplotlib.colors import LogNorm
    for idx, key in enumerate(keys):
        ax = fig.add_subplot(nrow, ncol, idx + 1)
        Z = grids[key]
        label, _ = NORMS[key]
        if log_scale:
            zpos = Z[np.isfinite(Z) & (Z > 0)]
            vmin = float(zpos.min()) if zpos.size else 1e-12
            vmax = float(Z.max()) if np.isfinite(Z).any() else 1.0
            Zc = np.clip(Z, vmin, None)
            norm = LogNorm(vmin=vmin, vmax=max(vmax, vmin * 10))
            im = ax.imshow(Zc, origin="lower", extent=extent, aspect="auto",
                           cmap=cmap, norm=norm)
        else:
            im = ax.imshow(Z, origin="lower", extent=extent, aspect="auto",
                           cmap=cmap)
        ax.set_title(label, fontsize=11)
        ax.set_xlabel(r"$\theta$ [rad]", fontsize=9)
        ax.set_ylabel(r"$\dot\theta$ [rad/s]", fontsize=9)
        ax.tick_params(labelsize=8)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.ax.tick_params(labelsize=7)

    scale = "log" if log_scale else "linear"
    fig.suptitle(
        rf"Hessian norm of $f(x,\,u{{=}}{grids['u']:.0f})$ on $(\theta,\dot\theta)$"
        rf"   ({scale} color)",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return fig
