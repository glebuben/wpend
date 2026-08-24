"""
Interactive launcher / hub  (the "main menu" of the project).

A single pygame window from which you can

* set the common run parameters (initial conditions, gains, torque limit,
  horizon) once, then
* open the interactive windows -- the segway **animation** and the clickable
  **recoverable-set map** -- and
* generate any of the static **figures** (the LQR/ROA, recoverable-set and
  Vdot scripts that drop ``fig_*.png`` into ``figures/``).

Design
------
Every action is launched as an *independent OS process*
(``python main.py --animate ...`` / ``python roa_lqr.py`` / ...), exactly the
way ``recoverable_map.py`` already spawns its animations.  Consequences:

* the menu never blocks -- you can keep clicking while figures render or
  several animation windows play at once;
* pygame only owns one display, so the animation has to be a separate
  process anyway; this design makes that the rule for *everything*;
* the parameters set here are passed on the command line to the runs that
  understand them (the animation and the ``main.py`` static plots).  The
  standalone figure scripts use their own built-in defaults.

Run
---
    python main.py --menu
    python -m src.menu
"""

from __future__ import annotations

import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np

try:
    import pygame
except ImportError as exc:                                           # pragma: no cover
    raise RuntimeError(
        "pygame is not installed.  Install it with:  pip install pygame"
    ) from exc

# Reuse the look-and-feel widgets from the animation module so the hub
# matches the rest of the project visually.
from .animation import Slider, Button, COL

# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent     # project root (has main.py)
MAIN = ROOT / "main.py"
FIG_DIR = ROOT / "figures"

WIDTH, HEIGHT = 1180, 820
LEFT_X        = 40            # parameter column
RIGHT_X       = 470          # action column
PANEL_X       = RIGHT_X - 20


def _open_path(path: Path) -> None:
    """Open a file or folder with the OS default handler (best effort)."""
    try:
        if platform.system() == "Windows":
            os.startfile(str(path))                     # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as exc:                                         # pragma: no cover
        print(f"[menu] could not open {path}: {exc}")


# ---------------------------------------------------------------------------
class MenuApp:
    """The launcher window."""

    # Figure scripts: (button label, script path from ROOT, representative fig).
    FIG_SCRIPTS = [
        ("Recoverable set (numeric)",  "scripts/figures/recoverable_set.py",          "fig_recoverable_set.png"),
        ("Recoverable set (symbolic)", "scripts/figures/recoverable_set_symbolic.py", "fig_recoverable_set_symbolic.png"),
        ("LQR ROA  (psi design)",      "scripts/figures/roa_lqr.py",                  "fig_lqr_roa.png"),
        ("LQR ROA  (2-D tilt)",        "scripts/figures/roa_lqr_tilt.py",             "fig_lqr_roa_tilt.png"),
        ("LQR ROA  (physical)",        "scripts/figures/roa_lqr_phys.py",             "fig_lqr_roa_phys.png"),
        ("LQR ROA  (phi slices)",      "scripts/figures/roa_lqr_slices.py",           "fig_lqr_roa_phi_slices.png"),
        ("Vdot portraits",             "scripts/figures/vdot_portraits.py",           "fig_vdot_2d_portraits.png"),
        ("LQR Vdot heatmap",           "scripts/figures/vdot_heatmap.py",             "fig_lqr_vdot_heatmap.png"),
        ("Hessian-norm heatmaps",      "scripts/figures/hessian_norms.py",            "fig_hessian_norms.png"),
    ]

    def __init__(self, *, fps: int = 60):
        pygame.init()
        pygame.display.set_caption("Wheeled-Pendulum -- main menu / launcher")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        self.clock  = pygame.time.Clock()
        self.fps    = fps

        self.font        = pygame.font.SysFont("dejavusans,arial", 14)
        self.small_font  = pygame.font.SysFont("dejavusans,arial", 11)
        self.title_font  = pygame.font.SysFont("dejavusans,arial", 20, bold=True)
        self.header_font = pygame.font.SysFont("dejavusans,arial", 13, bold=True)
        self.mono_font   = pygame.font.SysFont("dejavusansmono,consolas,courier", 12)

        # toggles
        self.controller_kind = "lyapunov"   # or "lqr"
        self.autostart = True               # animation runs immediately on open
        self.open_png  = True               # open the PNG when a figure is done

        # running children: list of dicts {name, proc, fig, t0, interactive}
        self.jobs: List[dict] = []
        self.status: List[tuple] = [
            ("Set the parameters, then launch a window or generate a figure.",
             COL["info"]),
        ]

        self._build_ui()

    # ----- UI ---------------------------------------------------------------
    def _build_ui(self) -> None:
        x, w = LEFT_X, 360
        y = 96
        step = 42

        def slider(label, lo, hi, val, fmt):
            nonlocal y
            s = Slider((x, y, w, 6), label, lo, hi, val, fmt)
            y += step
            return s

        self.s_th0  = slider("theta0  [rad]",    -1.55, 1.55, 0.30, "%+.3f")
        self.s_ps0  = slider("psi0  [rad]",      -3.14, 3.14, 0.00, "%+.3f")
        self.s_dth0 = slider("dtheta0  [rad/s]", -15.0, 15.0, 0.00, "%+.3f")
        self.s_dps0 = slider("dpsi0  [rad/s]",   -15.0, 15.0, 0.00, "%+.3f")
        y += 8
        self.s_kp   = slider("kp  (Lyapunov)",    50.0, 400.0, 100.0, "%.0f")
        self.s_ki   = slider("ki  (Lyapunov)",     0.05,  5.0,   1.0, "%.2f")
        self.s_kl   = slider("kL  (Lyapunov)",     0.05,  2.0,   0.5, "%.2f")
        self.s_lam1 = slider("lambda1  (LQR Q=l1*I)", 1.0, 200.0, 50.0, "%.1f")
        self.s_lam2 = slider("lambda2  (LQR R=l2*I)", 0.1,  20.0,  1.0, "%.2f")
        y += 8
        self.s_umax = slider("u_max [Nm]  (0 = unbounded)", 0.0, 60.0, 0.0, "%.1f")
        self.s_T    = slider("T  [s]",             2.0,  20.0,   8.0, "%.1f")

        self.sliders = [self.s_th0, self.s_ps0, self.s_dth0, self.s_dps0,
                        self.s_kp, self.s_ki, self.s_kl,
                        self.s_lam1, self.s_lam2, self.s_umax, self.s_T]

        # ---- action buttons (right column) --------------------------------
        bw, bh, gap = 290, 34, 10
        col1, col2 = RIGHT_X, RIGHT_X + bw + gap

        self.buttons: List[Button] = []

        # Interactive windows
        self.y_interactive = 96
        yb = self.y_interactive + 24
        self.buttons += [
            Button((col1, yb, bw, bh), "Open animation", self._open_animation),
            Button((col2, yb, bw, bh), "Open recoverable map", self._open_map),
        ]
        yb2 = yb + bh + gap
        self.buttons += [
            Button((col1, yb2, bw, bh), "Open LQR recoverable map",
                   self._open_lqr_map),
            Button((col2, yb2, bw, bh), "Open LQR Vdot heatmap",
                   self._open_vdot_heatmap),
        ]
        yb3 = yb2 + bh + gap
        self.buttons += [
            Button((col1, yb3, bw, bh), "Open Hessian map",
                   self._open_hessian_map),
        ]

        # Figures
        self.y_figures = yb3 + bh + 26
        yf = self.y_figures + 24
        # "Static plots" first (it honours the parameters), then the scripts.
        self.buttons.append(
            Button((col1, yf, bw, bh), "Static plots (states/phase/V)",
                   self._static_plots))
        self.buttons.append(
            Button((col2, yf, bw, bh), "Open figures folder",
                   self._open_fig_folder, style="save"))
        yf += bh + gap
        for i, (label, script, fig) in enumerate(self.FIG_SCRIPTS):
            cx = col1 if i % 2 == 0 else col2
            self.buttons.append(
                Button((cx, yf, bw, bh), label,
                       (lambda s=script, f=fig, n=label: self._run_figure(n, s, f))))
            if i % 2 == 1:
                yf += bh + gap
        if len(self.FIG_SCRIPTS) % 2 == 1:
            yf += bh + gap

        # Options / toggles
        self.y_options = yf + 14
        yo = self.y_options + 24
        self.b_ctrl = Button((col1, yo, bw, bh), self._ctrl_label(),
                             self._toggle_ctrl)
        self.b_auto = Button((col2, yo, bw, bh), self._auto_label(),
                             self._toggle_auto)
        yo += bh + gap
        self.b_png = Button((col1, yo, bw, bh), self._png_label(),
                            self._toggle_png)
        self.b_quit = Button((col2, yo, bw, bh), "Quit", self._quit)
        self.buttons += [self.b_ctrl, self.b_auto, self.b_png, self.b_quit]

        # Status panel sits below the option row.
        self.status_y = yo + bh + 18

    # ----- toggle labels ----------------------------------------------------
    def _ctrl_label(self) -> str:
        nice = "LQR" if self.controller_kind == "lqr" else "Lyapunov"
        return f"Controller: {nice}  (click to switch)"

    def _auto_label(self) -> str:
        return f"Animation autostart: {'ON' if self.autostart else 'off'}"

    def _png_label(self) -> str:
        return f"Open PNG when done: {'ON' if self.open_png else 'off'}"

    def _toggle_ctrl(self) -> None:
        self.controller_kind = "lqr" if self.controller_kind == "lyapunov" else "lyapunov"
        self.b_ctrl.label = self._ctrl_label()

    def _toggle_auto(self) -> None:
        self.autostart = not self.autostart
        self.b_auto.label = self._auto_label()

    def _toggle_png(self) -> None:
        self.open_png = not self.open_png
        self.b_png.label = self._png_label()

    def _quit(self) -> None:
        self._running = False

    # ----- parameters -------------------------------------------------------
    def _params(self) -> dict:
        return dict(
            theta0=self.s_th0.value, psi0=self.s_ps0.value,
            dtheta0=self.s_dth0.value, dpsi0=self.s_dps0.value,
            kp=self.s_kp.value, ki=self.s_ki.value, kl=self.s_kl.value,
            lam1=self.s_lam1.value, lam2=self.s_lam2.value,
            umax=self.s_umax.value, T=self.s_T.value,
        )

    # ----- launching --------------------------------------------------------
    def _spawn(self, cmd: List[str], name: str, *,
               fig: Optional[str] = None, interactive: bool = False) -> None:
        try:
            proc = subprocess.Popen(cmd, cwd=str(ROOT))
        except Exception as exc:                                     # pragma: no cover
            self._say(f"failed to launch {name}: {exc}", COL["warn"])
            return
        self.jobs.append(dict(name=name, proc=proc, fig=fig, t0=time.time(),
                              interactive=interactive))
        kind = "opened" if interactive else "generating"
        self._say(f"{kind}: {name}  (pid {proc.pid})", COL["info"])
        print(f"[menu] {kind}: {name}  ->  {' '.join(cmd)}")

    def _open_animation(self) -> None:
        p = self._params()
        cmd = [sys.executable, str(MAIN), "--animate",
               "--theta0", f"{p['theta0']:.6f}", "--psi0", f"{p['psi0']:.6f}",
               "--dtheta0", f"{p['dtheta0']:.6f}", "--dpsi0", f"{p['dpsi0']:.6f}",
               "--kp", f"{p['kp']:.6f}", "--ki", f"{p['ki']:.6f}",
               "--kl", f"{p['kl']:.6f}",
               "--u-max", f"{p['umax']:.6f}",
               "--controller", self.controller_kind,
               "--lam1", f"{p['lam1']:.6f}", "--lam2", f"{p['lam2']:.6f}",
               "--T", f"{p['T']:.6f}"]
        if self.autostart:
            cmd.append("--autostart")
        self._spawn(cmd, "animation", interactive=True)

    def _open_map(self) -> None:
        cmd = [sys.executable, str(MAIN), "--map"]
        self._spawn(cmd, "recoverable map", interactive=True)

    def _open_lqr_map(self) -> None:
        p = self._params()
        cmd = [sys.executable, str(MAIN), "--lqr-map",
               "--lam1", f"{p['lam1']:.6f}", "--lam2", f"{p['lam2']:.6f}",
               "--u-max", f"{p['umax']:.6f}"]
        self._spawn(cmd, "LQR recoverable map", interactive=True)

    def _open_vdot_heatmap(self) -> None:
        cmd = [sys.executable, str(ROOT / "scripts" / "figures" / "vdot_heatmap.py"), "--interactive"]
        self._spawn(cmd, "LQR Vdot heatmap", interactive=True)

    def _open_hessian_map(self) -> None:
        cmd = [sys.executable, str(MAIN), "--hessian-map"]
        self._spawn(cmd, "Hessian map", interactive=True)

    def _static_plots(self) -> None:
        p = self._params()
        cmd = [sys.executable, str(MAIN),
               "--theta0", f"{p['theta0']:.6f}", "--psi0", f"{p['psi0']:.6f}",
               "--dtheta0", f"{p['dtheta0']:.6f}", "--dpsi0", f"{p['dpsi0']:.6f}",
               "--kp", f"{p['kp']:.6f}", "--ki", f"{p['ki']:.6f}",
               "--kl", f"{p['kl']:.6f}", "--T", f"{p['T']:.6f}"]
        self._spawn(cmd, "static plots", fig="fig_states.png")

    def _run_figure(self, name: str, script: str, fig: str) -> None:
        cmd = [sys.executable, str(ROOT / script)]
        self._spawn(cmd, name, fig=fig)

    def _open_fig_folder(self) -> None:
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        _open_path(FIG_DIR)
        self._say(f"opened folder: {FIG_DIR}", COL["info"])

    # ----- status -----------------------------------------------------------
    def _say(self, text: str, color=None) -> None:
        self.status.append((text, color or COL["text"]))
        self.status = self.status[-9:]          # keep the last few lines

    def _poll_jobs(self) -> None:
        for job in list(self.jobs):
            rc = job["proc"].poll()
            if rc is None:
                continue
            self.jobs.remove(job)
            dt = time.time() - job["t0"]
            if job["interactive"]:
                self._say(f"closed: {job['name']}  (after {dt:.0f} s)", COL["sub_text"])
                continue
            if rc == 0:
                self._say(f"done: {job['name']}  ({dt:.1f} s)", COL["ok"])
                if self.open_png and job["fig"]:
                    fpath = FIG_DIR / job["fig"]
                    if fpath.exists():
                        _open_path(fpath)
            else:
                self._say(f"FAILED ({rc}): {job['name']} -- see console",
                          COL["warn"])

    # ----- main loop --------------------------------------------------------
    def run(self) -> None:
        self._running = True
        while self._running:
            for ev in pygame.event.get():
                if ev.type == pygame.QUIT:
                    self._running = False
                elif ev.type == pygame.KEYDOWN and ev.key == pygame.K_ESCAPE:
                    self._running = False
                for w in self.sliders + self.buttons:
                    w.handle(ev)

            self._poll_jobs()
            self._draw()
            pygame.display.flip()
            self.clock.tick(self.fps)

        pygame.quit()

    # ----- drawing ----------------------------------------------------------
    def _draw(self) -> None:
        scr = self.screen
        scr.fill(COL["bg"])

        # title
        t = self.title_font.render("Wheeled-Pendulum  --  control sandbox menu",
                                   True, COL["section"])
        scr.blit(t, (LEFT_X, 28))
        sub = self.small_font.render(
            "Parameters below feed the animation and the static plots; "
            "the figure scripts use their own defaults.",
            True, COL["sub_text"])
        scr.blit(sub, (LEFT_X, 58))

        # left column header
        ph = self.header_font.render("Run parameters", True, COL["section"])
        scr.blit(ph, (LEFT_X, 74))
        for s in self.sliders:
            s.draw(scr, self.font)

        # right column: section headers + buttons
        for y, label in [(self.y_interactive, "Interactive windows"),
                         (self.y_figures,     "Generate figures  (-> figures/)"),
                         (self.y_options,     "Options")]:
            h = self.header_font.render(label, True, COL["section"])
            scr.blit(h, (RIGHT_X, y))
        for b in self.buttons:
            b.draw(scr, self.font)

        # status panel
        pygame.draw.line(scr, COL["axis"],
                         (RIGHT_X, self.status_y - 8),
                         (WIDTH - 30, self.status_y - 8), 1)
        sh = self.header_font.render(
            f"Status   ({len(self.jobs)} running)", True, COL["section"])
        scr.blit(sh, (RIGHT_X, self.status_y))
        oy = self.status_y + 22
        for text, color in self.status:
            scr.blit(self.mono_font.render(text, True, color), (RIGHT_X, oy))
            oy += 16

        hint = self.small_font.render(
            "Click a button to launch an independent window/job.  "
            "Esc or Quit to close the menu (running windows keep going).",
            True, COL["sub_text"])
        scr.blit(hint, (LEFT_X, HEIGHT - 24))



# ---------------------------------------------------------------------------
def build_menu(*, fps: int = 60) -> "MenuApp":
    """Open the launcher window and run until quit."""
    app = MenuApp(fps=fps)
    app.run()
    return app


if __name__ == "__main__":
    build_menu()
