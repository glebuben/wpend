"""Дымовой тест окна: запускается headless (SDL dummy) и обязан нарисовать кадр.

Не проверяет, КАК выглядит окно -- только что приложение собирается, считает
сетку, выбирает клетку и рисует без исключений.
"""

import os

import pytest

pygame = pytest.importorskip("pygame")


@pytest.mark.slow
def test_explorer_renders_a_frame(tmp_path):
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from wpend.viz.explorer import build_parser, run, Explorer

    shot = tmp_path / "frame.png"
    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "1.0", "--stride", "20", "--frames", "2"]
    )
    app = Explorer(args)
    assert app.batch is not None and len(app.batch) == 49
    assert app.traj is not None
    run(app, max_frames=2, screenshot=str(shot))
    assert shot.exists() and shot.stat().st_size > 0


@pytest.mark.slow
def test_no_precompute_computes_on_click():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from wpend.viz.explorer import build_parser, Explorer

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "1.0", "--stride", "20", "--no-precompute"]
    )
    app = Explorer(args)
    assert app.batch is None
    assert (app.outcome >= 0).sum() == 1      # посчитана только стартовая клетка
    app.select(0, 0)
    assert (app.outcome >= 0).sum() == 2
    assert app.traj is not None
