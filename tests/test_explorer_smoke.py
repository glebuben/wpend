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


@pytest.mark.slow
def test_two_controllers_render_together(tmp_path):
    """Окно с регулятором сравнения: обе траектории есть и они РАЗНЫЕ.

    Проверяется не картинка, а то, что призрак действительно считается своим
    регулятором: если бы траектория сравнения бралась из основной, окно
    рисовало бы одно и то же дважды и никто бы не заметил.
    """
    import numpy as np

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from wpend.viz.explorer import build_parser, run, Explorer

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "2.0", "--stride", "20",
         "--main", "bang", "--compare", "lqr", "--frames", "2"]
    )
    app = Explorer(args)
    assert app.traj is not None and app.traj_cmp is not None
    assert not np.allclose(app.traj.x, app.traj_cmp.x)

    shot = tmp_path / "compare.png"
    run(app, max_frames=2, screenshot=str(shot))
    assert shot.exists() and shot.stat().st_size > 0


@pytest.mark.slow
def test_switching_controllers_recomputes_the_map():
    """Основной регулятор красит карту, значит смена основного обязана
    пересчитать сетку; смена сравнения -- только одну клетку."""
    import numpy as np

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from wpend.viz.explorer import build_parser, Explorer

    args = build_parser().parse_args(
        ["--grid", "9", "--horizon", "2.0", "--stride", "20", "--main", "lqr"]
    )
    app = Explorer(args)
    outcome_lqr = app.outcome.copy()
    assert app.traj_cmp is None

    app.set_main("bang")
    assert not np.array_equal(app.outcome, outcome_lqr)

    app.set_compare("bang-map")               # использует маску чистого ЛКР
    assert app.traj_cmp is not None
    app.set_compare(None)
    assert app.traj_cmp is None


@pytest.mark.slow
def test_picker_buttons_cover_every_controller():
    """Раскладка кнопок -- чистая функция, и она обязана предлагать ровно те
    же варианты, что и командная строка: иначе мышь и --main разойдутся."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import pygame as pg

    from wpend.viz.explorer import (CONTROLLER_KEYS, build_parser, Explorer,
                                    picker_layout)

    pg.init()
    font = pg.font.SysFont("consolas,dejavusansmono,monospace", 15)
    args = build_parser().parse_args(["--grid", "5", "--horizon", "1.0",
                                      "--stride", "20", "--no-precompute"])
    app = Explorer(args)
    layout = picker_layout(app, font)
    assert [k for r, row, k in [(a, b, c) for a, b, c in layout] if row == 0] \
        == CONTROLLER_KEYS
    assert [k for _, row, k in layout if row == 1] == CONTROLLER_KEYS + [None]
    assert all(x + w <= 1280 for (x, _, w, _), _, _ in layout)   # влезает в окно
    pg.quit()


@pytest.mark.slow
def test_unavailable_controller_is_refused_and_explained(tmp_path):
    """Без scipy эллипсоида нет. Окно обязано отказать в выборе И сказать,
    чего не хватает: кнопка, которая молча ничего не делает, -- баг."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
    from wpend.viz.explorer import build_parser, run, Explorer

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "1.0", "--stride", "20", "--frames", "2"]
    )
    app = Explorer(args)
    app.P = None                          # как будто scipy не установлен
    app.design_hint = "ellipsoid needs scipy:  uv sync --extra dev"

    assert not app.available("bang-ell")
    assert app.available("lqr") and app.available("bang-map")

    app.set_main("bang-ell")
    assert app.main_key == "lqr"           # выбор отклонён, а не применён молча
    app.set_compare("bang-ell")
    assert app.cmp_key is None

    shot = tmp_path / "disabled.png"       # и подсказка рисуется без исключений
    run(app, max_frames=2, screenshot=str(shot))
    assert shot.exists() and shot.stat().st_size > 0


@pytest.mark.slow
def test_estimator_row_changes_the_map_and_records_the_estimate():
    """Оцениватель красит карту наравне с регулятором.

    Не тавтология: проверяется, что смена слоя МЕНЯЕТ исход клеток (иначе ряд
    кнопок был бы декорацией) и что оценка попадает в Trajectory, откуда её и
    читает окно -- правило 6 при этом цело.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import numpy as np

    from wpend.viz.explorer import build_parser, Explorer
    from wpend.viz.grid import HELD

    def held(est):
        args = build_parser().parse_args(
            ["--grid", "9", "--horizon", "2.0", "--stride", "20",
             "--main", "lqr", "--estimator", est]
        )
        app = Explorer(args)
        return app, int((app.outcome == HELD).sum())

    ideal, n_ideal = held("ideal")
    noisy, n_noisy = held("acc")
    assert n_ideal != n_noisy, "ряд оценивателей ни на что не влияет"
    # идеальный: писать нечего, x_hat == x побитово
    assert ideal.batch.x_hat is None
    assert noisy.batch.x_hat is not None
    assert noisy.batch.x_hat.shape[:2] == noisy.batch.u.shape[:2]
    assert noisy.traj.x_hat is not None
    err = noisy.traj.x_hat[:, 0] - noisy.traj.x[:-1, 0]
    assert np.max(np.abs(err[np.isfinite(err)])) > 1e-6


@pytest.mark.slow
def test_estimator_cycles_and_hint_names_the_wheel_problem():
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from wpend.viz.explorer import build_parser, Explorer, ESTIMATOR_KEYS

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "1.0", "--stride", "20", "--main", "lqr"]
    )
    app = Explorer(args)
    assert app.est_key == "ideal" and app.design_hint == ""
    app.set_estimator("gyro")
    assert app.est_key == "gyro"
    # K полного ЛКР спрашивает колесо, которого ИДУ не видит -- окно обязано
    # сказать об этом вслух, а не молча покрасить карту
    assert "wheel" in app.design_hint
    for key in ESTIMATOR_KEYS:
        app.set_estimator(key)
        assert app.est_key == key


@pytest.mark.slow
def test_second_estimator_runs_only_for_the_selected_cell():
    """Второй оцениватель -- ровно как второй регулятор: один прогон
    выбранной клетки, карта остаётся за основным.

    Проверяется и то, ради чего сравнение существует: при одном и том же
    регуляторе и одном начальном условии ИСТИННЫЕ траектории расходятся --
    значит расхождение принадлежит слою оценки, и больше нечему.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import numpy as np

    from wpend.viz.explorer import build_parser, Explorer

    args = build_parser().parse_args(
        ["--grid", "9", "--horizon", "2.0", "--stride", "20", "--main", "lqr",
         "--estimator", "ideal", "--estimator-compare", "acc"]
    )
    app = Explorer(args)
    assert app.traj_est_cmp is not None
    assert app.traj.x.shape == app.traj_est_cmp.x.shape
    assert not np.allclose(app.traj.x[:, 0], app.traj_est_cmp.x[:, 0])
    # у идеального основного оценка не пишется, у сравнения -- пишется
    assert app.traj.x_hat is None and app.traj_est_cmp.x_hat is not None

    app.set_estimator_compare("acc")          # повторный выбор выключает
    assert app.est_cmp_key is None and app.traj_est_cmp is None


@pytest.mark.slow
def test_encoder_estimator_is_offered_and_closes_the_wheel_gap():
    """Вариант IMU+encoder обязан присутствовать и обязан давать ограниченную
    ошибку колеса там, где счисление пути расходится."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import numpy as np

    from wpend.viz.explorer import build_parser, Explorer, ESTIMATOR_KEYS

    assert "enc" in ESTIMATOR_KEYS
    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "3.0", "--stride", "20", "--main", "lqr",
         "--estimator", "enc"]
    )
    app = Explorer(args)
    assert app.traj.x_hat is not None
    err = np.abs(app.traj.x_hat[:, 1] - app.traj.x[:-1, 1])   # phi
    assert np.max(err[np.isfinite(err)]) < 1.0                # не расходится


@pytest.mark.slow
def test_noise_sliders_drive_the_sensor_and_the_analytic_mark():
    """Слайдеры -- не декорация: значение уходит в датчик, а метка оптимума
    двигается по формуле, а не по подгонке.

    Проверяется закон масштабирования: tau* ~ sigma_a^(2/3) при неизменном
    смещении.  Если метка начнёт считаться иначе, тест это поймает.
    """
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import numpy as np

    from wpend.viz.explorer import build_parser, Explorer

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "1.0", "--stride", "20",
         "--estimator", "comp", "--sigma-g", "0", "--sigma-a", "1e-3", "--b0-g", "1e-2"]
    )
    app = Explorer(args)
    base = app.tau_star_model
    assert base == pytest.approx((1e-6 / (4 * 9.8 ** 2 * 1e-4)) ** (1 / 3), rel=1e-6)

    app.preview_noise("sigma_a", 8e-3)
    assert app.noise["sigma_a"] == pytest.approx(8e-3)
    assert app.tau_star_model / base == pytest.approx(8 ** (2 / 3), rel=1e-6)
    assert app.stale is True                       # карта помечена устаревшей

    # значение действительно доезжает до датчика
    assert app._sensor().sigma_w[2] == pytest.approx(app.noise["sigma_g"])
    assert app._sensor().sigma_w[0] == pytest.approx(8e-3)


@pytest.mark.slow
def test_measured_optimum_differs_from_the_analytic_one():
    """Главный смысл двух меток: в замкнутом контуре оптимум сдвинут, потому
    что ошибка акселерометра не белая -- там кажущаяся вертикаль."""
    os.environ["SDL_VIDEODRIVER"] = "dummy"
    from wpend.viz.explorer import build_parser, Explorer

    args = build_parser().parse_args(
        ["--grid", "7", "--horizon", "2.0", "--stride", "20", "--estimator", "comp"]
    )
    app = Explorer(args)
    app.measure_tau_star(n_points=5)
    assert app.tau_star_measured is not None
    assert app.tau_star_measured > 3 * app.tau_star_model
    app.set_estimator("ideal")
    app.measure_tau_star()
    assert app.tau_star_measured is None           # мерить нечего
