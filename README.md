# wpend

Минимальная лаборатория динамики и управления. Пять слоёв, один прогон,
одна зависимость (numpy).

```
System      dx/dt = f(t, x, u)   + ограничение мотора u_bounds
Sensor      x, u₋₁ ──▶ y        что реально видно
Estimator   y  ──▶ x̂             что мы думаем о состоянии
Controller  x̂ ──▶ u             что мы хотим приложить
Integrator  (f, x, u, dt) ──▶ x'

rollout(...) ──▶ Trajectory(t, x, u)
```

Всё, что идёт после прогона — графики, анимация, метрики, обучение — читает
только `Trajectory`.

Датчику передаётся ПРОШЛОЕ управление: акселерометр меряет удельную силу, в
которую входит ускорение корпуса, то есть момент. Модель ошибки инерциального
датчика — `IMUSensor`, разбор с формулами и числами — `docs/imu_noise.md`.

## Быстрый старт

Окружение заморожено: `.python-version` задаёт интерпретатор, `uv.lock` —
версии пакетов. `uv sync` воспроизводит и то, и другое.

```bash
uv sync --extra dev            # CPython 3.12.13 + numpy/pytest/pygame из uv.lock
uv run pytest tests -q
uv run python examples/balance.py
```

## Окно исследователя

```bash
uv sync --extra viz                                  # или --extra dev, там pygame тоже есть
uv run python -m wpend.viz.explorer                  # сетка 21x21 считается заранее
uv run python -m wpend.viz.explorer --grid 41        # 1681 НУ за ~2 с
uv run python -m wpend.viz.explorer --no-precompute  # считать клетку по клику
uv run python -m wpend.viz.explorer --help           # u_max, горизонт, dt, границы карты
```

Слева карта начальных условий `(theta0, dtheta0)`, справа робот и графики
`theta(t)`, `u(t)`. Клик по клетке проигрывает её траекторию.

## Версии

`requires-python` остаётся `>=3.10`: библиотека совместима с 3.10+. Но
разработка и проверки идут на одной сборке — `.python-version` = `3.12.13`,
uv скачает именно её. Пакеты приколочены в `uv.lock` (он коммитится).

```bash
uv lock --upgrade              # осознанно обновить всё
uv lock --upgrade-package numpy
uv sync --extra dev            # привести .venv в соответствие с локом
```

## Документы

* `ARCHITECTURE.md` — контракт слоёв и правила, которые не обсуждаются.
* `PROPOSALS.md` — метка добавок: что добавлено сверх схемы и что предлагается
  добавить дальше, с ссылками на теорию.
* `docs/Key_Formulas.md` — вывод уравнений колёсного маятника.

## Что где

```
wpend/            ядро: пять слоёв + rollout
wpend/models/     конкретные системы: маятник, колёсный маятник
wpend/viz/        окно pygame и классификация исходов (читает только Trajectory)
tests/            оракулы: порядок схем, сохранение энергии и первого интеграла
examples/         запускаемые сценарии
src/, scripts/    код второго семестра (numpy), достался от master, не трогаем
```

Прошлая версия библиотеки (`dynlab`) живёт на ветке `refactor/architecture` и
сюда не переносится.
