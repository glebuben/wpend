# wpend

Минимальная лаборатория динамики и управления. Пять слоёв, один прогон,
одна зависимость (numpy).

```
System      dx/dt = f(t, x, u)   + ограничение мотора u_bounds
Sensor      x  ──▶ y             что реально видно
Estimator   y  ──▶ x̂             что мы думаем о состоянии
Controller  x̂ ──▶ u             что мы хотим приложить
Integrator  (f, x, u, dt) ──▶ x'

rollout(...) ──▶ Trajectory(t, x, u)
```

Всё, что идёт после прогона — графики, анимация, метрики, обучение — читает
только `Trajectory`.

## Быстрый старт

```bash
pip install -e ".[dev]"
python -m pytest tests -q
python examples/balance.py
```

## Окно исследователя

```bash
pip install "wpend[viz]"
python -m wpend.viz.explorer                  # сетка 21x21 считается заранее
python -m wpend.viz.explorer --grid 41        # 1681 НУ за ~2 с
python -m wpend.viz.explorer --no-precompute  # считать клетку по клику
python -m wpend.viz.explorer --help           # u_max, горизонт, dt, границы карты
```

Слева карта начальных условий `(theta0, dtheta0)`, справа робот и графики
`theta(t)`, `u(t)`. Клик по клетке проигрывает её траекторию.

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
