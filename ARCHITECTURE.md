# Архитектура ветки `arch/wpend`

Пять слоёв и один прогон. Больше в ядре ничего нет.

```
                    ┌─────────── истинное состояние x ───────────┐
                    │                                            │
   System.f(t,x,u) ─┤                                            │
                    │                                            ▼
                    │   Sensor      x ──▶ y      (что видно)
                    │   Estimator   y ──▶ x̂      (что думаем)
                    │   Controller  x̂ ──▶ u      (что хотим)
                    │   System.clip_action(u)    (что можем)
                    │                                            │
                    └── Integrator.step(f, t, x, u, dt) ─▶ x' ───┘

   rollout(...) ──▶ Trajectory(t, x, u)
```

## Файлы

| Файл | Что в нём |
|---|---|
| `wpend/system.py` | `System`: `f(t,x,u)`, `n_state`, `n_action`, `u_bounds`, `clip_action` |
| `wpend/sensor.py` | `Sensor`: `measure(t,x) -> y`. Реализации: `FullStateSensor`, `GaussianNoiseSensor` |
| `wpend/estimator.py` | `Estimator`: `estimate(t,y,u_prev) -> x̂`. Реализация: `PassthroughEstimator` |
| `wpend/controller.py` | `Controller`: `act(t,x̂) -> u`. Реализации: `Zero`, `Constant`, `LinearFeedback` |
| `wpend/integrator.py` | `Integrator`: `step(f,t,x,u,dt) -> x'`. Реализации: `Euler`, `RK4` |
| `wpend/rollout.py` | `Trajectory(t,x,u)` и `rollout(...)` |
| `wpend/models/` | Конкретные системы: `pendulum.py`, `wheeled_pendulum.py` |

## Правила, которые не обсуждаются

1. **`System.f` — чистая функция.** Никакой внутренней памяти: RK4 вызывает `f`
   четыре раза за шаг с промежуточными `x`, и любое состояние внутри `f` сломало
   бы схему.

2. **Память живёт только в `Estimator` и `Controller`.** Фильтр, интегральная
   часть ПИД, любая история — там и нигде больше. `reset()` вызывается
   rollout-ом один раз перед прогоном.

3. **Ограничение мотора — свойство системы, а не регулятора.** `u_bounds` живёт
   в `System`; `rollout` клиппует выход регулятора перед подстановкой в `f`.
   В `Trajectory.u` записывается **реально приложенное** управление.

4. **Мир развивается по истинному `x`, регулятор видит только `x̂`.** В `f`
   подставляется истинное состояние. Именно это разделение делает осмысленным
   вопрос «насколько плохая оценка ломает регулятор».

5. **Один `dt` на всё.** Регулятор пересчитывается на каждом шаге интегратора.
   Разделение на `dt_control`/`dt_sim` (zero-order hold) отложено осознанно —
   см. `PROPOSALS.md`.

6. **Всё, что идёт после прогона, читает только `Trajectory`.** Графики,
   анимация, метрики, обучение не должны знать, какой датчик и какой регулятор
   породили траекторию.

## Форма результата

```python
traj.t   # (N+1,)      моменты времени
traj.x   # (N+1, n_x)  состояния
traj.u   # (N,   n_u)  управления на интервалах [t_k, t_{k+1})
```

Управлений на одно меньше, чем состояний: `u_k` действует **между** `x_k` и
`x_{k+1}`, а у последнего состояния интервала впереди нет.

## Пример

```python
import numpy as np
from wpend import LinearFeedbackController, RK4Integrator, rollout
from wpend.models import WheeledPendulum

system = WheeledPendulum(u_max=15.0)
K = np.array([[95.122221, 1.0, 19.807871, 1.449624]])   # ЛКР, Q=diag(100,1,10,1), R=1

traj = rollout(
    system,
    LinearFeedbackController(K),
    RK4Integrator(),
    x0=[0.15, 0.0, 0.0, 0.0],
    dt=1e-3,
    n_steps=5000,
)
print(traj.x[-1])   # ≈ 0: корпус удержан
```

## Модели

Обе модели — в координатах, где `theta = 0` соответствует **верхней вертикали**.

* `Pendulum`, состояние `(theta, dtheta)`: `I·θ̈ = m g l sin θ − c θ̇ + u`.
  Оракул: `energy(x)` сохраняется при `u = 0, c = 0`.
* `WheeledPendulum`, состояние `(theta, phi, dtheta, dphi)`: уравнения из
  `docs/Key_Formulas.md` (ветка `master`), §1.3. Оракулы: `first_integral(x, u)`
  сохраняется при постоянном `u` (§3), `ddtheta` не зависит от `phi`, `dphi` (§1.4).

## Тесты

```
python -m pytest tests -q
```

Тесты — не формальность, а оракулы: численно измеренный порядок схем
(1 для Эйлера, 4 для RK4), сохранение энергии и первого интеграла, совпадение
аналитической линеаризации с конечными разностями.
