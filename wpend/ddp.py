"""Оптимизация траектории методом iLQR (итеративный ЛКР).

Этот модуль, как и `wpend/lqr.py`, -- НЕ шестой слой. Классов здесь нет, есть
функции, которые ОДИН раз до прогона считают числа: номинальную траекторию
`x_bar`, номинальное управление `u_bar` и матрицы обратной связи `K_k` для
каждого шага. Потом эти числа уезжают аргументами в
`TrajectoryTrackingController`, и `rollout` про этот файл ничего не знает.
Ядру нужен только numpy.

Что решается
------------
Дискретная задача на горизонте N шагов:

    min  J = sum_{k=0}^{N-1} dt [ e_k^T Q e_k + u_k^T R u_k ] + e_N^T Q_f e_N,
    e_k = x_k - x_goal,
    x_{k+1} = F(x_k, u_k) = integrator.step(system.f, t_k, x_k, u_k, dt).

Множитель dt при бегущей цене нужен, чтобы сумма приближала интеграл
∫ (x^T Q x + u^T R u) dt, то есть ту же цену, которую минимизирует `lqr()`.
Иначе Q и R в двух модулях значили бы разное, и сравнивать `K` было бы нельзя.

F -- это ровно тот шаг, которым `rollout` двигает мир. Поэтому план, выполненный
без возмущений тем же интегратором и с тем же dt, воспроизводится до ошибки
округления (это проверяет `tests/test_ddp.py`).

Как решается (одна итерация)
----------------------------
1. Линеаризация вдоль текущей траектории: F_x, F_u на каждом шаге
   (центральные разности, см. `linearize_step` и `linearize_trajectory`).
2. Обратный проход -- уравнение Риккати, но вдоль траектории, а не в точке.
   Для функции ценности V_{k+1} ≈ V_x^T δx + ½ δx^T V_xx δx:

       Q_x  = l_x  + F_x^T V_x          Q_u  = l_u  + F_u^T V_x
       Q_xx = l_xx + F_x^T V_xx F_x     Q_uu = l_uu + F_u^T V_xx F_u
       Q_ux = F_u^T V_xx F_x

       δu = d + G δx,   d = -Q_uu^-1 Q_u,   G = -Q_uu^-1 Q_ux

       V_x  = Q_x  + G^T Q_uu d + G^T Q_u + Q_ux^T d
       V_xx = Q_xx + G^T Q_uu G + G^T Q_ux + Q_ux^T G

   Полный DDP добавил бы в Q_xx, Q_uu, Q_ux слагаемые V_x · F_xx (вторые
   производные динамики, тензоры). iLQR их отбрасывает: это метод Гаусса--
   Ньютона вместо Ньютона. Сходится медленнее, но нужны только F_x, F_u.
3. Прямой проход по НАСТОЯЩЕЙ F:
       u_k = u_bar_k + α d_k + G_k (x_k - x_bar_k),
   α уменьшается вдвое, пока цена не уменьшится (линейный поиск).

Проверка «на пальцах». Если F линейна, а цена квадратична, то квадратичная
модель точна, и одна итерация с α = 1 даёт оптимум сразу. Вторая итерация
ничего не улучшает -- это оракул в тестах.

Знак матрицы обратной связи
---------------------------
В литературе пишут δu = G δx. В проекте принято u = -K x
(`LinearFeedbackController`), поэтому наружу отдаётся K = -G:

    u = u_bar_k - K_k (x - x_bar_k).

Тогда при длинном горизонте и x_goal = 0 матрица K_0 сравнима с `K` из `lqr()`
без смены знака.

Предел момента: box-DDP (Tassa, Mansard, Todorov, ICRA 2014)
-------------------------------------------------------------
С аргументом `u_bounds = (u_min, u_max)` задача получает ограничение
u_min <= u_k <= u_max, и меняются ровно три места.

1. Обратный проход. Поправка k -- уже не безусловный минимум, а минимум
   квадратичной модели на отрезке допустимых поправок (ур. 11 статьи):

       k = argmin ½ δu^T Q_uu δu + δu^T Q_u,   u_min - ū <= δu <= u_max - ū.

   В общем случае это задача с коробкой (проекционный Ньютон, приложение
   статьи). У нас одно управление, m = 1, и выпуклая парабола на отрезке
   минимизируется обрезкой:

       k = clip(-Q_u / Q_uu, u_min - ū, u_max - ū).

   Если обрезка сработала, управление «зажато», и строка обратной связи
   равна нулю (статья, §III-C): на пределе малый сдвиг состояния не
   меняет управление -- мотор и так отдаёт всё.

       G = -Q_ux / Q_uu,  если k не обрезан;     G = 0,  если обрезан.

   Обновление V_x, V_xx -- общая форма из пункта 2. Короткая форма статьи
   (V_x = Q_x - K^T Q_uu k) верна только для безусловного минимума, а
   зажатый шаг им не является.
2. Прямой проход: u_k = clip(ū_k + α k_k + G_k (x_k - x̄_k), u_min, u_max).
   ū + α k допустимо при любом α из [0, 1] (отрезок между допустимыми
   точками), выйти за предел может только слагаемое G δx -- нелинейная
   траектория уходит от номинала.
3. Начальное приближение обрезается: план стартует допустимым.

Для m > 1 нужен проекционный Ньютон; в проекте управление одно, поэтому
`ilqr` с пределом при m > 1 честно отказывается.

Чего здесь нет
--------------
* Произвольной цены. Только матрицы Q, R, Q_f (решение Глеба 15.09).
* Учёта предела в терминальной цене. Q_f = P оценивает хвост за горизонтом
  ценой ЛКР БЕЗ предела -- это оптимистично.

Источники: Jacobson & Mayne, *Differential Dynamic Programming* (1970),
гл. 2; Li & Todorov, «Iterative linear quadratic regulator design for
nonlinear biological movement systems», ICINCO 2004; Tassa, Erez & Todorov,
«Synthesis and stabilization of complex behaviors through online trajectory
optimization», IROS 2012, §II (регуляризация и линейный поиск); Anderson &
Moore, *Optimal Control: Linear Quadratic Methods*, гл. 2 (дискретный Риккати).
"""

from __future__ import annotations

import numpy as np


def rollout_open_loop(system, integrator, x0, u_seq, dt, t0=0.0):
    """Прогнать F по заданной последовательности управлений, БЕЗ клиппинга.

    Возвращает x формы (N+1, n). Отдельно от `rollout`, потому что здесь нет ни
    датчика, ни регулятора, ни `clip_action`: это модель планировщика, а не мир.
    """
    u_seq = np.asarray(u_seq, dtype=float)
    n_steps = u_seq.shape[0]
    x = np.empty((n_steps + 1, system.n_state))
    x[0] = np.asarray(x0, dtype=float)
    for k in range(n_steps):
        x[k + 1] = integrator.step(system.f, t0 + k * dt, x[k], u_seq[k], dt)
    return x


def linearize_step(system, integrator, t, x, u, dt, h=1e-6):
    """(F_x, F_u) дискретного шага F(x, u) центральными разностями.

    Почему разности, а не аналитика: F -- это шаг RK4, а не f. Аналитическая
    производная шага RK4 -- громоздкая цепочка из четырёх якобианов f, и её
    пришлось бы переписывать для каждого интегратора. Разность по самому шагу
    верна для любой пары (system, integrator) и совпадает с тем, чем реально
    двигается мир. Ошибка центральной разности O(h^2) ≈ 1e-12 при h = 1e-6.

    Все 2(n+m) возмущённых точек идут в F одной пачкой (n+m, n) -- соглашение
    о пачках A8, -- поэтому шаг интегратора вызывается два раза, а не 2(n+m).
    """
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    n, m = x.size, u.size
    E = h * np.eye(n + m)                      # строка j -- возмущение j-й переменной
    X_plus = x + E[:, :n]
    U_plus = u + E[:, n:]
    X_minus = x - E[:, :n]
    U_minus = u - E[:, n:]
    F_plus = integrator.step(system.f, t, X_plus, U_plus, dt)     # (n+m, n)
    F_minus = integrator.step(system.f, t, X_minus, U_minus, dt)
    J = ((F_plus - F_minus) / (2.0 * h)).T                        # (n, n+m)
    return J[:, :n], J[:, n:]


def linearize_trajectory(system, integrator, x_bar, u_bar, dt, t0=0.0, h=1e-6):
    """(F_x, F_u) формы (N, n, n), (N, n, m) на всех шагах траектории сразу.

    То же, что `linearize_step` в цикле по k, но все N·(n+m) возмущённых точек
    идут в интегратор ОДНОЙ пачкой. Замер (N = 100, колёсный маятник): это
    была половина времени итерации iLQR, 100 пар вызовов RK4 из Python.

    Цена: время тоже идёт пачкой, t формы (N·(n+m),), и `system.f` получает
    массив t, а не число. Модели проекта t не читают вовсе, а формула вида
    np.sin(t) транслируется по строкам так же, как x[..., i]. Совпадение с
    `linearize_step` построчно проверяет `tests/test_ddp.py`.
    """
    x_bar = np.asarray(x_bar, dtype=float)
    u_bar = np.asarray(u_bar, dtype=float)
    n_steps, m = u_bar.shape
    n = x_bar.shape[1]
    p = n + m
    E = h * np.eye(p)
    # (N, p, n) и (N, p, m): на шаге k строка j -- возмущение j-й переменной
    X = x_bar[:-1, None, :] + np.zeros((1, p, 1))
    U = u_bar[:, None, :] + np.zeros((1, p, 1))
    T = np.repeat(t0 + dt * np.arange(n_steps), p)
    X_plus = (X + E[None, :, :n]).reshape(-1, n)
    X_minus = (X - E[None, :, :n]).reshape(-1, n)
    U_plus = (U + E[None, :, n:]).reshape(-1, m)
    U_minus = (U - E[None, :, n:]).reshape(-1, m)
    F_plus = integrator.step(system.f, T, X_plus, U_plus, dt).reshape(n_steps, p, n)
    F_minus = integrator.step(system.f, T, X_minus, U_minus, dt).reshape(n_steps, p, n)
    J = ((F_plus - F_minus) / (2.0 * h)).transpose(0, 2, 1)       # (N, n, p)
    return J[:, :, :n], J[:, :, n:]


def trajectory_cost(x, u, Q, R, Q_f, dt, x_goal=None):
    """J из докстринга модуля для готовых массивов x (N+1, n) и u (N, m)."""
    e = x if x_goal is None else x - x_goal
    running = np.einsum("ki,ij,kj->", e[:-1], Q, e[:-1]) \
        + np.einsum("ki,ij,kj->", u, R, u)
    return float(dt * running + e[-1] @ Q_f @ e[-1])


def ilqr(system, integrator, x0, dt, n_steps, Q, R, Q_f, *, x_goal=None,
         u_init=None, t0=0.0, max_iter=100, tol=1e-10, mu=1e-6, u_bounds=None):
    """Оптимальная траектория из x0 и обратная связь вдоль неё.

    Параметры
    ---------
    system, integrator
        Модель планировщика. Шаг F -- `integrator.step(system.f, ...)`, то есть
        тот же, что в `rollout`. Предел берётся из аргумента `u_bounds`, а не
        из `system.u_bounds`: планировщик с пределом и без -- разные опыты, и
        выбор должен быть виден в вызове.
    x0 : (n,)
    dt, n_steps
        Шаг и горизонт. Горизонт в секундах -- dt * n_steps.
    Q, R, Q_f : (n, n), (m, m), (n, n)
        Веса. Q_f -- терминальная цена; разумный выбор -- P из `lqr()`: тогда
        хвост за горизонтом оценён ценой бесконечного ЛКР.
    x_goal : (n,) | None
        Целевое состояние, None -- ноль. Удобно для phi: колесо можно вести в
        любую точку, не меняя модель (phi циклична, Key_Formulas §1.4).
    u_init : (N, m) | None
        Начальное приближение, None -- нули. Для MPC сюда пойдёт сдвинутый
        прошлый план.
    max_iter, tol
        Остановка, когда относительное улучшение цены меньше tol.
    mu
        Регуляризация: к Q_uu добавляется mu·I. Q_uu обязана быть
        положительно определённой, иначе шаг d идёт не к минимуму. При R > 0
        это так вблизи оптимума, но вдали от него Гаусс--Ньютон может дать
        плохо обусловленную матрицу. Если обратный проход ломается, mu
        увеличивается в 10 раз (Tassa et al. 2012, §II-F).
    u_bounds : (u_min, u_max) | None
        Предел управления, каждый формы (m,) или скаляр. None -- без предела,
        поведение побитово прежнее. Только для m = 1 (см. модуль, box-DDP).

    Возвращает
    ----------
    x_bar : (N+1, n)   номинальная траектория
    u_bar : (N, m)     номинальное управление
    K     : (N, m, n)  обратная связь: u = u_bar_k - K_k (x - x_bar_k)
    costs : list[float]  цена после каждой принятой итерации; costs[0] --
                         цена начального приближения
    """
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    Q_f = np.atleast_2d(np.asarray(Q_f, dtype=float))
    n, m = system.n_state, system.n_action
    goal = np.zeros(n) if x_goal is None else np.asarray(x_goal, dtype=float)

    bounds = None
    if u_bounds is not None:
        if m != 1:
            raise NotImplementedError("ilqr: предел реализован для одного управления "
                                      "(m = 1); для m > 1 нужен проекционный Ньютон, "
                                      "Tassa et al. 2014, приложение")
        bounds = (np.broadcast_to(np.asarray(u_bounds[0], dtype=float), (m,)),
                  np.broadcast_to(np.asarray(u_bounds[1], dtype=float), (m,)))

    u_bar = (np.zeros((n_steps, m)) if u_init is None
             else np.array(u_init, dtype=float).reshape(n_steps, m))
    if bounds is not None:
        u_bar = np.clip(u_bar, bounds[0], bounds[1])
    x_bar = rollout_open_loop(system, integrator, x0, u_bar, dt, t0)
    cost = trajectory_cost(x_bar, u_bar, Q, R, Q_f, dt, goal)
    costs = [cost]

    # Вторые производные цены постоянны: цена квадратична.
    l_xx = 2.0 * dt * Q
    l_uu = 2.0 * dt * R

    G = np.zeros((n_steps, m, n))
    for _ in range(max_iter):
        # --- 1. линеаризация вдоль текущей траектории -----------------------
        F_x, F_u = linearize_trajectory(system, integrator, x_bar, u_bar, dt, t0)

        # --- 2. обратный проход (с ростом mu при неудаче) --------------------
        while True:
            d, G, ok = _backward_pass(x_bar, u_bar, F_x, F_u, Q, R, Q_f,
                                      l_xx, l_uu, dt, goal, mu, bounds)
            if ok:
                break
            mu *= 10.0
            if mu > 1e10:
                raise RuntimeError("ilqr: Q_uu не удаётся сделать положительно "
                                   "определённой -- проверь R > 0")

        # --- 3. прямой проход с линейным поиском по α ------------------------
        accepted = False
        alpha = 1.0
        while alpha > 1e-8:
            x_new, u_new = _forward_pass(system, integrator, x_bar, u_bar,
                                         d, G, alpha, dt, t0, bounds)
            new_cost = trajectory_cost(x_new, u_new, Q, R, Q_f, dt, goal)
            if np.isfinite(new_cost) and new_cost < cost:
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            # Улучшить нельзя даже крошечным шагом: мы в (локальном) минимуме
            # с точностью квадратичной модели.
            break

        improvement = (cost - new_cost) / max(abs(cost), 1e-300)
        x_bar, u_bar, cost = x_new, u_new, new_cost
        costs.append(cost)
        if improvement < tol:
            break

    # Наружу -- в соглашении проекта u = u_bar - K (x - x_bar), см. модуль.
    return x_bar, u_bar, -G, costs


def _backward_pass(x_bar, u_bar, F_x, F_u, Q, R, Q_f, l_xx, l_uu, dt, goal, mu,
                   bounds=None):
    """Формулы из докстринга модуля, шаги k = N-1 ... 0.

    bounds -- (u_min, u_max) или None; с пределом k обрезается, а G на
    зажатых шагах обнуляется (box-DDP, см. модуль).

    Возвращает (d, G, ok); ok = False, если Q_uu + mu·I не положительно
    определена на каком-то шаге (проверка через Холецкого).
    """
    n_steps, m = u_bar.shape
    n = x_bar.shape[1]
    d = np.empty((n_steps, m))
    G = np.empty((n_steps, m, n))

    e_N = x_bar[-1] - goal
    V_x = 2.0 * Q_f @ e_N
    V_xx = 2.0 * Q_f

    for k in range(n_steps - 1, -1, -1):
        A, B = F_x[k], F_u[k]
        l_x = 2.0 * dt * Q @ (x_bar[k] - goal)
        l_u = 2.0 * dt * R @ u_bar[k]

        Q_x = l_x + A.T @ V_x
        Q_u = l_u + B.T @ V_x
        Q_xx = l_xx + A.T @ V_xx @ A
        Q_uu = l_uu + B.T @ V_xx @ B
        Q_ux = B.T @ V_xx @ A

        Q_uu_reg = Q_uu + mu * np.eye(m)
        try:
            # Холецкий -- одновременно проверка положительной определённости и
            # самый устойчивый способ решить систему с симметричной матрицей.
            L = np.linalg.cholesky(Q_uu_reg)
        except np.linalg.LinAlgError:
            return d, G, False
        d[k] = -_chol_solve(L, Q_u)
        G[k] = -_chol_solve(L, Q_ux)
        if bounds is not None:
            # m = 1: минимум параболы на отрезке -- обрезка безусловного
            # минимума. Зажатое управление не реагирует на δx, поэтому G = 0.
            clipped = np.clip(d[k], bounds[0] - u_bar[k], bounds[1] - u_bar[k])
            if clipped[0] != d[k][0]:
                G[k] = 0.0
            d[k] = clipped

        V_x = Q_x + G[k].T @ Q_uu @ d[k] + G[k].T @ Q_u + Q_ux.T @ d[k]
        V_xx = Q_xx + G[k].T @ Q_uu @ G[k] + G[k].T @ Q_ux + Q_ux.T @ G[k]
        V_xx = 0.5 * (V_xx + V_xx.T)     # симметризация: снимаем шум округления

    return d, G, True


def _chol_solve(L, b):
    """Решить (L L^T) z = b; b -- вектор (m,) или матрица (m, n)."""
    y = np.linalg.solve(L, b)
    return np.linalg.solve(L.T, y)


def _forward_pass(system, integrator, x_bar, u_bar, d, G, alpha, dt, t0, bounds=None):
    n_steps = u_bar.shape[0]
    x = np.empty_like(x_bar)
    u = np.empty_like(u_bar)
    x[0] = x_bar[0]
    for k in range(n_steps):
        u[k] = u_bar[k] + alpha * d[k] + G[k] @ (x[k] - x_bar[k])
        if bounds is not None:
            u[k] = np.clip(u[k], bounds[0], bounds[1])
        x[k + 1] = integrator.step(system.f, t0 + k * dt, x[k], u[k], dt)
    return x, u
