"""Синтез ЛКР и сертификация его области притяжения.

Этот модуль -- НЕ шестой слой. Здесь нет ни одного класса: только функции,
которые считают числа `K`, `P`, `c*` ОДИН раз, до прогона. Дальше эти числа
уезжают в `LinearFeedbackController` / `BangBangLQRController` как обычные
аргументы, и ядру (`rollout`) про существование этого файла знать не нужно.

Поэтому `scipy` импортируется ВНУТРИ `lqr()`, а не на уровне модуля:
`import wpend` и любой прогон остаются на чистом numpy (инвариант из
CLAUDE.md). Ставится как `uv sync --extra design`.

Что здесь есть
--------------
* `lqr(A, B, Q, R)` -- решение уравнения Риккати (§B3 из PROPOSALS.md).
* `closed_loop_u`, `vdot` -- скорость функции Ляпунова V = x^T P x вдоль
  НЕЛИНЕЙНОЙ замкнутой системы, с учётом насыщения мотора.
* `certified_level(...)` -- наибольшее c, при котором эллипсоид
  {x : x^T P x <= c} целиком лежит в {V̇ < 0}, то есть является доказанной
  (консервативной) областью притяжения.

Теория: Anderson & Moore, *Optimal Control: Linear Quadratic Methods*, гл. 2
(уравнение Риккати); Khalil, *Nonlinear Systems*, 3-е изд., гл. 4.1 и
пример 8.4 (оценка области притяжения уровнем функции Ляпунова).
"""

from __future__ import annotations

import numpy as np


def lqr(A, B, Q, R):
    """Вернуть (K, P) для задачи min ∫ (x^T Q x + u^T R u) dt при dx/dt = Ax + Bu.

    Стабилизирующее P -- симметричное положительно определённое решение
    непрерывного алгебраического уравнения Риккати

        A^T P + P A - P B R^-1 B^T P + Q = 0,                          (CARE)

    а закон управления -- u = -K x с K = R^-1 B^T P.

    Считает `scipy.linalg.solve_continuous_are` (упорядоченное разложение
    Шура). Ручной вариант через собственные векторы гамильтоновой матрицы
        H = [[A, -B R^-1 B^T], [-Q, -A^T]],   P = X2 X1^-1
    (он лежит в `src/lqr.py` со второго семестра) даёт тот же ответ на хорошо
    обусловленных задачах, но теряет точность, когда собственные значения H
    близки: собственные векторы почти вырожденной матрицы плохо обусловлены,
    а разложение Шура -- нет. Оракул один и тот же: подставить P обратно в
    CARE и посмотреть на невязку (`tests/test_lqr.py`).

    Источник: Anderson & Moore, гл. 2; Åström & Murray, *Feedback Systems*,
    гл. 7. Численный метод -- Laub, «A Schur method for solving algebraic
    Riccati equations», IEEE TAC 24 (1979), 913-921.
    """
    # Импорт внутри функции: ядру scipy не нужен, см. докстринг модуля.
    from scipy.linalg import solve_continuous_are

    A = np.asarray(A, dtype=float)
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))

    P = solve_continuous_are(A, B, Q, R)
    P = 0.5 * (P + P.T)                       # симметризация: снимаем шум O(eps)
    K = np.linalg.solve(R, B.T @ P)
    return K, P


def lqr_tilt(system, Q, R, coords=(0, 2)):
    """ЛКР для ОДНОЙ подсистемы; возвращает (K, P), вложенные в полное
    пространство состояний нулями.

    Задуман для колёсного маятника, где подсистема наклона (theta, dtheta)
    замкнута сама на себя: `system.linearize_tilt()` даёт её пару (A, B),
    здесь решается 2x2 Риккати, а результат раскладывается по индексам
    `coords` в матрицы полного размера. Дальше `K` уезжает в любой регулятор,
    принимающий обратную связь по состоянию, безо всяких оговорок: нули в
    столбцах колеса означают ровно то, что управление колесо не видит.

    `P` получается вырожденной (ранг 2 в 4-мерном пространстве), и множество
    `{x : x^T P x <= c}` -- не эллипсоид, а ЦИЛИНДР: по phi и dphi оно
    бесконечно. Это не дефект, а содержание: сертификат утверждает сходимость
    наклона при любом состоянии колеса. Из-за вырожденности `certified_level`
    для такого `P` обязана искать только в подпространстве -- ей передаётся
    тот же `coords`, иначе луч вдоль колеса даст `c* = 0`.

    Теория: Khalil, *Nonlinear Systems*, гл. 11 (каскадные и двухтемповые
    системы -- почему подсистему можно проектировать отдельно); Anderson &
    Moore, гл. 2.
    """
    A_t, B_t = system.linearize_tilt()
    K_t, P_t = lqr(A_t, B_t, Q, R)
    coords = tuple(int(i) for i in coords)
    n = system.n_state
    K = np.zeros((K_t.shape[0], n))
    K[:, coords] = K_t
    P = np.zeros((n, n))
    P[np.ix_(coords, coords)] = P_t
    return K, P


def care_residual(A, B, Q, R, P):
    """Невязка уравнения Риккати ||A^T P + P A - P B R^-1 B^T P + Q||_max.

    Оракул для `lqr`: величина, которая обязана быть нулём по определению
    решения, а не по совпадению с прошлым прогоном.
    """
    A = np.asarray(A, dtype=float)
    B = np.atleast_2d(np.asarray(B, dtype=float))
    Q = np.atleast_2d(np.asarray(Q, dtype=float))
    R = np.atleast_2d(np.asarray(R, dtype=float))
    P = np.asarray(P, dtype=float)
    G = B @ np.linalg.solve(R, B.T)
    return float(np.abs(A.T @ P + P @ A - P @ G @ P + Q).max())


def closed_loop_u(K, x, u_max=None):
    """u = -K x, при необходимости обрезанное по |u| <= u_max.

    Обрезка здесь -- не дублирование `System.clip_action`, а часть ВОПРОСА:
    сертификат «V̇ < 0» обязан считаться для того управления, которое реально
    попадёт в систему. Если внутри эллипсоида ЛКР просит больше, чем может
    мотор, то незасатурированный сертификат просто неверен.
    """
    x = np.asarray(x, dtype=float)
    u = -x @ np.atleast_2d(np.asarray(K, dtype=float)).T
    if u_max is not None:
        u = np.clip(u, -u_max, u_max)
    return u


def vdot(system, x, K, P, u_max=None):
    """V̇ = 2 x^T P f(t, x, -K x) вдоль НЕЛИНЕЙНОЙ замкнутой системы.

    V = x^T P x убывает вблизи нуля по построению P (там динамика близка к
    A - BK), но нелинейные слагаемые растут быстрее линейных, и на некотором
    удалении V̇ меняет знак. Множество {V̇ < 0} -- это и есть то, что
    ограничивает сертифицированную область.

    Работает и с одним состоянием (n,), и с пачкой (M, n) -- соглашение A8.
    """
    x = np.asarray(x, dtype=float)
    P = np.asarray(P, dtype=float)
    u = closed_loop_u(K, x, u_max)
    f = system.f(0.0, x, u)
    return 2.0 * np.einsum("...i,ij,...j->...", x, P, f)


def certified_level(system, K, P, *, u_max=None, n_dirs=20000, s_max=8.0,
                    ds=5e-3, seed=0, theta_max=None, i_theta=0, coords=None):
    """Наибольшее c, при котором {x^T P x <= c} \\ {0} лежит в {V̇ < 0}.

    Метод -- луч из начала координат по многим случайным направлениям: идём
    наружу с шагом `ds`, пока V̇ не станет неотрицательной, и записываем
    значение V в этой точке. Минимум по направлениям и есть c*.

    Почему это законно: множество {x^T P x <= c} -- эллипсоид, а V вдоль луча
    монотонно растёт как s^2 d^T P d, поэтому первая точка смены знака вдоль
    луча даёт наибольший уровень, до которого этот луч «чистый». Минимум по
    достаточно плотному набору направлений -- оценка сверху для c*, и она тем
    точнее, чем больше `n_dirs`. Это НЕ доказательство (направлений конечное
    число); строгая версия -- SOS-релаксация, её здесь нет.

    Параметры
    ---------
    u_max : float | None
        Предел момента. None -- незасатурированный ЛКР (оптимистичный ответ).
    theta_max : float | None
        Если задан, направление перестаёт учитываться, как только |x[i_theta]|
        выходит за этот угол: там корпус уже лежит, и знак V̇ бессмысленен.
    coords : кортеж индексов | None
        Искать только в этом подпространстве. Нужен для вырожденной P от
        `lqr_tilt`: там множество уровня -- цилиндр, и луч вдоль оси цилиндра
        даёт d^T P d = 0, то есть c* = 0 при любом уровне. Ограничив
        направления плоскостью наклона, получаем уровень именно того
        цилиндра. None (по умолчанию) -- всё пространство, как раньше.
    s_max : float
        Луч должен доставать дальше границы эллипсоида в ЛЮБОМ направлении,
        иначе почти касательные направления обрываются раньше границы и c*
        получается завышенным.

    Возвращает (c_star, worst_direction).

    Источник: Khalil, *Nonlinear Systems*, гл. 8.2 и пример 8.4 -- оценка
    области притяжения наибольшим множеством уровня функции Ляпунова,
    целиком лежащим в области отрицательности производной.
    """
    P = np.asarray(P, dtype=float)
    n = P.shape[0]
    rng = np.random.default_rng(seed)
    if coords is None:
        D = rng.standard_normal((n_dirs, n))
    else:
        # Направления строго в подпространстве: остальные координаты -- нули,
        # поэтому луч никогда не выходит из цилиндра вдоль его оси.
        D = np.zeros((n_dirs, n))
        D[:, tuple(int(i) for i in coords)] = rng.standard_normal(
            (n_dirs, len(tuple(coords))))
    D /= np.linalg.norm(D, axis=1, keepdims=True)
    dPd = np.einsum("ij,jk,ik->i", D, P, D)

    c_dir = np.full(n_dirs, np.inf)
    active = np.ones(n_dirs, dtype=bool)

    s = ds
    while s <= s_max and active.any():
        idx = np.flatnonzero(active)
        X = s * D[idx]
        vd = vdot(system, X, K, P, u_max)
        crossed = vd >= 0.0
        if theta_max is not None:
            left = np.abs(X[:, i_theta]) >= theta_max
        else:
            left = np.zeros_like(crossed)
        hit = idx[crossed & ~left]
        c_dir[hit] = s * s * dPd[hit]
        active[idx[crossed | left]] = False
        s += ds

    if not np.isfinite(c_dir).any():
        raise ValueError("certified_level: ни одно направление не пересекло "
                         "V̇ = 0 -- увеличь s_max")
    j = int(np.argmin(c_dir))
    return float(c_dir[j]), D[j].copy()
