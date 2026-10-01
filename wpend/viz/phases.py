"""Дорожка фаз: в какой фазе регулятор считал каждое управление (ER-016).

Зачем. `BangBangLQRController` работает фазами: реле по сепаратрисе ->
(`bang-eps`) мягкий ЛКР в полосе |psi| < eps -> (A27) базовый ЛКР в
сертифицированном эллипсоиде. На графиках видно только u(t), и момент
передачи приходится угадывать по излому кривой. При сравнении двух
регуляторов или при шуме угадывание перестаёт работать: непонятно, упал
прогон в релейной фазе или уже после передачи.

Почему фаза восстанавливается, а не записывается. `Trajectory` хранит
t, x, u, x_hat и фазы не хранит, а `wpend/viz/` по правилу 6 читает только
`Trajectory`. Поле `phase` в `Trajectory` -- правка контракта прогона, то есть
«да» Глеба; карточка ER-016 выбрала путь без него (предложение записано в
`PROPOSALS.md` §B13). Здесь фаза получается ПРОГОНОМ САМОГО РЕГУЛЯТОРА по
записанной оценке: предикат региона и защёлка -- чистые функции состояния, а
код фазы отдаёт сам регулятор (`BangBangLQRController.phase`). Предикаты
здесь не повторяются: второй источник правды разошёлся бы с первым молча.

Фаза считается по `x_hat`, а не по `x`: регулятор решал по оценке, и при шуме
дорожка по истине показывала бы передачу не там, где она была. Если оценка не
записана (`record_estimate=False`), берётся `x[:-1]` -- для идеального датчика
это то же самое, для шумного -- приближение, и об этом сказано в `sources`.

Чего дорожка НЕ знает. Прогон сохраняется с прореживанием (`stride`), поэтому
повтор видит не каждый шаг: передача, случившаяся между сохранёнными кадрами,
на дорожке появится позже, а вход-и-выход между кадрами повтор пропустит
целиком. Молчать об этом нельзя, поэтому `track_summary` считает `u_mismatch` --
расхождение управления повтора с записанным. Ноль (точнее, шум машинной
арифметики) означает, что повтор воспроизвёл прогон шаг в шаг; заметное число
означает, что дорожке верить нельзя, и это видно числом, а не догадкой.
"""

from __future__ import annotations

import numpy as np

from ..controller import PHASE_BANG, PHASE_FINAL, PHASE_LQR

__all__ = ["PHASE_BANG", "PHASE_LQR", "PHASE_FINAL", "PHASE_LABEL",
           "phase_replay", "phase_track", "handover_times", "relay_switches",
           "track_summary", "track_line"]

#: Подписи фаз для легенды. Цвета -- в окне: палитра принадлежит рисунку.
PHASE_LABEL = {
    PHASE_BANG: "bang",
    PHASE_LQR: "LQR",
    PHASE_FINAL: "LQR final",
}


def phase_replay(traj, controller):
    """Прогнать регулятор по записанной траектории: (фазы, запрошенное u).

    Возвращает массив кодов фаз длины `traj.n_steps` (выровнен по `traj.u`,
    то есть на один отсчёт короче `traj.x`) и то управление, которое регулятор
    ПРОСИТ на этих шагах -- до обрезки `System.clip_action`.

    Регулятор здесь не считает новую траекторию: состояния берутся из готовой
    записи, двигается только его собственная память -- защёлки фаз. `reset()`
    зовётся и до, и после: до -- чтобы повтор начался с той же чистой памяти,
    что и прогон, после -- чтобы объект остался пригодным для следующего
    прогона (rollout и сам зовёт reset, но оставлять за собой защёлкнутый
    регулятор -- значит полагаться на чужую вежливость).

    У регулятора без фаз (обычный `LinearFeedbackController`) метода `phase`
    нет, и вся дорожка -- PHASE_LQR: он всегда в одной фазе, и это честный
    ответ, а не заглушка.
    """
    x_hat = traj.x_hat if traj.x_hat is not None else traj.x[:-1]
    n = int(traj.u.shape[0])
    if len(x_hat) < n:
        raise ValueError(f"состояний {len(x_hat)}, а управлений {n}: "
                         "траектория неполна")
    phase_of = getattr(controller, "phase", None)
    phases = np.full(n, PHASE_LQR, dtype=int)
    u_ask = np.empty_like(np.asarray(traj.u, dtype=float))
    controller.reset()
    # Разошедшийся прогон -- это inf и nan в состоянии. Регулятор их переживает
    # (сравнения с nan дают False, то есть «в регион не вошли»), а numpy на них
    # ругается; глушим так же, как это делает окно на прогоне.
    with np.errstate(all="ignore"):
        for k in range(n):
            u_ask[k] = np.asarray(controller.act(float(traj.t[k]), x_hat[k]),
                                  dtype=float)
            if phase_of is not None:
                phases[k] = int(np.asarray(phase_of()))
    controller.reset()
    return phases, u_ask


def phase_track(traj, controller) -> np.ndarray:
    """Коды фаз длины `traj.n_steps`. Короткая форма `phase_replay`."""
    return phase_replay(traj, controller)[0]


def handover_times(t, phases) -> dict:
    """Моменты первых передач: {PHASE_LQR: t1, PHASE_FINAL: t2}.

    Время отсчитывается от начала прогона и указывает на ПЕРВЫЙ шаг, который
    регулятор посчитал уже в новой фазе. Фазы, которой не было, в словаре нет:
    NaN пришлось бы проверять на каждом чтении.
    """
    t = np.asarray(t, dtype=float)
    phases = np.asarray(phases, dtype=int)
    out = {}
    for code in (PHASE_LQR, PHASE_FINAL):
        hit = np.flatnonzero(phases >= code)
        if hit.size:
            out[code] = float(t[hit[0]] - t[0])
    return out


def relay_switches(u, phases) -> int:
    """Сколько раз реле сменило знак до первой передачи.

    Считается по ЗАПИСАННОМУ u (что мир получил), а не по запрошенному: на
    релейной фазе это одно и то же (|u| = u_max уже допустимо), зато число
    остаётся числом про прогон. Нули знака не считаются сменой: u = 0 у реле
    не бывает (sign(0) уведён в ветвь торможения сознательно), но прореженная
    запись может поймать шов между фазами.
    """
    u = np.asarray(u, dtype=float)[:, 0]
    phases = np.asarray(phases, dtype=int)
    first = np.flatnonzero(phases > PHASE_BANG)
    stop = int(first[0]) if first.size else len(u)
    s = np.sign(u[:stop])
    s = s[np.isfinite(s) & (s != 0.0)]
    return int((np.diff(s) != 0).sum())


def track_summary(traj, controller, system=None) -> dict:
    """Дорожка фаз и числа к ней -- одним проходом.

    Ключи:
      `phases`       -- коды фаз, выровненные по `traj.u`;
      `t_handover`   -- время первой передачи ЛКР (NaN, если её не было);
      `t_final`      -- время передачи третьей фазе (NaN, если её нет вовсе);
      `n_relay`      -- число смен знака реле до первой передачи;
      `back_to_bang` -- фаза когда-нибудь УБЫВАЛА. Защёлка этого не допускает,
                        поэтому True означает не «регулятор вернулся в реле», а
                        «дорожка восстановлена неверно» -- проверка на себя;
      `u_mismatch`   -- максимум |u повтора - u прогона|. Управление повтора
                        обрезается `system.clip_action`, если система дана:
                        в записи лежит то, что мир получил, а регулятор просит
                        необрезанное. Без системы сравниваются как есть, и на
                        клетках, где ЛКР упирается в предел, число будет
                        большим законно.
    """
    phases, u_ask = phase_replay(traj, controller)
    t = handover_times(traj.t, phases)
    u_cmp = u_ask if system is None else np.asarray(system.clip_action(u_ask),
                                                    dtype=float)
    diff = np.abs(u_cmp - np.asarray(traj.u, dtype=float))
    diff = diff[np.isfinite(diff)]
    return {
        "phases": phases,
        "t_handover": t.get(PHASE_LQR, float("nan")),
        "t_final": t.get(PHASE_FINAL, float("nan")),
        "n_relay": relay_switches(traj.u, phases),
        "back_to_bang": bool((np.diff(phases) < 0).any()),
        "u_mismatch": float(diff.max()) if diff.size else float("nan"),
    }


def track_line(summary, tol: float = 1e-9) -> str:
    """Сводка одной строкой -- для шапки графика и для печати в примере.

    Одна формулировка на окно и на скрипт: разойдясь, они дали бы два разных
    числа под одним именем. Жалобы (`(!)`) печатаются рядом с числами, а не
    вместо них: дорожка, которой нельзя верить, всё равно нарисована, и
    сказать об этом надо там же, где её читают.
    """
    parts = []
    t = summary["t_handover"]
    parts.append("handover %.2f s" % t if np.isfinite(t) else "no handover")
    parts.append("relay switches %d" % summary["n_relay"])
    if np.isfinite(summary["t_final"]):
        parts.append("final %.2f s" % summary["t_final"])
    if summary["back_to_bang"]:
        parts.append("(!) phase went back to bang")
    off = summary["u_mismatch"]
    if np.isfinite(off) and off > tol:
        parts.append("(!) replay off by %.3g N*m" % off)
    return "   |   ".join(parts)
