"""Рыночные вероятности и смешивание модели с рынком."""
from __future__ import annotations

import math
from typing import Sequence

from scipy.optimize import brentq


def devig(odds: Sequence[float], method: str = "power") -> list[float]:
    """Вероятности без маржи по полному набору исходов рынка.

    power — устойчивее к перекосу маржи в сторону андердогов, чем multiplicative.
    """
    inv = [1.0 / o for o in odds]
    if method == "multiplicative":
        s = sum(inv)
        return [p / s for p in inv]
    if method == "power":
        if abs(sum(inv) - 1.0) < 1e-12:
            return inv
        k = brentq(lambda k: sum(p ** k for p in inv) - 1.0, 0.5, 3.0)
        return [p ** k for p in inv]
    raise ValueError(method)


def implied_single(odds: float, overround: float) -> float:
    """Оценка вероятности без маржи, когда известна только одна сторона рынка."""
    return min(1.0 / (odds * overround), 0.999)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def blend(p_model: float, p_market: float, w_model: float) -> float:
    """Логарифмический пулинг для бинарного события (= взвешивание в логит-шкале)."""
    z = w_model * _logit(p_model) + (1 - w_model) * _logit(p_market)
    return 1.0 / (1.0 + math.exp(-z))


def blend_with_push(p: tuple[float, float, float], q_win: float, w_model: float) -> tuple[float, float, float]:
    """Смешивание для ставки с возможным возвратом: смешиваем P(win | не возврат)."""
    win, push, loss = p
    if push <= 1e-9:
        b = blend(win, q_win, w_model)
        return b, 0.0, 1.0 - b
    cond = win / (win + loss)
    q_cond = min(q_win / (1.0 - push), 0.999)
    b = blend(cond, q_cond, w_model) * (1.0 - push)
    return b, push, 1.0 - push - b
