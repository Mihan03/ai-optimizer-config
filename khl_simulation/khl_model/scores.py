"""Распределение счёта за 60 минут и вероятности рынков.

Отличия от v1.0:
* точный расчёт по матрице счетов вместо Monte Carlo (нет шума симуляции);
* поправка на ничьи в основное время (draw inflation θ) — независимый
  Пуассон систематически занижает X, из-за чего завышались П1/П2;
* поправка на голы в пустые ворота — часть побед в 1 шайбу превращается
  в +2, что важно для фор −1.5 и верхнего хвоста тотала;
* победитель с учётом ОТ/Б зависит от силы команд, но сжат к 50/50.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .config import Params

# Стандартизованные рынки:
#   1X2        selection 1 / X / 2                 (60 минут)
#   DC         selection 1X / X2 / 12              (60 минут)
#   TOTAL      selection O / U, line               (60 минут)
#   HCP        selection 1 / 2, line (фора команде) (60 минут)
#   TEAM_TOTAL selection 1O / 1U / 2O / 2U, line   (60 минут)
#   ML_OT      selection 1 / 2                     (победитель матча с ОТ и буллитами)
MARKETS = ("1X2", "DC", "TOTAL", "HCP", "TEAM_TOTAL", "ML_OT")


def _poisson_pmf(lam: float, n: int) -> np.ndarray:
    k = np.arange(n)
    logp = -lam + k * math.log(lam) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(logp)


def score_matrix(lam_home: float, lam_away: float, params: Params) -> np.ndarray:
    n = params.max_goals + 2
    m = np.outer(_poisson_pmf(lam_home, n), _poisson_pmf(lam_away, n))

    # Пустые ворота: проигрывающий в 1 шайбу снимает вратаря.
    p_en = params.empty_net_shift
    if p_en > 0:
        shifted = m.copy()
        for i in range(n - 1):
            j = i + 1  # гости ведут в 1 шайбу (счёт i:j)
            if j + 1 < n:
                moved = m[i, j] * p_en
                shifted[i, j] -= moved
                shifted[i, j + 1] += moved
            moved = m[j, i] * p_en  # хозяева ведут в 1 шайбу (счёт j:i)
            if j + 1 < n:
                shifted[j, i] -= moved
                shifted[j + 1, i] += moved
        m = shifted

    # Поправка на ничьи.
    if params.draw_inflation:
        m[np.diag_indices(n)] *= 1.0 + params.draw_inflation

    m = m[: params.max_goals + 1, : params.max_goals + 1]
    return m / m.sum()


@dataclass
class MatchModel:
    lam_home: float
    lam_away: float
    params: Params

    def __post_init__(self) -> None:
        self.m = score_matrix(self.lam_home, self.lam_away, self.params)
        g = np.arange(self.m.shape[0])
        self._diff = g[:, None] - g[None, :]
        self._total = g[:, None] + g[None, :]
        self._home_g = np.broadcast_to(g[:, None], self.m.shape)
        self._away_g = np.broadcast_to(g[None, :], self.m.shape)

    # --- основные величины ---
    def p_1x2(self) -> tuple[float, float, float]:
        m, d = self.m, self._diff
        return float(m[d > 0].sum()), float(m[d == 0].sum()), float(m[d < 0].sum())

    def p_home_after_tie(self) -> float:
        p = self.params
        share = self.lam_home / (self.lam_home + self.lam_away)
        p_ot_home = 0.5 + p.ot_strength_shrink * (share - 0.5)
        return p.ot_goal_prob * p_ot_home + (1 - p.ot_goal_prob) * p.shootout_home

    def p_winner(self) -> tuple[float, float]:
        h, x, a = self.p_1x2()
        ph = self.p_home_after_tie()
        return h + x * ph, a + x * (1 - ph)

    def total_dist(self) -> np.ndarray:
        out = np.zeros(self._total.max() + 1)
        np.add.at(out, self._total.ravel(), self.m.ravel())
        return out

    def total_interval(self, coverage: float = 0.8) -> tuple[int, int]:
        cdf = np.cumsum(self.total_dist())
        lo_q, hi_q = (1 - coverage) / 2, 1 - (1 - coverage) / 2
        return int(np.searchsorted(cdf, lo_q)), int(np.searchsorted(cdf, hi_q))

    def top_scores(self, n: int = 5) -> list[tuple[str, float]]:
        flat = np.argsort(self.m, axis=None)[::-1][:n]
        rows, cols = np.unravel_index(flat, self.m.shape)
        return [(f"{r}:{c}", float(self.m[r, c])) for r, c in zip(rows, cols)]

    # --- рынки ---
    def _line(self, values: np.ndarray, line: float) -> tuple[float, float, float]:
        """(win, push, loss) для события values + line > 0."""
        x = values + line
        m = self.m
        return float(m[x > 1e-9].sum()), float(m[np.abs(x) <= 1e-9].sum()), float(m[x < -1e-9].sum())

    def market_prob(self, market: str, selection: str, line: float | None = None) -> tuple[float, float, float]:
        """Вероятности (выигрыш, возврат, проигрыш) ставки."""
        market, selection = market.upper(), str(selection).upper()
        if market == "1X2":
            h, x, a = self.p_1x2()
            p = {"1": h, "X": x, "2": a}[selection]
            return p, 0.0, 1 - p
        if market == "DC":
            h, x, a = self.p_1x2()
            p = {"1X": h + x, "X2": x + a, "12": h + a}[selection]
            return p, 0.0, 1 - p
        if market == "ML_OT":
            ph, pa = self.p_winner()
            p = ph if selection == "1" else pa
            return p, 0.0, 1 - p
        if line is None:
            raise ValueError(f"{market} требует line")
        if market == "TOTAL":
            if selection == "O":
                return self._line(self._total, -line)
            return self._line(-self._total, line)
        if market == "HCP":
            diff = self._diff if selection == "1" else -self._diff
            return self._line(diff, line)
        if market == "TEAM_TOTAL":
            goals = self._home_g if selection[0] == "1" else self._away_g
            if selection[1] == "O":
                return self._line(goals, -line)
            return self._line(-goals, line)
        raise ValueError(f"Неизвестный рынок {market}")


def adjusted_lambdas(lam_home: float, lam_away: float, params: Params, *,
                     home_backup_goalie: bool = False, away_backup_goalie: bool = False,
                     home_b2b: bool = False, away_b2b: bool = False,
                     home_tz_shift: float = 0.0, away_tz_shift: float = 0.0) -> tuple[float, float]:
    """Поправки на запасного вратаря, второй матч подряд и смену часовых поясов."""
    lh, la = math.log(lam_home), math.log(lam_away)
    if away_backup_goalie:
        lh += params.backup_goalie_effect
    if home_backup_goalie:
        la += params.backup_goalie_effect
    if home_b2b:
        lh += params.b2b_attack_effect
        la += params.b2b_defense_effect
    if away_b2b:
        la += params.b2b_attack_effect
        lh += params.b2b_defense_effect
    lh += params.travel_tz_effect * min(abs(home_tz_shift), 4)
    la += params.travel_tz_effect * min(abs(away_tz_shift), 4)
    return math.exp(lh), math.exp(la)
