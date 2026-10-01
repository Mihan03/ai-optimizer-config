"""Рейтинги атаки/обороны: регуляризованная Пуассоновская регрессия.

log λ_home = μ + h + att_home − def_away
log λ_away = μ +     att_away − def_home

* только матчи СТРОГО до момента прогноза (защита от утечки);
* вес матча = 0.5^(возраст / half_life) × season_carryover для прошлых сезонов —
  в начале сезона рейтинги опираются на прошлый сезон, но сжаты к среднему;
* L2-регуляризация к нулю (к среднему по лиге).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .config import Params

RESULT_COLUMNS = ["date", "season", "home", "away", "home_goals_60", "away_goals_60",
                  "home_goals_final", "away_goals_final", "decided"]


def load_results(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in RESULT_COLUMNS[:6] if c not in df.columns]
    if missing:
        raise ValueError(f"В {path} нет колонок: {missing}")
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


@dataclass
class Ratings:
    mu: float
    home: float
    att: dict[str, float]
    defense: dict[str, float]
    as_of: str
    games: dict[str, float] = field(default_factory=dict)

    def lambdas(self, home: str, away: str) -> tuple[float, float]:
        for t in (home, away):
            if t not in self.att:
                raise KeyError(f"Нет рейтинга для команды «{t}»")
        lh = math.exp(self.mu + self.home + self.att[home] - self.defense[away])
        la = math.exp(self.mu + self.att[away] - self.defense[home])
        return lh, la

    def to_frame(self) -> pd.DataFrame:
        rows = []
        for t in sorted(self.att, key=lambda t: self.att[t] + self.defense[t], reverse=True):
            rows.append({
                "team": t, "att": round(self.att[t], 4), "def": round(self.defense[t], 4),
                "gf60_vs_avg": round(math.exp(self.mu + self.att[t]), 3),
                "ga60_vs_avg": round(math.exp(self.mu - self.defense[t]), 3),
                "weighted_games": round(self.games.get(t, 0.0), 1),
            })
        return pd.DataFrame(rows)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.__dict__, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Ratings":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


def fit_ratings(results: pd.DataFrame, as_of: str | pd.Timestamp, params: Params) -> Ratings:
    as_of = pd.Timestamp(as_of)
    df = results[results["date"] < as_of]
    if df.empty:
        raise ValueError("Нет матчей до даты прогноза")

    teams = sorted(set(df["home"]) | set(df["away"]))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    hi = df["home"].map(idx).to_numpy()
    ai = df["away"].map(idx).to_numpy()
    yh = df["home_goals_60"].to_numpy(float)
    ya = df["away_goals_60"].to_numpy(float)

    age = (as_of - df["date"]).dt.days.to_numpy(float)
    w = 0.5 ** (age / params.half_life_days)
    if "season" in df.columns:
        current = df["season"].iloc[-1]
        w = w * np.where(df["season"].to_numpy() == current, 1.0, params.season_carryover)

    def unpack(x):
        return x[0], x[1], x[2:2 + n], x[2 + n:]

    def nll(x):
        mu, h, att, dfn = unpack(x)
        eh = mu + h + att[hi] - dfn[ai]
        ea = mu + att[ai] - dfn[hi]
        lh, la = np.exp(eh), np.exp(ea)
        val = np.sum(w * (lh - yh * eh)) + np.sum(w * (la - ya * ea))
        val += 0.5 * params.ridge * (att @ att + dfn @ dfn)
        rh, ra = w * (lh - yh), w * (la - ya)
        g = np.empty_like(x)
        g[0] = rh.sum() + ra.sum()
        g[1] = rh.sum()
        g[2:2 + n] = np.bincount(hi, rh, n) + np.bincount(ai, ra, n) + params.ridge * att
        g[2 + n:] = -np.bincount(ai, rh, n) - np.bincount(hi, ra, n) + params.ridge * dfn
        return val, g

    x0 = np.zeros(2 + 2 * n)
    x0[0] = math.log(max(0.5 * (yh @ w + ya @ w) / w.sum(), 0.1))
    res = minimize(nll, x0, jac=True, method="L-BFGS-B")
    if not res.success:
        raise RuntimeError(f"Оптимизация не сошлась: {res.message}")
    mu, h, att, dfn = unpack(res.x)
    games = np.bincount(hi, w, n) + np.bincount(ai, w, n)
    return Ratings(
        mu=float(mu), home=float(h),
        att={t: float(att[i]) for t, i in idx.items()},
        defense={t: float(dfn[i]) for t, i in idx.items()},
        as_of=str(as_of.date()), games={t: float(games[i]) for t, i in idx.items()},
    )
