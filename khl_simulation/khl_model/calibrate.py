"""Walk-forward калибровка параметров, которые в v2.0 стоят как PROVISIONAL.

1. Для каждой игровой даты рейтинги фитятся только на прошлых матчах
   (обновление раз в refit_days), прогнозируются λ на этот день.
2. По честным прогнозам подбираются draw_inflation и empty_net_shift
   (максимум правдоподобия точного счёта за 60 минут).
3. ОТ-параметры — частоты по историческим ничьим.
4. Вес модели при смешивании с рынком — минимум log loss по 1X2,
   если в истории есть рыночные вероятности (колонки q1, qx, q2).
"""
from __future__ import annotations

import itertools
import math

import numpy as np
import pandas as pd

from .config import Params
from .ratings import fit_ratings
from .scores import score_matrix


def walk_forward_lambdas(results: pd.DataFrame, params: Params, *, min_history_days: int = 60,
                         refit_days: int = 7) -> pd.DataFrame:
    start = results["date"].min() + pd.Timedelta(days=min_history_days)
    out, ratings, last_fit = [], None, None
    for day, games in results[results["date"] >= start].groupby("date"):
        if ratings is None or (day - last_fit).days >= refit_days:
            ratings, last_fit = fit_ratings(results, day, params), day
        for _, g in games.iterrows():
            if g["home"] not in ratings.att or g["away"] not in ratings.att:
                continue
            lh, la = ratings.lambdas(g["home"], g["away"])
            out.append({**g.to_dict(), "lam_home": lh, "lam_away": la})
    return pd.DataFrame(out)


def score_loglik(pred: pd.DataFrame, params: Params) -> float:
    ll = 0.0
    for _, r in pred.iterrows():
        m = score_matrix(r["lam_home"], r["lam_away"], params)
        h, a = int(r["home_goals_60"]), int(r["away_goals_60"])
        if h < m.shape[0] and a < m.shape[1]:
            ll += math.log(max(m[h, a], 1e-12))
    return ll


def fit_score_params(pred: pd.DataFrame, params: Params) -> tuple[float, float, float]:
    best = (-math.inf, params.draw_inflation, params.empty_net_shift)
    for theta, p_en in itertools.product(np.arange(0.0, 0.85, 0.05), np.arange(0.0, 0.31, 0.03)):
        ll = score_loglik(pred, params.replace(draw_inflation=float(theta), empty_net_shift=float(p_en)))
        if ll > best[0]:
            best = (ll, float(theta), float(p_en))
    return best[1], best[2], best[0]


def fit_ot_params(results: pd.DataFrame) -> dict:
    if "decided" not in results.columns:
        return {}
    ties = results[results["home_goals_60"] == results["away_goals_60"]]
    ties = ties[ties["decided"].isin(["OT", "SO"])]
    if len(ties) < 30:
        return {}
    so = ties[ties["decided"] == "SO"]
    out = {"ot_goal_prob": round(float((ties["decided"] == "OT").mean()), 3)}
    if len(so) >= 30:
        out["shootout_home"] = round(float((so["home_goals_final"] > so["away_goals_final"]).mean()), 3)
    return out


def fit_market_weight(pred: pd.DataFrame, params: Params) -> float | None:
    """Нужны колонки q1, qx, q2 — вероятности 1X2 без маржи на момент прогноза."""
    if not {"q1", "qx", "q2"} <= set(pred.columns):
        return None
    rows = pred.dropna(subset=["q1", "qx", "q2"])
    if len(rows) < 100:
        return None
    probs = []
    for _, r in rows.iterrows():
        m = score_matrix(r["lam_home"], r["lam_away"], params)
        d = np.subtract.outer(np.arange(m.shape[0]), np.arange(m.shape[1]))
        probs.append((m[d > 0].sum(), m[d == 0].sum(), m[d < 0].sum()))
    pm = np.array(probs)
    q = rows[["q1", "qx", "q2"]].to_numpy(float)
    y = np.select([rows["home_goals_60"] > rows["away_goals_60"],
                   rows["home_goals_60"] == rows["away_goals_60"]], [0, 1], 2)
    best_w, best_ll = 0.0, -math.inf
    for w in np.arange(0.0, 1.01, 0.05):
        p = pm ** w * q ** (1 - w)
        p /= p.sum(axis=1, keepdims=True)
        ll = np.log(p[np.arange(len(y)), y]).sum()
        if ll > best_ll:
            best_w, best_ll = float(w), ll
    return best_w


def calibrate(results: pd.DataFrame, params: Params) -> tuple[Params, dict]:
    pred = walk_forward_lambdas(results, params)
    report = {"n_predictions": len(pred)}
    theta, p_en, ll = fit_score_params(pred, params)
    report.update(draw_inflation=theta, empty_net_shift=p_en, loglik=round(ll, 2))
    tie_rate = float((results["home_goals_60"] == results["away_goals_60"]).mean())
    report["tie_rate_actual"] = round(tie_rate, 4)
    new = params.replace(draw_inflation=theta, empty_net_shift=p_en, **fit_ot_params(results))
    w = fit_market_weight(pred, new)
    if w is not None:
        new = new.replace(market_weight_model=w)
        report["market_weight_model"] = w
    report.update(fit_ot_params(results))
    new.notes = {**params.notes, "calibrated_on": f"{len(results)} матчей, {results['date'].min().date()}–{results['date'].max().date()}"}
    return new, report

