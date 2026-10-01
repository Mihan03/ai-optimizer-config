"""Метрики качества: доходность, CLV, Brier / log loss против рынка, калибровка.

Win rate без учёта коэффициента ничего не говорит, поэтому основная метрика —
yield (P&L / оборот) со стандартной ошибкой, а самый быстрый сигнал —
CLV: EV ставки по вероятности без маржи на закрытии линии.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd


def bet_summary(j: pd.DataFrame) -> pd.DataFrame:
    """j: журнал со столбцами pool, odds, stake, result (WIN/LOSS/PUSH/пусто), pnl,
    p_final, q_market и, если есть, closing_odds, closing_q."""
    rows = []
    for pool, g in list(j.groupby("pool")) + [("ИТОГО", j)]:
        s = g[g["result"].isin(["WIN", "LOSS", "PUSH"])]
        staked = s["stake"].sum()
        unit = (s["pnl"] / s["stake"]).to_numpy() if len(s) else np.array([])
        row = {
            "pool": pool, "bets": len(g), "settled": len(s), "pending": len(g) - len(s),
            "wins": int((s["result"] == "WIN").sum()), "losses": int((s["result"] == "LOSS").sum()),
            "pushes": int((s["result"] == "PUSH").sum()), "staked": staked, "pnl": s["pnl"].sum(),
            "yield": s["pnl"].sum() / staked if staked else np.nan,
            "yield_se": unit.std(ddof=1) / math.sqrt(len(unit)) if len(unit) > 1 else np.nan,
            "avg_odds": s["odds"].mean() if len(s) else np.nan,
            "avg_ev_expected": (g["p_final"] * g["odds"] - 1).mean() if "p_final" in g else np.nan,
        }
        if "closing_q" in g and g["closing_q"].notna().any():
            c = g.dropna(subset=["closing_q"])
            row["clv_ev"] = (c["odds"] * c["closing_q"] - 1).mean()
            row["clv_beat_close_share"] = (c["odds"] > 1 / c["closing_q"]).mean()
        elif "closing_odds" in g and g["closing_odds"].notna().any():
            c = g.dropna(subset=["closing_odds"])
            row["clv_raw"] = (c["odds"] / c["closing_odds"] - 1).mean()
        if len(s) and {"p_final", "q_market"} <= set(s.columns):
            y = (s["result"] == "WIN").astype(float).to_numpy()
            keep = s["result"] != "PUSH"
            row["brier_model"] = brier(s.loc[keep, "p_final"], y[keep.to_numpy()])
            row["brier_market"] = brier(s.loc[keep, "q_market"], y[keep.to_numpy()])
        rows.append(row)
    return pd.DataFrame(rows)


def brier(p, y) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    return float(np.mean((p - y) ** 2)) if len(p) else float("nan")


def logloss(p, y) -> float:
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))) if len(p) else float("nan")


def forecast_scores(f: pd.DataFrame) -> pd.DataFrame:
    """Качество прогнозов 1X2 по ВСЕМ матчам (не только по ставкам).
    Колонки: p1, px, p2 (модель), q1, qx, q2 (рынок без маржи), home_goals_60, away_goals_60."""
    y = np.select([f["home_goals_60"] > f["away_goals_60"], f["home_goals_60"] == f["away_goals_60"]], [0, 1], 2)
    out = {}
    for name, cols in (("model", ["p1", "px", "p2"]), ("market", ["q1", "qx", "q2"])):
        if not set(cols) <= set(f.columns):
            continue
        p = f[cols].to_numpy(float)
        onehot = np.eye(3)[y]
        out[name] = {
            "brier": float(np.mean(np.sum((p - onehot) ** 2, axis=1))),
            "logloss": float(-np.mean(np.log(np.clip(p[np.arange(len(y)), y], 1e-9, 1)))),
            "mean_p_draw": float(p[:, 1].mean()),
        }
    out["actual_draw_rate"] = {"value": float((y == 1).mean())}
    return pd.DataFrame(out).T


def calibration_table(p, y, bins: int = 10) -> pd.DataFrame:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        mask = idx == b
        if mask.any():
            rows.append({"bin": f"{edges[b]:.1f}–{edges[b + 1]:.1f}", "n": int(mask.sum()),
                         "mean_p": float(p[mask].mean()), "hit_rate": float(y[mask].mean())})
    return pd.DataFrame(rows)
