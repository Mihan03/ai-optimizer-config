"""Прогноз на игровой день: λ → полный прогноз матча → оценка всех предложений БК → отбор."""
from __future__ import annotations

import pandas as pd

from .betting import POOLS, Offer, evaluate_offer, select_bets, stake_for
from .config import Params
from .market import devig
from .ratings import Ratings
from .scores import MatchModel, adjusted_lambdas

ODDS_COLUMNS = ["match_id", "market", "selection", "line", "odds", "bookmaker", "taken_at"]


def _group_key(r) -> tuple:
    m, sel, line = r["market"], str(r["selection"]), r["line"]
    if m == "HCP":
        return (r["match_id"], m, round(line if sel == "1" else -line, 2))
    if m == "TEAM_TOTAL":
        return (r["match_id"], m, sel[0], line)
    if m in ("TOTAL",):
        return (r["match_id"], m, line)
    return (r["match_id"], m)


_FULL = {"1X2": 3, "TOTAL": 2, "HCP": 2, "TEAM_TOTAL": 2, "ML_OT": 2}


def build_offers(odds: pd.DataFrame, params: Params) -> list[Offer]:
    """Вероятности без маржи считаются по полному рынку, если обе (все) стороны есть в файле.
    Для двойного шанса берутся из 1X2; иначе — оценка по default_overround."""
    odds = odds.copy()
    odds["market"] = odds["market"].str.upper()
    odds["selection"] = odds["selection"].astype(str).str.upper()
    odds["line"] = pd.to_numeric(odds.get("line"), errors="coerce")
    q = {}
    for key, g in odds.groupby(odds.apply(_group_key, axis=1)):
        market = key[1]
        if _FULL.get(market) == len(g):
            for i, p in zip(g.index, devig(g["odds"].tolist())):
                q[i] = p
    x12 = {}
    for i, r in odds[odds["market"] == "1X2"].iterrows():
        if i in q:
            x12.setdefault(r["match_id"], {})[r["selection"]] = q[i]
    for i, r in odds[odds["market"] == "DC"].iterrows():
        parts = x12.get(r["match_id"], {})
        if len(parts) == 3:
            q[i] = sum(parts[c] for c in r["selection"])
    offers = []
    for i, r in odds.iterrows():
        line = None if pd.isna(r["line"]) else float(r["line"])
        offers.append(Offer(
            match_id=str(r["match_id"]), market=r["market"], selection=r["selection"], line=line,
            odds=float(r["odds"]), bookmaker=str(r.get("bookmaker", "") or ""),
            taken_at=str(r.get("taken_at", "") or ""), q_market=q.get(i),
        ))
    return offers


def _flag(row, col) -> bool:
    v = row.get(col)
    return bool(v) and not pd.isna(v) and str(v).strip().lower() not in ("0", "false", "нет", "")


def match_lambdas(row, ratings: Ratings | None, params: Params) -> tuple[float, float]:
    if ratings is not None and row["home"] in ratings.att and row["away"] in ratings.att:
        lh, la = ratings.lambdas(row["home"], row["away"])
    elif not pd.isna(row.get("lam_home")) and not pd.isna(row.get("lam_away")):
        lh, la = float(row["lam_home"]), float(row["lam_away"])
    else:
        raise ValueError(f"Нет рейтингов и λ для матча {row['match_id']}")
    return adjusted_lambdas(
        lh, la, params,
        home_backup_goalie=_flag(row, "home_backup_goalie"), away_backup_goalie=_flag(row, "away_backup_goalie"),
        home_b2b=_flag(row, "home_b2b"), away_b2b=_flag(row, "away_b2b"),
        home_tz_shift=float(row.get("home_tz_shift") or 0), away_tz_shift=float(row.get("away_tz_shift") or 0),
    )


def run(matches: pd.DataFrame, odds: pd.DataFrame, params: Params, ratings: Ratings | None = None,
        banks: dict[str, float] | None = None):
    banks = dict(banks or {p: params.start_bank for p in POOLS})
    lam = {}
    forecasts = []
    for _, row in matches.iterrows():
        lh, la = match_lambdas(row, ratings, params)
        lam[str(row["match_id"])] = (lh, la)
        mm = MatchModel(lh, la, params)
        h, x, a = mm.p_1x2()
        w1, w2 = mm.p_winner()
        lo, hi = mm.total_interval()
        forecasts.append({
            "match_id": row["match_id"], "home": row["home"], "away": row["away"],
            "lam_home": round(lh, 3), "lam_away": round(la, 3),
            "p1_60": round(h, 4), "px_60": round(x, 4), "p2_60": round(a, 4),
            "p1_incl_ot": round(w1, 4), "p2_incl_ot": round(w2, 4),
            "top_scores": ", ".join(f"{s} ({p:.1%})" for s, p in mm.top_scores(5)),
            "total_80": f"[{lo}, {hi}]",
        })
    evals = [evaluate_offer(*lam[o.match_id], o, params) for o in build_offers(odds, params) if o.match_id in lam]
    chosen = select_bets(evals, params)
    picks = []
    for e in sorted(chosen, key=lambda e: e.ev, reverse=True):
        stake = stake_for(e, banks[e.pool], params)
        if stake <= 0:
            continue
        picks.append({**e.as_row(), "stake": stake})
    all_rows = pd.DataFrame([{**e.as_row(), "selected": e in chosen} for e in evals])
    return pd.DataFrame(forecasts), all_rows, pd.DataFrame(picks)
