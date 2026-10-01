"""Оценка ставок, отбор, размер позиции и расчёт.

Правила v2.0 (замена «Главное/Надёжное/Смелое» из v1.0):
* EV считается по вероятности ПОСЛЕ смешивания с рынком;
* один и тот же фильтр EV ≥ ev_min и проверка устойчивости для всех пулов;
* пул определяется коэффициентом (Надёжное < 1.60 ≤ Главное < 2.20 ≤ Смелое);
* не больше max_bets_per_match ставок на матч (ставки на один матч коррелированы);
* размер — доля Келли от ТЕКУЩЕГО банка пула с потолком, банк не уходит в минус.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .config import Params
from .market import blend_with_push, implied_single
from .scores import MatchModel

POOL_MAIN, POOL_RELIABLE, POOL_BOLD = "Главное", "Надёжное", "Смелое"
POOLS = (POOL_MAIN, POOL_RELIABLE, POOL_BOLD)


@dataclass
class Offer:
    match_id: str
    market: str
    selection: str
    line: float | None
    odds: float
    bookmaker: str = ""
    taken_at: str = ""
    q_market: float | None = None  # вероятность без маржи; None → оценка по default_overround


@dataclass
class Evaluation:
    offer: Offer
    p_model: tuple[float, float, float]
    q_market: float
    p_final: tuple[float, float, float]
    ev: float
    ev_model_only: float
    ev_worst: float
    kelly: float
    pool: str

    def as_row(self) -> dict:
        row = asdict(self.offer)
        row.update(
            p_model=round(self.p_model[0], 4), p_push=round(self.p_final[1], 4),
            q_market=round(self.q_market, 4), p_final=round(self.p_final[0], 4),
            fair_odds=round(1 / self.p_final[0], 3) if self.p_final[0] > 0 else None,
            ev=round(self.ev, 4), ev_model_only=round(self.ev_model_only, 4),
            ev_worst=round(self.ev_worst, 4), kelly=round(self.kelly, 4), pool=self.pool,
        )
        return row


def ev_of(p: tuple[float, float, float], odds: float) -> float:
    win, push, _ = p
    return win * odds + push - 1.0


def kelly_of(p: tuple[float, float, float], odds: float) -> float:
    win, _, loss = p
    b = odds - 1.0
    return max(0.0, (b * win - loss) / b) if b > 0 else 0.0


def pool_for(odds: float, params: Params) -> str:
    if odds < params.reliable_max_odds:
        return POOL_RELIABLE
    if odds >= params.bold_min_odds:
        return POOL_BOLD
    return POOL_MAIN


def _final_prob(lh: float, la: float, params: Params, offer: Offer, q: float) -> tuple:
    mm = MatchModel(lh, la, params)
    p_model = mm.market_prob(offer.market, offer.selection, offer.line)
    return p_model, blend_with_push(p_model, q, params.market_weight_model)


def stress_scenarios(lh: float, la: float, params: Params):
    k = 1.07
    yield lh * k, la / k, params
    yield lh / k, la * k, params
    yield lh * k, la * k, params
    yield lh / k, la / k, params
    yield lh, la, params.replace(draw_inflation=params.draw_inflation * 1.5)
    yield lh, la, params.replace(draw_inflation=params.draw_inflation * 0.5)
    yield lh, la, params.replace(market_weight_model=max(0.0, params.market_weight_model - 0.2))


def evaluate_offer(lh: float, la: float, offer: Offer, params: Params) -> Evaluation:
    q = offer.q_market if offer.q_market is not None else implied_single(offer.odds, params.default_overround)
    p_model, p_final = _final_prob(lh, la, params, offer, q)
    ev = ev_of(p_final, offer.odds)
    worst = min(ev_of(_final_prob(a, b, p, offer, q)[1], offer.odds) for a, b, p in stress_scenarios(lh, la, params))
    return Evaluation(
        offer=offer, p_model=p_model, q_market=q, p_final=p_final, ev=ev,
        ev_model_only=ev_of(p_model, offer.odds), ev_worst=min(worst, ev),
        kelly=kelly_of(p_final, offer.odds), pool=pool_for(offer.odds, params),
    )


def passes(e: Evaluation, params: Params) -> bool:
    if e.ev < params.ev_min:
        return False
    if params.require_robust and e.ev_worst < 0:
        return False
    return True


def select_bets(evals: list[Evaluation], params: Params) -> list[Evaluation]:
    by_match: dict[str, list[Evaluation]] = {}
    for e in evals:
        if passes(e, params):
            by_match.setdefault(e.offer.match_id, []).append(e)
    chosen: list[Evaluation] = []
    for items in by_match.values():
        items.sort(key=lambda e: e.ev, reverse=True)
        chosen.extend(items[: params.max_bets_per_match])
    return chosen


def stake_for(e: Evaluation, bank: float, params: Params) -> float:
    if bank <= 0:
        return 0.0
    frac = min(params.kelly_fraction * e.kelly, params.max_stake_frac)
    stake = math.floor(frac * bank)
    return float(stake) if stake >= params.min_stake else 0.0


# --- расчёт ---

def parse_score(s: str) -> tuple[int, int]:
    h, a = str(s).strip().split()[0].split(":")
    return int(h), int(a)


def settle(market: str, selection: str, line: float | None, score_60: str, score_final: str | None = None) -> str:
    """WIN / LOSS / PUSH. score_final нужен только для ML_OT (итоговый счёт с ОТ/Б)."""
    h, a = parse_score(score_60)
    market, selection = market.upper(), str(selection).upper()
    if market == "ML_OT":
        fh, fa = parse_score(score_final or score_60)
        if fh == fa:
            raise ValueError("Итоговый счёт не может быть ничейным")
        return "WIN" if (fh > fa) == (selection == "1") else "LOSS"

    def by_value(x: float) -> str:
        return "WIN" if x > 1e-9 else ("PUSH" if abs(x) <= 1e-9 else "LOSS")

    if market == "1X2":
        res = "1" if h > a else ("X" if h == a else "2")
        return "WIN" if res == selection else "LOSS"
    if market == "DC":
        res = "1" if h > a else ("X" if h == a else "2")
        return "WIN" if res in selection else "LOSS"
    if market == "TOTAL":
        return by_value((h + a) - line if selection == "O" else line - (h + a))
    if market == "HCP":
        return by_value((h - a if selection == "1" else a - h) + line)
    if market == "TEAM_TOTAL":
        g = h if selection[0] == "1" else a
        return by_value(g - line if selection[1] == "O" else line - g)
    raise ValueError(market)


def pnl(result: str, stake: float, odds: float) -> float:
    return {"WIN": stake * (odds - 1), "LOSS": -stake, "PUSH": 0.0}.get(result, 0.0)


def describe(market: str, selection: str, line: float | None, home: str = "П1", away: str = "П2") -> str:
    """Человекочитаемое описание ставки для журнала."""
    market, selection = market.upper(), str(selection).upper()
    team = {"1": home, "2": away}
    if market == "1X2":
        return {"1": f"П1 {home}", "X": "Ничья", "2": f"П2 {away}"}[selection] + " (60 мин)"
    if market == "DC":
        return f"{selection} (60 мин)"
    if market == "TOTAL":
        return f"{'ТБ' if selection == 'O' else 'ТМ'} {line:g} (60 мин)"
    if market == "HCP":
        return f"Ф{selection}({line:+g}) {team[selection]} (60 мин)"
    if market == "TEAM_TOTAL":
        return f"ИТ{selection[0]} {'Б' if selection[1] == 'O' else 'М'} {line:g} {team[selection[0]]} (60 мин)"
    if market == "ML_OT":
        return f"Победа {team[selection]} с учётом ОТ/Б"
    return f"{market} {selection} {line}"
