import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from khl_model import MatchModel, Params  # noqa: E402
from khl_model.betting import Offer, evaluate_offer, kelly_of, select_bets, settle, stake_for  # noqa: E402
from khl_model.calibrate import fit_score_params, walk_forward_lambdas  # noqa: E402
from khl_model.forecast import build_offers, run  # noqa: E402
from khl_model.market import blend, devig  # noqa: E402
from khl_model.ratings import fit_ratings  # noqa: E402
from khl_model.scores import score_matrix  # noqa: E402

PLAIN = Params(draw_inflation=0.0, empty_net_shift=0.0)


def test_plain_poisson_matches_v1_sheet():
    # v1.0: ЦСКА–СКА λ 2.82:2.14 → П1 0.5287, ТМ5.5 0.6232 (Monte Carlo)
    mm = MatchModel(2.82, 2.14, PLAIN)
    h, x, a = mm.p_1x2()
    assert h == pytest.approx(0.5287, abs=0.002)
    assert mm.market_prob("TOTAL", "U", 5.5)[0] == pytest.approx(0.6232, abs=0.002)


def test_matrix_normalised_and_draw_inflation_raises_x():
    m = score_matrix(3.0, 2.0, Params())
    assert m.sum() == pytest.approx(1.0)
    x_plain = MatchModel(3.0, 2.0, PLAIN).p_1x2()[1]
    x_v2 = MatchModel(3.0, 2.0, Params()).p_1x2()[1]
    assert x_v2 > x_plain + 0.03


def test_empty_net_moves_one_goal_wins_to_two():
    base = MatchModel(3.0, 2.0, PLAIN)
    en = MatchModel(3.0, 2.0, PLAIN.replace(empty_net_shift=0.2))
    assert en.market_prob("HCP", "1", -1.5)[0] > base.market_prob("HCP", "1", -1.5)[0]
    assert en.p_1x2()[0] == pytest.approx(base.p_1x2()[0], abs=1e-9)  # победитель не меняется


def test_market_probabilities_consistent():
    mm = MatchModel(2.9, 2.3, Params())
    h, x, a = mm.p_1x2()
    assert h + x + a == pytest.approx(1)
    assert mm.market_prob("DC", "1X")[0] == pytest.approx(h + x)
    o, u = mm.market_prob("TOTAL", "O", 5.5)[0], mm.market_prob("TOTAL", "U", 5.5)[0]
    assert o + u == pytest.approx(1)
    w, p, l = mm.market_prob("TOTAL", "O", 5.0)
    assert p > 0 and w + p + l == pytest.approx(1)
    w1, w2 = mm.p_winner()
    assert w1 + w2 == pytest.approx(1) and w1 > h


def test_settlement():
    assert settle("TOTAL", "U", 5.5, "6:5") == "LOSS"
    assert settle("DC", "1X", None, "2:2") == "WIN"
    assert settle("HCP", "1", -1.5, "3:1") == "WIN"
    assert settle("HCP", "2", 1.5, "3:1") == "LOSS"
    assert settle("TOTAL", "O", 5.0, "3:2") == "PUSH"
    assert settle("ML_OT", "2", None, "2:2", "2:3 ОТ") == "WIN"
    assert settle("TEAM_TOTAL", "2O", 2.5, "1:3") == "WIN"


def test_devig_and_blend():
    q = devig([1.76, 2.05])
    assert sum(q) == pytest.approx(1)
    assert blend(0.6, 0.5, 0.4) == pytest.approx(1 / (1 + math.exp(-(0.4 * math.log(1.5)))))


def test_kelly_and_stake_caps():
    assert kelly_of((0.5, 0, 0.5), 2.0) == 0
    assert kelly_of((0.6, 0, 0.4), 2.0) == pytest.approx(0.2)
    p = Params()
    e = evaluate_offer(3.4, 1.55, Offer("m", "1X2", "1", None, 1.48, q_market=0.6435), p)
    assert stake_for(e, 1000, p) <= p.max_stake_frac * 1000
    assert stake_for(e, -200, p) == 0


def test_v1_reliable_bet_is_negative_ev():
    # 1X @1.18 Автомобилист–Лада: по самой модели v1 EV < 0
    e = evaluate_offer(3.15, 1.70, Offer("m", "DC", "1X", None, 1.18), Params())
    assert e.ev < 0 and not select_bets([e], Params())


def _synthetic_league(seed=1, seasons=2, params=Params()):
    rng = np.random.default_rng(seed)
    teams = [f"T{i}" for i in range(12)]
    att = dict(zip(teams, rng.normal(0, 0.15, len(teams))))
    dfn = dict(zip(teams, rng.normal(0, 0.15, len(teams))))
    rows, day = [], pd.Timestamp("2024-09-01")
    for s in range(seasons):
        for rnd in range(30):
            perm = rng.permutation(teams)
            for h, a in zip(perm[::2], perm[1::2]):
                lh = math.exp(params.mu + params.home_adv + att[h] - dfn[a])
                la = math.exp(params.mu + att[a] - dfn[h])
                m = score_matrix(lh, la, params)
                k = rng.choice(m.size, p=m.ravel())
                gh, ga = divmod(k, m.shape[1])
                rows.append({"date": day, "season": 2024 + s, "home": h, "away": a,
                             "home_goals_60": gh, "away_goals_60": ga})
            day += pd.Timedelta(days=3)
    return pd.DataFrame(rows), att, dfn


def test_ratings_recover_truth_and_no_leakage():
    df, att, dfn = _synthetic_league(seasons=1)
    r = fit_ratings(df, df["date"].max() + pd.Timedelta(days=1), Params(half_life_days=1e6))
    est = np.array([r.att[t] for t in att])
    assert np.corrcoef(est, list(att.values()))[0, 1] > 0.6
    cutoff = df["date"].iloc[len(df) // 2]
    r2 = fit_ratings(df, cutoff, Params())
    assert pd.Timestamp(r2.as_of) == cutoff  # используются только матчи до cutoff


def test_calibration_recovers_draw_inflation():
    truth = Params(draw_inflation=0.4, empty_net_shift=0.12)
    df, *_ = _synthetic_league(seed=3, seasons=2, params=truth)
    pred = walk_forward_lambdas(df, Params(draw_inflation=0.0, empty_net_shift=0.0), refit_days=15)
    theta, p_en, _ = fit_score_params(pred, Params())
    assert 0.15 <= theta <= 0.7


def test_run_end_to_end():
    matches = pd.DataFrame([{"match_id": "A", "home": "X", "away": "Y", "lam_home": 2.8, "lam_away": 2.1}])
    odds = pd.DataFrame([
        {"match_id": "A", "market": "1X2", "selection": "1", "line": None, "odds": 1.95},
        {"match_id": "A", "market": "1X2", "selection": "X", "line": None, "odds": 4.2},
        {"match_id": "A", "market": "1X2", "selection": "2", "line": None, "odds": 3.6},
        {"match_id": "A", "market": "DC", "selection": "1X", "line": None, "odds": 1.33},
        {"match_id": "A", "market": "TOTAL", "selection": "U", "line": 5.5, "odds": 1.80},
        {"match_id": "A", "market": "TOTAL", "selection": "O", "line": 5.5, "odds": 2.00},
    ])
    offers = build_offers(odds, Params())
    assert all(o.q_market is not None for o in offers)  # DC взят из 1X2
    fc, all_offers, picks = run(matches, odds, Params())
    assert len(fc) == 1 and len(all_offers) == 6
    assert len(picks) <= 1
