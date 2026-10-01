"""CLI: python -m khl_model {fit,calibrate,predict,evaluate} ..."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from .betting import POOLS
from .calibrate import calibrate
from .config import Params
from .evaluate import bet_summary
from .forecast import run
from .ratings import Ratings, fit_ratings, load_results

DEFAULT_PARAMS = Path(__file__).resolve().parent.parent / "params.json"


def _banks_from_journal(path: str | None, params: Params) -> dict[str, float]:
    banks = {p: params.start_bank for p in POOLS}
    if path and Path(path).exists():
        j = pd.read_csv(path)
        for pool, pnl in j.groupby("pool")["pnl"].sum().items():
            banks[pool] = params.start_bank + pnl
    return banks


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="khl_model")
    ap.add_argument("--params", default=str(DEFAULT_PARAMS))
    sub = ap.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fit", help="рейтинги команд на дату")
    f.add_argument("--results", required=True)
    f.add_argument("--as-of", required=True, help="YYYY-MM-DD, учитываются только матчи до этой даты")
    f.add_argument("--out", default="ratings.json")

    c = sub.add_parser("calibrate", help="walk-forward калибровка θ, пустых ворот, ОТ и веса рынка")
    c.add_argument("--results", required=True)
    c.add_argument("--out", default=None, help="куда сохранить params (по умолчанию --params)")

    p = sub.add_parser("predict", help="прогноз и отбор ставок")
    p.add_argument("--matches", required=True)
    p.add_argument("--odds", required=True)
    p.add_argument("--ratings", default=None)
    p.add_argument("--journal", default=None, help="журнал для текущих банков пулов")
    p.add_argument("--out-dir", default=".")

    e = sub.add_parser("evaluate", help="метрики по журналу ставок")
    e.add_argument("--journal", required=True)

    a = ap.parse_args(argv)
    params = Params.load(a.params)

    if a.cmd == "fit":
        r = fit_ratings(load_results(a.results), a.as_of, params)
        r.save(a.out)
        print(f"μ={r.mu:.3f} (λ̄={math.exp(r.mu):.2f}), h={r.home:.3f}")
        print(r.to_frame().to_string(index=False))
    elif a.cmd == "calibrate":
        new, report = calibrate(load_results(a.results), params)
        new.save(a.out or a.params)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif a.cmd == "predict":
        ratings = Ratings.load(a.ratings) if a.ratings else None
        fc, all_offers, picks = run(pd.read_csv(a.matches), pd.read_csv(a.odds), params, ratings,
                                    _banks_from_journal(a.journal, params))
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        fc.to_csv(out / "forecast.csv", index=False)
        all_offers.to_csv(out / "offers_evaluated.csv", index=False)
        picks.to_csv(out / "picks.csv", index=False)
        print(fc.to_string(index=False))
        print("\nОтобрано:" if len(picks) else "\nНет ставок, прошедших фильтр — это нормальный исход.")
        if len(picks):
            print(picks[["match_id", "pool", "market", "selection", "line", "odds", "p_final", "ev", "ev_worst", "stake"]].to_string(index=False))
    elif a.cmd == "evaluate":
        print(bet_summary(pd.read_csv(a.journal)).to_string(index=False))


if __name__ == "__main__":
    main()
