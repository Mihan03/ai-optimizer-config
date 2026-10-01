"""Параметры модели v2.0.

Значения с пометкой PROVISIONAL выставлены вручную (по рыночным ценам и
общим знаниям о хоккее) и должны быть перекалиброваны командой
`python -m khl_model calibrate` на истории КХЛ, как только она будет загружена.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


@dataclass
class Params:
    version: str = "2.0"

    # --- Рейтинги команд (регуляризованный Пуассон) ---
    mu: float = math.log(2.55)          # базовый log-λ за 60 минут; фитится по данным
    home_adv: float = 0.06              # log-преимущество площадки; фитится по данным
    ridge: float = 6.0                  # сила L2-регуляризации att/def (в «матчах»)
    half_life_days: float = 120.0       # затухание веса старых матчей
    season_carryover: float = 0.6       # доля прошлосезонного рейтинга в приоре

    # --- Структура счёта за 60 минут ---
    draw_inflation: float = 0.40        # PROVISIONAL: θ, множитель (1+θ) на диагональ матрицы счетов
    empty_net_shift: float = 0.12       # PROVISIONAL: доля побед в 1 шайбу, превращающихся в +2 (пустые ворота)
    max_goals: int = 15

    # --- Овертайм 3×3 и буллиты ---
    ot_goal_prob: float = 0.58          # PROVISIONAL: P(гол в ОТ | ничья за 60 мин)
    ot_strength_shrink: float = 0.5     # насколько сила команд переносится в ОТ (0 = монетка)
    shootout_home: float = 0.5          # P(хозяева выигрывают серию буллитов)

    # --- Поправки к λ (лог-множители), PROVISIONAL ---
    backup_goalie_effect: float = 0.05  # +5% к λ соперника, если в воротах запасной
    b2b_attack_effect: float = -0.03    # второй матч подряд: −3% к своей λ
    b2b_defense_effect: float = 0.03    # второй матч подряд: +3% к λ соперника
    travel_tz_effect: float = -0.01     # −1% к своей λ за каждый час смены пояса (до 4 ч)

    # --- Смешивание с рынком ---
    market_weight_model: float = 0.40   # PROVISIONAL: вес модели при логарифмическом пулинге с рынком
    default_overround: float = 1.05     # маржа, если известна только одна сторона рынка

    # --- Отбор ставок ---
    ev_min: float = 0.03                # порог EV после смешивания с рынком
    require_robust: bool = True         # EV ≥ 0 во всех стресс-сценариях
    max_bets_per_match: int = 1
    reliable_max_odds: float = 1.60     # пул «Надёжное»: кэф < 1.60
    bold_min_odds: float = 2.20         # пул «Смелое»: кэф ≥ 2.20, иначе «Главное»

    # --- Банк ---
    start_bank: float = 1000.0          # стартовый банк каждого пула
    kelly_fraction: float = 0.25
    max_stake_frac: float = 0.03        # потолок ставки, доля текущего банка пула
    min_stake: float = 5.0

    notes: dict = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None) -> "Params":
        if path is None or not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in raw.items() if k in known})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")

    def replace(self, **kw) -> "Params":
        d = asdict(self)
        d.update(kw)
        return Params(**d)
