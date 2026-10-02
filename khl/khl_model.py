"""
KHL model v2.0 — модель голов КХЛ, обучаемая на истории, с честной проверкой против рынка.

Что изменилось относительно v1 (khl_simulation_v1.py):
  * λ не задаются вручную, а оцениваются на истории: регуляризованный пуассоновский GLM
    (атака / оборона / домашний лёд / вратарь / игра «второй день подряд») с временным затуханием.
  * Поправка на концовку матча (пустые ворота): лидер +1 забивает в пустые, отстающий
    сравнивает счёт, лидер +2 забивает в пустые. Параметры подбираются по истории вместе с
    масштабом λ, поэтому голы в пустые ворота не считаются дважды. Исправляет занижение ничьих
    за 60 минут и искажение фор ±1.5 и тоталов у независимого Пуассона.
  * Овертайм/буллиты: вероятность победы хозяев зависит от силы команд (логистическая модель).
  * Распределение счёта считается точно (аналитически), Monte Carlo не нужен; неопределённость
    параметров — через выборку из апостериорного (Лапласова) распределения коэффициентов.
  * Снятие маржи, смешивание модели с рынком (логарифмический пул), EV, нижняя оценка EV,
    дробный Келли с потолком, одна ставка на матч.
  * Walk-forward бэктест: log loss / Brier против базовой модели и рынка без маржи,
    калибровка, доля ничьих, стратегия, CLV.

Команды:
  python khl_model.py selfcheck
  python khl_model.py backtest --matches matches.csv [--odds odds.csv] [--start 2025-09-01]
  python khl_model.py tune     --matches matches.csv [--start ...]
  python khl_model.py predict  --matches matches.csv --fixtures fixtures.csv [--odds odds.csv]

Формат CSV — см. README.md рядом с файлом.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, asdict, replace

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, gammaln

MODEL_VERSION = "v2.0"
K = 15  # голы одной команды до поправки на концовку: 0..14
K1 = K + 1  # после поправки: 0..15


# ----------------------------------------------------------------------------------------
# Настройки
# ----------------------------------------------------------------------------------------
@dataclass
class Config:
    half_life_days: float = 240.0  # период полураспада веса старых матчей
    l2_team: float = 8.0  # усадка атаки/обороны к среднему лиги
    l2_goalie: float = 30.0  # усадка вратарей (сильнее: выборки маленькие)
    l2_cov: float = 2.0  # усадка коэффициентов b2b
    n_param_draws: int = 200  # выборки параметров для диапазона вероятностей и EV_low
    low_quantile: float = 0.25  # консервативная оценка EV = 25-й перцентиль по неопределённости параметров
    blend_w: float = 0.5  # вес модели в пуле с рынком (1 = только модель, 0 = только рынок)
    ev_min: float = 0.03  # минимальный точечный EV
    max_gap: float = 0.05  # максимальное расхождение модели и рынка без маржи
    max_odds: float = 4.0  # длинные коэффициенты хуже откалиброваны — не берём
    kelly_fraction: float = 0.25
    stake_cap: float = 0.02  # не больше 2% банка на событие
    seed: int = 42


# ----------------------------------------------------------------------------------------
# Загрузка данных
# ----------------------------------------------------------------------------------------
MATCH_REQUIRED = ["match_id", "date", "home", "away", "home_60", "away_60"]
FIXTURE_REQUIRED = ["match_id", "date", "home", "away"]


def _parse_dates(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, format="mixed", dayfirst=True)


def _norm_ot(v) -> str:
    v = "" if pd.isna(v) else str(v).strip().upper()
    if v in ("OT", "ОТ"):
        return "OT"
    if v in ("SO", "Б", "Б.", "БУЛЛИТЫ"):
        return "SO"
    return ""


def load_matches(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in MATCH_REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path}: нет колонок {missing}")
    df["date"] = _parse_dates(df["date"])
    df["stage"] = df.get("stage", "regular")
    df["stage"] = df["stage"].fillna("regular")
    df["ot_so"] = df.get("ot_so", "").map(_norm_ot) if "ot_so" in df else ""
    for c in ("home_final", "away_final", "home_en", "away_en"):
        if c not in df:
            df[c] = np.nan
    for c in ("home_goalie", "away_goalie"):
        if c not in df:
            df[c] = None
    # матч решён в основное время -> итоговый счёт равен счёту за 60 минут
    reg = df.home_60 != df.away_60
    df.loc[reg, "home_final"] = df.loc[reg, "home_final"].fillna(df.loc[reg, "home_60"])
    df.loc[reg, "away_final"] = df.loc[reg, "away_final"].fillna(df.loc[reg, "away_60"])
    if df["match_id"].duplicated().any():
        sys.exit(f"{path}: повторяющиеся match_id: {df.loc[df.match_id.duplicated(), 'match_id'].tolist()[:5]}")
    bad = df[(df.home_60 == df.away_60) & df.home_final.notna() & (df.home_final == df.away_final)]
    if len(bad):
        sys.exit(f"{path}: ничья в итоговом счёте у {bad.match_id.tolist()[:5]} — проверьте home_final/away_final")
    return df.sort_values("date").reset_index(drop=True)


def load_fixtures(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FIXTURE_REQUIRED if c not in df.columns]
    if missing:
        sys.exit(f"{path}: нет колонок {missing}")
    df["date"] = _parse_dates(df["date"])
    df["stage"] = df.get("stage", "regular")
    df["stage"] = df["stage"].fillna("regular")
    for c in ("home_goalie", "away_goalie"):
        if c not in df:
            df[c] = None
    return df.reset_index(drop=True)


MARKET_ALIASES = {
    ("1x2", "60"): "1x2_60", ("dc", "60"): "dc_60",
    ("total", "60"): "total_60", ("total", "incl_ot"): "total_ot",
    ("handicap", "60"): "handicap_60", ("handicap", "incl_ot"): "handicap_ot",
    ("win", "incl_ot"): "win_ot", ("win_incl_ot", "incl_ot"): "win_ot", ("win_incl_ot", "60"): "win_ot",
}
MARKETS = {"1x2_60", "dc_60", "total_60", "total_ot", "handicap_60", "handicap_ot", "win_ot"}


def load_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("match_id", "market", "odds_1", "odds_2"):
        if c not in df:
            sys.exit(f"{path}: нет колонки {c}")

    def key(row):
        m = str(row["market"]).strip().lower()
        if m in MARKETS:
            return m
        p = str(row.get("period", "60")).strip().lower()
        p = "incl_ot" if p in ("incl_ot", "ot", "с от", "с от/б") else "60"
        return MARKET_ALIASES.get((m, p), m)

    df["market"] = df.apply(key, axis=1)
    unknown = set(df.market) - MARKETS
    if unknown:
        sys.exit(f"{path}: неизвестные рынки {unknown}; допустимо {sorted(MARKETS)}")
    df["line"] = pd.to_numeric(df.get("line", 0), errors="coerce").fillna(0.0)
    df["odds_X"] = pd.to_numeric(df.get("odds_X", np.nan), errors="coerce")
    df["is_closing"] = df.get("is_closing", 0)
    df["is_closing"] = df["is_closing"].map(lambda v: str(v).strip().lower() in ("1", "true", "да", "yes"))
    df["timestamp"] = _parse_dates(df["timestamp"]) if "timestamp" in df else pd.NaT
    df["bookmaker"] = df.get("bookmaker", "?")
    return df


def add_rest_features(df: pd.DataFrame) -> pd.DataFrame:
    """b2b = команда играла накануне (≤1 дня назад). Расписание известно заранее — это не утечка."""
    long = pd.concat([
        df[["match_id", "date", "home"]].rename(columns={"home": "team"}).assign(side="home"),
        df[["match_id", "date", "away"]].rename(columns={"away": "team"}).assign(side="away"),
    ]).sort_values(["team", "date"])
    long["rest"] = long.groupby("team")["date"].diff().dt.days
    long["b2b"] = (long["rest"] <= 1).astype(float)
    piv = long.pivot(index="match_id", columns="side", values="b2b")
    out = df.copy()
    out["home_b2b"] = out["match_id"].map(piv["home"]).fillna(0.0)
    out["away_b2b"] = out["match_id"].map(piv["away"]).fillna(0.0)
    return out


# ----------------------------------------------------------------------------------------
# Распределение счёта
# ----------------------------------------------------------------------------------------
_GOALS = np.arange(K)


def poisson_pmf(lam: np.ndarray) -> np.ndarray:
    lam = np.asarray(lam, dtype=float)[..., None]
    return np.exp(_GOALS * np.log(lam) - lam - gammaln(_GOALS + 1))


def base_matrix(lh: np.ndarray, la: np.ndarray) -> np.ndarray:
    """(N, K, K): P(хозяева i, гости j) при независимых Пуассонах (хвост ≥K отброшен, нормировано)."""
    P = poisson_pmf(lh)[:, :, None] * poisson_pmf(la)[:, None, :]
    return P / P.sum(axis=(1, 2), keepdims=True)


_I1 = np.arange(1, K)
_I2 = np.arange(2, K)


def late_transform(P: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Концовка матча. Из счёта с разницей 1: лидер забивает в пустые (a), отстающий
    сравнивает (b). С разницей 2: лидер забивает в пустые (c). (N,K,K) -> (N,K1,K1)."""
    N = P.shape[0]
    Q = np.zeros((N, K1, K1))
    Q[:, :K, :K] = P
    m = P[:, _I1, _I1 - 1]  # хозяева +1
    Q[:, _I1, _I1 - 1] -= (a + b) * m
    Q[:, _I1 + 1, _I1 - 1] += a * m
    Q[:, _I1, _I1] += b * m
    m = P[:, _I1 - 1, _I1]  # гости +1
    Q[:, _I1 - 1, _I1] -= (a + b) * m
    Q[:, _I1 - 1, _I1 + 1] += a * m
    Q[:, _I1, _I1] += b * m
    m = P[:, _I2, _I2 - 2]  # хозяева +2
    Q[:, _I2, _I2 - 2] -= c * m
    Q[:, _I2 + 1, _I2 - 2] += c * m
    m = P[:, _I2 - 2, _I2]  # гости +2
    Q[:, _I2 - 2, _I2] -= c * m
    Q[:, _I2 - 2, _I2 + 1] += c * m
    return Q


# Матрицы перехода «ячейка счёта -> значение разницы / тотала» для векторного подсчёта.
_H, _A = np.indices((K1, K1))
_DIFF_VALUES = np.arange(-(K1 - 1), K1)
_TOT_VALUES = np.arange(0, 2 * K1)
_M_DIFF = np.zeros((K1 * K1, len(_DIFF_VALUES)))
_M_DIFF[np.arange(K1 * K1), (_H - _A).ravel() + K1 - 1] = 1
_M_TOT = np.zeros((K1 * K1, len(_TOT_VALUES)))
_M_TOT[np.arange(K1 * K1), (_H + _A).ravel()] = 1
_IS_DRAW = (_H == _A).ravel().astype(float)
_HOME_WIN = (_H > _A).ravel().astype(float)
_AWAY_WIN = (_H < _A).ravel().astype(float)
# тотал с ОТ/Б: при ничьей за 60 минут итоговый счёт +1 (решающий гол или буллит)
_M_TOT_OT = np.zeros_like(_M_TOT)
_M_TOT_OT[np.arange(K1 * K1), (_H + _A).ravel() + (_H == _A).ravel()] = 1


def distributions(F: np.ndarray, p_ot: np.ndarray) -> dict:
    """F: (D,K1,K1) — счёт за 60 минут, p_ot: (D,) — P(хозяева выигрывают ОТ/Б | ничья)."""
    D = F.shape[0]
    flat = F.reshape(D, -1)
    p1, px, p2 = flat @ _HOME_WIN, flat @ _IS_DRAW, flat @ _AWAY_WIN
    diff60 = flat @ _M_DIFF
    diff_ot = diff60.copy()
    zero = K1 - 1
    diff_ot[:, zero] = 0.0
    diff_ot[:, zero + 1] += px * p_ot
    diff_ot[:, zero - 1] += px * (1 - p_ot)
    return {
        "F": F, "p1": p1, "px": px, "p2": p2, "p_ot": p_ot,
        "win1_ot": p1 + px * p_ot, "win2_ot": p2 + px * (1 - p_ot),
        "diff60": diff60, "diff_ot": diff_ot,
        "total60": flat @ _M_TOT, "total_ot": flat @ _M_TOT_OT,
    }


# ----------------------------------------------------------------------------------------
# Контракты: выигрыш / половина / возврат / половина проигрыша / проигрыш
# ----------------------------------------------------------------------------------------
CATS = ("win", "half_win", "push", "half_loss", "loss")


def _line_cat_matrix(values: np.ndarray, threshold: float, over: bool) -> np.ndarray:
    """(n_values, 5). Ставка «больше threshold» (или «меньше»). Четвертные линии делятся на две половины."""
    q = threshold * 4
    halves = [threshold - 0.25, threshold + 0.25] if abs(q - round(q)) < 1e-9 and int(round(q)) % 2 == 1 else [threshold]
    score = np.zeros(len(values))
    for t in halves:
        s = np.sign(values - t) if over else np.sign(t - values)
        score += s / len(halves)
    M = np.zeros((len(values), 5))
    for k, target in enumerate((1.0, 0.5, 0.0, -0.5, -1.0)):
        M[:, k] = np.isclose(score, target)
    return M


def payoff(odds: float) -> np.ndarray:
    return np.array([odds - 1, (odds - 1) / 2, 0.0, -0.5, -1.0])


def is_quarter(line: float) -> bool:
    q = line * 4
    return abs(q - round(q)) < 1e-9 and int(round(q)) % 2 == 1


def devig(odds: list[float]) -> np.ndarray:
    inv = 1 / np.asarray(odds, dtype=float)
    return inv / inv.sum()


def log_pool(p: np.ndarray, q: np.ndarray | None, w: float) -> np.ndarray:
    """Логарифмический пул: p^w * q^(1-w), нормировка по последней оси."""
    if q is None or w >= 1:
        return p
    z = w * np.log(np.clip(p, 1e-12, 1)) + (1 - w) * np.log(np.clip(q, 1e-12, 1))
    z = np.exp(z - z.max(axis=-1, keepdims=True))
    return z / z.sum(axis=-1, keepdims=True)


def contract_cats(dist: dict, market: str, line: float, side: str,
                  q: np.ndarray | None = None, w: float = 1.0) -> np.ndarray:
    """Вероятности категорий (D,5) для контракта с учётом пула с рынком q (без маржи).
    side: 1x2 -> '1','X','2'; dc -> '1X','12','X2'; win_ot -> '1','2';
          total -> 'over','under'; handicap -> 'home','away' (line — фора хозяев)."""
    D = dist["p1"].shape[0]
    out = np.zeros((D, 5))
    if market in ("1x2_60", "dc_60"):
        p3 = log_pool(np.stack([dist["p1"], dist["px"], dist["p2"]], -1), q, w)
        pick = {"1": [0], "X": [1], "2": [2], "1X": [0, 1], "12": [0, 2], "X2": [1, 2]}[side]
        out[:, 0] = p3[:, pick].sum(-1)
        out[:, 4] = 1 - out[:, 0]
        return out
    if market == "win_ot":
        p2w = log_pool(np.stack([dist["win1_ot"], dist["win2_ot"]], -1), q, w)
        out[:, 0] = p2w[:, 0] if side == "1" else p2w[:, 1]
        out[:, 4] = 1 - out[:, 0]
        return out
    if market.startswith("total"):
        probs, values = dist["total60" if market == "total_60" else "total_ot"], _TOT_VALUES
        over_c, under_c = _line_cat_matrix(values, line, True), _line_cat_matrix(values, line, False)
    else:  # handicap: хозяева выигрывают, если diff + line > 0, т.е. diff > -line
        probs, values = dist["diff60" if market == "handicap_60" else "diff_ot"], _DIFF_VALUES
        over_c, under_c = _line_cat_matrix(values, -line, True), _line_cat_matrix(values, -line, False)
    cats_over, cats_under = probs @ over_c, probs @ under_c
    if q is not None and w < 1 and not is_quarter(line):
        # пул по условным (без возврата) вероятностям, затем обратно
        push = cats_over[:, 2]
        nz = np.clip(1 - push, 1e-12, None)
        pc = log_pool(np.stack([cats_over[:, 0] / nz, cats_under[:, 0] / nz], -1), q, w)
        cats_over = np.stack([pc[:, 0] * nz, 0 * push, push, 0 * push, pc[:, 1] * nz], -1)
        cats_under = np.stack([pc[:, 1] * nz, 0 * push, push, 0 * push, pc[:, 0] * nz], -1)
    return cats_over if side in ("over", "home") else cats_under


def settle(market: str, line: float, side: str, h60: int, a60: int, hf: float, af: float) -> int | None:
    """Индекс категории CATS по факту матча; None — нельзя рассчитать."""
    if market in ("1x2_60", "dc_60"):
        res = "1" if h60 > a60 else "2" if h60 < a60 else "X"
        return 0 if res in side else 4
    if market.endswith("_ot") or market == "win_ot":
        if pd.isna(hf) or pd.isna(af):
            return None
        h, a = hf, af
    else:
        h, a = h60, a60
    if market == "win_ot":
        return 0 if (h > a) == (side == "1") else 4
    if market.startswith("total"):
        M = _line_cat_matrix(np.array([h + a]), line, side == "over")
    else:
        M = _line_cat_matrix(np.array([h - a]), -line, side == "home")
    return int(M[0].argmax())


def kelly(cats: np.ndarray, odds: float) -> float:
    r = payoff(odds)
    if cats @ r <= 0:
        return 0.0
    f = minimize_scalar(lambda f: -(cats * np.log1p(np.clip(f * r, -0.999999, None))).sum(),
                        bounds=(0, 0.99), method="bounded")
    return float(f.x)


# ----------------------------------------------------------------------------------------
# Модель
# ----------------------------------------------------------------------------------------
class KHLModel:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    # --- дизайн-матрица: строка = (атакующая команда, обороняющаяся команда) ---
    def _design(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        n, P = len(df), self.n_params
        Xh, Xa = np.zeros((n, P)), np.zeros((n, P))
        Xh[:, 0] = Xa[:, 0] = 1.0
        Xh[:, 1] = 1.0
        T, G = len(self.teams), len(self.goalies)
        for r, (h, a, hg, ag) in enumerate(zip(df.home, df.away, df.home_goalie, df.away_goalie)):
            if h in self.team_idx:
                Xh[r, 2 + self.team_idx[h]] = 1.0
                Xa[r, 2 + T + self.team_idx[h]] = -1.0
            if a in self.team_idx:
                Xa[r, 2 + self.team_idx[a]] = 1.0
                Xh[r, 2 + T + self.team_idx[a]] = -1.0
            if ag in self.goalie_idx:
                Xh[r, 2 + 2 * T + self.goalie_idx[ag]] = -1.0
            if hg in self.goalie_idx:
                Xa[r, 2 + 2 * T + self.goalie_idx[hg]] = -1.0
        b = 2 + 2 * T + G
        Xh[:, b], Xh[:, b + 1] = df.home_b2b.values, df.away_b2b.values
        Xa[:, b], Xa[:, b + 1] = df.away_b2b.values, df.home_b2b.values
        return Xh, Xa

    def fit(self, train: pd.DataFrame, as_of: pd.Timestamp) -> "KHLModel":
        cfg = self.cfg
        train = train[train.date < as_of]
        if len(train) < 100:
            raise ValueError(f"мало матчей для обучения: {len(train)}")
        self.as_of = as_of
        self.teams = sorted(set(train.home) | set(train.away))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        gk = pd.concat([train.home_goalie, train.away_goalie]).dropna()
        self.goalies = sorted(gk.unique())
        self.goalie_idx = {g: i for i, g in enumerate(self.goalies)}
        T, G = len(self.teams), len(self.goalies)
        self.n_params = 2 + 2 * T + G + 2
        pen = np.r_[0.0, 0.0, np.full(2 * T, cfg.l2_team), np.full(G, cfg.l2_goalie), np.full(2, cfg.l2_cov)]

        Xh, Xa = self._design(train)
        X = np.vstack([Xh, Xa])
        # голы без пустых ворот, если они известны: концовку добавит late_transform
        yh = train.home_60 - train.home_en.fillna(0)
        ya = train.away_60 - train.away_en.fillna(0)
        y = np.r_[yh.values, ya.values].astype(float)
        age = (as_of - train.date).dt.days.values
        w1 = 0.5 ** (age / cfg.half_life_days)
        w = np.r_[w1, w1]

        beta = np.zeros(self.n_params)
        beta[0] = math.log(max(np.average(y, weights=w), 1e-3))
        for _ in range(100):
            lam = np.exp(X @ beta)
            g = X.T @ (w * (y - lam)) - pen * beta
            H = (X * (w * lam)[:, None]).T @ X + np.diag(pen)
            step = np.linalg.solve(H, g)
            mx = np.abs(step).max()
            beta += step if mx < 1 else step / mx
            if mx < 1e-9:
                break
        lam = np.exp(X @ beta)
        H = (X * (w * lam)[:, None]).T @ X + np.diag(pen)
        self.beta, self.cov = beta, np.linalg.inv(H)
        self.chol = np.linalg.cholesky((self.cov + self.cov.T) / 2 + 1e-12 * np.eye(self.n_params))

        lh, la = np.exp(Xh @ beta), np.exp(Xa @ beta)
        self._fit_late(lh, la, train.home_60.values, train.away_60.values, w1)
        self._fit_ot(lh, la, train, w1)
        return self

    def _fit_late(self, lh, la, h60, a60, w):
        hi, ai = np.clip(h60, 0, K1 - 1).astype(int), np.clip(a60, 0, K1 - 1).astype(int)
        idx = np.arange(len(lh))

        def unpack(t):
            e = np.exp([0.0, t[1], t[2]])
            return math.exp(t[0]), e[1] / e.sum(), e[2] / e.sum(), expit(t[3])

        def nll(t):
            s, a, b, c = unpack(t)
            F = late_transform(base_matrix(s * lh, s * la), a, b, c)
            return -np.sum(w * np.log(np.clip(F[idx, hi, ai], 1e-300, None)))

        t0 = np.array([0.0, math.log(0.12 / 0.78), math.log(0.10 / 0.78), math.log(0.15 / 0.85)])
        res = minimize(nll, t0, method="Nelder-Mead", options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
        self.scale, self.a_en, self.b_eq, self.c_en2 = unpack(res.x)

    def _fit_ot(self, lh, la, train, w):
        m = ((train.home_60 == train.away_60) & train.home_final.notna()).values
        self.ot_alpha, self.ot_beta = 0.0, 0.0
        if m.sum() < 30:
            return
        x = np.log(lh[m] / la[m])
        y = (train.home_final.values[m] > train.away_final.values[m]).astype(float)
        ww = w[m]

        def nll(t):
            p = np.clip(expit(t[0] + t[1] * x), 1e-12, 1 - 1e-12)
            return -np.sum(ww * (y * np.log(p) + (1 - y) * np.log(1 - p))) + 0.5 * (0.1 * t[0] ** 2 + 1.0 * t[1] ** 2)

        self.ot_alpha, self.ot_beta = minimize(nll, [0.0, 0.0], method="BFGS").x

    # --- прогноз ---
    def lambdas(self, fx: pd.DataFrame, draws: int = 0, rng=None) -> tuple[np.ndarray, np.ndarray]:
        """(D, n): D = 1 + draws; строка 0 — точечная оценка."""
        Xh, Xa = self._design(fx)
        betas = self.beta[None, :]
        if draws:
            z = rng.standard_normal((draws, self.n_params))
            betas = np.vstack([betas, self.beta + z @ self.chol.T])
        return np.exp(betas @ Xh.T), np.exp(betas @ Xa.T)

    def match_dist(self, lh: np.ndarray, la: np.ndarray, stage: str = "regular") -> dict:
        """lh, la: (D,) базовые λ одного матча по выборкам параметров."""
        F = late_transform(base_matrix(self.scale * lh, self.scale * la), self.a_en, self.b_eq, self.c_en2)
        p_ot = expit(self.ot_alpha + self.ot_beta * np.log(lh / la))
        if stage != "regular":
            p_ot = 0.5 + (p_ot - 0.5) * 0.5  # плей-офф: модель ОТ регулярки не переносим, сжимаем к 50%
        return distributions(F, p_ot)

    def params_summary(self) -> dict:
        T = len(self.teams)
        att = dict(zip(self.teams, np.round(self.beta[2:2 + T], 3)))
        dfn = dict(zip(self.teams, np.round(self.beta[2 + T:2 + 2 * T], 3)))
        return {
            "version": MODEL_VERSION, "as_of": str(self.as_of.date()),
            "mu": round(float(self.beta[0]), 4), "home": round(float(self.beta[1]), 4),
            "b2b_att": round(float(self.beta[-2]), 4), "b2b_def": round(float(self.beta[-1]), 4),
            "scale": round(float(self.scale), 4), "p_en_lead1": round(float(self.a_en), 4),
            "p_equalize": round(float(self.b_eq), 4), "p_en_lead2": round(float(self.c_en2), 4),
            "ot_alpha": round(float(self.ot_alpha), 4), "ot_beta": round(float(self.ot_beta), 4),
            "attack": att, "defence": dfn, "n_goalies": len(self.goalies),
        }


# ----------------------------------------------------------------------------------------
# Котировки и отбор ставок
# ----------------------------------------------------------------------------------------
SIDES = {
    "1x2_60": [("1", "odds_1"), ("X", "odds_X"), ("2", "odds_2")],
    "dc_60": [("1X", "odds_1"), ("12", "odds_X"), ("X2", "odds_2")],
    "win_ot": [("1", "odds_1"), ("2", "odds_2")],
    "total_60": [("over", "odds_1"), ("under", "odds_2")],
    "total_ot": [("over", "odds_1"), ("under", "odds_2")],
    "handicap_60": [("home", "odds_1"), ("away", "odds_2")],
    "handicap_ot": [("home", "odds_1"), ("away", "odds_2")],
}


def pick_snapshot(odds_m: pd.DataFrame, closing: bool, cutoff=None) -> pd.DataFrame:
    """Последний срез по (БК, рынок, линия): предматчевый (closing=False) или закрытие."""
    d = odds_m[odds_m.is_closing == closing]
    if cutoff is not None and not closing and d.timestamp.notna().any():
        d = d[d.timestamp.isna() | (d.timestamp <= cutoff)]
    if d.empty:
        return d
    return d.sort_values("timestamp").groupby(["bookmaker", "market", "line"], as_index=False).tail(1)


def market_q(row) -> np.ndarray | None:
    """Вероятности рынка без маржи для пула; None для двойного шанса (не разбиение)."""
    if row.market == "dc_60":
        return None
    if row.market == "1x2_60":
        if pd.isna(row.odds_X):
            return None
        return devig([row.odds_1, row.odds_X, row.odds_2])
    return devig([row.odds_1, row.odds_2])


def evaluate_candidates(dist: dict, odds_rows: pd.DataFrame, cfg: Config,
                        q_1x2: np.ndarray | None) -> list[dict]:
    out = []
    for row in odds_rows.itertuples(index=False):
        q = market_q(row)
        overround = None
        if row.market == "1x2_60" and not pd.isna(row.odds_X):
            overround = float((1 / np.array([row.odds_1, row.odds_X, row.odds_2])).sum() - 1)
        elif row.market != "dc_60":
            overround = float(1 / row.odds_1 + 1 / row.odds_2 - 1)
        for k, (side, col) in enumerate(SIDES[row.market]):
            odds = getattr(row, col)
            if pd.isna(odds) or odds <= 1:
                continue
            qq = q_1x2 if row.market == "dc_60" else q
            cats_model = contract_cats(dist, row.market, row.line, side)
            cats = contract_cats(dist, row.market, row.line, side, qq, cfg.blend_w)
            r = payoff(odds)
            ev_all = cats @ r
            ev_point, ev_low = float(ev_all[0]), float(np.quantile(ev_all[1:], cfg.low_quantile)) if len(ev_all) > 1 else float(ev_all[0])
            pw, pp = float(cats_model[0, 0] + cats_model[0, 1]), float(cats_model[0, 2])
            p_model_c = pw / (1 - pp) if pp < 1 else float("nan")
            q_side = None
            if row.market == "dc_60" and q_1x2 is not None:
                q_side = float(q_1x2[{"1X": [0, 1], "12": [0, 2], "X2": [1, 2]}[side]].sum())
            elif q is not None:
                q_side = float(q[k])
            gap = p_model_c - q_side if q_side is not None else None
            reasons = []
            if ev_point < cfg.ev_min:
                reasons.append(f"EV {ev_point:+.1%} < {cfg.ev_min:.0%}")
            if ev_low < 0:
                reasons.append(f"EV_low {ev_low:+.1%} < 0")
            if gap is not None and abs(gap) > cfg.max_gap:
                reasons.append(f"расхождение с рынком {gap:+.1%} > {cfg.max_gap:.0%} — проверить данные")
            if odds > cfg.max_odds:
                reasons.append(f"кэф {odds} > {cfg.max_odds}")
            if is_quarter(row.line):
                reasons.append("четвертная линия: без пула с рынком")
            c0 = cats[0]
            f = kelly(c0, odds) * cfg.kelly_fraction
            out.append({
                "bookmaker": row.bookmaker, "market": row.market, "line": row.line, "side": side,
                "odds": float(odds), "timestamp": row.timestamp,
                "p_win": float(c0[0] + c0[1]), "p_push": float(c0[2]), "p_loss": float(c0[3] + c0[4]),
                "p_model_cond": p_model_c, "q_market": q_side, "gap": gap, "overround": overround,
                "fair": float((1 - c0[2]) / (c0[0] + c0[1])) if c0[0] + c0[1] > 0 else float("inf"),
                "ev": ev_point, "ev_low": ev_low, "kelly_stake": min(f, cfg.stake_cap),
                "pass": not reasons and not is_quarter(row.line), "reasons": "; ".join(reasons),
            })
    return out


def choose_one(cands: list[dict]) -> dict | None:
    ok = [c for c in cands if c["pass"]]
    return max(ok, key=lambda c: (c["ev_low"], c["ev"])) if ok else None


# ----------------------------------------------------------------------------------------
# Метрики
# ----------------------------------------------------------------------------------------
def logloss3(P: np.ndarray, y: np.ndarray) -> float:
    return float(-np.mean(np.log(np.clip(P[np.arange(len(y)), y], 1e-12, None))))


def brier3(P: np.ndarray, y: np.ndarray) -> float:
    Y = np.eye(3)[y]
    return float(np.mean(((P - Y) ** 2).sum(1)))


def central_interval(probs: np.ndarray, values: np.ndarray, level=0.8) -> tuple[int, int, float]:
    cdf = np.cumsum(probs)
    tail = (1 - level) / 2
    lo = values[np.searchsorted(cdf, tail, side="right")]
    hi = values[np.searchsorted(cdf, 1 - tail, side="left")]
    cov = probs[(values >= lo) & (values <= hi)].sum()
    return int(lo), int(hi), float(cov)


def max_drawdown(pnl: np.ndarray) -> float:
    eq = np.cumsum(pnl)
    return float((np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()) if len(eq) else 0.0


# ----------------------------------------------------------------------------------------
# Walk-forward бэктест
# ----------------------------------------------------------------------------------------
def walk_forward(matches: pd.DataFrame, cfg: Config, start=None, refit_days=7, odds=None, verbose=True,
                 draws: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    matches = add_rest_features(matches)
    rng = np.random.default_rng(cfg.seed)
    draws = cfg.n_param_draws if draws is None else draws
    if start is None:
        seasons = matches.season.unique() if "season" in matches else []
        start = matches[matches.season == seasons[1]].date.min() if len(seasons) > 1 else matches.date.iloc[len(matches) // 2]
    start = pd.Timestamp(start)
    preds, bets = [], []
    t = start
    end = matches.date.max()
    while t <= end:
        t_next = t + pd.Timedelta(days=refit_days)
        test = matches[(matches.date >= t) & (matches.date < t_next)]
        if len(test):
            model = KHLModel(cfg).fit(matches, as_of=t)
            base = matches[matches.date < t]
            freq = np.array([(base.home_60 > base.away_60).mean(), (base.home_60 == base.away_60).mean(),
                             (base.home_60 < base.away_60).mean()])
            LH, LA = model.lambdas(test, draws=draws if odds is not None else 0, rng=rng)
            for j, m in enumerate(test.itertuples(index=False)):
                dist = model.match_dist(LH[:, j], LA[:, j], m.stage)
                y = 0 if m.home_60 > m.away_60 else 1 if m.home_60 == m.away_60 else 2
                rec = {"match_id": m.match_id, "date": m.date, "home": m.home, "away": m.away,
                       "lam_h": model.scale * LH[0, j], "lam_a": model.scale * LA[0, j],
                       "p1": dist["p1"][0], "px": dist["px"][0], "p2": dist["p2"][0],
                       "b1": freq[0], "bx": freq[1], "b2": freq[2], "y": y,
                       "p_over55": float(dist["total60"][0] @ (_TOT_VALUES > 5.5)),
                       "exp_total": float(dist["total60"][0] @ _TOT_VALUES),
                       "over55": int(m.home_60 + m.away_60 > 5.5), "total60": int(m.home_60 + m.away_60)}
                if odds is not None:
                    om = odds[odds.match_id == m.match_id]
                    close = pick_snapshot(om, True)
                    c1x2 = close[close.market == "1x2_60"].dropna(subset=["odds_X"])
                    if len(c1x2):
                        r0 = c1x2.iloc[0]
                        rec["q1"], rec["qx"], rec["q2"] = devig([r0.odds_1, r0.odds_X, r0.odds_2])
                    entry = pick_snapshot(om, False)
                    entry_is_close = entry.empty
                    if entry_is_close:
                        entry = close
                    e1x2 = entry[entry.market == "1x2_60"].dropna(subset=["odds_X"])
                    q1x2 = devig(e1x2.iloc[0][["odds_1", "odds_X", "odds_2"]].astype(float)) if len(e1x2) else None
                    best = choose_one(evaluate_candidates(dist, entry, cfg, q1x2))
                    if best:
                        cat = settle(best["market"], best["line"], best["side"], m.home_60, m.away_60,
                                     m.home_final, m.away_final)
                        if cat is not None:
                            r = payoff(best["odds"])[cat]
                            clv = None
                            cl = close[(close.market == best["market"]) & (close.line == best["line"]) &
                                       (close.bookmaker == best["bookmaker"])]
                            if len(cl) and not entry_is_close:
                                col = dict(SIDES[best["market"]])[best["side"]]
                                k_close = cl.iloc[0][col]
                                if not pd.isna(k_close):
                                    clv = best["odds"] / k_close - 1
                            bets.append({**best, "match_id": m.match_id, "date": m.date, "result": CATS[cat],
                                         "pnl_flat": r, "pnl_kelly": r * best["kelly_stake"], "clv": clv,
                                         "entry_is_closing": entry_is_close})
                preds.append(rec)
            if verbose:
                print(f"  {t.date()}: обучено на {len(matches[matches.date < t])}, прогнозов {len(test)}", file=sys.stderr)
        t = t_next
    return pd.DataFrame(preds), pd.DataFrame(bets)


def report(preds: pd.DataFrame, bets: pd.DataFrame, cfg: Config) -> str:
    L = []
    y = preds.y.values
    P = preds[["p1", "px", "p2"]].values
    B = preds[["b1", "bx", "b2"]].values
    L.append(f"Матчей в проверке: {len(preds)}  ({preds.date.min().date()} … {preds.date.max().date()})")
    L.append("\n1X2 за 60 минут (меньше — лучше):")
    L.append(f"  модель          log loss {logloss3(P, y):.4f}   Brier(0–2) {brier3(P, y):.4f}")
    L.append(f"  частоты лиги    log loss {logloss3(B, y):.4f}   Brier(0–2) {brier3(B, y):.4f}")
    if "q1" in preds and preds.q1.notna().any():
        mk = preds.dropna(subset=["q1"])
        Q, ym, Pm = mk[["q1", "qx", "q2"]].values, mk.y.values, mk[["p1", "px", "p2"]].values
        L.append(f"  рынок (закр.)   log loss {logloss3(Q, ym):.4f}   (n={len(mk)})")
        L.append(f"  модель на тех же log loss {logloss3(Pm, ym):.4f}")
        best_w, best_ll = 1.0, 1e9
        for w in np.linspace(0, 1, 21):
            ll = logloss3(log_pool(Pm, Q, w), ym)
            if ll < best_ll:
                best_w, best_ll = w, ll
        L.append(f"  лучший вес модели в пуле w={best_w:.2f} (log loss {best_ll:.4f}); "
                 f"оценка оптимистична — подобрана на этой же выборке")
        if logloss3(Pm, ym) > logloss3(Q, ym):
            L.append("  ВНИМАНИЕ: модель хуже закрытия рынка. Ставить по ней без пула с рынком нельзя.")
    L.append("\nДоля ничьих за 60 минут: прогноз {:.3f}, факт {:.3f}".format(preds.px.mean(), (y == 1).mean()))
    L.append("Средний тотал 60 мин: прогноз {:.2f}, факт {:.2f}".format(preds.exp_total.mean(), preds.total60.mean()))
    po = preds.p_over55.values
    L.append("ТБ 5.5: log loss модель {:.4f} / частота {:.4f}".format(
        -np.mean(preds.over55 * np.log(po) + (1 - preds.over55) * np.log(1 - po)),
        -np.mean(preds.over55 * np.log(preds.over55.mean()) + (1 - preds.over55) * np.log(1 - preds.over55.mean()))))
    L.append("\nКалибровка П1 (60 мин):  диапазон | n | прогноз | факт")
    bins = np.arange(0, 1.01, 0.1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (preds.p1 >= lo) & (preds.p1 < hi)
        if m.sum() >= 10:
            L.append(f"  {lo:.1f}–{hi:.1f} | {m.sum():4d} | {preds.p1[m].mean():.3f} | {(y[m] == 0).mean():.3f}")
    if len(bets):
        settled = bets[bets.result != "push"]
        n = len(bets)
        L.append(f"\nСтратегия (одна ставка на матч, фильтр EV≥{cfg.ev_min:.0%}, EV_low≥0, |p−q|≤{cfg.max_gap:.0%}, w={cfg.blend_w}):")
        L.append(f"  ставок {n}; выигрыш {(bets.result == 'win').sum()}, проигрыш {(bets.result == 'loss').sum()}, "
                 f"прочее {n - (bets.result == 'win').sum() - (bets.result == 'loss').sum()}")
        y_flat = bets.pnl_flat.sum() / n
        # доверительный интервал с кластеризацией по дню
        daily = bets.groupby(bets.date.dt.date).pnl_flat.agg(["sum", "count"]).values
        idx = np.random.default_rng(0).integers(0, len(daily), (2000, len(daily)))
        bs = daily[idx, 0].sum(1) / daily[idx, 1].sum(1)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        L.append(f"  yield (плоская ставка) {y_flat:+.2%}, 95% ДИ [{lo:+.1%}; {hi:+.1%}]  — если ДИ включает 0, преимущество не доказано")
        L.append(f"  средний заявленный EV {bets.ev.mean():+.2%}; P&L по Келли {bets.pnl_kelly.sum():+.3f} банка; "
                 f"макс. просадка (плоско) {max_drawdown(bets.pnl_flat.values):.1f} ставок")
        if bets.clv.notna().any():
            L.append(f"  CLV средний {bets.clv.mean():+.2%} (n={bets.clv.notna().sum()}); доля с CLV>0 {(bets.clv > 0).mean():.0%}")
        if bets.entry_is_closing.any():
            L.append(f"  ВНИМАНИЕ: {bets.entry_is_closing.sum()} ставок рассчитаны по закрывающим кэфам — "
                     f"результат оптимистичен, CLV для них не определён")
        L.append("  по рынкам: " + ", ".join(f"{k}: {len(g)} шт, {g.pnl_flat.sum() / len(g):+.1%}"
                                             for k, g in bets.groupby("market")))
    return "\n".join(L)


# ----------------------------------------------------------------------------------------
# Самопроверка: аналитика против Monte Carlo того же процесса
# ----------------------------------------------------------------------------------------
def selfcheck(n=400_000, seed=7) -> str:
    rng = np.random.default_rng(seed)
    lh, la, a, b, c, p_ot = 3.0, 2.2, 0.12, 0.09, 0.14, 0.55
    F = late_transform(base_matrix(np.array([lh]), np.array([la])), a, b, c)
    d = distributions(F, np.array([p_ot]))
    h, g = rng.poisson(lh, n), rng.poisson(la, n)
    u = rng.random(n)
    diff = h - g
    h = h + ((diff == 1) & (u < a)) + ((diff == -1) & (u >= a) & (u < a + b)) + ((diff == 2) & (u < c))
    g = g + ((diff == -1) & (u < a)) + ((diff == 1) & (u >= a) & (u < a + b)) + ((diff == -2) & (u < c))
    tot = h + g
    win_ot = (h > g) | ((h == g) & (rng.random(n) < p_ot))
    checks = {
        "П1": (d["p1"][0], np.mean(h > g)), "Х": (d["px"][0], np.mean(h == g)), "П2": (d["p2"][0], np.mean(h < g)),
        "ТБ5.5": (d["total60"][0] @ (_TOT_VALUES > 5.5), np.mean(tot > 5.5)),
        "Ф1(-1.5)": (contract_cats(d, "handicap_60", -1.5, "home")[0, 0], np.mean(h - g >= 2)),
        "Поб1 ОТ/Б": (d["win1_ot"][0], np.mean(win_ot)),
        "ТБ4.5 с ОТ": (d["total_ot"][0] @ (_TOT_VALUES > 4.5), np.mean(tot + (h == g) > 4.5)),
        "Ф1(-1) с ОТ, возврат": (contract_cats(d, "handicap_ot", -1.0, "home")[0, 2],
                                 np.mean((h - g == 1) | ((h == g) & win_ot))),
    }
    lines, ok = [], True
    for k, (an, mc) in checks.items():
        se = math.sqrt(mc * (1 - mc) / n)
        good = abs(an - mc) < 4 * se
        ok &= good
        lines.append(f"  {k:20s} аналитика {an:.4f}  MC {mc:.4f}  |Δ|/SE {abs(an - mc) / se:.1f}  {'OK' if good else 'FAIL'}")
    s = F.sum()
    ok &= abs(s - 1) < 1e-6
    lines.append(f"  сумма вероятностей {s:.8f}")
    # контракты: сумма категорий = 1, четвертная линия = среднее двух половин
    c_q = contract_cats(d, "total_60", 5.25, "over")[0]
    c_a, c_b = contract_cats(d, "total_60", 5.0, "over")[0], contract_cats(d, "total_60", 5.5, "over")[0]
    ev_q, ev_ab = c_q @ payoff(1.9), 0.5 * (c_a @ payoff(1.9) + c_b @ payoff(1.9))
    ok &= abs(ev_q - ev_ab) < 1e-9 and abs(c_q.sum() - 1) < 1e-9
    lines.append(f"  ТБ5.25: EV {ev_q:+.5f} = среднее ТБ5 и ТБ5.5 {ev_ab:+.5f}")
    return ("САМОПРОВЕРКА ПРОЙДЕНА\n" if ok else "САМОПРОВЕРКА НЕ ПРОЙДЕНА\n") + "\n".join(lines)


# ----------------------------------------------------------------------------------------
# Прогноз на будущие матчи
# ----------------------------------------------------------------------------------------
def predict(matches: pd.DataFrame, fixtures: pd.DataFrame, odds: pd.DataFrame | None, cfg: Config,
            cutoff: pd.Timestamp, bank: float) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    allm = add_rest_features(pd.concat([matches, fixtures], ignore_index=True))
    fx = allm[allm.match_id.isin(fixtures.match_id)].reset_index(drop=True)
    model = KHLModel(cfg).fit(allm[allm.match_id.isin(matches.match_id)], as_of=fx.date.min())
    rng = np.random.default_rng(cfg.seed)
    LH, LA = model.lambdas(fx, draws=cfg.n_param_draws, rng=rng)
    rows, bet_rows = [], []
    lq, hq = 0.10, 0.90  # диапазон вероятностей от неопределённости параметров
    for j, m in enumerate(fx.itertuples(index=False)):
        d = model.match_dist(LH[:, j], LA[:, j], m.stage)
        Fm = d["F"][0]
        top = np.argsort(-Fm.ravel())[:3]
        lo, hi, cov = central_interval(d["total60"][0], _TOT_VALUES)
        notes = []
        for t in (m.home, m.away):
            if t not in model.team_idx:
                notes.append(f"{t}: нет в истории — средний уровень лиги")
        for side, gk in (("хозяева", m.home_goalie), ("гости", m.away_goalie)):
            if gk is None or pd.isna(gk):
                notes.append(f"вратарь ({side}) не указан — сценарий без поправки на вратаря")
            elif gk not in model.goalie_idx:
                notes.append(f"вратарь {gk}: нет в истории — средний уровень")
        rec = {
            "match_id": m.match_id, "date": m.date.date(), "home": m.home, "away": m.away,
            "cutoff": str(cutoff), "version": MODEL_VERSION, "seed": cfg.seed, "param_draws": cfg.n_param_draws,
            "lam_h": round(model.scale * LH[0, j], 3), "lam_a": round(model.scale * LA[0, j], 3),
            "p1": d["p1"][0], "px": d["px"][0], "p2": d["p2"][0],
            "p1_range": f"{np.quantile(d['p1'][1:], lq):.3f}–{np.quantile(d['p1'][1:], hq):.3f}",
            "p2_range": f"{np.quantile(d['p2'][1:], lq):.3f}–{np.quantile(d['p2'][1:], hq):.3f}",
            "win1_ot": d["win1_ot"][0], "win2_ot": d["win2_ot"][0],
            "p_over_4.5": float(d["total60"][0] @ (_TOT_VALUES > 4.5)),
            "p_over_5.5": float(d["total60"][0] @ (_TOT_VALUES > 5.5)),
            "p_over_6.5": float(d["total60"][0] @ (_TOT_VALUES > 6.5)),
            "p_h1_-1.5": float(d["diff60"][0] @ (_DIFF_VALUES >= 2)),
            "p_h2_-1.5": float(d["diff60"][0] @ (_DIFF_VALUES <= -2)),
            "top3": ", ".join(f"{i // K1}:{i % K1} ({Fm.ravel()[i]:.1%})" for i in top),
            "total_interval": f"[{lo}, {hi}] покрытие {cov:.0%}",
            "notes": "; ".join(notes),
        }
        assert abs(rec["p1"] + rec["px"] + rec["p2"] - 1) < 1e-6
        rows.append(rec)
        if odds is not None:
            om = odds[odds.match_id == m.match_id]
            entry = pick_snapshot(om, False, cutoff)
            if entry.empty:
                bet_rows.append({"match_id": m.match_id, "decision": "НЕТ ЛИНИИ"})
                continue
            e1 = entry[entry.market == "1x2_60"].dropna(subset=["odds_X"])
            q1 = devig(e1.iloc[0][["odds_1", "odds_X", "odds_2"]].astype(float)) if len(e1) else None
            cands = evaluate_candidates(d, entry, cfg, q1)
            best = choose_one(cands)
            for c in cands:
                chosen = best is not None and c is best
                bet_rows.append({"match_id": m.match_id, "home": m.home, "away": m.away, **c,
                                 "decision": "ВИРТУАЛЬНЫЙ ВЫБОР" if chosen else ("НАБЛЮДЕНИЕ" if c["pass"] else "ПРОПУСК"),
                                 "stake": round(c["kelly_stake"] * bank, 2) if chosen else 0.0,
                                 "notes": rec["notes"]})
    return pd.DataFrame(rows), pd.DataFrame(bet_rows), model.params_summary()


# ----------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------
def _cfg_from_args(a) -> Config:
    cfg = Config()
    for k in asdict(cfg):
        v = getattr(a, k, None)
        if v is not None:
            cfg = replace(cfg, **{k: type(getattr(cfg, k))(v)})
    return cfg


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selfcheck")
    for name in ("backtest", "tune", "predict"):
        p = sub.add_parser(name)
        p.add_argument("--matches", required=True)
        p.add_argument("--odds")
        p.add_argument("--out", default=".")
        for k, v in asdict(Config()).items():
            p.add_argument("--" + k.replace("_", "-"), dest=k, type=type(v))
        if name in ("backtest", "tune"):
            p.add_argument("--start")
            p.add_argument("--refit-days", type=int, default=7)
        if name == "predict":
            p.add_argument("--fixtures", required=True)
            p.add_argument("--cutoff", help="момент фиксации прогноза, по умолчанию — сейчас")
            p.add_argument("--bank", type=float, default=1000.0)
    a = ap.parse_args(argv)

    if a.cmd == "selfcheck":
        print(selfcheck())
        return
    cfg = _cfg_from_args(a)
    matches = load_matches(a.matches)
    odds = load_odds(a.odds) if a.odds else None

    if a.cmd == "backtest":
        preds, bets = walk_forward(matches, cfg, a.start, a.refit_days, odds)
        preds.to_csv(f"{a.out}/backtest_predictions.csv", index=False)
        if len(bets):
            bets.to_csv(f"{a.out}/backtest_bets.csv", index=False)
        print(report(preds, bets, cfg))
    elif a.cmd == "tune":
        print("Подбор half_life_days × l2_team по log loss 1X2 (walk-forward, без котировок)")
        res = []
        for hl in (90, 180, 365, 730):
            for l2 in (3, 8, 20):
                c = replace(cfg, half_life_days=hl, l2_team=l2)
                p, _ = walk_forward(matches, c, a.start, a.refit_days, None, verbose=False)
                ll = logloss3(p[["p1", "px", "p2"]].values, p.y.values)
                res.append((ll, hl, l2))
                print(f"  half_life={hl:4d}  l2_team={l2:3d}  log loss {ll:.4f}")
        ll, hl, l2 = min(res)
        print(f"Лучшее: --half-life-days {hl} --l2-team {l2} (log loss {ll:.4f}). "
              f"Это подбор на проверочном отрезке — подтверждайте на следующем сезоне.")
    elif a.cmd == "predict":
        fixtures = load_fixtures(a.fixtures)
        cutoff = pd.Timestamp(a.cutoff) if a.cutoff else pd.Timestamp.now()
        preds, bets, params = predict(matches, fixtures, odds, cfg, cutoff, a.bank)
        stamp = cutoff.strftime("%Y%m%d_%H%M")
        preds.to_csv(f"{a.out}/predictions_{stamp}.csv", index=False)
        with open(f"{a.out}/model_params_{stamp}.json", "w", encoding="utf-8") as fh:
            json.dump({"config": asdict(cfg), "params": params}, fh, ensure_ascii=False, indent=1, default=str)
        pd.set_option("display.width", 200)
        print(preds[["home", "away", "lam_h", "lam_a", "p1", "px", "p2", "p1_range", "win1_ot", "p_over_5.5", "top3", "notes"]]
              .round(3).to_string(index=False))
        if len(bets):
            bets.to_csv(f"{a.out}/bets_{stamp}.csv", index=False)
            ch = bets[bets.decision == "ВИРТУАЛЬНЫЙ ВЫБОР"]
            print(f"\nВиртуальных выборов: {len(ch)} из {bets.match_id.nunique()} матчей")
            if len(ch):
                print(ch[["home", "away", "market", "line", "side", "odds", "p_win", "q_market", "fair", "ev", "ev_low", "stake"]]
                      .round(3).to_string(index=False))
        print(f"\nФайлы: predictions_{stamp}.csv, bets_{stamp}.csv, model_params_{stamp}.json в {a.out}")


if __name__ == "__main__":
    main()
