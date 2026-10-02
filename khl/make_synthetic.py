"""
Синтетические данные КХЛ для проверки конвейера (НЕ для выводов о реальной лиге).

Генерирует matches.csv, odds.csv и fixtures.csv в формате, который ждёт khl_model.py.
Истинные силы команд известны, поэтому можно проверить, что модель их восстанавливает,
и что на «честном» рынке стратегия не находит несуществующего перевеса.

  python make_synthetic.py --out data_synth
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.special import expit

TEAMS = ["Авангард", "Автомобилист", "Адмирал", "Ак Барс", "Амур", "Барыс", "Динамо Мн", "Динамо М",
         "Лада", "Локомотив", "Металлург Мг", "Нефтехимик", "Салават Юлаев", "Северсталь", "Сибирь",
         "СКА", "Спартак", "Торпедо", "Трактор", "ХК Сочи", "ЦСКА", "Шанхайские Драконы"]


def simulate(seed=1, seasons=(2023, 2024, 2025), games_per_pair=2, margin=0.07):
    rng = np.random.default_rng(seed)
    T = len(TEAMS)
    att, dfn = rng.normal(0, 0.12, T), rng.normal(0, 0.12, T)
    goalies = {t: [f"{t}-G{k}" for k in range(1, 3)] for t in TEAMS}
    gk_eff = {g: rng.normal(0, 0.07) for gs in goalies.values() for g in gs}
    mu, home, b2b_att, b2b_def = np.log(2.45), 0.06, -0.05, -0.04
    a_en, b_eq, c_en2 = 0.13, 0.10, 0.12
    rows, odds = [], []
    mid = 0
    for season in seasons:
        att += rng.normal(0, 0.06, T)  # межсезонные изменения составов
        dfn += rng.normal(0, 0.06, T)
        pairs = [(h, a) for h in range(T) for a in range(T) if h != a] * games_per_pair
        rng.shuffle(pairs)
        start = pd.Timestamp(f"{season}-09-01")
        last_played = {}
        for n, (h, a) in enumerate(pairs):
            date = start + pd.Timedelta(days=int(n * 200 / len(pairs)))
            att += rng.normal(0, 0.004, T)
            hb = float(last_played.get(h) is not None and (date - last_played[h]).days <= 1)
            ab = float(last_played.get(a) is not None and (date - last_played[a]).days <= 1)
            last_played[h] = last_played[a] = date
            hg = goalies[TEAMS[h]][int(rng.random() < 0.3)]
            ag = goalies[TEAMS[a]][int(rng.random() < 0.3)]
            lh = np.exp(mu + home + att[h] - dfn[a] - gk_eff[ag] + b2b_att * hb + b2b_def * ab)
            la = np.exp(mu + att[a] - dfn[h] - gk_eff[hg] + b2b_att * ab + b2b_def * hb)
            x, y = rng.poisson(lh), rng.poisson(la)
            u, hen, aen = rng.random(), 0, 0
            d = x - y
            if d == 1 and u < a_en: x, hen = x + 1, 1
            elif d == -1 and u < a_en: y, aen = y + 1, 1
            elif d == 1 and u < a_en + b_eq: y += 1
            elif d == -1 and u < a_en + b_eq: x += 1
            elif d == 2 and u < c_en2: x, hen = x + 1, 1
            elif d == -2 and u < c_en2: y, aen = y + 1, 1
            hf, af, ot = x, y, ""
            p_ot = expit(0.08 + 0.5 * np.log(lh / la))
            if x == y:
                ot = "OT" if rng.random() < 0.6 else "SO"
                if rng.random() < p_ot: hf += 1
                else: af += 1
            mid += 1
            m_id = f"KHL-{season}-{mid:05d}"
            rows.append(dict(match_id=m_id, season=season, stage="regular", date=date.strftime("%d.%m.%Y"),
                             time_msk="19:30", home=TEAMS[h], away=TEAMS[a], home_60=x, away_60=y, ot_so=ot,
                             home_final=hf, away_final=af, home_en=hen, away_en=aen,
                             home_goalie=hg, away_goalie=ag))
            # «истинные» вероятности 1X2 за 60 минут и ТБ 5.5 для котировок
            sim_h, sim_a = rng.poisson(lh, 4000), rng.poisson(la, 4000)
            p = np.array([np.mean(sim_h > sim_a), np.mean(sim_h == sim_a), np.mean(sim_h < sim_a)])
            p[1] += 0.035; p /= p.sum()  # ничьих в реальности больше, чем у Пуассона
            pov = np.mean(sim_h + sim_a > 5.5) + 0.02
            for closing, noise in ((False, 0.06), (True, 0.025)):
                z = np.log(p) + rng.normal(0, noise, 3)
                q = np.exp(z) / np.exp(z).sum()
                k = 1 / (q * (1 + margin))
                ts = (date - pd.Timedelta(hours=26)) if not closing else (date + pd.Timedelta(hours=19))
                odds.append(dict(match_id=m_id, bookmaker="BK", market="1x2", period="60", line="",
                                 odds_1=round(k[0], 2), odds_X=round(k[1], 2), odds_2=round(k[2], 2),
                                 timestamp=ts.strftime("%Y-%m-%d %H:%M"), is_closing=int(closing)))
                qo = expit(np.log(pov / (1 - pov)) + rng.normal(0, noise))
                ko = 1 / (np.array([qo, 1 - qo]) * (1 + margin * 0.8))
                odds.append(dict(match_id=m_id, bookmaker="BK", market="total", period="60", line=5.5,
                                 odds_1=round(ko[0], 2), odds_X="", odds_2=round(ko[1], 2),
                                 timestamp=ts.strftime("%Y-%m-%d %H:%M"), is_closing=int(closing)))
    truth = pd.DataFrame({"team": TEAMS, "attack": att - att.mean(), "defence": dfn - dfn.mean()})
    return pd.DataFrame(rows), pd.DataFrame(odds), truth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synth")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    m, o, truth = simulate(a.seed)
    # последние 10 матчей — «будущие» (без счёта) для команды predict
    fut = m.tail(10)
    m.iloc[:-10].to_csv(f"{a.out}/matches.csv", index=False)
    fut[["match_id", "season", "stage", "date", "time_msk", "home", "away", "home_goalie", "away_goalie"]] \
        .to_csv(f"{a.out}/fixtures.csv", index=False)
    o.to_csv(f"{a.out}/odds.csv", index=False)
    truth.to_csv(f"{a.out}/truth_end.csv", index=False)
    print(f"{len(m) - 10} матчей, {len(fut)} будущих, {len(o)} строк котировок -> {a.out}/")


if __name__ == "__main__":
    main()
