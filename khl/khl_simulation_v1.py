import numpy as np
import scipy.stats as stats
import json

def run_simulation(seed=42, n_sims=100000):
    np.random.seed(seed)
    
    # Match: CSKA (Home) vs SKA (Away)
    # Date: 29.09.2026, 19:30 MSK (CSKA Arena)
    # Model parameters for KHL v1.0:
    # Baseline expected goals for 60 min
    lambda_home = 2.82
    lambda_away = 2.14
    
    # 100,000 Monte Carlo runs
    home_goals = np.random.poisson(lambda_home, n_sims)
    away_goals = np.random.poisson(lambda_away, n_sims)
    
    total_goals = home_goals + away_goals
    diff = home_goals - away_goals
    
    # 60 min outcomes
    p1_60 = np.mean(home_goals > away_goals)
    px_60 = np.mean(home_goals == away_goals)
    p2_60 = np.mean(home_goals < away_goals)
    
    # Overtime & Shootout simulation
    ties_idx = np.where(home_goals == away_goals)[0]
    n_ties = len(ties_idx)
    home_ot_win_prob = 0.54
    ot_winner = np.random.binomial(1, home_ot_win_prob, n_ties)
    
    home_final_wins = (home_goals > away_goals).astype(int)
    home_final_wins[ties_idx] = ot_winner
    
    p_home_incl_ot = np.mean(home_final_wins == 1)
    p_away_incl_ot = np.mean(home_final_wins == 0)
    
    # Top 5 most probable exact scores
    scores, counts = np.unique(list(zip(home_goals, away_goals)), axis=0, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top_scores = []
    for i in range(5):
        idx = sorted_indices[i]
        sc = scores[idx]
        pr = counts[idx] / n_sims
        top_scores.append((f"{sc[0]}:{sc[1]}", float(pr)))
        
    # Totals
    p_over_4_5 = np.mean(total_goals > 4.5)
    p_under_4_5 = np.mean(total_goals < 4.5)
    
    p_over_5_5 = np.mean(total_goals > 5.5)
    p_under_5_5 = np.mean(total_goals < 5.5)
    
    p_over_6_5 = np.mean(total_goals > 6.5)
    p_under_6_5 = np.mean(total_goals < 6.5)
    
    # Handicaps
    p_h1_minus_1_5 = np.mean(diff > 1.5)
    p_h2_plus_1_5 = np.mean(diff < 1.5)
    
    # 80% central predictive interval for total goals
    tot_p10 = int(np.percentile(total_goals, 10))
    tot_p90 = int(np.percentile(total_goals, 90))
    
    se_p1 = np.sqrt(p1_60 * (1 - p1_60) / n_sims)
    
    return {
        "lambda_home": lambda_home,
        "lambda_away": lambda_away,
        "p1_60": float(p1_60),
        "px_60": float(px_60),
        "p2_60": float(p2_60),
        "p_home_incl_ot": float(p_home_incl_ot),
        "p_away_incl_ot": float(p_away_incl_ot),
        "p_over_4_5": float(p_over_4_5),
        "p_under_4_5": float(p_under_4_5),
        "p_over_5_5": float(p_over_5_5),
        "p_under_5_5": float(p_under_5_5),
        "p_over_6_5": float(p_over_6_5),
        "p_under_6_5": float(p_under_6_5),
        "p_h1_minus_1_5": float(p_h1_minus_1_5),
        "p_h2_plus_1_5": float(p_h2_plus_1_5),
        "tot_80_pi": [tot_p10, tot_p90],
        "top_scores": top_scores,
        "se_p1": float(se_p1),
        "n_sims": n_sims,
        "seed": seed
    }

if __name__ == "__main__":
    res = run_simulation()
    print(json.dumps(res, indent=2))

