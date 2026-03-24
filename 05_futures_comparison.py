###############################################################################
# FUTURES ODDS vs MODEL PREDICTIONS COMPARISON
# 2026 Brandeis Datathon — Team ASA
#
# Merges Covers.com pre-tournament futures odds with our cascading bracket
# simulation results to compare: what did the market think, what did our
# models predict, and what actually happened?
#
# Reads:
#   scraped/covers_futures.csv   (from scrape_covers_futures.py)
#   outputs/games_merged.csv     (from 01_merge_and_eda.py)
#   datathon set/*.csv           (for team feature lookup)
#
# Run from ~/Desktop/Datathon:
#   python3.12 05_futures_comparison.py
###############################################################################

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

DATA_DIR = "datathon set"

print("="*70)
print("FUTURES ODDS vs MODEL PREDICTIONS")
print("="*70)

# ========================================================================== #
# 1. LOAD DATA
# ========================================================================== #

games = pd.read_csv("outputs/games_merged.csv")
games = games[games['A_WIN'].notna()].copy()
games['A_WIN'] = games['A_WIN'].astype(int)

cf = pd.read_csv("scraped/covers_futures.csv")
kenpom = pd.read_csv(f"{DATA_DIR}/KenPom Barttorvik.csv")
resumes = pd.read_csv(f"{DATA_DIR}/Resumes.csv")
teamrankings = pd.read_csv(f"{DATA_DIR}/TeamRankings.csv")

for df in [kenpom, resumes, teamrankings]:
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()

print(f"Games: {len(games)} | Futures: {len(cf)} rows across {cf['tournament_year'].nunique()} years")

# ========================================================================== #
# 2. EXTRACT MARKET FAVORITES AND ACTUAL WINNERS
# ========================================================================== #

# Find actual winners from our game data
actual_winners = {}
for year in sorted(games['YEAR'].unique()):
    final = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 2)]
    if len(final) == 0:
        continue
    g = final.iloc[0]
    champ = g['A_TEAM'] if g['A_WIN'] == 1 else g['B_TEAM']
    actual_winners[year] = champ

# Find actual F4 teams
actual_f4 = {}
for year in sorted(games['YEAR'].unique()):
    f4g = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 4)]
    if len(f4g) == 0:
        continue
    teams = set()
    for _, g in f4g.iterrows():
        teams.add(g['A_TEAM'])
        teams.add(g['B_TEAM'])
    actual_f4[year] = teams

# Find market favorite (lowest pre-tournament odds) per year
market_favorites = {}
for year in sorted(cf['tournament_year'].unique()):
    yr = cf[cf['tournament_year'] == year].copy()
    yr['pre_tournament_odds'] = pd.to_numeric(yr['pre_tournament_odds'], errors='coerce')
    valid = yr[yr['pre_tournament_odds'].notna() & (yr['pre_tournament_odds'] > 0)]
    if len(valid) > 0:
        fav = valid.loc[valid['pre_tournament_odds'].idxmin()]
        market_favorites[year] = {
            'team': fav['team'],
            'odds': int(fav['pre_tournament_odds']),
        }

# Find actual winner's pre-tournament odds
winner_odds = {}
for year in sorted(cf['tournament_year'].unique()):
    yr = cf[cf['tournament_year'] == year]
    for _, r in yr.iterrows():
        if 'WINNER' in str(r['all_columns']).upper():
            winner_odds[year] = {
                'team': r['team'],
                'odds': int(r['pre_tournament_odds']) if pd.notna(r['pre_tournament_odds']) else None,
            }
            break

# ========================================================================== #
# 3. TRAIN MODELS + RUN CASCADING BRACKET SIM (same as Script 02)
# ========================================================================== #

print("\nTraining models for bracket simulation...")

model_features = [
    'DIFF_KADJ EM', 'DIFF_TR RATING', 'DIFF_BARTHAG',
    'DIFF_KADJ O', 'DIFF_KADJ D', 'DIFF_KADJ T',
    'DIFF_Z_EFG%', 'DIFF_Z_TOV%', 'DIFF_Z_OREB%', 'DIFF_Z_FTR',
    'DIFF_Z_EFG%D', 'DIFF_Z_TOV%D', 'DIFF_Z_DREB%', 'DIFF_Z_FTRD',
    'DIFF_Z_3PT%', 'DIFF_Z_2PT%', 'DIFF_Z_3PT%D', 'DIFF_Z_2PT%D',
    'DIFF_Z_FT%',
    'DIFF_ELITE SOS', 'DIFF_WAB', 'DIFF_SOS RATING',
    'DIFF_Q1 PLUS Q2 W', 'DIFF_R SCORE', 'DIFF_V 1-25 WINS',
    'DIFF_LUCK RATING', 'DIFF_CONSISTENCY TR RATING',
    'DIFF_TALENT', 'DIFF_EXP', 'DIFF_AVG HGT',
    'DIFF_SEED', 'DIFF_ELO', 'DIFF_Z_WIN%',
]

train = games[games['YEAR'].between(2008, 2019)]
X_train, y_train = train[model_features], train['A_WIN']
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

xgb_reg = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5, reg_alpha=8.0, reg_lambda=15.0,
    min_child_weight=8, random_state=42, eval_metric='logloss', verbosity=0)
xgb_reg.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
    max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

nn = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    alpha=0.01, learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42)
nn.fit(X_train_scaled, y_train)

print("Models trained ✓")

# --- Build team feature lookup ---
kp_cols = ['YEAR', 'TEAM NO', 'KADJ O', 'KADJ D', 'KADJ EM', 'BADJ EM', 'BADJ O', 'BADJ D',
           'BARTHAG', 'KADJ T', 'RAW T', 'EFG%', 'TOV%', 'OREB%', 'FTR',
           'EFG%D', 'TOV%D', 'DREB%', 'FTRD', '2PT%', '3PT%', '2PT%D', '3PT%D',
           '2PTR', '3PTR', 'BLK%', 'AST%', 'WIN%', 'ELITE SOS', 'WAB',
           'EXP', 'TALENT', 'AVG HGT', 'EFF HGT', 'FT%', 'SEED']
team_features = kenpom[kp_cols].copy()
team_features = team_features.merge(resumes[['YEAR','TEAM NO','NET RPI','ELO','Q1 PLUS Q2 W','R SCORE']],
                                     on=['YEAR','TEAM NO'], how='left')
team_features = team_features.merge(teamrankings[['YEAR','TEAM NO','TR RATING','SOS RATING','LUCK RATING',
                                                    'CONSISTENCY TR RATING','V 1-25 WINS']],
                                     on=['YEAR','TEAM NO'], how='left')
raw_stats = ['EFG%','TOV%','OREB%','FTR','EFG%D','TOV%D','DREB%','FTRD',
             '2PT%','3PT%','2PT%D','3PT%D','2PTR','3PTR','BLK%','AST%','FT%','WIN%']
season_stats = kenpom.groupby('YEAR')[raw_stats].agg(['mean','std'])
for stat in raw_stats:
    team_features[f'Z_{stat}'] = team_features.apply(
        lambda row: ((row[stat] - season_stats.loc[row['YEAR'],(stat,'mean')]) /
                     season_stats.loc[row['YEAR'],(stat,'std')]
                     if season_stats.loc[row['YEAR'],(stat,'std')] > 0 else 0.0), axis=1)
team_features = team_features.set_index(['YEAR','TEAM NO'])

# --- Pick functions ---
def build_matchup_features(year, a, b):
    try:
        fa = team_features.loc[(year, a)]
        fb = team_features.loc[(year, b)]
    except KeyError:
        return None
    fmap = {'KADJ EM':'DIFF_KADJ EM','TR RATING':'DIFF_TR RATING','BARTHAG':'DIFF_BARTHAG',
            'KADJ O':'DIFF_KADJ O','KADJ D':'DIFF_KADJ D','KADJ T':'DIFF_KADJ T',
            'Z_EFG%':'DIFF_Z_EFG%','Z_TOV%':'DIFF_Z_TOV%','Z_OREB%':'DIFF_Z_OREB%','Z_FTR':'DIFF_Z_FTR',
            'Z_EFG%D':'DIFF_Z_EFG%D','Z_TOV%D':'DIFF_Z_TOV%D','Z_DREB%':'DIFF_Z_DREB%','Z_FTRD':'DIFF_Z_FTRD',
            'Z_3PT%':'DIFF_Z_3PT%','Z_2PT%':'DIFF_Z_2PT%','Z_3PT%D':'DIFF_Z_3PT%D','Z_2PT%D':'DIFF_Z_2PT%D',
            'Z_FT%':'DIFF_Z_FT%','ELITE SOS':'DIFF_ELITE SOS','WAB':'DIFF_WAB','SOS RATING':'DIFF_SOS RATING',
            'Q1 PLUS Q2 W':'DIFF_Q1 PLUS Q2 W','R SCORE':'DIFF_R SCORE','V 1-25 WINS':'DIFF_V 1-25 WINS',
            'LUCK RATING':'DIFF_LUCK RATING','CONSISTENCY TR RATING':'DIFF_CONSISTENCY TR RATING',
            'TALENT':'DIFF_TALENT','EXP':'DIFF_EXP','AVG HGT':'DIFF_AVG HGT',
            'SEED':'DIFF_SEED','ELO':'DIFF_ELO','Z_WIN%':'DIFF_Z_WIN%'}
    return pd.Series({d: fa[r] - fb[r] for r, d in fmap.items()})

def chalk_pick(yr, a, b):
    return a if team_features.loc[(yr,a),'SEED'] <= team_features.loc[(yr,b),'SEED'] else b
def metric_pick(yr, a, b, m):
    return a if team_features.loc[(yr,a),m] >= team_features.loc[(yr,b),m] else b
def ml_pick(yr, a, b, model, use_scaler=False):
    feat = build_matchup_features(yr, a, b)
    if feat is None: return a
    X = feat[model_features].values.reshape(1, -1)
    if use_scaler: X = scaler.transform(X)
    return a if model.predict(X)[0] == 1 else b

def get_team_name(yr, tno):
    r = kenpom[(kenpom['YEAR']==yr) & (kenpom['TEAM NO']==tno)]
    return r.iloc[0]['TEAM'] if len(r) > 0 else f"Team {tno}"

# --- Bracket simulation ---
def simulate_bracket(year, pick_func):
    r64 = games[(games['YEAR']==year) & (games['CURRENT ROUND']==64)].reset_index(drop=True)
    r32_data = games[(games['YEAR']==year) & (games['CURRENT ROUND']==32)].reset_index(drop=True)
    
    r64_teams = set()
    for _, g in r64.iterrows():
        r64_teams.add(int(g['A_TEAM_NO'])); r64_teams.add(int(g['B_TEAM_NO']))
    r32_teams = set()
    for _, g in r32_data.iterrows():
        r32_teams.add(int(g['A_TEAM_NO'])); r32_teams.add(int(g['B_TEAM_NO']))
    bye_teams = r32_teams - r64_teams
    
    r64_winners = []
    for _, g in r64.iterrows():
        w = pick_func(year, int(g['A_TEAM_NO']), int(g['B_TEAM_NO']))
        r64_winners.append(w)
    
    r32_matchups = []
    used = set()
    for _, g in r32_data.iterrows():
        a_no, b_no = int(g['A_TEAM_NO']), int(g['B_TEAM_NO'])
        sides = []
        for tno in [a_no, b_no]:
            if tno in bye_teams:
                sides.append(tno)
            else:
                found = False
                for _, r64g in r64.iterrows():
                    ra, rb = int(r64g['A_TEAM_NO']), int(r64g['B_TEAM_NO'])
                    if tno in (ra, rb):
                        for w in r64_winners:
                            if w in (ra, rb) and w not in used:
                                sides.append(w); used.add(w); found = True; break
                        if found: break
                if not found:
                    for w in r64_winners:
                        if w not in used:
                            sides.append(w); used.add(w); break
        if len(sides) == 2:
            r32_matchups.append((sides[0], sides[1]))
    
    current = r32_matchups
    results = {'f4_teams': [], 'champion': None}
    for rnd in ['R32','S16','E8','F4','Final']:
        winners = [pick_func(year, a, b) for a, b in current]
        if rnd == 'E8':
            results['f4_teams'] = [get_team_name(year, t) for t in winners]
        if len(winners) == 1:
            results['champion'] = get_team_name(year, winners[0])
        if len(winners) >= 2:
            current = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
    return results

# ========================================================================== #
# 4. RUN SIMULATIONS AND BUILD COMPARISON TABLE
# ========================================================================== #

methods = {
    'XGB (reg)':     lambda yr, a, b: ml_pick(yr, a, b, xgb_reg),
    'Random Forest': lambda yr, a, b: ml_pick(yr, a, b, rf),
    'Neural Net':    lambda yr, a, b: ml_pick(yr, a, b, nn, use_scaler=True),
    'Chalk':         lambda yr, a, b: chalk_pick(yr, a, b),
    'KADJ EM':       lambda yr, a, b: metric_pick(yr, a, b, 'KADJ EM'),
    'BARTHAG':       lambda yr, a, b: metric_pick(yr, a, b, 'BARTHAG'),
    'TR RATING':     lambda yr, a, b: metric_pick(yr, a, b, 'TR RATING'),
}

print("\nSimulating brackets for 2021-2025...")

sim_results = {}
for year in range(2021, 2026):
    sim_results[year] = {}
    for name, pfunc in methods.items():
        sim = simulate_bracket(year, pfunc)
        sim_results[year][name] = sim

# ========================================================================== #
# 5. CHAMPION COMPARISON TABLE
# ========================================================================== #

print(f"\n{'='*70}")
print("CHAMPION PREDICTIONS: MARKET vs MODELS vs REALITY")
print(f"{'='*70}")
print(f"\n{'Year':<6} {'Market Fav':>18} {'Odds':>7} {'Actual Champ':>18} {'Odds':>7} {'RF Pick':>18} {'Chalk':>18}")
print("-" * 100)

for year in range(2008, 2026):
    if year == 2020: continue
    
    mf = market_favorites.get(year, {})
    wo = winner_odds.get(year, {})
    aw = actual_winners.get(year, "?")
    
    # Model picks (only for 2021-2025 where we have OOS sims)
    rf_pick = sim_results.get(year, {}).get('Random Forest', {}).get('champion', '—')
    chalk = sim_results.get(year, {}).get('Chalk', {}).get('champion', '—')
    
    print(f"{year:<6} {mf.get('team', '?'):>18} {'+' + str(mf.get('odds', '?')):>7} "
          f"{aw:>18} {'+' + str(wo.get('odds', '?')):>7} "
          f"{rf_pick:>18} {chalk:>18}")

# ========================================================================== #
# 6. HYPOTHETICAL FUTURES BETTING ANALYSIS
# ========================================================================== #

print(f"\n{'='*70}")
print("HYPOTHETICAL FUTURES BETTING (2021-2025)")
print(f"{'='*70}")
print("\nStrategy: Before each tournament, place a $100 futures bet on each")
print("method's predicted champion at the pre-tournament odds from Covers.\n")

# Build name mapping from our data to Covers names
# (Covers uses slightly different names)
covers_name_map = {
    'Connecticut': 'Connecticut', 'UConn': 'Connecticut',
    'North Carolina St.': 'NC State', 'Michigan St.': 'Michigan State',
    'Iowa St.': 'Iowa State', 'Ohio St.': 'Ohio State',
    'Saint Mary\'s': 'Saint Mary\'s',
}

def find_covers_odds(year, team_name):
    """Look up pre-tournament odds for a team from Covers data."""
    yr_data = cf[cf['tournament_year'] == year]
    
    # Try exact match first
    match = yr_data[yr_data['team'].str.lower() == team_name.lower()]
    if len(match) > 0:
        odds = pd.to_numeric(match.iloc[0]['pre_tournament_odds'], errors='coerce')
        return int(odds) if pd.notna(odds) else None
    
    # Try mapped name
    mapped = covers_name_map.get(team_name, team_name)
    match = yr_data[yr_data['team'].str.lower() == mapped.lower()]
    if len(match) > 0:
        odds = pd.to_numeric(match.iloc[0]['pre_tournament_odds'], errors='coerce')
        return int(odds) if pd.notna(odds) else None
    
    # Try substring match
    for _, r in yr_data.iterrows():
        covers_team = str(r['team']).lower()
        our_team = team_name.lower()
        if our_team in covers_team or covers_team in our_team:
            odds = pd.to_numeric(r['pre_tournament_odds'], errors='coerce')
            return int(odds) if pd.notna(odds) else None
    
    return None

print(f"{'Method':<18} {'Year':<6} {'Predicted Champ':>20} {'Odds':>8} {'Actual Champ':>20} {'Won?':>6} {'P/L':>10}")
print("-" * 95)

method_totals = {name: {'bets': 0, 'wins': 0, 'profit': 0} for name in methods}
method_totals['Market Fav'] = {'bets': 0, 'wins': 0, 'profit': 0}

for year in range(2021, 2026):
    actual = actual_winners.get(year, "?")
    
    # Market favorite bet
    mf = market_favorites.get(year, {})
    if mf:
        odds = mf.get('odds', 0)
        won = mf['team'] == actual or mf['team'] in actual or actual in mf['team']
        pl = odds if won else -100
        method_totals['Market Fav']['bets'] += 1
        method_totals['Market Fav']['profit'] += pl
        if won: method_totals['Market Fav']['wins'] += 1
        marker = "✓" if won else "✗"
        print(f"{'Market Fav':<18} {year:<6} {mf['team']:>20} {'+' + str(odds):>8} {actual:>20} {marker:>6} {pl:>+10}")
    
    # Model bets
    for name in methods:
        sim = sim_results[year][name]
        champ = sim['champion']
        if not champ:
            continue
        
        odds = find_covers_odds(year, champ)
        if odds is None:
            print(f"{name:<18} {year:<6} {champ:>20} {'N/A':>8} {actual:>20} {'—':>6} {'—':>10}")
            continue
        
        won = champ == actual
        pl = odds if won else -100
        method_totals[name]['bets'] += 1
        method_totals[name]['profit'] += pl
        if won: method_totals[name]['wins'] += 1
        marker = "✓" if won else "✗"
        print(f"{name:<18} {year:<6} {champ:>20} {'+' + str(odds):>8} {actual:>20} {marker:>6} {pl:>+10}")
    
    print()

# Summary
print(f"\n{'='*70}")
print("FUTURES BETTING SUMMARY (2021-2025, $100 per bet)")
print(f"{'='*70}")
print(f"\n{'Method':<18} {'Bets':>6} {'Wins':>6} {'Profit':>10} {'ROI':>8}")
print("-" * 52)

all_methods_list = ['Market Fav'] + list(methods.keys())
for name in all_methods_list:
    t = method_totals[name]
    if t['bets'] == 0:
        continue
    roi = t['profit'] / (t['bets'] * 100) * 100
    print(f"{name:<18} {t['bets']:>6} {t['wins']:>6} {t['profit']:>+10} {roi:>+7.1f}%")

# ========================================================================== #
# 7. HISTORICAL MARKET FAVORITE PERFORMANCE (2008-2025)
# ========================================================================== #

print(f"\n{'='*70}")
print("HISTORICAL MARKET FAVORITE PERFORMANCE (2008-2025)")
print(f"{'='*70}")

mf_wins = 0
mf_total = 0
mf_profit = 0

for year in range(2008, 2026):
    if year == 2020: continue
    mf = market_favorites.get(year, {})
    aw = actual_winners.get(year, None)
    if not mf or not aw:
        continue
    
    mf_total += 1
    won = mf['team'] == aw or mf['team'].lower() in aw.lower() or aw.lower() in mf['team'].lower()
    if won:
        mf_wins += 1
        mf_profit += mf['odds']
    else:
        mf_profit -= 100

print(f"\nMarket favorite won: {mf_wins}/{mf_total} tournaments ({mf_wins/mf_total*100:.1f}%)")
print(f"If you bet $100 on the market favorite every year: {mf_profit:+d} ({mf_profit/(mf_total*100)*100:+.1f}% ROI)")

# What if you bet on the actual winner at pre-tournament odds?
print(f"\n--- What the actual winners were priced at ---")
total_return = 0
for year in range(2008, 2026):
    if year == 2020: continue
    wo = winner_odds.get(year, {})
    if wo and wo.get('odds'):
        total_return += wo['odds']

avg_winner_odds = total_return / 17
print(f"Average pre-tournament odds of actual winner: +{avg_winner_odds:.0f}")
print(f"Median: +{sorted([winner_odds[y]['odds'] for y in winner_odds if winner_odds[y].get('odds')])[8]}")

print(f"\n{'='*70}")
print("ANALYSIS COMPLETE")
print(f"{'='*70}")