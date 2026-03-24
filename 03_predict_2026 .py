###############################################################################
# PHASE 3: 2026 BRACKET PREDICTION
# 2026 Brandeis Datathon — Team ASA (Anokh, Samiya, Aastha)
#
# Retrains all models on FULL 2008-2025 data (no holdout — this is deployment).
# Predicts the 2026 NCAA Tournament bracket from First Four through Championship.
# Outputs predicted brackets for all 7 methods + win probabilities.
#
# Run: cd ~/Downloads && python3.12 phase3_predict_2026.py
# Requires: outputs/games_merged.csv from Phase 1 (re-run with updated data)
#           datathon set/ folder with updated 2026 Kaggle data
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

os.makedirs("outputs", exist_ok=True)

# --------------------------------------------------------------------------- #
# 1. LOAD ALL DATA SOURCES
# --------------------------------------------------------------------------- #

print("="*70)
print("PHASE 3: 2026 MARCH MADNESS PREDICTIONS")
print("="*70)

kenpom = pd.read_csv("datathon set/KenPom Barttorvik.csv")
resumes = pd.read_csv("datathon set/Resumes.csv")
teamrankings = pd.read_csv("datathon set/TeamRankings.csv")
matchups = pd.read_csv("datathon set/Tournament Matchups.csv")

for df in [kenpom, resumes, teamrankings, matchups]:
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

print(f"KenPom: {len(kenpom)} rows, years {kenpom['YEAR'].min()}-{kenpom['YEAR'].max()}")
print(f"2026 teams in KenPom: {len(kenpom[kenpom['YEAR']==2026])}")

# --------------------------------------------------------------------------- #
# 2. BUILD TEAM FEATURE LOOKUP (same as Phase 2b)
# --------------------------------------------------------------------------- #

kp_cols = ['YEAR', 'TEAM NO', 'KADJ O', 'KADJ D', 'KADJ EM', 'BADJ EM', 'BADJ O', 'BADJ D',
           'BARTHAG', 'KADJ T', 'RAW T', 'EFG%', 'TOV%', 'OREB%', 'FTR',
           'EFG%D', 'TOV%D', 'DREB%', 'FTRD', '2PT%', '3PT%', '2PT%D', '3PT%D',
           '2PTR', '3PTR', 'BLK%', 'AST%', 'WIN%', 'ELITE SOS', 'WAB',
           'EXP', 'TALENT', 'AVG HGT', 'EFF HGT', 'FT%', 'SEED']

team_features = kenpom[kp_cols].copy()

res_cols = ['YEAR', 'TEAM NO', 'NET RPI', 'ELO', 'Q1 PLUS Q2 W', 'R SCORE']
team_features = team_features.merge(resumes[res_cols], on=['YEAR', 'TEAM NO'], how='left')

tr_cols = ['YEAR', 'TEAM NO', 'TR RATING', 'SOS RATING', 'LUCK RATING',
           'CONSISTENCY TR RATING', 'V 1-25 WINS']
team_features = team_features.merge(teamrankings[tr_cols], on=['YEAR', 'TEAM NO'], how='left')

# Z-scores for raw stats
raw_stats_to_zscore = [
    'EFG%', 'TOV%', 'OREB%', 'FTR', 'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    '2PT%', '3PT%', '2PT%D', '3PT%D', '2PTR', '3PTR',
    'BLK%', 'AST%', 'FT%', 'WIN%',
]

season_stats = kenpom.groupby('YEAR')[raw_stats_to_zscore].agg(['mean', 'std'])

for stat in raw_stats_to_zscore:
    team_features[f'Z_{stat}'] = team_features.apply(
        lambda row: (
            (row[stat] - season_stats.loc[row['YEAR'], (stat, 'mean')]) /
            season_stats.loc[row['YEAR'], (stat, 'std')]
            if season_stats.loc[row['YEAR'], (stat, 'std')] > 0 else 0.0
        ), axis=1
    )

team_features = team_features.set_index(['YEAR', 'TEAM NO'])
print(f"Team feature lookup: {len(team_features)} team-seasons (including 2026)")

# --------------------------------------------------------------------------- #
# 3. REBUILD TRAINING DATA FROM 2008-2025 GAMES
# --------------------------------------------------------------------------- #

# We need to rebuild games_merged with the updated data
# Load the existing one which has 2008-2025 game results
games = pd.read_csv("outputs/games_merged.csv")

# Filter to only games with actual results (exclude 2026 which has NaN scores)
games = games[games['A_WIN'].notna()].copy()
games['A_WIN'] = games['A_WIN'].astype(int)
print(f"Training games (2008-2025): {len(games)}")

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

# --------------------------------------------------------------------------- #
# 4. TRAIN MODELS ON ALL 2008-2025 DATA
# --------------------------------------------------------------------------- #

print("\nTraining models on ALL 2008-2025 data (deployment mode)...")

X_train = games[model_features]
y_train = games['A_WIN']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

xgb = XGBClassifier(n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=8.0, reg_lambda=15.0, min_child_weight=8,
    random_state=42, eval_metric='logloss', verbosity=0)
xgb.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=300, max_depth=6, min_samples_leaf=10,
    max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

nn = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    alpha=0.01, learning_rate='adaptive', learning_rate_init=0.001, max_iter=500,
    early_stopping=True, validation_fraction=0.15, random_state=42)
nn.fit(X_train_scaled, y_train)

print(f"  XGBoost trained on {len(X_train)} games ✓")
print(f"  Random Forest trained on {len(X_train)} games ✓")
print(f"  Neural Network trained on {len(X_train)} games ✓")

# --------------------------------------------------------------------------- #
# 5. MATCHUP FEATURE BUILDER
# --------------------------------------------------------------------------- #

def build_matchup_features(year, team_a_no, team_b_no):
    """Compute differential features for a hypothetical matchup."""
    try:
        a = team_features.loc[(year, team_a_no)]
        b = team_features.loc[(year, team_b_no)]
    except KeyError:
        return None

    feature_map = {
        'KADJ EM': 'DIFF_KADJ EM', 'TR RATING': 'DIFF_TR RATING',
        'BARTHAG': 'DIFF_BARTHAG', 'KADJ O': 'DIFF_KADJ O',
        'KADJ D': 'DIFF_KADJ D', 'KADJ T': 'DIFF_KADJ T',
        'Z_EFG%': 'DIFF_Z_EFG%', 'Z_TOV%': 'DIFF_Z_TOV%',
        'Z_OREB%': 'DIFF_Z_OREB%', 'Z_FTR': 'DIFF_Z_FTR',
        'Z_EFG%D': 'DIFF_Z_EFG%D', 'Z_TOV%D': 'DIFF_Z_TOV%D',
        'Z_DREB%': 'DIFF_Z_DREB%', 'Z_FTRD': 'DIFF_Z_FTRD',
        'Z_3PT%': 'DIFF_Z_3PT%', 'Z_2PT%': 'DIFF_Z_2PT%',
        'Z_3PT%D': 'DIFF_Z_3PT%D', 'Z_2PT%D': 'DIFF_Z_2PT%D',
        'Z_FT%': 'DIFF_Z_FT%',
        'ELITE SOS': 'DIFF_ELITE SOS', 'WAB': 'DIFF_WAB',
        'SOS RATING': 'DIFF_SOS RATING',
        'Q1 PLUS Q2 W': 'DIFF_Q1 PLUS Q2 W', 'R SCORE': 'DIFF_R SCORE',
        'V 1-25 WINS': 'DIFF_V 1-25 WINS',
        'LUCK RATING': 'DIFF_LUCK RATING',
        'CONSISTENCY TR RATING': 'DIFF_CONSISTENCY TR RATING',
        'TALENT': 'DIFF_TALENT', 'EXP': 'DIFF_EXP',
        'AVG HGT': 'DIFF_AVG HGT', 'SEED': 'DIFF_SEED',
        'ELO': 'DIFF_ELO', 'Z_WIN%': 'DIFF_Z_WIN%',
    }

    diff = {}
    for raw_name, diff_name in feature_map.items():
        diff[diff_name] = a[raw_name] - b[raw_name]
    return pd.Series(diff)

# --------------------------------------------------------------------------- #
# 6. PREDICTION FUNCTIONS
# --------------------------------------------------------------------------- #

def get_team_name(team_no):
    row = kenpom[(kenpom['YEAR'] == 2026) & (kenpom['TEAM NO'] == team_no)]
    return row.iloc[0]['TEAM'] if len(row) > 0 else f"Team {team_no}"

def get_team_seed(team_no):
    try:
        return int(team_features.loc[(2026, team_no), 'SEED'])
    except:
        return 0

def chalk_pick(team_a_no, team_b_no):
    a_seed = team_features.loc[(2026, team_a_no), 'SEED']
    b_seed = team_features.loc[(2026, team_b_no), 'SEED']
    return team_a_no if a_seed <= b_seed else team_b_no

def metric_pick(team_a_no, team_b_no, metric):
    a_val = team_features.loc[(2026, team_a_no), metric]
    b_val = team_features.loc[(2026, team_b_no), metric]
    return team_a_no if a_val >= b_val else team_b_no

def ml_pick(team_a_no, team_b_no, model, use_scaler=False):
    features = build_matchup_features(2026, team_a_no, team_b_no)
    if features is None:
        return team_a_no
    X = features[model_features].values.reshape(1, -1)
    if use_scaler:
        X = scaler.transform(X)
    pred = model.predict(X)[0]
    return team_a_no if pred == 1 else team_b_no

def ml_prob(team_a_no, team_b_no, model, use_scaler=False):
    """Get win probability for team A."""
    features = build_matchup_features(2026, team_a_no, team_b_no)
    if features is None:
        return 0.5
    X = features[model_features].values.reshape(1, -1)
    if use_scaler:
        X = scaler.transform(X)
    return model.predict_proba(X)[0][1]

# --------------------------------------------------------------------------- #
# 7. BUILD 2026 BRACKET
# --------------------------------------------------------------------------- #

print("\n" + "="*70)
print("2026 BRACKET STRUCTURE")
print("="*70)

tm_2026 = matchups[matchups['YEAR'] == 2026]
r64 = tm_2026[tm_2026['CURRENT ROUND'] == 64].sort_values('BY YEAR NO', ascending=False)

# Identify play-in games and build clean bracket
playin_games = []
r64_games = []  # (team_a_no, team_b_no) for R64 — PLACEHOLDER for play-in slots

# Convert to list of dicts for easier access
rows = [row.to_dict() for _, row in r64.iterrows()]
i = 0
while i < len(rows):
    if i + 3 < len(rows):
        # Check if this is a play-in slot (same team appears 2 rows later)
        if rows[i]['TEAM'] == rows[i+2]['TEAM'] and rows[i]['SEED'] == rows[i+2]['SEED']:
            host = rows[i]
            pa = rows[i+1]
            pb = rows[i+3]
            playin_idx = len(r64_games)  # record the bracket position
            playin_games.append({
                'host_team_no': int(host['TEAM NO']),
                'host_team': host['TEAM'],
                'host_seed': int(host['SEED']),
                'playin_a_no': int(pa['TEAM NO']),
                'playin_a': pa['TEAM'],
                'playin_b_no': int(pb['TEAM NO']),
                'playin_b': pb['TEAM'],
                'playin_seed': int(pa['SEED']),
                'bracket_position': playin_idx,  # where in r64_games this belongs
            })
            r64_games.append(None)  # placeholder — will be filled after play-in
            i += 4
            continue
    
    # Normal R64 pair
    if i + 1 < len(rows):
        t_top = rows[i]
        t_bot = rows[i+1]
        r64_games.append((int(t_top['TEAM NO']), int(t_bot['TEAM NO'])))
        i += 2
    else:
        i += 1

print(f"\nFirst Four (Play-in) Games: {len(playin_games)}")
for pg in playin_games:
    print(f"  ({pg['playin_seed']}) {pg['playin_a']:20s} vs ({pg['playin_seed']}) {pg['playin_b']:20s} "
          f"→ winner faces ({pg['host_seed']}) {pg['host_team']}")

print(f"\nR64 Games (without play-in slots): {len(r64_games)}")

# --------------------------------------------------------------------------- #
# 8. SIMULATE 2026 BRACKET FOR ALL METHODS
# --------------------------------------------------------------------------- #

methods = {
    'XGBoost': lambda a, b: ml_pick(a, b, xgb),
    'Random Forest': lambda a, b: ml_pick(a, b, rf),
    'Neural Net': lambda a, b: ml_pick(a, b, nn, use_scaler=True),
    'Chalk': lambda a, b: chalk_pick(a, b),
    'KADJ EM': lambda a, b: metric_pick(a, b, 'KADJ EM'),
    'BARTHAG': lambda a, b: metric_pick(a, b, 'BARTHAG'),
    'TR RATING': lambda a, b: metric_pick(a, b, 'TR RATING'),
}

# Also store XGBoost probabilities for each game
xgb_probs = {}

all_predictions = {}

for method_name, pick_func in methods.items():
    print(f"\n{'='*70}")
    print(f"2026 BRACKET — {method_name}")
    print(f"{'='*70}")
    
    # Step 1: Predict play-in games
    bracket_games = list(r64_games)  # copy the base bracket (has None placeholders)
    playin_results = []
    
    for pg in playin_games:
        winner_no = pick_func(pg['playin_a_no'], pg['playin_b_no'])
        winner_name = get_team_name(winner_no)
        loser_no = pg['playin_b_no'] if winner_no == pg['playin_a_no'] else pg['playin_a_no']
        loser_name = get_team_name(loser_no)
        
        # Get XGBoost probability for this game
        if method_name == 'XGBoost':
            prob = ml_prob(pg['playin_a_no'], pg['playin_b_no'], xgb)
            xgb_probs[f"First Four: {pg['playin_a']} vs {pg['playin_b']}"] = {
                'team_a': pg['playin_a'], 'team_b': pg['playin_b'],
                'prob_a': prob, 'winner': winner_name
            }
        
        print(f"  First Four: ({pg['playin_seed']}) {winner_name:20s} "
              f"beats ({pg['playin_seed']}) {loser_name}")
        
        # Insert winner into bracket at the CORRECT position (not appended)
        bracket_games[pg['bracket_position']] = (pg['host_team_no'], winner_no)
        playin_results.append(winner_no)
    
    # Step 2: Simulate R64 through Championship
    round_names = ['R64', 'R32', 'S16', 'E8', 'F4', 'Final']
    current_matchups = bracket_games
    
    bracket_results = {'method': method_name, 'rounds': {}}
    
    for rnd_name in round_names:
        winners = []
        rnd_results = []
        
        print(f"\n  --- {rnd_name} ---")
        for team_a_no, team_b_no in current_matchups:
            winner_no = pick_func(team_a_no, team_b_no)
            loser_no = team_b_no if winner_no == team_a_no else team_a_no
            
            winner_name = get_team_name(winner_no)
            loser_name = get_team_name(loser_no)
            winner_seed = get_team_seed(winner_no)
            loser_seed = get_team_seed(loser_no)
            
            # Get XGBoost probs
            if method_name == 'XGBoost':
                prob = ml_prob(team_a_no, team_b_no, xgb)
                actual_winner_prob = prob if winner_no == team_a_no else 1 - prob
                xgb_probs[f"{rnd_name}: {get_team_name(team_a_no)} vs {get_team_name(team_b_no)}"] = {
                    'team_a': get_team_name(team_a_no), 'team_b': get_team_name(team_b_no),
                    'prob_a': round(prob, 3), 'winner': winner_name,
                    'winner_prob': round(actual_winner_prob, 3)
                }
            
            upset = " *** UPSET ***" if winner_seed > loser_seed else ""
            print(f"    ({winner_seed}) {winner_name:20s} over ({loser_seed}) {loser_name:20s}{upset}")
            
            rnd_results.append({
                'winner': winner_name, 'winner_no': winner_no,
                'winner_seed': winner_seed,
                'loser': loser_name, 'loser_seed': loser_seed,
            })
            winners.append(winner_no)
        
        bracket_results['rounds'][rnd_name] = rnd_results
        
        # Pair winners for next round
        if len(winners) >= 2:
            current_matchups = [(winners[i], winners[i+1]) for i in range(0, len(winners), 2)]
        
        if len(winners) == 1:
            bracket_results['champion'] = get_team_name(winners[0])
            bracket_results['champion_seed'] = get_team_seed(winners[0])
            print(f"\n  🏆 CHAMPION: ({bracket_results['champion_seed']}) {bracket_results['champion']}")
    
    all_predictions[method_name] = bracket_results

# --------------------------------------------------------------------------- #
# 9. SUMMARY COMPARISON
# --------------------------------------------------------------------------- #

print("\n" + "="*70)
print("2026 PREDICTION SUMMARY")
print("="*70)

print(f"\n{'Method':<18} {'Champion':>20} {'Seed':>5}")
print("-" * 45)
for method_name, results in all_predictions.items():
    print(f"{method_name:<18} {results['champion']:>20} {results['champion_seed']:>5}")

print(f"\n{'Method':<18} {'Final Four Teams'}")
print("-" * 75)
for method_name, results in all_predictions.items():
    if 'E8' in results['rounds']:
        f4 = [f"({r['winner_seed']}) {r['winner']}" for r in results['rounds']['E8']]
        print(f"{method_name:<18} {', '.join(f4)}")

# --------------------------------------------------------------------------- #
# 10. SAVE OUTPUTS
# --------------------------------------------------------------------------- #

# Save XGBoost bracket with probabilities
print(f"\n{'='*70}")
print("XGBOOST GAME-BY-GAME PROBABILITIES")
print(f"{'='*70}")

xgb_rows = []
for game_key, info in xgb_probs.items():
    print(f"  {game_key:50s} | P({info['team_a'][:15]})={info['prob_a']:.3f} | Winner: {info['winner']}")
    xgb_rows.append({
        'game': game_key,
        'team_a': info['team_a'],
        'team_b': info['team_b'],
        'prob_team_a': info['prob_a'],
        'winner': info['winner'],
    })

xgb_df = pd.DataFrame(xgb_rows)
xgb_df.to_csv("outputs/predictions_2026_xgb.csv", index=False)

# Save all method predictions
summary_rows = []
for method_name, results in all_predictions.items():
    for rnd_name, rnd_results in results['rounds'].items():
        for game in rnd_results:
            summary_rows.append({
                'method': method_name,
                'round': rnd_name,
                'winner': game['winner'],
                'winner_seed': game['winner_seed'],
                'loser': game['loser'],
                'loser_seed': game['loser_seed'],
            })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv("outputs/predictions_2026_all_methods.csv", index=False)

print(f"\nOutputs saved:")
print(f"  outputs/predictions_2026_xgb.csv — XGBoost picks + probabilities")
print(f"  outputs/predictions_2026_all_methods.csv — All methods, all rounds")

# --------------------------------------------------------------------------- #
# 11. CONSENSUS ANALYSIS
# --------------------------------------------------------------------------- #

print(f"\n{'='*70}")
print("CONSENSUS ANALYSIS — WHERE DO METHODS AGREE/DISAGREE?")
print(f"{'='*70}")

for rnd_name in ['E8', 'F4', 'Final']:
    print(f"\n  --- {rnd_name} ---")
    all_winners = {}
    for method_name, results in all_predictions.items():
        if rnd_name in results['rounds']:
            for i, game in enumerate(results['rounds'][rnd_name]):
                key = f"Game {i+1}"
                if key not in all_winners:
                    all_winners[key] = {}
                winner = game['winner']
                if winner not in all_winners[key]:
                    all_winners[key][winner] = []
                all_winners[key][winner].append(method_name)
    
    for game_key, winners in all_winners.items():
        print(f"  {game_key}:")
        for team, methods_list in sorted(winners.items(), key=lambda x: -len(x[1])):
            count = len(methods_list)
            print(f"    {team:20s}: {count}/7 methods ({', '.join(methods_list)})")

print(f"\n{'='*70}")
print("PHASE 3 COMPLETE")
print(f"{'='*70}")