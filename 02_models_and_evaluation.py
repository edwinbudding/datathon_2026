###############################################################################
# SCRIPT 02: MODEL TRAINING, EVALUATION, AND BETTING ANALYSIS
# 2026 Brandeis Datathon — Team ASA (Anokh, Samiya, Aastha)
#
# Unified pipeline:
#   Part A: Load data + Vegas lines
#   Part B: Feature selection + temporal split
#   Part C: Train models (XGB original, XGB regularized, RF, NN)
#   Part D: Game-level evaluation (accuracy, log-loss, upset detection)
#   Part E: Feature importance
#   Part F: Game-independent bracket scoring (2021-2025)
#   Part G: Cascading bracket simulation (2021-2025)
#   Part H: Vegas comparison + betting strategy
#   Part I: Summary + save outputs
#
# Run from ~/Desktop/Datathon:
#   python3.12 02_models_and_evaluation.py
###############################################################################

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, log_loss
from sklearn.metrics import precision_score, recall_score, f1_score
from xgboost import XGBClassifier

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150

DATA_DIR = "datathon set"

###############################################################################
# PART A: DATA LOADING
###############################################################################

print("="*70)
print("SCRIPT 02: MODELS + EVALUATION + BETTING ANALYSIS")
print("="*70)

games = pd.read_csv("outputs/games_merged.csv")
print(f"Games: {len(games)} (years {games['YEAR'].min()}-{games['YEAR'].max()})")

# Load raw data for cascading bracket sim (needs team feature lookup)
kenpom = pd.read_csv(f"{DATA_DIR}/KenPom Barttorvik.csv")
resumes = pd.read_csv(f"{DATA_DIR}/Resumes.csv")
teamrankings = pd.read_csv(f"{DATA_DIR}/TeamRankings.csv")

# Strip whitespace from string columns
for df in [kenpom, resumes, teamrankings]:
    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.strip()

# Load Vegas lines if available
vegas_path = "scraped/vegas_lines.csv"
HAS_VEGAS = os.path.exists(vegas_path)
if HAS_VEGAS:
    vl = pd.read_csv(vegas_path)
    games = games.merge(
        vl[['YEAR', 'A_TEAM_NO', 'B_TEAM_NO', 'FAVORED_TEAM_SR', 'SPREAD', 'OVER_UNDER']],
        on=['YEAR', 'A_TEAM_NO', 'B_TEAM_NO'], how='left'
    )
    print(f"Vegas lines merged: {games['SPREAD'].notna().sum()} with spreads")

    sr_to_kaggle = {
        'Washington State': 'Washington St.', 'Kent State': 'Kent St.',
        'Southern California': 'USC', 'Mississippi State': 'Mississippi St.',
        'Michigan State': 'Michigan St.', 'Miami (FL)': 'Miami FL',
        'Ohio State': 'Ohio St.', 'Virginia Commonwealth': 'VCU',
        'North Carolina State': 'North Carolina St.', 'Brigham Young': 'BYU',
        'Arizona State': 'Arizona St.', 'Iowa State': 'Iowa St.',
        'Colorado State': 'Colorado St.', 'Florida State': 'Florida St.',
        'Kansas State': 'Kansas St.', 'Wichita State': 'Wichita St.',
        'Oklahoma State': 'Oklahoma St.', 'Oregon State': 'Oregon St.',
        'Boise State': 'Boise St.', 'Fresno State': 'Fresno St.',
        'San Diego State': 'San Diego St.', 'Penn State': 'Penn St.',
        'Murray State': 'Murray St.', 'Morehead State': 'Morehead St.',
        'Norfolk State': 'Norfolk St.', 'Weber State': 'Weber St.',
        'South Dakota State': 'South Dakota St.', 'Wright State': 'Wright St.',
        'Kennesaw State': 'Kennesaw St.', 'Cleveland State': 'Cleveland St.',
        'Miami (OH)': 'Miami OH', 'Nevada-Las Vegas': 'UNLV',
        'Louisiana State': 'LSU', 'Texas-San Antonio': 'UTSA',
        'Massachusetts': 'UMass', 'Southern Methodist': 'SMU',
        'Central Florida': 'UCF', 'Loyola (IL)': 'Loyola Chicago',
        'Cal State Fullerton': 'Cal St. Fullerton', 'Gardner-Webb': 'Gardner Webb',
        'North Carolina-Wilmington': 'UNC Wilmington',
        'North Carolina-Asheville': 'UNC Asheville',
        'UC-Irvine': 'UC Irvine', 'UC-Santa Barbara': 'UC Santa Barbara',
        'UC-San Diego': 'UC San Diego', 'Alabama-Birmingham': 'UAB',
        'Arkansas-Little Rock': 'Little Rock', 'Texas Christian': 'TCU',
        'Middle Tennessee State': 'Middle Tennessee',
        'Georgia State': 'Georgia St.', 'Tennessee State': 'Tennessee St.',
        'Montana State': 'Montana St.', 'Utah State': 'Utah St.',
        'New Mexico State': 'New Mexico St.', 'NC State': 'North Carolina St.',
        "Saint Mary's (CA)": "Saint Mary's", "Saint Peter's": "Saint Peter's",
        'Long Island University': 'LIU Brooklyn', 'Prairie View': 'Prairie View A&M',
        'Texas A&M-Corpus Christi': 'Texas A&M Corpus Christi',
    }

    def get_vegas_pick_a(row):
        if pd.isna(row.get('FAVORED_TEAM_SR')): return -1
        fav = sr_to_kaggle.get(str(row['FAVORED_TEAM_SR']).strip(),
                               str(row['FAVORED_TEAM_SR']).strip())
        a, b = str(row['A_TEAM']).strip(), str(row['B_TEAM']).strip()
        if fav == a: return 1
        elif fav == b: return 0
        elif fav.lower() in a.lower() or a.lower() in fav.lower(): return 1
        elif fav.lower() in b.lower() or b.lower() in fav.lower(): return 0
        else: return -1

    games['VEGAS_PICK_A'] = games.apply(get_vegas_pick_a, axis=1)
    print(f"Vegas picks matched: {(games['VEGAS_PICK_A'] != -1).sum()}")
else:
    print("Vegas lines not found — skipping Vegas sections")
    games['VEGAS_PICK_A'] = -1
    games['SPREAD'] = np.nan

###############################################################################
# PART B: FEATURES + TEMPORAL SPLIT
###############################################################################

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

print(f"\nFeatures: {len(model_features)}")

train = games[games['YEAR'].between(2008, 2019)]
val   = games[games['YEAR'].between(2021, 2024)]
test  = games[games['YEAR'] == 2025]

X_train, y_train = train[model_features], train['A_WIN']
X_val,   y_val   = val[model_features],   val['A_WIN']
X_test,  y_test  = test[model_features],  test['A_WIN']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)
X_test_scaled  = scaler.transform(X_test)

print(f"Train: {len(train)} (2008-2019) | Val: {len(val)} (2021-2024) | Test: {len(test)} (2025)")

###############################################################################
# PART C: TRAIN MODELS
###############################################################################

print("\n" + "="*70)
print("MODEL TRAINING")
print("="*70)

print("\n[1/4] XGBoost (original)...")
xgb_orig = XGBClassifier(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0,
    min_child_weight=1, random_state=42, eval_metric='logloss', verbosity=0)
xgb_orig.fit(X_train, y_train)

print("[2/4] XGBoost (regularized — L1+L2, depth=2)...")
xgb_reg = XGBClassifier(
    n_estimators=100, max_depth=2, learning_rate=0.05,
    subsample=0.7, colsample_bytree=0.5,
    reg_alpha=8.0, reg_lambda=15.0, min_child_weight=8,
    random_state=42, eval_metric='logloss', verbosity=0)
xgb_reg.fit(X_train, y_train)

print("[3/4] Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300, max_depth=6, min_samples_leaf=10,
    max_features='sqrt', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

print("[4/4] Neural Network...")
nn = MLPClassifier(
    hidden_layer_sizes=(64, 32), activation='relu', solver='adam',
    alpha=0.01, learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=500, early_stopping=True, validation_fraction=0.15, random_state=42)
nn.fit(X_train_scaled, y_train)

print("\nAll models trained ✓")

# Regularization comparison
print(f"\n  {'Model':20s} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8}")
print(f"  {'-'*50}")
for name, mdl, scl in [
    ('XGB (original)', xgb_orig, False), ('XGB (regularized)', xgb_reg, False),
    ('Random Forest', rf, False), ('Neural Net', nn, True)]:
    tr = mdl.score(X_train_scaled if scl else X_train, y_train)
    va = mdl.score(X_val_scaled if scl else X_val, y_val)
    te = mdl.score(X_test_scaled if scl else X_test, y_test)
    print(f"  {name:20s} {tr:>8.3f} {va:>8.3f} {te:>8.3f} {tr-va:>8.3f}")

###############################################################################
# PART D: GAME-LEVEL EVALUATION
###############################################################################

print("\n" + "="*70)
print("GAME-LEVEL EVALUATION")
print("="*70)

def chalk_predict(row): return 1 if row['DIFF_SEED'] < 0 else 0
def kadj_predict(row):  return 1 if row['DIFF_KADJ EM'] > 0 else 0
def barthag_predict(row): return 1 if row['DIFF_BARTHAG'] > 0 else 0
def tr_predict(row):    return 1 if row['DIFF_TR RATING'] > 0 else 0

# Accuracy + Log-Loss for ML models
ml_configs = {
    'XGB (original)':    (xgb_orig, False),
    'XGB (regularized)': (xgb_reg, False),
    'Random Forest':     (rf, False),
    'Neural Net':        (nn, True),
}

for split_name, df_split, X_s, y_s in [
    ('Val (2021-24)', val, X_val, y_val), ('Test (2025)', test, X_test, y_test)]:
    print(f"\n  --- {split_name} ({len(df_split)} games) ---")
    print(f"  {'Method':<22} {'Accuracy':>10} {'Log-Loss':>10}")
    print(f"  {'-'*44}")
    for name, (mdl, scl) in ml_configs.items():
        X = scaler.transform(X_s) if scl else X_s
        acc = accuracy_score(y_s, mdl.predict(X))
        ll = log_loss(y_s, mdl.predict_proba(X)[:, 1])
        print(f"  {name:<22} {acc:>10.3f} {ll:>10.3f}")
    for bl_name, bl_func in [('Chalk (Seed)', chalk_predict), ('KADJ EM', kadj_predict),
                              ('BARTHAG', barthag_predict), ('TR RATING', tr_predict)]:
        bl_pred = df_split.apply(bl_func, axis=1)
        acc = accuracy_score(y_s, bl_pred)
        print(f"  {bl_name:<22} {acc:>10.3f} {'—':>10}")
    if HAS_VEGAS:
        vg = df_split[df_split['VEGAS_PICK_A'] != -1]
        if len(vg) > 0:
            vg_correct = ((vg['VEGAS_PICK_A'] == 1) & (vg['A_WIN'] == 1)) | \
                         ((vg['VEGAS_PICK_A'] == 0) & (vg['A_WIN'] == 0))
            print(f"  {'Vegas Favorite':<22} {vg_correct.mean():>10.3f} {'—':>10}")

# --- Upset Detection ---
print(f"\n{'='*70}")
print("UPSET DETECTION (Val + Test combined)")
print(f"{'='*70}")

oos = games[games['YEAR'].between(2021, 2025)].copy()
oos_mask = oos['A_SEED'] != oos['B_SEED']
oos_diff = oos[oos_mask].copy()

is_upset = (
    ((oos_diff['A_WIN'] == 1) & (oos_diff['A_SEED'] > oos_diff['B_SEED'])) |
    ((oos_diff['A_WIN'] == 0) & (oos_diff['A_SEED'] < oos_diff['B_SEED']))
)

total_upsets = is_upset.sum()
print(f"\nUpsets in OOS data: {total_upsets}/{len(oos_diff)} ({total_upsets/len(oos_diff)*100:.1f}%)")

print(f"\n{'Method':<22} {'Predicted':>10} {'Correct':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-" * 68)

for name, (mdl, scl) in ml_configs.items():
    X_oos = scaler.transform(oos_diff[model_features]) if scl else oos_diff[model_features]
    pred = mdl.predict(X_oos)
    pred_upset = (
        ((pred == 1) & (oos_diff['A_SEED'].values > oos_diff['B_SEED'].values)) |
        ((pred == 0) & (oos_diff['A_SEED'].values < oos_diff['B_SEED'].values))
    )
    n_pred = pred_upset.sum()
    n_correct = (pred_upset & is_upset.values).sum()
    prec = n_correct / n_pred if n_pred > 0 else 0
    rec = n_correct / total_upsets
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{name:<22} {n_pred:>10} {n_correct:>8} {prec:>8.1%} {rec:>8.1%} {f1:>8.3f}")

for bl_name, bl_func in [('Chalk (Seed)', chalk_predict), ('KADJ EM', kadj_predict),
                          ('BARTHAG', barthag_predict), ('TR RATING', tr_predict)]:
    bl_pred = oos_diff.apply(bl_func, axis=1).values
    pred_upset = (
        ((bl_pred == 1) & (oos_diff['A_SEED'].values > oos_diff['B_SEED'].values)) |
        ((bl_pred == 0) & (oos_diff['A_SEED'].values < oos_diff['B_SEED'].values))
    )
    n_pred = pred_upset.sum()
    n_correct = (pred_upset & is_upset.values).sum()
    prec = n_correct / n_pred if n_pred > 0 else 0
    rec = n_correct / total_upsets
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"{bl_name:<22} {n_pred:>10} {n_correct:>8} {prec:>8.1%} {rec:>8.1%} {f1:>8.3f}")

###############################################################################
# PART E: FEATURE IMPORTANCE
###############################################################################

print(f"\n{'='*70}")
print("FEATURE IMPORTANCE")
print(f"{'='*70}")

xgb_imp = pd.Series(xgb_reg.feature_importances_, index=model_features).sort_values(ascending=False)
rf_imp = pd.Series(rf.feature_importances_, index=model_features).sort_values(ascending=False)

print("\n--- XGBoost (regularized) Top 15 ---")
print(xgb_imp.head(15).to_string())
print("\n--- Random Forest Top 15 ---")
print(rf_imp.head(15).to_string())

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
xgb_imp.head(15).plot(kind='barh', ax=axes[0], color='#3498db', edgecolor='white')
axes[0].set_title('XGBoost (Regularized) — Top 15 Features')
axes[0].invert_yaxis()
rf_imp.head(15).plot(kind='barh', ax=axes[1], color='#2ecc71', edgecolor='white')
axes[1].set_title('Random Forest — Top 15 Features')
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig("outputs/feature_importance.png")
plt.close()
print("Saved: outputs/feature_importance.png")

###############################################################################
# PART F: GAME-INDEPENDENT BRACKET SCORING (2021-2025)
###############################################################################

print(f"\n{'='*70}")
print("GAME-INDEPENDENT BRACKET SCORING")
print(f"{'='*70}")
print("(Each game scored independently — no cascading)")

round_points = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16, 2: 32}

all_method_names = ['XGB (orig)', 'XGB (reg)', 'RF', 'NN', 'Chalk', 'KADJ EM', 'BARTHAG', 'TR RATING']
yearly_scores = {n: [] for n in all_method_names}

print(f"\n{'Year':<6}", end="")
for n in all_method_names:
    print(f" {n:>9}", end="")
print()
print("-" * 82)

for year in sorted(games['YEAR'].unique()):
    if year < 2021 or year > 2025: continue
    yr = games[games['YEAR'] == year].reset_index(drop=True)
    X_yr = yr[model_features]

    preds = {
        'XGB (orig)': xgb_orig.predict(X_yr),
        'XGB (reg)':  xgb_reg.predict(X_yr),
        'RF':         rf.predict(X_yr),
        'NN':         nn.predict(scaler.transform(X_yr)),
        'Chalk':      yr.apply(chalk_predict, axis=1).values,
        'KADJ EM':    yr.apply(kadj_predict, axis=1).values,
        'BARTHAG':    yr.apply(barthag_predict, axis=1).values,
        'TR RATING':  yr.apply(tr_predict, axis=1).values,
    }

    line = f"{year:<6}"
    for name in all_method_names:
        score = 0
        for rnd, pts in round_points.items():
            rnd_games = yr[yr['CURRENT ROUND'] == rnd]
            if len(rnd_games) == 0: continue
            rnd_preds = preds[name][rnd_games.index - yr.index[0]]
            correct = (rnd_preds == rnd_games['A_WIN'].values).sum()
            score += correct * pts
        yearly_scores[name].append(score)
        line += f" {score:>9}"
    print(line)

print("-" * 82)
line = f"{'Avg':<6}"
for name in all_method_names:
    line += f" {np.mean(yearly_scores[name]):>9.1f}"
print(line)

###############################################################################
# PART G: CASCADING BRACKET SIMULATION (2021-2025)
###############################################################################

print(f"\n{'='*70}")
print("CASCADING BRACKET SIMULATION")
print(f"{'='*70}")
print("Simulates full brackets forward from R64. Early misses cascade.\n")

# Build team feature lookup for matchup prediction
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

raw_stats_to_zscore = [
    'EFG%', 'TOV%', 'OREB%', 'FTR', 'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    '2PT%', '3PT%', '2PT%D', '3PT%D', '2PTR', '3PTR', 'BLK%', 'AST%', 'FT%', 'WIN%']
season_stats = kenpom.groupby('YEAR')[raw_stats_to_zscore].agg(['mean', 'std'])
for stat in raw_stats_to_zscore:
    team_features[f'Z_{stat}'] = team_features.apply(
        lambda row: ((row[stat] - season_stats.loc[row['YEAR'], (stat, 'mean')]) /
                     season_stats.loc[row['YEAR'], (stat, 'std')]
                     if season_stats.loc[row['YEAR'], (stat, 'std')] > 0 else 0.0), axis=1)

team_features = team_features.set_index(['YEAR', 'TEAM NO'])
print(f"Team feature lookup: {len(team_features)} team-seasons")

# --- Matchup builder and pick functions ---
def build_matchup_features(year, team_a_no, team_b_no):
    try:
        a = team_features.loc[(year, team_a_no)]
        b = team_features.loc[(year, team_b_no)]
    except KeyError:
        return None
    fmap = {
        'KADJ EM': 'DIFF_KADJ EM', 'TR RATING': 'DIFF_TR RATING',
        'BARTHAG': 'DIFF_BARTHAG', 'KADJ O': 'DIFF_KADJ O',
        'KADJ D': 'DIFF_KADJ D', 'KADJ T': 'DIFF_KADJ T',
        'Z_EFG%': 'DIFF_Z_EFG%', 'Z_TOV%': 'DIFF_Z_TOV%',
        'Z_OREB%': 'DIFF_Z_OREB%', 'Z_FTR': 'DIFF_Z_FTR',
        'Z_EFG%D': 'DIFF_Z_EFG%D', 'Z_TOV%D': 'DIFF_Z_TOV%D',
        'Z_DREB%': 'DIFF_Z_DREB%', 'Z_FTRD': 'DIFF_Z_FTRD',
        'Z_3PT%': 'DIFF_Z_3PT%', 'Z_2PT%': 'DIFF_Z_2PT%',
        'Z_3PT%D': 'DIFF_Z_3PT%D', 'Z_2PT%D': 'DIFF_Z_2PT%D',
        'Z_FT%': 'DIFF_Z_FT%', 'ELITE SOS': 'DIFF_ELITE SOS',
        'WAB': 'DIFF_WAB', 'SOS RATING': 'DIFF_SOS RATING',
        'Q1 PLUS Q2 W': 'DIFF_Q1 PLUS Q2 W', 'R SCORE': 'DIFF_R SCORE',
        'V 1-25 WINS': 'DIFF_V 1-25 WINS', 'LUCK RATING': 'DIFF_LUCK RATING',
        'CONSISTENCY TR RATING': 'DIFF_CONSISTENCY TR RATING',
        'TALENT': 'DIFF_TALENT', 'EXP': 'DIFF_EXP',
        'AVG HGT': 'DIFF_AVG HGT', 'SEED': 'DIFF_SEED',
        'ELO': 'DIFF_ELO', 'Z_WIN%': 'DIFF_Z_WIN%',
    }
    return pd.Series({diff_name: a[raw] - b[raw] for raw, diff_name in fmap.items()})

def chalk_pick_sim(yr, a, b):
    return a if team_features.loc[(yr, a), 'SEED'] <= team_features.loc[(yr, b), 'SEED'] else b

def metric_pick_sim(yr, a, b, metric):
    return a if team_features.loc[(yr, a), metric] >= team_features.loc[(yr, b), metric] else b

def ml_pick_sim(yr, a, b, model, use_scaler=False):
    feat = build_matchup_features(yr, a, b)
    if feat is None: return a
    X = feat[model_features].values.reshape(1, -1)
    if use_scaler: X = scaler.transform(X)
    return a if model.predict(X)[0] == 1 else b

def get_team_name(yr, tno):
    r = kenpom[(kenpom['YEAR'] == yr) & (kenpom['TEAM NO'] == tno)]
    return r.iloc[0]['TEAM'] if len(r) > 0 else f"Team {tno}"

def get_team_seed(yr, tno):
    try: return int(team_features.loc[(yr, tno), 'SEED'])
    except: return 0

# --- Bracket simulation engine ---
def simulate_bracket(year, pick_func):
    r64 = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 64)].reset_index(drop=True)
    r32_data = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 32)].reset_index(drop=True)

    # Find bye teams (in R32 but not R64)
    r64_teams = set()
    for _, g in r64.iterrows():
        r64_teams.add(int(g['A_TEAM_NO'])); r64_teams.add(int(g['B_TEAM_NO']))
    r32_teams = set()
    for _, g in r32_data.iterrows():
        r32_teams.add(int(g['A_TEAM_NO'])); r32_teams.add(int(g['B_TEAM_NO']))
    bye_teams = r32_teams - r64_teams

    # Predict R64 winners
    r64_winners = []
    for _, g in r64.iterrows():
        w = pick_func(year, int(g['A_TEAM_NO']), int(g['B_TEAM_NO']))
        r64_winners.append(w)

    # Build R32 matchups using actual R32 bracket structure
    # For each R32 game: if a team is a bye team, use it directly;
    # otherwise find the predicted R64 winner from that side
    r32_matchups = []
    used_r64_winners = set()
    
    for _, g in r32_data.iterrows():
        a_no, b_no = int(g['A_TEAM_NO']), int(g['B_TEAM_NO'])
        
        # For each side of the R32 game, find the predicted team
        sides = []
        for team_no in [a_no, b_no]:
            if team_no in bye_teams:
                sides.append(team_no)
            else:
                # Find which R64 winner goes here
                # Check if this team played in R64 — if they won, they're in r64_winners
                # If they lost, their opponent is in r64_winners
                found = False
                for _, r64g in r64.iterrows():
                    r64_a, r64_b = int(r64g['A_TEAM_NO']), int(r64g['B_TEAM_NO'])
                    if team_no in (r64_a, r64_b):
                        # This R64 game feeds into this R32 slot
                        # Use the predicted winner
                        for w in r64_winners:
                            if w in (r64_a, r64_b) and w not in used_r64_winners:
                                sides.append(w)
                                used_r64_winners.add(w)
                                found = True
                                break
                        if found: break
                if not found:
                    # Fallback: just use the first available R64 winner
                    for w in r64_winners:
                        if w not in used_r64_winners:
                            sides.append(w)
                            used_r64_winners.add(w)
                            break
        
        if len(sides) == 2:
            r32_matchups.append((sides[0], sides[1]))

    # Simulate R32 through Championship
    results = {'rounds': {}, 'champion': None, 'f4_teams': []}
    current = r32_matchups

    for rnd_name in ['R32', 'S16', 'E8', 'F4', 'Final']:
        rnd_winners = []
        for a, b in current:
            w = pick_func(year, a, b)
            rnd_winners.append(w)
        results['rounds'][rnd_name] = rnd_winners

        if rnd_name == 'E8':
            results['f4_teams'] = [get_team_name(year, t) for t in rnd_winners]
        if len(rnd_winners) == 1:
            results['champion'] = get_team_name(year, rnd_winners[0])
        if len(rnd_winners) >= 2:
            current = [(rnd_winners[i], rnd_winners[i+1]) for i in range(0, len(rnd_winners), 2)]

    return results

# --- Run simulations ---
sim_methods = {
    'XGB (reg)':   lambda yr, a, b: ml_pick_sim(yr, a, b, xgb_reg),
    'Random Forest': lambda yr, a, b: ml_pick_sim(yr, a, b, rf),
    'Neural Net':  lambda yr, a, b: ml_pick_sim(yr, a, b, nn, use_scaler=True),
    'Chalk':       lambda yr, a, b: chalk_pick_sim(yr, a, b),
    'KADJ EM':     lambda yr, a, b: metric_pick_sim(yr, a, b, 'KADJ EM'),
    'BARTHAG':     lambda yr, a, b: metric_pick_sim(yr, a, b, 'BARTHAG'),
    'TR RATING':   lambda yr, a, b: metric_pick_sim(yr, a, b, 'TR RATING'),
}

champ_card = {n: [] for n in sim_methods}
f4_card = {n: [] for n in sim_methods}
holdout_years = [yr for yr in sorted(games['YEAR'].unique()) if 2021 <= yr <= 2025]

for year in holdout_years:
    # Get actual results
    f4g = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 4)]
    fg = games[(games['YEAR'] == year) & (games['CURRENT ROUND'] == 2)]
    actual_f4 = set()
    for _, g in f4g.iterrows(): actual_f4.add(g['A_TEAM']); actual_f4.add(g['B_TEAM'])
    actual_champ = fg.iloc[0]['A_TEAM'] if fg.iloc[0]['A_WIN'] == 1 else fg.iloc[0]['B_TEAM']

    print(f"\n{'='*70}")
    print(f"{year} — Actual F4: {actual_f4} | Champion: {actual_champ}")
    print(f"{'='*70}")

    for name, pfunc in sim_methods.items():
        sim = simulate_bracket(year, pfunc)
        cc = 1 if sim['champion'] == actual_champ else 0
        champ_card[name].append(cc)
        f4_overlap = len(set(sim['f4_teams']) & actual_f4)
        f4_card[name].append(f4_overlap)
        mark = "✓" if cc else "✗"
        f4_str = ", ".join(sim['f4_teams'])
        print(f"  {name:15s} | F4: {f4_str:55s} ({f4_overlap}/4) | Champ: {sim['champion']:18s} {mark}")

# Scorecards
print(f"\n--- Champion Scorecard (Cascading) ---")
print(f"{'Method':<18} {'Total':>6}")
print("-" * 26)
for n in sim_methods:
    print(f"{n:<18} {sum(champ_card[n]):>4}/{len(champ_card[n])}")

print(f"\n--- F4 Teams Correct (Cascading) ---")
print(f"{'Method':<18} {'Total':>8}")
print("-" * 28)
for n in sim_methods:
    print(f"{n:<18} {sum(f4_card[n]):>4}/{len(holdout_years)*4}")

###############################################################################
# PART H: VEGAS COMPARISON + BETTING STRATEGY
###############################################################################

if HAS_VEGAS:
    print(f"\n{'='*70}")
    print("VEGAS COMPARISON + BETTING STRATEGY")
    print(f"{'='*70}")

    # Out-of-sample with Vegas
    oos_v = games[(games['YEAR'].between(2021, 2025)) &
                  (games['SPREAD'].notna()) & (games['VEGAS_PICK_A'] != -1)].copy()

    oos_v['VEGAS_CORRECT'] = ((oos_v['VEGAS_PICK_A'] == 1) & (oos_v['A_WIN'] == 1)) | \
                              ((oos_v['VEGAS_PICK_A'] == 0) & (oos_v['A_WIN'] == 0))
    oos_v['XGB_REG_PRED'] = xgb_reg.predict(oos_v[model_features])
    oos_v['RF_PRED'] = rf.predict(oos_v[model_features])
    oos_v['NN_PRED'] = nn.predict(scaler.transform(oos_v[model_features]))

    print(f"\nOOS games with Vegas lines: {len(oos_v)}")
    print(f"\n  --- Out-of-Sample Accuracy Ranking ---")
    ranking = [
        ('XGB (regularized)', accuracy_score(oos_v['A_WIN'], oos_v['XGB_REG_PRED'])),
        ('Random Forest', accuracy_score(oos_v['A_WIN'], oos_v['RF_PRED'])),
        ('Neural Net', accuracy_score(oos_v['A_WIN'], oos_v['NN_PRED'])),
        ('Vegas Favorite', oos_v['VEGAS_CORRECT'].mean()),
        ('Chalk (Seed)', accuracy_score(oos_v['A_WIN'],
            oos_v.apply(chalk_predict, axis=1))),
        ('KADJ EM', accuracy_score(oos_v['A_WIN'],
            oos_v.apply(kadj_predict, axis=1))),
        ('BARTHAG', accuracy_score(oos_v['A_WIN'],
            oos_v.apply(barthag_predict, axis=1))),
        ('TR RATING', accuracy_score(oos_v['A_WIN'],
            oos_v.apply(tr_predict, axis=1))),
    ]
    for rank, (n, acc) in enumerate(sorted(ranking, key=lambda x: -x[1]), 1):
        bar = "█" * int(acc * 40)
        print(f"  {rank}. {n:<22} {acc:.3f}  {bar}")

    # --- Betting Strategy ---
    print(f"\n  --- Hypothetical Betting (OOS, flat $100 bets against Vegas) ---")

    def spread_to_ml_odds(spread):
        s = abs(spread)
        if s <= 1: return (95, 105)
        elif s <= 2.5: return (85, 120)
        elif s <= 4: return (75, 140)
        elif s <= 6: return (65, 175)
        elif s <= 8: return (55, 220)
        elif s <= 10: return (45, 270)
        elif s <= 14: return (35, 350)
        elif s <= 20: return (20, 500)
        else: return (10, 800)

    print(f"\n  {'Method':<22} {'Bets':>6} {'Wins':>6} {'Win%':>7} {'Profit':>10} {'ROI':>8}")
    print(f"  {'-'*63}")

    for name, pred_col in [('XGB (regularized)', 'XGB_REG_PRED'),
                            ('Random Forest', 'RF_PRED'), ('Neural Net', 'NN_PRED')]:
        model_pick_a = oos_v[pred_col] == 1
        vegas_pick_a = oos_v['VEGAS_PICK_A'] == 1
        disagree = model_pick_a != vegas_pick_a
        bet_games = oos_v[disagree]

        if len(bet_games) == 0: continue
        total_profit, wins = 0, 0
        for _, g in bet_games.iterrows():
            _, dog_pay = spread_to_ml_odds(g['SPREAD'])
            mp_a = (oos_v.loc[g.name, pred_col] == 1)
            correct = (mp_a and g['A_WIN'] == 1) or (not mp_a and g['A_WIN'] == 0)
            if correct:
                total_profit += dog_pay; wins += 1
            else:
                total_profit -= 100
        n_bets = len(bet_games)
        roi = total_profit / (n_bets * 100) * 100
        print(f"  {name:<22} {n_bets:>6} {wins:>6} {wins/n_bets*100:>6.1f}% {total_profit:>+10.0f} {roi:>+7.1f}%")

    # Add baseline betting too
    for name, bl_func in [('KADJ EM', kadj_predict), ('TR RATING', tr_predict)]:
        bl_pred = oos_v.apply(bl_func, axis=1)
        model_pick_a = bl_pred == 1
        vegas_pick_a = oos_v['VEGAS_PICK_A'] == 1
        disagree = model_pick_a != vegas_pick_a
        bet_games = oos_v[disagree]
        if len(bet_games) == 0: continue
        total_profit, wins = 0, 0
        for _, g in bet_games.iterrows():
            _, dog_pay = spread_to_ml_odds(g['SPREAD'])
            mp_a = (bl_pred.loc[g.name] == 1)
            correct = (mp_a and g['A_WIN'] == 1) or (not mp_a and g['A_WIN'] == 0)
            if correct: total_profit += dog_pay; wins += 1
            else: total_profit -= 100
        n_bets = len(bet_games)
        roi = total_profit / (n_bets * 100) * 100
        print(f"  {name:<22} {n_bets:>6} {wins:>6} {wins/n_bets*100:>6.1f}% {total_profit:>+10.0f} {roi:>+7.1f}%")

    # Spread magnitude breakdown
    print(f"\n  --- Accuracy by Spread Magnitude ---")
    oos_v['ABS_SPREAD'] = oos_v['SPREAD'].abs()
    bins = [(0, 3, '0-3 (toss-up)'), (3, 7, '3-7 (lean)'), (7, 12, '7-12 (solid)'),
            (12, 20, '12-20 (heavy)'), (20, 40, '20+ (blowout)')]
    print(f"  {'Range':<17} {'Games':>6} {'XGB_reg':>8} {'RF':>8} {'Vegas':>8} {'Chalk':>8}")
    print(f"  {'-'*54}")
    for lo, hi, label in bins:
        mask = (oos_v['ABS_SPREAD'] >= lo) & (oos_v['ABS_SPREAD'] < hi)
        sub = oos_v[mask]
        if len(sub) == 0: continue
        xgb_a = accuracy_score(sub['A_WIN'], sub['XGB_REG_PRED'])
        rf_a = accuracy_score(sub['A_WIN'], sub['RF_PRED'])
        v_a = sub['VEGAS_CORRECT'].mean()
        c_a = accuracy_score(sub['A_WIN'], sub.apply(chalk_predict, axis=1))
        print(f"  {label:<17} {len(sub):>6} {xgb_a:>8.3f} {rf_a:>8.3f} {v_a:>8.3f} {c_a:>8.3f}")

###############################################################################
# PART I: SAVE OUTPUTS
###############################################################################

print(f"\n{'='*70}")
print("SAVING OUTPUTS")
print(f"{'='*70}")

# Save 2025 predictions
test_out = test[['YEAR', 'CURRENT ROUND', 'A_TEAM', 'A_SEED', 'B_TEAM', 'B_SEED', 'A_WIN']].copy().reset_index(drop=True)
test_reset = test.reset_index(drop=True)
test_out['XGB_REG_PRED'] = xgb_reg.predict(test_reset[model_features])
test_out['RF_PRED'] = rf.predict(test_reset[model_features])
test_out['NN_PRED'] = nn.predict(scaler.transform(test_reset[model_features]))
test_out['CHALK_PRED'] = test_reset.apply(chalk_predict, axis=1)
test_out.to_csv("outputs/predictions_2025.csv", index=False)
print("Saved: outputs/predictions_2025.csv")

# Save enriched games with Vegas + model predictions
games['XGB_REG_PRED'] = xgb_reg.predict(games[model_features])
games['RF_PRED'] = rf.predict(games[model_features])
games['NN_PRED'] = nn.predict(scaler.transform(games[model_features]))
games['XGB_REG_PROB'] = xgb_reg.predict_proba(games[model_features])[:, 1]
games['RF_PROB'] = rf.predict_proba(games[model_features])[:, 1]
games.to_csv("outputs/games_enriched.csv", index=False)
print("Saved: outputs/games_enriched.csv")

print(f"\n{'='*70}")
print("SCRIPT 02 COMPLETE")
print(f"{'='*70}")
print(f"\nModels: XGB (original + regularized), Random Forest, Neural Network")
print(f"Features: {len(model_features)}")
print(f"Train: 2008-2019 | Val: 2021-2024 | Test: 2025")
print(f"Output files: outputs/predictions_2025.csv, outputs/games_enriched.csv,")
print(f"  outputs/feature_importance.png")