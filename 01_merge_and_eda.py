'''
PHASE 1: DATA MERGE + EDA
2026 Brandeis Datathon — Team ASA (Anokh, Samiya, Aastha)
- Merges tournament outcomes with team metrics
- Engineers differential features
- Runs EDA to answer Q1 (seed performance) and Q2 (features)
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --------------------------------------------------------------------------- #
# 0. SETUP
# --------------------------------------------------------------------------- #

os.makedirs("outputs", exist_ok=True)
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['figure.dpi'] = 150
DATA_DIR = "datathon set"

# --------------------------------------------------------------------------- #
# 1. LOAD RAW DATA
# --------------------------------------------------------------------------- #

matchups = pd.read_csv(f"{DATA_DIR}/Tournament Matchups.csv")
print(f"Matchups loaded: {matchups.shape[0]} rows, years {matchups['YEAR'].min()}-{matchups['YEAR'].max()}")

kenpom = pd.read_csv(f"{DATA_DIR}/KenPom Barttorvik.csv")
print(f"KenPom loaded: {kenpom.shape[0]} rows, {kenpom.shape[1]} columns")

resumes = pd.read_csv(f"{DATA_DIR}/Resumes.csv")
print(f"Resumes loaded: {resumes.shape[0]} rows")

teamrankings = pd.read_csv(f"{DATA_DIR}/TeamRankings.csv")
print(f"TeamRankings loaded: {teamrankings.shape[0]} rows")

'''Note: excluded CSVs with incomplete year coverage (538 Ratings, EvanMiya, RPPF)'''

# --------------------------------------------------------------------------- #
# 2. DATA VALIDATION AND CLEANING
# --------------------------------------------------------------------------- #

print("\n" + "="*70)
print("DATA VALIDATION AND CLEANING")
print("="*70)

# --- 2a. Missing values ---
print("\n--- Missing Value Check ---")
for name, df in [("Matchups", matchups), ("KenPom", kenpom), ("Resumes", resumes), ("TeamRankings", teamrankings)]:
    n_missing = df.isnull().sum().sum()
    cols_missing = df.isnull().sum()[df.isnull().sum() > 0]
    if n_missing == 0:
        print(f"  {name}: No missing values")
    else:
        print(f"  {name}: {n_missing} missing values across {len(cols_missing)} columns")
        print(f"    {cols_missing.to_dict()}")

# --- 2b. Duplicate check ---
print("\n--- Duplicate Check ---")
print(f"  Matchups duplicate rows: {matchups.duplicated().sum()}")
print(f"  KenPom duplicate YEAR+TEAM NO: {kenpom.groupby(['YEAR','TEAM NO']).size().gt(1).sum()}")
print(f"  Resumes duplicate YEAR+TEAM NO: {resumes.groupby(['YEAR','TEAM NO']).size().gt(1).sum()}")
print(f"  TeamRankings duplicate YEAR+TEAM NO: {teamrankings.groupby(['YEAR','TEAM NO']).size().gt(1).sum()}")

# --- 2c. Range validation ---
print("\n--- Range Validation ---")
assert matchups['SEED'].between(1, 16).all(), "Invalid seed values in matchups"
assert matchups['SCORE'].dropna().gt(0).all(), "Non-positive scores found in matchups"
assert kenpom['SEED'].between(1, 16).all(), "Invalid seed values in KenPom"
print(f"  Seeds: all in [1, 16] range ✓")
print(f"  Scores: all positive (range {matchups['SCORE'].min()}-{matchups['SCORE'].max()}) ✓")

numeric_kenpom = kenpom.select_dtypes(include=[np.number])
n_inf = np.isinf(numeric_kenpom).sum().sum()
assert n_inf == 0, f"Found {n_inf} infinite values in KenPom"
print(f"  KenPom numeric columns: no infinite values ✓")

pct_cols = [c for c in kenpom.columns if '%' in c and 'RANK' not in c]
out_of_range = []
for col in pct_cols:
    if kenpom[col].min() < 0 or kenpom[col].max() > 100:
        out_of_range.append(col)
if out_of_range:
    print(f"  WARNING: Percentage columns outside [0,100]: {out_of_range}")
else:
    print(f"  Percentage columns: all within [0, 100] range ✓")

# --- 2d. Whitespace cleaning ---
print("\n--- Whitespace Cleaning ---")
ws_matchups = (matchups['TEAM'] != matchups['TEAM'].str.strip()).sum()
ws_kenpom = (kenpom['TEAM'] != kenpom['TEAM'].str.strip()).sum()
ws_resumes = (resumes['TEAM'] != resumes['TEAM'].str.strip()).sum()
ws_tr = (teamrankings['TEAM'] != teamrankings['TEAM'].str.strip()).sum()
print(f"  Team names with trailing whitespace: Matchups={ws_matchups}, KenPom={ws_kenpom}, Resumes={ws_resumes}, TeamRankings={ws_tr}")

for df in [matchups, kenpom, resumes, teamrankings]:
    str_cols = df.select_dtypes(include=['object', 'string']).columns
    for col in str_cols:
        df[col] = df[col].str.strip()

print(f"  Stripped whitespace from all string columns ✓")

# --- 2e. Structural validation ---
print("\n--- Structural Validation ---")
rows_per_year = matchups.groupby('YEAR').size()
odd_years = rows_per_year[rows_per_year % 2 != 0]
if len(odd_years) > 0:
    print(f"  WARNING: Odd row counts in years: {odd_years.to_dict()}")
else:
    print(f"  All years have even row counts (valid game pairs) ✓")

# 2021 had reduced First Four due to COVID bubble — 62 games instead of 63
games_per_year = rows_per_year // 2
anomalies = games_per_year[games_per_year != 63]
if len(anomalies) > 0:
    print(f"  Note: 2020 had no March Madness tournament")
    print(f"  Note: Non-standard game counts: {anomalies.to_dict()}")
    print(f"    (2021 had a reduced field due to COVID bubble format)")

# --- 2f. Cross-file key consistency ---
print("\n--- Cross-File Consistency ---")
m_keys = set(zip(matchups['YEAR'], matchups['TEAM NO']))
k_keys = set(zip(kenpom['YEAR'], kenpom['TEAM NO']))
r_keys = set(zip(resumes['YEAR'], resumes['TEAM NO']))
t_keys = set(zip(teamrankings['YEAR'], teamrankings['TEAM NO']))

missing_kenpom = m_keys - k_keys
missing_resume = m_keys - r_keys
missing_tr = m_keys - t_keys

print(f"  Matchup team-years: {len(m_keys)}")
print(f"  KenPom team-years: {len(k_keys)}")
print(f"  Resume team-years: {len(r_keys)}")
print(f"  TeamRankings team-years: {len(t_keys)}")

if missing_kenpom:
    print(f"  WARNING: {len(missing_kenpom)} matchup teams missing from KenPom: {missing_kenpom}")
else:
    print(f"  All matchup teams have KenPom data ✓")

if missing_resume:
    print(f"  WARNING: {len(missing_resume)} matchup teams missing from Resumes: {missing_resume}")
else:
    print(f"  All matchup teams have Resume data ✓")

if missing_tr:
    print(f"  WARNING: {len(missing_tr)} matchup teams missing from TeamRankings: {missing_tr}")
else:
    print(f"  All matchup teams have TeamRankings data ✓")

print("\nData validation complete — proceeding to game construction.\n")

# --------------------------------------------------------------------------- #
# 3. BUILD GAME-LEVEL DATASET
# --------------------------------------------------------------------------- #

# Each game = 2 consecutive rows in matchups (paired by BY YEAR NO).
# Higher BY YEAR NO = Team A, lower = Team B. Winner determined by score.

matchups_sorted = matchups.sort_values(
    ['YEAR', 'CURRENT ROUND', 'BY YEAR NO'], 
    ascending=[True, False, False]
).reset_index(drop=True)

games_list = []

for i in range(0, len(matchups_sorted), 2):
    a = matchups_sorted.iloc[i]
    b = matchups_sorted.iloc[i + 1]
    
    assert a['YEAR'] == b['YEAR'], f"Year mismatch at row {i}"
    assert a['CURRENT ROUND'] == b['CURRENT ROUND'], f"Round mismatch at row {i}"
    
    a_won = a['SCORE'] > b['SCORE']
    
    games_list.append({
        'YEAR': a['YEAR'],
        'CURRENT ROUND': a['CURRENT ROUND'],
        'A_TEAM_NO': a['TEAM NO'], 'A_TEAM': a['TEAM'],
        'A_SEED': a['SEED'], 'A_SCORE': a['SCORE'], 'A_ROUND': a['ROUND'],
        'B_TEAM_NO': b['TEAM NO'], 'B_TEAM': b['TEAM'],
        'B_SEED': b['SEED'], 'B_SCORE': b['SCORE'], 'B_ROUND': b['ROUND'],
        'A_WIN': int(a_won),
        'SCORE_DIFF': a['SCORE'] - b['SCORE'],
    })

games = pd.DataFrame(games_list)
print(f"\nGames built: {len(games)} total games across {games['YEAR'].nunique()} tournaments")
print(f"  Per-round counts:")
round_labels = {64: 'R64', 32: 'R32', 16: 'S16', 8: 'E8', 4: 'F4', 2: 'Final'}
for r in [64, 32, 16, 8, 4, 2]:
    ct = len(games[games['CURRENT ROUND'] == r])
    print(f"    {round_labels[r]}: {ct} games")

# --------------------------------------------------------------------------- #
# 4. MERGE KENPOM FEATURES ONTO GAMES
# --------------------------------------------------------------------------- #

feature_cols = [
    'YEAR', 'TEAM NO',
    # Adjusted efficiency
    'KADJ O', 'KADJ D', 'KADJ EM',
    'BADJ EM', 'BADJ O', 'BADJ D', 'BARTHAG',
    # Tempo
    'KADJ T', 'RAW T',
    # Four factors (offense)
    'EFG%', 'TOV%', 'OREB%', 'FTR',
    # Four factors (defense)
    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    # Shooting profile
    '2PT%', '3PT%', '2PT%D', '3PT%D', '2PTR', '3PTR',
    # Other
    'BLK%', 'AST%', 'WIN%', 'ELITE SOS', 'WAB',
    'EXP', 'TALENT', 'AVG HGT', 'EFF HGT', 'FT%',
    'GAMES', 'W', 'L',
]

kenpom_features = kenpom[feature_cols].copy()

# Merge for Team A
games_merged = games.merge(
    kenpom_features,
    left_on=['YEAR', 'A_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'],
    how='left', suffixes=('', '_drop')
).drop(columns=['TEAM NO'])
rename_a = {col: f'A_{col}' for col in feature_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_a)

# Merge for Team B
games_merged = games_merged.merge(
    kenpom_features,
    left_on=['YEAR', 'B_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'],
    how='left', suffixes=('', '_drop')
).drop(columns=['TEAM NO'])
rename_b = {col: f'B_{col}' for col in feature_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_b)

# --------------------------------------------------------------------------- #
# 5a. MERGE RESUME DATA
# --------------------------------------------------------------------------- #

resume_cols = ['YEAR', 'TEAM NO', 'NET RPI', 'ELO', 'Q1 W', 'Q2 W', 
               'Q1 PLUS Q2 W', 'Q3 Q4 L', 'R SCORE']
resume_features = resumes[resume_cols].copy()

games_merged = games_merged.merge(
    resume_features, left_on=['YEAR', 'A_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'], how='left'
).drop(columns=['TEAM NO'])
rename_a_res = {col: f'A_{col}' for col in resume_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_a_res)

games_merged = games_merged.merge(
    resume_features, left_on=['YEAR', 'B_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'], how='left'
).drop(columns=['TEAM NO'])
rename_b_res = {col: f'B_{col}' for col in resume_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_b_res)

# --------------------------------------------------------------------------- #
# 5b. MERGE TEAMRANKINGS DATA
# --------------------------------------------------------------------------- #

# Independent rating system with luck + consistency metrics KenPom doesn't capture
tr_cols = ['YEAR', 'TEAM NO', 
           'TR RATING', 'SOS RATING', 'LUCK RATING',
           'CONSISTENCY TR RATING', 'V 1-25 WINS', 'V 1-25 LOSS',
]
tr_features = teamrankings[tr_cols].copy()

games_merged = games_merged.merge(
    tr_features, left_on=['YEAR', 'A_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'], how='left'
).drop(columns=['TEAM NO'])
rename_a_tr = {col: f'A_{col}' for col in tr_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_a_tr)

games_merged = games_merged.merge(
    tr_features, left_on=['YEAR', 'B_TEAM_NO'],
    right_on=['YEAR', 'TEAM NO'], how='left'
).drop(columns=['TEAM NO'])
rename_b_tr = {col: f'B_{col}' for col in tr_cols if col not in ['YEAR', 'TEAM NO']}
games_merged = games_merged.rename(columns=rename_b_tr)

print(f"TeamRankings merged: added {len(tr_cols) - 2} features per team")

# --------------------------------------------------------------------------- #
# 5c. SEASONAL Z-SCORE NORMALIZATION FOR RAW STATS
# --------------------------------------------------------------------------- #

# Z-score each raw stat within its season: z = (value - season_mean) / season_std
raw_stats_to_zscore = [
    'EFG%', 'TOV%', 'OREB%', 'FTR',
    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    '2PT%', '3PT%', '2PT%D', '3PT%D', '2PTR', '3PTR',
    'BLK%', 'AST%', 'FT%', 'WIN%',
]

print(f"\n--- Seasonal Z-Score Normalization ---")
print(f"Normalizing {len(raw_stats_to_zscore)} raw stats within each season...")

season_stats = kenpom.groupby('YEAR')[raw_stats_to_zscore].agg(['mean', 'std'])

for stat in raw_stats_to_zscore:
    for prefix in ['A_', 'B_']:
        raw_col = f'{prefix}{stat}'
        z_col = f'{prefix}Z_{stat}'
        
        if raw_col not in games_merged.columns:
            continue
            
        games_merged[z_col] = games_merged.apply(
            lambda row: (
                (row[raw_col] - season_stats.loc[row['YEAR'], (stat, 'mean')]) / 
                season_stats.loc[row['YEAR'], (stat, 'std')]
                if season_stats.loc[row['YEAR'], (stat, 'std')] > 0 
                else 0.0
            ), axis=1
        )

z_count = len([c for c in games_merged.columns if '_Z_' in c])
print(f"Created {z_count} z-scored columns ({z_count // 2} per team)")

# Sanity check
sample_year = 2025
sample_stat = '3PT%'
a_raw = games_merged[games_merged['YEAR'] == sample_year][f'A_{sample_stat}']
a_z = games_merged[games_merged['YEAR'] == sample_year][f'A_Z_{sample_stat}']
print(f"\nSanity check — {sample_year} 3PT% (Team A):")
print(f"  Raw:    mean={a_raw.mean():.1f}, std={a_raw.std():.1f}")
print(f"  Z-score: mean={a_z.mean():.2f}, std={a_z.std():.2f}")
print(f"  (Z-scores should be ~centered at 0 with std ~1)")

# --------------------------------------------------------------------------- #
# 6. ENGINEER DIFFERENTIAL FEATURES
# --------------------------------------------------------------------------- #

# Calculate matchup differentials (A - B), which are more predictive than raw values.
diff_features = [
    # Adjusted metrics (already season-normalized)
    'KADJ O', 'KADJ D', 'KADJ EM', 'BADJ EM', 'BADJ O', 'BADJ D',
    'BARTHAG', 'KADJ T',
    # Raw four factors + shooting
    'EFG%', 'TOV%', 'OREB%', 'FTR',
    'EFG%D', 'TOV%D', 'DREB%', 'FTRD',
    '2PT%', '3PT%', '2PT%D', '3PT%D',
    'BLK%', 'AST%', 'WIN%', 'FT%',
    # Z-scored versions
    'Z_EFG%', 'Z_TOV%', 'Z_OREB%', 'Z_FTR',
    'Z_EFG%D', 'Z_TOV%D', 'Z_DREB%', 'Z_FTRD',
    'Z_2PT%', 'Z_3PT%', 'Z_2PT%D', 'Z_3PT%D',
    'Z_2PTR', 'Z_3PTR',
    'Z_BLK%', 'Z_AST%', 'Z_FT%', 'Z_WIN%',
    # Non-rate features
    'ELITE SOS', 'WAB', 'EXP', 'TALENT', 'AVG HGT', 'EFF HGT',
    # Resume features
    'NET RPI', 'ELO', 'Q1 PLUS Q2 W', 'R SCORE',
    # TeamRankings features
    'TR RATING', 'SOS RATING', 'LUCK RATING', 
    'CONSISTENCY TR RATING', 'V 1-25 WINS',
]

for feat in diff_features:
    a_col = f'A_{feat}'
    b_col = f'B_{feat}'
    if a_col in games_merged.columns and b_col in games_merged.columns:
        games_merged[f'DIFF_{feat}'] = games_merged[a_col] - games_merged[b_col]

games_merged['DIFF_SEED'] = games_merged['A_SEED'] - games_merged['B_SEED']

print(f"\nFinal merged dataset: {games_merged.shape[0]} games, {games_merged.shape[1]} columns")
print(f"Missing values in key diff features:")
diff_cols = [c for c in games_merged.columns if c.startswith('DIFF_')]
print(games_merged[diff_cols].isnull().sum()[games_merged[diff_cols].isnull().sum() > 0])

# --------------------------------------------------------------------------- #
# 7. SAVE MERGED DATASET
# --------------------------------------------------------------------------- #

games_merged.to_csv("outputs/games_merged.csv", index=False)
print(f"\nSaved: outputs/games_merged.csv")

###############################################################################
# EDA SECTION
###############################################################################

print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

# --------------------------------------------------------------------------- #
# EDA 1: SEED PERFORMANCE (Q1)
# --------------------------------------------------------------------------- #

# --- 1a. Overall win rate by seed ---
a_records = games_merged[['YEAR', 'A_TEAM', 'A_SEED', 'A_WIN', 'CURRENT ROUND']].copy()
a_records.columns = ['YEAR', 'TEAM', 'SEED', 'WIN', 'ROUND']
b_records = games_merged[['YEAR', 'B_TEAM', 'B_SEED', 'A_WIN', 'CURRENT ROUND']].copy()
b_records.columns = ['YEAR', 'TEAM', 'SEED', 'WIN', 'ROUND']
b_records['WIN'] = 1 - b_records['WIN']

all_records = pd.concat([a_records, b_records], ignore_index=True)

seed_stats = all_records.groupby('SEED').agg(
    GAMES=('WIN', 'count'),
    WINS=('WIN', 'sum'),
).reset_index()
seed_stats['WIN%'] = (seed_stats['WINS'] / seed_stats['GAMES'] * 100).round(1)

print("\n--- Seed Win Rates (All Rounds Combined) ---")
print(seed_stats.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(seed_stats['SEED'], seed_stats['WIN%'], 
              color=sns.color_palette("RdYlGn_r", 16), edgecolor='white')
ax.set_xlabel('Seed', fontsize=12)
ax.set_ylabel('Win Rate (%)', fontsize=12)
ax.set_title('Tournament Win Rate by Seed (2008–2025)', fontsize=14)
ax.set_xticks(range(1, 17))
ax.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
for bar, pct in zip(bars, seed_stats['WIN%']):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
            f'{pct}%', ha='center', va='bottom', fontsize=9)
ax.legend()
plt.tight_layout()
plt.savefig("outputs/eda_seed_winrate.png")
plt.close()
print("Saved: outputs/eda_seed_winrate.png")

# --- 1b. Seed win rate by round (heatmap) ---
seed_round = all_records.groupby(['SEED', 'ROUND']).agg(
    GAMES=('WIN', 'count'),
    WINS=('WIN', 'sum'),
).reset_index()
seed_round['WIN%'] = (seed_round['WINS'] / seed_round['GAMES'] * 100).round(1)

heatmap_data = seed_round.pivot(index='SEED', columns='ROUND', values='WIN%')
heatmap_data = heatmap_data.rename(columns=round_labels)
col_order = ['R64', 'R32', 'S16', 'E8', 'F4', 'Final']
heatmap_data = heatmap_data[[c for c in col_order if c in heatmap_data.columns]]

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='RdYlGn', 
            center=50, ax=ax, linewidths=0.5,
            cbar_kws={'label': 'Win Rate (%)'})
ax.set_title('Win Rate by Seed × Round (2008–2025)', fontsize=14)
ax.set_ylabel('Seed', fontsize=12)
ax.set_xlabel('Round', fontsize=12)
plt.tight_layout()
plt.savefig("outputs/eda_seed_round_heatmap.png")
plt.close()
print("Saved: outputs/eda_seed_round_heatmap.png")

# --- 1c. Over/underperformers vs seed expectation ---
seed_avg_wins = all_records.groupby('SEED')['WIN'].mean()

team_tourney = all_records.groupby(['YEAR', 'TEAM', 'SEED'])['WIN'].agg(['sum', 'count']).reset_index()
team_tourney.columns = ['YEAR', 'TEAM', 'SEED', 'ACTUAL_WINS', 'GAMES_PLAYED']
team_tourney['EXPECTED_WINRATE'] = team_tourney['SEED'].map(seed_avg_wins)
team_tourney['EXPECTED_WINS'] = team_tourney['EXPECTED_WINRATE'] * team_tourney['GAMES_PLAYED']
team_tourney['OVER_UNDER'] = team_tourney['ACTUAL_WINS'] - team_tourney['EXPECTED_WINS']

print("\n--- Top 15 Overperformers (Actual Wins - Expected Wins) ---")
top_over = team_tourney.nlargest(15, 'OVER_UNDER')[
    ['YEAR', 'TEAM', 'SEED', 'ACTUAL_WINS', 'GAMES_PLAYED', 'OVER_UNDER']
]
top_over['OVER_UNDER'] = top_over['OVER_UNDER'].round(2)
print(top_over.to_string(index=False))

print("\n--- Top 15 Underperformers ---")
top_under = team_tourney.nsmallest(15, 'OVER_UNDER')[
    ['YEAR', 'TEAM', 'SEED', 'ACTUAL_WINS', 'GAMES_PLAYED', 'OVER_UNDER']
]
top_under['OVER_UNDER'] = top_under['OVER_UNDER'].round(2)
print(top_under.to_string(index=False))

# --------------------------------------------------------------------------- #
# EDA 2: FEATURE ANALYSIS (Q2)
# --------------------------------------------------------------------------- #

# --- 2a. Correlation of differential features with outcome ---
diff_cols_for_corr = [c for c in games_merged.columns if c.startswith('DIFF_')]
correlations = games_merged[diff_cols_for_corr + ['A_WIN']].corr()['A_WIN'].drop('A_WIN')
correlations = correlations.dropna().sort_values(ascending=False)

print("\n--- Feature Correlations with Winning (Differential Features) ---")
print(correlations.to_string())

fig, ax = plt.subplots(figsize=(10, 8))
top_corr = pd.concat([correlations.head(10), correlations.tail(10)]) if len(correlations) > 20 else correlations
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in top_corr.values]
top_corr.plot(kind='barh', ax=ax, color=colors, edgecolor='white')
ax.set_xlabel('Correlation with Team A Winning', fontsize=12)
ax.set_title('Feature Correlations with Game Outcome', fontsize=14)
ax.axvline(0, color='black', linewidth=0.5)
plt.tight_layout()
plt.savefig("outputs/eda_feature_correlations.png")
plt.close()
print("Saved: outputs/eda_feature_correlations.png")

# --- 2b. Winner vs loser distributions ---
winners = games_merged[games_merged['A_WIN'] == 1][['A_KADJ EM', 'A_BARTHAG', 'A_WIN%', 'A_WAB']].copy()
winners.columns = ['KADJ EM', 'BARTHAG', 'WIN%', 'WAB']
winners['RESULT'] = 'Winner'

losers_a = games_merged[games_merged['A_WIN'] == 0][['A_KADJ EM', 'A_BARTHAG', 'A_WIN%', 'A_WAB']].copy()
losers_a.columns = ['KADJ EM', 'BARTHAG', 'WIN%', 'WAB']
losers_b = games_merged[games_merged['A_WIN'] == 1][['B_KADJ EM', 'B_BARTHAG', 'B_WIN%', 'B_WAB']].copy()
losers_b.columns = ['KADJ EM', 'BARTHAG', 'WIN%', 'WAB']
losers = pd.concat([losers_a, losers_b])
losers['RESULT'] = 'Loser'

wl = pd.concat([winners, losers], ignore_index=True)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
for ax, col in zip(axes.flatten(), ['KADJ EM', 'BARTHAG', 'WIN%', 'WAB']):
    for result, color in [('Winner', '#2ecc71'), ('Loser', '#e74c3c')]:
        subset = wl[wl['RESULT'] == result][col].dropna()
        ax.hist(subset, bins=30, alpha=0.6, label=result, color=color, edgecolor='white')
    ax.set_title(f'{col} Distribution', fontsize=12)
    ax.legend()
    ax.set_xlabel(col)
    ax.set_ylabel('Count')

plt.suptitle('Key Metrics: Winners vs Losers (2008–2025)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig("outputs/eda_winner_loser_distributions.png")
plt.close()
print("Saved: outputs/eda_winner_loser_distributions.png")

# --- 2c. Upset analysis ---
games_merged['UPSET'] = (
    ((games_merged['A_SEED'] > games_merged['B_SEED']) & (games_merged['A_WIN'] == 1)) |
    ((games_merged['B_SEED'] > games_merged['A_SEED']) & (games_merged['A_WIN'] == 0))
).astype(int)

diff_seed_games = games_merged[games_merged['A_SEED'] != games_merged['B_SEED']]
upset_rate = diff_seed_games.groupby('CURRENT ROUND')['UPSET'].mean() * 100

print("\n--- Upset Rate by Round ---")
for r in [64, 32, 16, 8, 4, 2]:
    if r in upset_rate.index:
        print(f"  {round_labels[r]}: {upset_rate[r]:.1f}%")

upsets = diff_seed_games[diff_seed_games['UPSET'] == 1]
non_upsets = diff_seed_games[diff_seed_games['UPSET'] == 0]

print("\n--- What characterizes upsets? ---")
print("  Median KADJ EM differential in upsets vs non-upsets:")
print(f"    Upsets:     DIFF_KADJ EM = {upsets['DIFF_KADJ EM'].median():.2f}")
print(f"    Non-upsets: DIFF_KADJ EM = {non_upsets['DIFF_KADJ EM'].median():.2f}")
print(f"    (Upsets have flatter efficiency gaps — seeds overstate the mismatch)")

# --- 2d. Seed vs Efficiency scatter ---
team_data = kenpom[kenpom['ROUND'] != 68].copy()
team_data['ROUNDS_WON'] = team_data['ROUND'].map({
    64: 0, 32: 1, 16: 2, 8: 3, 4: 4, 2: 5, 1: 6
})

fig, ax = plt.subplots(figsize=(12, 7))
scatter = ax.scatter(
    team_data['SEED'], team_data['KADJ EM'],
    c=team_data['ROUNDS_WON'], cmap='RdYlGn',
    alpha=0.5, s=30, edgecolors='gray', linewidth=0.3
)
cbar = plt.colorbar(scatter, ax=ax, label='Tournament Wins')
cbar.set_ticks([0, 1, 2, 3, 4, 5, 6])
ax.set_xlabel('Seed', fontsize=12)
ax.set_ylabel('KenPom Adj. Efficiency Margin', fontsize=12)
ax.set_title('Seed vs Efficiency: Does the "Best" Team Win? (2008–2025)', fontsize=14)
ax.set_xticks(range(1, 17))
plt.tight_layout()
plt.savefig("outputs/eda_seed_vs_efficiency.png")
plt.close()
print("Saved: outputs/eda_seed_vs_efficiency.png")

# --------------------------------------------------------------------------- #
# EDA 3: SUMMARY STATISTICS + BASELINES
# --------------------------------------------------------------------------- #

print("\n" + "="*70)
print("SUMMARY STATISTICS FOR REPORT")
print("="*70)

n_years = games['YEAR'].nunique()
n_games = len(games)
print(f"\nDataset: {n_years} tournaments (2008-2025, excluding 2020)")
print(f"Total games: {n_games}")
print(f"Features per team: {len(feature_cols) - 2} from KenPom + {len(resume_cols) - 2} from Resumes + {len(tr_cols) - 2} from TeamRankings")
print(f"Differential features engineered: {len(diff_cols)}")

# Baselines
chalk_correct = diff_seed_games.apply(
    lambda r: (r['A_SEED'] < r['B_SEED'] and r['A_WIN'] == 1) or
              (r['B_SEED'] < r['A_SEED'] and r['A_WIN'] == 0), axis=1
).mean() * 100
print(f"\nChalk baseline accuracy (higher seed wins): {chalk_correct:.1f}%")

kadj_correct = diff_seed_games.apply(
    lambda r: (r['A_KADJ EM'] > r['B_KADJ EM'] and r['A_WIN'] == 1) or
              (r['B_KADJ EM'] > r['A_KADJ EM'] and r['A_WIN'] == 0), axis=1
).dropna().mean() * 100
print(f"Higher KADJ EM wins accuracy: {kadj_correct:.1f}%")

barthag_correct = diff_seed_games.apply(
    lambda r: (r['A_BARTHAG'] > r['B_BARTHAG'] and r['A_WIN'] == 1) or
              (r['B_BARTHAG'] > r['A_BARTHAG'] and r['A_WIN'] == 0), axis=1
).dropna().mean() * 100
print(f"Higher BARTHAG wins accuracy: {barthag_correct:.1f}%")

tr_correct = diff_seed_games.apply(
    lambda r: (r['A_TR RATING'] > r['B_TR RATING'] and r['A_WIN'] == 1) or
              (r['B_TR RATING'] > r['A_TR RATING'] and r['A_WIN'] == 0), axis=1
).dropna().mean() * 100
print(f"Higher TR RATING wins accuracy: {tr_correct:.1f}%")

print("\n--- Baseline Performance Summary (Accuracy) ---")
print(f"  Always pick higher seed:      {chalk_correct:.1f}%")
print(f"  Always pick higher KADJ EM:   {kadj_correct:.1f}%")
print(f"  Always pick higher BARTHAG:   {barthag_correct:.1f}%")
print(f"  Always pick higher TR RATING: {tr_correct:.1f}%")

# --- Bracket score baselines ---
# Raw accuracy treats every game equally, but the real goal is predicting the
# Final Four and champion. Round-weighted bracket scoring rewards late-round
# picks more heavily: R64=1, R32=2, S16=4, E8=8, F4=16, Final=32.
# Max possible per tournament = 192 points.

round_weights = {64: 1, 32: 2, 16: 4, 8: 8, 4: 16, 2: 32}

def score_baseline(df, pick_fn, label):
    """Evaluate a baseline pick function across all metrics."""
    correct = df.apply(lambda r: pick_fn(r) == r['A_WIN'], axis=1)
    points = df['CURRENT ROUND'].map(round_weights) * correct.astype(int)
    yearly_pts = points.groupby(df['YEAR']).sum()
    yearly_max = df['CURRENT ROUND'].map(round_weights).groupby(df['YEAR']).sum()
    
    # Late-round accuracy (Sweet 16+)
    late = df['CURRENT ROUND'].isin([16, 8, 4, 2])
    late_acc = correct[late].mean() * 100 if late.sum() > 0 else 0
    
    # Final Four + Championship accuracy
    f4 = df['CURRENT ROUND'].isin([4, 2])
    f4_acc = correct[f4].mean() * 100 if f4.sum() > 0 else 0
    
    return {
        'label': label,
        'accuracy': correct.mean() * 100,
        'bracket_score': yearly_pts.mean(),
        'bracket_pct': (yearly_pts / yearly_max).mean() * 100,
        'sweet16_plus_acc': late_acc,
        'f4_champ_acc': f4_acc,
    }

baselines = [
    score_baseline(games_merged, lambda r: 1 if r['A_SEED'] < r['B_SEED'] else 0, "Chalk (higher seed)"),
    score_baseline(games_merged, lambda r: 1 if r['A_KADJ EM'] > r['B_KADJ EM'] else 0, "Higher KADJ EM"),
    score_baseline(games_merged, lambda r: 1 if r['A_BARTHAG'] > r['B_BARTHAG'] else 0, "Higher BARTHAG"),
    score_baseline(games_merged, lambda r: 1 if r['A_TR RATING'] > r['B_TR RATING'] else 0, "Higher TR RATING"),
]

print("\n--- Baseline Performance Summary (Bracket Score) ---")
print(f"  {'Baseline':<22} {'Accuracy':>8} {'Bracket':>8} {'S16+ Acc':>8} {'F4 Acc':>8}")
print(f"  {'-'*22} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
for b in baselines:
    print(f"  {b['label']:<22} {b['accuracy']:>7.1f}% {b['bracket_score']:>6.1f}/192 {b['sweet16_plus_acc']:>7.1f}% {b['f4_champ_acc']:>7.1f}%")

print(f"\n  Key insight: highest accuracy ≠ highest bracket score.")
print(f"  Models should be evaluated on bracket score, not just raw accuracy.\n")

print("="*70)
print("PHASE 1 COMPLETE")
print("="*70)
print(f"\nOutput files in outputs/:")
print(f"  games_merged.csv              - full game-level dataset ({games_merged.shape[1]} cols)")
print(f"  games_slim.csv                - differential features + outcomes only")
print(f"  eda_seed_winrate.png          - seed win rate bar chart")
print(f"  eda_seed_round_heatmap.png    - seed × round win rate heatmap")
print(f"  eda_feature_correlations.png  - feature correlation with outcomes")
print(f"  eda_winner_loser_distributions.png - metric distributions W vs L")
print(f"  eda_seed_vs_efficiency.png    - seed vs KADJ EM scatter")
print(f"\nNext step: Phase 2 — model training (XGBoost, Random Forest, NN)")