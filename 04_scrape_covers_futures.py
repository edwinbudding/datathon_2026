###############################################################################
# SCRAPE PRE-TOURNAMENT FUTURES ODDS FROM COVERS.COM (v2)
# 2026 Brandeis Datathon — Team ASA
#
# Fixed version: properly identifies the "prior to Round 1" column
# by reading the actual table headers, which vary by year.
#
# Output: scraped/covers_futures.csv
#
# Run from ~/Desktop/Datathon:
#   python3.12 04_scrape_covers_futures.py
###############################################################################

import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import re

DELAY = 3
OUTPUT_DIR = "scraped"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "covers_futures.csv")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}

YEARS = list(range(2008, 2026))

os.makedirs(OUTPUT_DIR, exist_ok=True)


def scrape_year(tournament_year):
    """
    Scrape pre-tournament futures odds for a given tournament year.
    Reads the actual column headers to find the correct columns.
    """
    season_start = tournament_year - 1
    url = f"https://www.covers.com/sportsoddshistory/cbb-main/?y={season_start}-{tournament_year}&sa=cbb&a=nc"

    print(f"  Fetching {tournament_year} ({season_start}-{tournament_year})... ", end="", flush=True)

    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except Exception as e:
        print(f"ERROR: {e}")
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')

    # Find the data table
    table = soup.find('table', class_=re.compile('soh'))
    if not table:
        wrapper = soup.find('div', class_='responsive-table-wrapper')
        if wrapper:
            table = wrapper.find('table')
    if not table:
        print("NO TABLE FOUND")
        return []

    # --- Parse ALL header rows to find column names ---
    # Covers uses a multi-row header: first row has grouped headers like
    # "Preseason, as of..." and "Tournament, prior to..."
    # Second row has specific dates or round names
    thead = table.find('thead')
    
    # Get all header text from all header rows
    header_cells = []
    if thead:
        # Get the LAST header row (most specific)
        header_rows = thead.find_all('tr')
        if header_rows:
            last_header = header_rows[-1]
            for cell in last_header.find_all(['th', 'td']):
                header_cells.append(cell.get_text(strip=True).lower())
    
    # Also check the first header row for grouped headers
    grouped_headers = []
    if thead and len(thead.find_all('tr')) > 1:
        first_header = thead.find_all('tr')[0]
        for cell in first_header.find_all(['th', 'td']):
            text = cell.get_text(strip=True).lower()
            colspan = int(cell.get('colspan', 1))
            grouped_headers.extend([text] * colspan)

    print(f"headers={header_cells[:8]}... ", end="")

    # --- Identify key columns ---
    # Strategy: find "round 1" or "tournament" in headers
    # The pre-tournament column is typically labeled with "round 1" or similar
    
    preseason_col = None
    pre_tournament_col = None
    result_col = None
    
    # Check last header row first
    for i, h in enumerate(header_cells):
        if 'nov' in h and preseason_col is None:
            preseason_col = i
        elif ('round 1' in h or 'round of 64' in h) and pre_tournament_col is None:
            pre_tournament_col = i
        elif 'result' in h:
            result_col = i
    
    # If we didn't find "round 1" in the last row, check grouped headers
    if pre_tournament_col is None:
        for i, h in enumerate(grouped_headers):
            if 'tournament' in h or 'round 1' in h or 'prior to' in h:
                # This grouped header spans some columns
                # The FIRST column under this group is the pre-tournament column
                # But we need to find where the tournament group starts
                # For the simple case, the pre-tournament odds is the first column
                # under the "Tournament" grouped header
                pre_tournament_col = i
                break
    
    # Fallback: for years with few columns (2008-2009), 
    # col 0 = team, col 1 = preseason, col 2 = pre-tournament
    # For years with many columns, try to find it from the data pattern
    if pre_tournament_col is None:
        # Use the "R64 loser" heuristic: find the last non-empty column 
        # for a team that doesn't appear as a winner
        pass  # Will handle below in data parsing
    
    # Adjust for the team column (column 0 in our data is column 1 in the table header
    # because header_cells might include "Team" as the first entry)
    # We need to figure out the offset
    team_col_offset = 0
    if header_cells and ('team' in header_cells[0] or header_cells[0] == ''):
        team_col_offset = 1  # data columns start at index 1 in headers
    
    # Convert header-based index to data-column index (subtract team column)
    if preseason_col is not None:
        preseason_col = preseason_col - team_col_offset
    if pre_tournament_col is not None:
        pre_tournament_col = pre_tournament_col - team_col_offset
    if result_col is not None:
        result_col = result_col - team_col_offset

    # --- Parse body rows ---
    tbody = table.find('tbody')
    if not tbody:
        tbody = table

    results = []
    for row in tbody.find_all('tr'):
        cells = row.find_all('td')
        if len(cells) < 2:
            continue

        # Team name
        team_cell = cells[0]
        link = team_cell.find('a')
        team_name = link.get_text(strip=True) if link else team_cell.get_text(strip=True)
        if not team_name or team_name.lower() in ['team', '']:
            continue

        # Get all data values (skip team column)
        data_cols = [c.get_text(strip=True) for c in cells[1:]]

        # Extract specific columns
        preseason_odds = ""
        pretournament_odds = ""
        result_text = ""

        if preseason_col is not None and 0 <= preseason_col < len(data_cols):
            preseason_odds = data_cols[preseason_col]

        if pre_tournament_col is not None and 0 <= pre_tournament_col < len(data_cols):
            pretournament_odds = data_cols[pre_tournament_col]

        if result_col is not None and 0 <= result_col < len(data_cols):
            result_text = data_cols[result_col]
        
        # Check for WINNER in any column
        is_winner = any('WINNER' in str(c).upper() for c in data_cols)

        results.append({
            'tournament_year': tournament_year,
            'team': team_name,
            'preseason_odds': preseason_odds,
            'pre_tournament_odds': pretournament_odds,
            'result': 'WINNER' if is_winner else '',
            'all_columns': '|'.join(data_cols),
            'preseason_col_idx': preseason_col,
            'pretournament_col_idx': pre_tournament_col,
        })

    # --- Post-processing: verify pre-tournament column ---
    # Sanity check: the winner's pre-tournament odds should be positive and reasonable
    winner_rows = [r for r in results if r['result'] == 'WINNER']
    if winner_rows and pre_tournament_col is not None:
        winner_odds = winner_rows[0]['pre_tournament_odds']
        print(f"pre_tourn=col{pre_tournament_col} | winner={winner_rows[0]['team']} odds={winner_odds} | ", end="")

    print(f"found {len(results)} teams")
    return results


def main():
    print("=" * 60)
    print("COVERS.COM FUTURES ODDS SCRAPER v2 (2008-2025)")
    print("=" * 60)

    all_results = []

    for year in YEARS:
        if year == 2020:
            print(f"  Skipping 2020 (tournament cancelled)")
            continue

        year_results = scrape_year(year)
        all_results.extend(year_results)
        time.sleep(DELAY)

    # Write CSV
    if all_results:
        with open(OUTPUT_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'tournament_year', 'team', 'preseason_odds',
                'pre_tournament_odds', 'result', 'all_columns',
                'preseason_col_idx', 'pretournament_col_idx'
            ])
            writer.writeheader()
            writer.writerows(all_results)

        print(f"\n{'=' * 60}")
        print(f"Saved {len(all_results)} rows to {OUTPUT_FILE}")

        # Verify winners
        print(f"\n--- Winner Verification ---")
        print(f"{'Year':<6} {'Winner':<25} {'Preseason':>10} {'Pre-Tourn':>10} {'Col Idx':>8}")
        print("-" * 65)
        for r in all_results:
            if r['result'] == 'WINNER':
                print(f"{r['tournament_year']:<6} {r['team']:<25} {r['preseason_odds']:>10} "
                      f"{r['pre_tournament_odds']:>10} {r['pretournament_col_idx']:>8}")

    else:
        print("\nNo data scraped!")


if __name__ == '__main__':
    main()