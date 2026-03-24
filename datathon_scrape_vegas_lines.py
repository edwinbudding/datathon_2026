###############################################################################
# SCRAPE VEGAS LINES FROM SPORTS REFERENCE
# 2026 Brandeis Datathon — Team ASA
#
# Two-pass scraper:
#   Pass 1: Hit 17 tournament bracket pages (2008-2025, no 2020) to collect
#           all box score URLs.
#   Pass 2: Hit each box score page and extract the Vegas line from the
#           Game Info table.
#
# Output: outputs/vegas_lines.csv
#
# Run: python3 scrape_vegas_lines_2.py
# Takes ~30-40 minutes with polite rate limiting
###############################################################################

import requests
from bs4 import BeautifulSoup
import csv
import time
import os
import re
import json
from datetime import datetime

# --- Configuration ---
DELAY_INDEX = 3        # seconds between bracket page requests
DELAY_BOXSCORE = 3    # seconds between box score requests
OUTPUT_DIR = "outputs"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "vegas_lines_raw.csv")
URLS_CACHE = os.path.join(OUTPUT_DIR, "boxscore_urls.json")
PROGRESS_FILE = os.path.join(OUTPUT_DIR, "scrape_progress.json")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36'
}

YEARS = [y for y in range(2008, 2026) if y != 2020]  # 17 tournament years

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========================================================================== #
# PASS 1: Collect box score URLs from tournament bracket pages
# ========================================================================== #

def collect_boxscore_urls():
    """
    Hit each year's tournament bracket page and extract all box score links.
    SR bracket pages contain links to individual game box scores.
    
    Returns dict: {year: [list of box score URLs]}
    """
    # Check cache first
    if os.path.exists(URLS_CACHE):
        print(f"Loading cached URLs from {URLS_CACHE}")
        with open(URLS_CACHE, 'r') as f:
            cached = json.load(f)
        # Convert string keys back to int
        return {int(k): v for k, v in cached.items()}
    
    all_urls = {}
    
    for year in YEARS:
        bracket_url = f"https://www.sports-reference.com/cbb/postseason/men/{year}-ncaa.html"
        print(f"[Pass 1] Fetching {year} bracket page... ", end="", flush=True)
        
        try:
            resp = requests.get(bracket_url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"ERROR: {e}")
            all_urls[year] = []
            time.sleep(DELAY_INDEX)
            continue
        
        soup = BeautifulSoup(resp.text, 'html.parser')
        
        # Find all links to box scores
        # SR box score links look like: /cbb/boxscores/2025-03-21-14-duke.html
        boxscore_links = []
        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            if '/cbb/boxscores/' in href and href.endswith('.html'):
                full_url = f"https://www.sports-reference.com{href}"
                if full_url not in boxscore_links:
                    boxscore_links.append(full_url)
        
        all_urls[year] = boxscore_links
        print(f"found {len(boxscore_links)} box score links")
        time.sleep(DELAY_INDEX)
    
    # Cache the URLs so we don't have to re-scrape bracket pages
    with open(URLS_CACHE, 'w') as f:
        json.dump(all_urls, f, indent=2)
    print(f"\nCached {sum(len(v) for v in all_urls.values())} URLs to {URLS_CACHE}")
    
    return all_urls


# ========================================================================== #
# PASS 2: Scrape Vegas lines from each box score page
# ========================================================================== #

def parse_vegas_line(soup, url, year, raw_html=""):
    """
    Extract the Vegas line and over/under from a box score page.
    
    SR hides the Game Info table inside HTML comments, so we use
    raw HTML regex instead of BeautifulSoup to find it.
    
    Returns dict with game info.
    """
    result = {
        'year': year,
        'url': url,
        'team1': None,
        'team1_score': None,
        'team2': None,
        'team2_score': None,
        'favored_team': None,
        'spread': None,
        'over_under': None,
    }
    
    # --- Extract team names and scores from the scorebox ---
    scorebox = soup.find('div', class_='scorebox')
    if scorebox:
        team_links = scorebox.find_all('a', href=True)
        teams_found = []
        for link in team_links:
            href = link['href']
            if '/cbb/schools/' in href:
                team_name = link.get_text(strip=True)
                if team_name and team_name not in teams_found:
                    teams_found.append(team_name)
        
        score_divs = scorebox.find_all('div', class_='score')
        scores_found = []
        for sd in score_divs:
            score_text = sd.get_text(strip=True)
            try:
                scores_found.append(int(score_text))
            except ValueError:
                pass
        
        if len(teams_found) >= 2:
            result['team1'] = teams_found[0]
            result['team2'] = teams_found[1]
        if len(scores_found) >= 2:
            result['team1_score'] = scores_found[0]
            result['team2_score'] = scores_found[1]
    
    # --- Extract Vegas line using raw HTML regex ---
    # The pattern in the HTML is:
    #   data-stat="info" >Vegas Line</th><td ... data-stat="stat" >North Carolina -25</td>
    
    vegas_match = re.search(
        r'>Vegas Line</th><td[^>]*>([^<]+)</td>',
        raw_html
    )
    if vegas_match:
        value = vegas_match.group(1).strip()
        result['spread_raw'] = value
        
        if value.lower() in ['pick', "pick'em", 'pick em', "pick 'em"]:
            result['favored_team'] = 'PICK'
            result['spread'] = 0.0
        else:
            # Parse "North Carolina -25" — team name then spread number
            spread_match = re.match(r'^(.+?)\s+([+-]?\d+\.?\d*)$', value)
            if spread_match:
                result['favored_team'] = spread_match.group(1).strip()
                result['spread'] = float(spread_match.group(2))
    
    # Over/Under pattern:
    #   data-stat="info" >Over/Under</th><td ... data-stat="stat" >150.0 ...
    ou_match = re.search(
        r'>Over/Under</th><td[^>]*>\s*([\d.]+)',
        raw_html
    )
    if ou_match:
        result['over_under'] = float(ou_match.group(1))
    
    return result


def scrape_all_lines(all_urls):
    """
    Hit each box score URL and extract the Vegas line.
    Supports resume from interruption using a progress file.
    """
    # Load progress if exists
    completed = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            completed = json.load(f)
        print(f"Resuming: {len(completed)} pages already scraped")
    
    results = []
    
    # Load any existing results
    if os.path.exists(OUTPUT_FILE) and completed:
        with open(OUTPUT_FILE, 'r') as f:
            reader = csv.DictReader(f)
            results = list(reader)
        print(f"Loaded {len(results)} existing results")
    
    total_urls = sum(len(urls) for urls in all_urls.values())
    current = len(completed)
    
    # Open CSV for appending
    write_header = not os.path.exists(OUTPUT_FILE) or not completed
    csvfile = open(OUTPUT_FILE, 'a' if not write_header else 'w', newline='')
    fieldnames = [
        'year', 'url', 'team1', 'team1_score', 'team2', 'team2_score',
        'favored_team', 'spread', 'spread_raw', 'over_under'
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
    
    for year in YEARS:
        urls = all_urls.get(year, [])
        for url in urls:
            if url in completed:
                continue
            
            current += 1
            print(f"[Pass 2] ({current}/{total_urls}) {year}: {url.split('/')[-1][:50]}... ",
                  end="", flush=True)
            
            try:
                resp = requests.get(url, headers=HEADERS, timeout=30)
                resp.raise_for_status()
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                result = parse_vegas_line(soup, url, year, raw_html=resp.text)
                
                # Write to CSV
                row = {k: result.get(k, '') for k in fieldnames}
                writer.writerow(row)
                csvfile.flush()
                
                spread_str = result.get('spread_raw', 'NO LINE')
                print(f"{result.get('team1', '?')} vs {result.get('team2', '?')} | {spread_str}")
                
            except Exception as e:
                print(f"ERROR: {e}")
                # Write error row
                writer.writerow({
                    'year': year, 'url': url, 'team1': 'ERROR',
                    'team1_score': '', 'team2': str(e)[:50],
                    'team2_score': '', 'favored_team': '', 'spread': '',
                    'spread_raw': '', 'over_under': ''
                })
                csvfile.flush()
            
            # Save progress
            completed[url] = True
            with open(PROGRESS_FILE, 'w') as f:
                json.dump(completed, f)
            
            time.sleep(DELAY_BOXSCORE)
    
    csvfile.close()
    print(f"\n{'='*60}")
    print(f"Scraping complete! Results saved to {OUTPUT_FILE}")
    print(f"Total pages scraped: {current}")


# ========================================================================== #
# PASS 3: Clean and merge with existing data
# ========================================================================== #

def clean_and_merge():
    """
    After scraping, clean the raw output and create a merge-ready CSV.
    Maps SR team names to TEAM NO values from the Kaggle dataset.
    """
    import pandas as pd
    
    if not os.path.exists(OUTPUT_FILE):
        print("No raw data found. Run the scraper first.")
        return
    
    raw = pd.read_csv(OUTPUT_FILE)
    print(f"\nLoaded {len(raw)} raw scraped rows")
    print(f"  - With spreads: {raw['spread'].notna().sum()}")
    print(f"  - Missing spreads: {raw['spread'].isna().sum()}")
    print(f"  - Errors: {(raw['team1'] == 'ERROR').sum()}")
    
    # Load KenPom for team name mapping
    kenpom = pd.read_csv("datathon_set/KenPom Barttorvik.csv")
    kenpom_names = kenpom[['YEAR', 'TEAM NO', 'TEAM']].drop_duplicates()
    
    # Load games_merged for joining
    games = pd.read_csv("outputs/games_merged.csv")
    
    # For each scraped game, try to match to games_merged using:
    # year + team names + scores
    print("\nMatching scraped games to existing dataset...")
    
    matched = []
    unmatched = []
    
    for _, row in raw.iterrows():
        if row['team1'] == 'ERROR' or pd.isna(row['team1']):
            continue
        
        year = int(row['year'])
        t1 = str(row['team1']).strip()
        t2 = str(row['team2']).strip()
        s1 = row.get('team1_score')
        s2 = row.get('team2_score')
        
        # Try to find this game in games_merged by year and scores
        # (scores are the most reliable match key since names might differ)
        game_match = None
        if pd.notna(s1) and pd.notna(s2):
            s1, s2 = int(s1), int(s2)
            # Check both orientations (A/B could be either team)
            mask1 = (
                (games['YEAR'] == year) & 
                (games['A_SCORE'] == s1) & (games['B_SCORE'] == s2)
            )
            mask2 = (
                (games['YEAR'] == year) & 
                (games['A_SCORE'] == s2) & (games['B_SCORE'] == s1)
            )
            
            candidates = games[mask1 | mask2]
            
            if len(candidates) == 1:
                game_match = candidates.iloc[0]
            elif len(candidates) > 1:
                # Multiple games with same score in same year — 
                # use team name fuzzy match to disambiguate
                for _, cand in candidates.iterrows():
                    a_name = str(cand['A_TEAM']).lower()
                    b_name = str(cand['B_TEAM']).lower()
                    if (t1.lower() in a_name or a_name in t1.lower() or
                        t1.lower() in b_name or b_name in t1.lower()):
                        game_match = cand
                        break
        
        if game_match is not None:
            matched.append({
                'YEAR': year,
                'A_TEAM_NO': int(game_match['A_TEAM_NO']),
                'B_TEAM_NO': int(game_match['B_TEAM_NO']),
                'A_TEAM': game_match['A_TEAM'],
                'B_TEAM': game_match['B_TEAM'],
                'CURRENT_ROUND': int(game_match['CURRENT ROUND']),
                'FAVORED_TEAM_SR': row.get('favored_team', ''),
                'SPREAD': row.get('spread', ''),
                'SPREAD_RAW': row.get('spread_raw', ''),
                'OVER_UNDER': row.get('over_under', ''),
            })
        else:
            unmatched.append({
                'year': year, 'team1': t1, 'team2': t2,
                'score1': s1, 'score2': s2,
                'spread_raw': row.get('spread_raw', '')
            })
    
    print(f"  Matched: {len(matched)}")
    print(f"  Unmatched: {len(unmatched)}")
    
    if unmatched:
        print(f"\n  First 10 unmatched games:")
        for um in unmatched[:10]:
            print(f"    {um['year']}: {um['team1']} ({um['score1']}) vs "
                  f"{um['team2']} ({um['score2']}) | {um['spread_raw']}")
    
    # Save clean merged file
    if matched:
        clean_df = pd.DataFrame(matched)
        clean_file = os.path.join(OUTPUT_DIR, "vegas_lines.csv")
        clean_df.to_csv(clean_file, index=False)
        print(f"\nClean data saved to {clean_file}")
        print(f"  {len(clean_df)} games with Vegas lines ready to merge")
        
        # Summary stats
        has_spread = clean_df['SPREAD'].notna() & (clean_df['SPREAD'] != '')
        print(f"  Games with spread: {has_spread.sum()}")
        print(f"\n  Sample:")
        print(clean_df.head(10).to_string())


# ========================================================================== #
# MAIN
# ========================================================================== #

if __name__ == '__main__':
    print("="*60)
    print("MARCH MADNESS VEGAS LINE SCRAPER")
    print("Sports Reference Box Scores (2008-2025)")
    print("="*60)
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--clean':
        # Just run the cleaning/merge step (after scraping is done)
        clean_and_merge()
    else:
        # Full run: collect URLs, scrape lines, then clean
        print("\n--- Pass 1: Collecting box score URLs from bracket pages ---")
        all_urls = collect_boxscore_urls()
        
        total = sum(len(v) for v in all_urls.values())
        print(f"\nTotal box score URLs found: {total}")
        for year in YEARS:
            print(f"  {year}: {len(all_urls.get(year, []))} games")
        
        print(f"\n--- Pass 2: Scraping Vegas lines ({total} pages) ---")
        print(f"Estimated time: {total * DELAY_BOXSCORE // 60} - {total * (DELAY_BOXSCORE+1) // 60} minutes")
        print(f"Progress is saved — safe to interrupt and resume.\n")
        
        scrape_all_lines(all_urls)
        
        print(f"\n--- Pass 3: Cleaning and matching to dataset ---")
        clean_and_merge()
