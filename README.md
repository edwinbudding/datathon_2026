How to Make Money and Lose Friends Betting on College Basketball
2026 Brandeis Datathon — Team ASA

Our submission to the 2026 Brandeis Datathon, in which we built a five-script pipeline to predict March Madness outcomes using machine learning, efficiency metrics, and scraped Vegas closing lines. We found that ML models can beat Vegas on individual game predictions (+29.8% ROI for Random Forest across 32 out-of-sample bets), but championship prediction is still ruled by the market — the pre-tournament favorite wins 41% of the time and is profitable to bet on blindly.
Pipeline

Run order: 01 → 02 → 03. Scrapers (04, scrape_vegas_lines) run independently. Script 05 requires outputs from 01, 02, and 04.

Key Findings

1) All methods converge on 67–72% game-level accuracy. Vegas closing lines finish 7th out of 8 methods.
2_ When models disagree with Vegas, betting the underdog side is profitable (4/5 methods positive ROI).
3) The pre-tournament market favorite has won 7/17 tournaments — champion prediction is where the market is most efficient.

Our 2026 Predicted Final Four: Duke, Houston, Arizona, Michigan
Real Final Four: Duke

Setup
Requires Python 3.12 with pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, requests, beautifulsoup4.

cd ~/Desktop/Datathon
pip install pandas numpy scikit-learn xgboost matplotlib seaborn requests beautifulsoup4

Data: download the Nishaan Amin Kaggle dataset into a folder called datathon set like we did. 

Note: File naming conventions in the scripts may reference "datathon set" (with a space). Ensure your local folder name matches.

Sincerely,
Anokh Palakurthi — MS Business Analytics, Brandeis '26
Samiya Islam — MS Business Analytics, Brandeis '26
Aastha Chavan — MS Business Analytics, Brandeis '26
