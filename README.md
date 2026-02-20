# TSLA Next-Day Direction Predictor

**Course:** Machine Learning (IDSS32204) — Mini Project 2: Trading

Binary classification — predict whether Tesla (TSLA) stock closes **higher or lower** the next trading day using alternative data and ensemble models.

---

## Results (Out-of-Sample 2024–2026)

| Model | L/F Return | L/F Sharpe | Max Drawdown | Hit Rate |
|-------|-----------|------------|--------------|----------|
| Random Forest | +451% | 3.34 | -21% | 61% |
| Neural Net (MLP) | +295% | 2.27 | -24% | 67% |
| XGBoost | +240% | 1.77 | -25% | 60% |
| Logistic Regression | +82% | 0.73 | -39% | 55% |
| **Buy & Hold** | **+65%** | — | -54% | — |

Tested across 5 independent years (2021–2025) — 16/20 year-model combinations show positive Sharpe ratios.

---

## Project Structure

```
├── data/
│   └── processed/          # Cleaned & merged datasets (model-ready)
│       ├── master_dataset.csv      # All sources merged (52 columns)
│       └── features_ready.csv     # Final 25 features after selection
├── notebooks/
│   ├── 01_data_collection.ipynb        # Download & save all data sources
│   ├── 02_data_exploration.ipynb       # EDA, visualizations, signal discovery
│   ├── 03_feature_engineering.ipynb    # Feature creation, selection (109 → 25)
│   ├── 04_tweet_signal_exploration.ipynb # Tweet NLP, word analysis
│   ├── 05_modeling.ipynb               # Model training, Optuna tuning
│   ├── 06_backtesting.ipynb            # Walk-forward backtest, equity curves
│   ├── 07_robustness.ipynb             # Multi-year robustness heatmaps
│   └── presentation.ipynb             # Clean end-to-end presentation notebook
├── src/
│   └── helpers.py          # Shared paths, target creation utility
├── outputs/
│   └── figures/            # Saved plots
├── requirements.txt
└── README.md
```

---

## Data Sources

| Source | Method | Period | Notes |
|--------|--------|--------|-------|
| TSLA OHLCV | `yfinance` | 2010–2026 | Daily price & volume |
| Technical Indicators | `pandas_ta` | 2010–2026 | RSI, MACD, Bollinger, ATR, OBV, Stochastic |
| Elon Musk Tweets | Kaggle CSV | 2012–2023 | 55K raw → 14.9K original tweets |
| Google Trends | `pytrends` | 2016–2026 | 13 search terms, weekly → daily |
| Fundamentals | `yfinance` | 2012–2025 | Quarterly EPS & revenue, forward-filled |
| SEC Filings | `sec-edgar-downloader` | 2010–2026 | 10-K and 8-K filings |

> **Note:** Raw data is excluded from this repo due to size. Run `01_data_collection.ipynb` to regenerate, except for tweets which require a manual Kaggle download (see below).

---

## Setup

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/tsla-direction-predictor.git
cd tsla-direction-predictor

# 2. Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Elon Musk Tweets
Download from Kaggle and place at `data/external/elonmusk_tweets.csv`:
- [Elon Musk Tweets Dataset](https://www.kaggle.com/datasets/andradaolteanu/all-elon-musks-tweets)

---

## How to Run

Run notebooks **in order**. Each notebook saves outputs that the next one loads.

```
01 → 02 → 03 → 04 → 05 → 06 → 07
```

Or open `presentation.ipynb` for a clean end-to-end walkthrough.

---

## Pipeline

```
Raw Data → EDA → Feature Engineering (109 → 25 features)
       → Modeling (LR, RF, XGBoost, MLP + Optuna tuning)
       → Walk-Forward Backtesting (6-month retrain windows)
       → Robustness Testing (5 independent years)
```

**Key design decisions:**
- Strict temporal train/test split at `2024-01-01` — no shuffling
- `TimeSeriesSplit` cross-validation (5 folds) for all CV
- StandardScaler fit on training data only — applied to test
- Feature selection (Mutual Information + XGBoost importance) on training data only
- Tweet word analysis derived from pre-2024 data only
- Walk-forward backtest uses expanding window, retrains every 6 months

---

## Key Findings

1. **Tweet word score is the strongest single predictor** — swings P(Up next day) from 42% (bearish tweets) to 80% (bullish tweets)
2. **Random Forest dominates** — +451% Long/Flat return vs +65% Buy & Hold, Sharpe 3.34
3. **The edge is statistically significant** — 4.8 standard deviations above 100 random baselines (Monte Carlo)
4. **Robust across market regimes** — profitable in bull (2021, 2023, 2024) and defensive in bear (2022, −65% B&H)
5. **Long/Flat beats Long/Short** — similar Sharpe but dramatically lower drawdowns (−21% vs −55%)
