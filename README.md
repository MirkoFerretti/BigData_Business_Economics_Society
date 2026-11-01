# Do Trading Volumes Help Predict Stock Returns?

A 3-CFU MSc Data Science project for *Big Data in Business, Economics and Society*,
University of Milano-Bicocca.

The accompanying written report is submitted separately.

**Research question:** Does information contained in trading volume improve the
out-of-sample prediction of next-day stock returns relative to a baseline using
only past stock returns?

The study uses 12 large-cap US stocks over 2015–2025: 2015–2022 for initial
estimation and diagnostics, 2023–2024 for validation and model selection, and
2025 for chronological final evaluation. Expanding-window linear models predict
the next observed trading-day adjusted-close log return. Each stock's return-only
Model A is frozen; Model B is selected among dollar-activity candidates B1–B3 and
robustness Model S among share-volume candidates S1–S3 using validation MSE.
AAPL selects B2/S2, TSLA B3/S3, and the other ten stocks B1/S1.
MSE is primary; RMSE and MAE are complementary.

In 2025, B improves MSE for 6 stocks and worsens it for 6, with equal-stock mean
improvement **−0.073135%**, median **+0.019773%**, and validation/test sign agreement
6/12. S improves 7 and worsens 5: mean **−0.113084%**, median **+0.055716%**,
agreement 7/12. The gains are small and mixed; trading activity does not provide
broad improvement in this sample. See the
[current numerical report](outputs/final_2025_stock_specific_volume_report.md).

## Repository

- [PROJECT.md](PROJECT.md): final design, definitions and limitations.
- `config/`: frozen universe and Model A mapping.
- `src/data/`, `src/features/`, `src/analysis/`: acquisition, features and analysis.
- `tests/`: synthetic analytical and publication regressions.
- [data/raw/README.md](data/raw/README.md): exact Yahoo CSV snapshot and SHA-256 checksums; caches remain ignored.
- `data/processed/`: generated market/features, ignored.
- `outputs/tables/`, `outputs/figures/`: curated current results; development outputs ignored.
- [outputs/report/README.md](outputs/report/README.md): descriptive EDA and final-result index.

## Setup

Use Python **3.12** (verified with 3.12.5). From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On macOS/Linux activate with `source .venv/bin/activate`. If activation is
unavailable, replace `python` below with `.\.venv\Scripts\python.exe` on Windows
or `.venv/bin/python` on macOS/Linux. Direct dependencies, including NumPy, are
pinned to the versions used for the saved results.

## Reproduce

Run from the repository root in the environment above. To reproduce the saved
data vintage, use the supplied raw CSVs and skip downloading. For a **fresh Yahoo
download** (network required; replaces raw CSVs):

```text
python -B src/data/download_market_data.py
```

Then run these stages in order:

```text
python -B src/data/build_processed_data.py
python -B src/analysis/model_a_return_diagnostics.py
python -B src/features/build_model_a_features.py
python -B src/analysis/model_a_feature_validation.py
python -B src/features/build_model_b_features.py
python -B src/analysis/model_b_volume_validation.py
python -B src/analysis/model_b_brkb_sensitivity.py
python -B src/features/build_share_volume_features.py
python -B src/analysis/model_b_share_volume_validation.py
python -B src/analysis/final_2025_stock_specific_volume_evaluation.py
python -B src/analysis/report_eda.py
```

These build the base data, diagnose AR return candidates, construct/validate
return features, construct/validate dollar-volume candidates, check BRK.B
source anomalies, repeat the activity comparison with share volume, evaluate
the selected models on 2025, and prepare descriptive EDA. Model A stays fixed
by `config/model_a_frozen.csv`; selection uses saved validation MSE only.
Reruns replace generated outputs. The final evaluation uses the stored market
and feature tables plus the two volume-validation metric tables.

## Tests

Tests use temporary synthetic fixtures and can run before generating data:

```text
python -B -m unittest discover -s tests -v
```

## Current final outputs

- [Numerical report](outputs/final_2025_stock_specific_volume_report.md)
- [Selected B/S models and validation evidence](outputs/tables/stock_specific_volume_model_selection.csv)
- [All MSE/RMSE/MAE metrics](outputs/tables/final_2025_stock_specific_volume_metrics.csv)
- [A versus B/S and validation/test signs](outputs/tables/final_2025_stock_specific_volume_comparison.csv)
- [Equal-stock summary](outputs/tables/final_2025_stock_specific_volume_cross_stock_summary.csv)
- [Training histories](outputs/tables/final_2025_stock_specific_volume_training_counts.csv)
- [9,000 dated forecasts](outputs/tables/final_2025_stock_specific_volume_forecasts.csv)
- [Evaluation-snapshot integrity record](outputs/tables/final_2025_stock_specific_volume_integrity.json)
- [Final figures and descriptive EDA](outputs/report/README.md)

## Data and interpretation

Yahoo Finance data are downloaded through yfinance, with dates 2015-01-01
inclusive to 2026-01-01 exclusive; BRK.B maps to BRK-B. Vendor-preserved Close
and Volume are not guaranteed to reconstruct historically unadjusted tape
prices/share counts. DollarVolume is their product; returns use Adjusted Close.
Yahoo can revise history, so a fresh download need not reproduce the saved
numbers. Preserve the supplied snapshot and its hashes for exact reproduction;
source redistribution must be confirmed before publishing it. Cookies and
local certificate/cache files are excluded. Certificate problems should be
resolved with a trusted CA bundle, without disabling verification.

2025 was excluded from predictive model fitting, feature selection, validation
and model selection during development. Final expanding refits can use
already-realized earlier 2025 outcomes. Earlier full-period descriptive
diagnostics exposed 2025 outcomes, and the stock-specific comparison followed
the historical 2025 evaluation. This is not a claim of preregistration or an
unseen-test history. The universe was selected retrospectively in October 2026;
results describe this sample, not the whole US market. See PROJECT for details.

