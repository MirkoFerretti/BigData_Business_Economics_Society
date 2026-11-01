# Do Trading Volumes Help Predict Stock Returns?

3-CFU MSc Data Science project, *Big Data in Business, Economics and Society*,
University of Milano-Bicocca.

## Research question and scope

Does information contained in trading volume improve the out-of-sample
prediction of next-day stock returns relative to a baseline using only
past stock returns?

The frozen universe contains 12 large-cap US stocks, two per sector:
NVDA/AAPL (Information Technology), BRK.B/JPM (Financials), LLY/JNJ (Health Care),
AMZN/TSLA (Consumer Discretionary), CAT/GE (Industrials), XOM/CVX (Energy).
Identifiers are in `config/universe.csv`; Yahoo uses BRK-B for BRK.B.
The sample was frozen on 2026-10-23, retrospectively using then-current
large-cap eligibility and full-period availability.

Daily Yahoo Finance data through yfinance cover 2015-01-01–2025-12-31
(download end 2026-01-01 is exclusive). Source fields are preserved without
inventing corrections. Yahoo Close/Volume may already reflect historical
split adjustments; “raw” means vendor-preserved, not independently reconstructed
unadjusted tape data. Returns are `r_t = ln(AdjClose_t / AdjClose_(t-1))`.
Dollar activity is `DollarVolume_t = Close_t × Volume_t` from those vendor fields.

## Chronology and target

- 2015–2022: initial estimation, return diagnostics and candidate construction.
- 2023–2024: expanding-window validation and model selection.
- 2025: chronological final evaluation.

At the close of observed trading day t, the target is the next observed
trading-day log return r_(t+1). Features use information available by that close.
A historical row can enter fitting only when its TargetDate is no later than
the current information date. All models refit with expanding histories;
already-realized earlier 2025 outcomes enter subsequent final-year fits.

2025 was excluded from predictive model fitting, feature selection, validation
and model selection during development. Earlier full-period descriptive
return/corporate-action diagnostics exposed 2025 outcomes. The stock-specific
comparison was implemented after the historical common-B1/S1 evaluation.
The repository does not demonstrate preregistration or pristine unseen-test
history; this limitation does not change the target-aware fitting rules.

## Return-only Model A

Model A is fixed per stock by the lowest 2023–2024 validation MSE among the
return-only candidates considered in Phases 2A/2B. Diagnostics examined
AR(0)–AR(10), with training BIC primary and AIC secondary. Only AR(0) is
intercept-only; AR(p) for p > 0 includes an intercept and p lagged returns.
The compact A0–A4 candidates add signed-return horizons, absolute return and
20-day volatility successively.

| Stocks | Frozen Model A |
| --- | --- |
| NVDA, AAPL, BRK.B, LLY, JNJ, XOM, CVX | A0 |
| JPM, CAT, GE | AR(9) |
| AMZN, TSLA | A3 |

A0 is an expanding historical mean. AR(9) uses the latest nine observed returns.
A3 has an intercept plus LastReturn = r_t, Prior4DayReturn = sum(r_(t-1)…r_(t-4)),
Prior15DayReturn = sum(r_(t-5)…r_(t-19)), and AbsLastReturn = |r_t|.
The machine-readable mapping is `config/model_a_frozen.csv`.
It is not changed in response to volume or test results.

## Volume candidates and final selection

B0 is the frozen Model A. B1 adds AbnormalDollarVolume20; B2 also adds
DollarVolumeChange1; B3 also adds DollarVolumePersistence5. Using log activity L:

- AbnormalDollarVolume20 = (L_t − mean(L_(t-20)…L_(t-1))) / sample_std(L_(t-20)…L_(t-1)).
- DollarVolumeChange1 = L_t − L_(t-1).
- DollarVolumePersistence5 = mean of today's and previous four abnormality values.

The prior-20 reference excludes today; sample standard deviation uses ddof=1.
S0–S3 repeat the same definitions with vendor share Volume rather than
DollarVolume. All models use ordinary linear regression with an intercept.

Final B is selected independently per stock among B1–B3 by
saved 2023–2024 validation MSE. Final robustness S is selected independently
among S1–S3 by the same rule. B0/S0 contain no volume information and are
excluded from this selection; exact ties favor the simpler candidate.

| Stocks | Final B | Final S |
| --- | --- | --- |
| AAPL | B2 | S2 |
| TSLA | B3 | S3 |
| Other ten stocks | B1 | S1 |

Selection never uses 2025 outcomes.

Within each stock, A/B/S use the same eligible historical rows complete for
the union of their selected predictors, with identical final target dates.
TSLA's selected persistence features require four additional warm-up rows:
its first training target is 2015-02-09 (2,491 observations initially), versus
2015-02-03 (2,495) for the other stocks. All have 250 final targets, for
12 × 250 × 3 = 9,000 forecasts. Saved validation used complete B3/S3 histories;
validation/test sign comparisons therefore reflect period and sample-rule
differences and are descriptive.

## Final results and robustness

MSE is the primary selection/evaluation metric; RMSE and MAE are complementary.
Positive improvement means 100 × (MSE_A − MSE_volume) / MSE_A.
Cross-stock summaries give equal weight to each stock.

| Model | Improved / worsened | Mean improvement | Median improvement | Validation/test sign agreement |
| --- | --- | --- | --- | --- |
| B | 6 / 6 | −0.073135% | +0.019773% | 6/12 |
| S | 7 / 5 | −0.113084% | +0.055716% | 7/12 |

The evidence is small, mixed and unstable across stocks rather than broad
predictive improvement. S is activity-representation robustness, not independent
replication. A separate BRK.B sensitivity check masks features depending on the
2021-03-16/17 volume anomalies without changing primary source observations.
Conventional training OLS p-values/standard errors are diagnostics, not evidence
of final forecast improvement; no new final significance test is imposed.

## Reproduction and presentation

Follow the root [README](README.md) for the complete ordered workflow:
raw snapshot/download → processed data → return diagnostics/features/validation →
dollar-volume features/validation → BRK.B sensitivity and share-volume robustness →
stock-specific final evaluation → descriptive EDA.
Source and feature construction remain in `src/`; presentation consumes outputs.

Current results are in
[the numerical report](outputs/final_2025_stock_specific_volume_report.md)
and [the report asset index](outputs/report/README.md).
The accompanying written report is submitted separately.

Exact reproduction requires the same source snapshot and numerical environment;
a new Yahoo download can revise historical values. The retrospective survivor
sample, changing businesses/sector labels, Yahoo corporate-action semantics,
BRK.B anomalies and descriptive holdout exposure limit interpretation.
The stored vendor snapshot is not a point-in-time archive; GE corporate-action
adjustments and the two BRK.B volume anomalies remain unresolved source
qualifications. Dollar activity includes the current price component, so the
comparison does not isolate volume beyond all possible price information.
Validation was reused across the bounded AR/return-feature and volume candidate
searches; winning validation margins can be negligible and selection minima
optimistic. Conventional nonrobust OLS inference and repeated diagnostics do not
establish forecast significance. The historical record has no independently
versioned pre-test decision timeline.
Do not generalize to all equities, infer causality, or treat tiny gains as
established signals. No dashboard or additional modeling layer is required.
