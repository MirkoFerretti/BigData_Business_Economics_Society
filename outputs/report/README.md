# Descriptive EDA for the written report

This presentation package uses all twelve frozen stocks and stored data only.
It is descriptive and did not alter model selection, features, windows, frozen
Model A/B/S, validation results or 2025 results. It contains no predictive fit,
new statistical test, download or dependency. Sector panels organize the stocks;
they do not estimate sector effects. Stock order follows config/universe.csv.

## Reproduce

From the repository root, using the existing environment:

```powershell
.\.venv\Scripts\python.exe -B src/analysis/report_eda.py
```

The script reads source fields as strings and selects dates before numeric
conversion. Same-day EDA uses Date in 2015-01-01 through 2024-12-31. The actual
source dates are 2015-01-02 through 2024-12-31. Next-day EDA requires BOTH Date
and TargetDate in that period; the 2024-12-31 row targeting 2025-01-02 is excluded
before its target is parsed. The ACF appendix, optional for presentation but
required by this generator, keeps its original 2015–2022 training sample.
No new EDA consumes a 2025 value. Existing 2025
results are linked separately below and are not used to compute the EDA.

Only outputs/report/ is written. Re-running replaces this script's presentation
assets; it never rewrites analytical data, earlier output files or the final
integrity JSON. Matplotlib uses a temporary configuration directory so existing
font caches remain unchanged. PNGs use 240 dpi, restrained colors, consistent
fonts and full uncropped data ranges; large multi-panel figures suit report
page width rather than a narrow single column.

## Artifact manifest and captions

All paths below are relative to this index. The [raw snapshot](../../data/raw/README.md)
is supplied with checksums. Generated source tables named below are recreated by
the root README pipeline and are not included as large release intermediates.
Every item is descriptive and did not alter model selection.
CSV return values are in log-return units, not percentages; figure return axes
show 100 × log return, not the simple-return transformation exp(r) − 1. No return
or volatility is annualized.

| Artifact | Source and period | Variables, alignment and interpretation |
| --- | --- | --- |
| [Indexed adjusted prices](figures/eda_indexed_adjusted_close.png) | `market_daily.csv`, 2015–2024 | Same-day Adj Close, Date, ticker, sector; 100 × adjusted price / first displayed price within stock, for display only. Log y-axis; sector grouping is organizational, not sector-performance analysis or investment strategy. Dashed boundary is 1 January 2023. No normalized modeling feature is stored. |
| [Descriptive statistics CSV](tables/eda_stock_descriptive_statistics.csv) | market_daily.csv and [universe.csv](../../config/universe.csv), 2015–2024 | One row per stock: identifiers, dates, defined-return count, daily mean/median/sample std (ddof=1), 1st/99th percentiles of LogReturn, median Volume and DollarVolume. Same-day summaries; no ranking, Sharpe ratio or annualization. Activity medians use all source dates; returns omit only the initial structural missing value. |
| [Return ECDFs](figures/eda_return_distributions.png) | market_daily.csv, 2015–2024 | Defined same-day LogReturn, common return axes and zero reference. Empirical cumulative probability reveals different dispersion and tails; every extreme is retained. No distribution fitting, normality testing or predictability claim. |
| [Existing 20-day volatility](figures/eda_realized_vol20.png) | `model_a_features.csv`, 2015–2024 | Stored same-day RealizedVol20: sample std of r[t]…r[t−19] with ddof=1. Existing historical Model A candidate shown descriptively, not a new final predictor. Initial 20 undefined rows remain gaps. Dashed boundary is 1 January 2023. |
| [Activity distributions](figures/eda_activity_distributions.png) | `model_b_features.csv`, `model_b_share_volume_features.csv`, 2015–2024 | Same-day stored LogDollarVolume, LogShareVolume, AbnormalDollarVolume20, AbnormalShareVolume20. Abnormality = (today's log activity − prior-20 mean) / prior-20 sample std, excluding today from the reference. Boxplots show median/Q1/Q3 and observed whisker endpoints within 1.5 IQR, with every outlier retained. Recent-history scale is not a normality assumption. No new correlation calculation. |
| [Activity versus same-day magnitude](figures/eda_activity_vs_same_day_magnitude.png) | Dollar features, checked against market_daily.csv, 2015–2024 | AbnormalDollarVolume20[t] versus stored AbsLastReturn[t] = abs(LogReturn[t]); every defined pair. The relationship is contemporaneous, does not imply causality, and is not a forecasting result. Dollar activity includes the current Yahoo Close component. No fit, smoother, correlation, bins or test. |
| [Activity versus next-day signed return](figures/eda_activity_vs_next_day_signed_return.png) | Dollar features, BOTH Date and TargetDate through 2024-12-31 | AbnormalDollarVolume20[t] versus stored TargetNextReturn = r[t+1] on the next observed trading day. Unconditional descriptive development-period visualization, not an out-of-sample predictive test and not a basis for changing the frozen model. All defined pairs; no fit, threshold analysis, groups, correlation or test. |
| [Existing training ACF appendix](figures/eda_acf_heatmap.png) | [model_a_acf_pacf.csv](../tables/model_a_acf_pacf.csv); training dates checked against [return diagnostics](../tables/model_a_return_diagnostics.csv), 2015–2022 | Saved ACF values at trading lags 1–20, twelve stocks in configured order; lag zero/PACF omitted. No ACF/PACF recalculation, significance overlay, lag selection or new predictive evidence. |

The optional statistics PNG is deliberately omitted: all 13 required columns
(including company/sector/date metadata) would produce a very wide report table.
Use the exact CSV for a LaTeX table, placing metadata in its caption or appendix
when typesetting. This avoids a cramped and redundant raster table.

## Rows retained per stock

Counts below are variable-specific, not model common-B3/S3 fit samples.
Price/log-activity and raw activity medians use every development source row.
Return distributions omit only the first undefined return. Volatility and
abnormalities omit only their existing 20-row warm-up. Same-day pairs include
the 2024-12-31 information date; next-day pairs end at information 2024-12-30
and target 2024-12-31. No persistence-feature availability restriction is imposed.

| stock | price / activity levels | returns / ECDF | volatility20 | abnormal dollar | abnormal share | same-day pair | next-day pair |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NVDA | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| AAPL | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| BRK.B | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| JPM | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| LLY | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| JNJ | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| AMZN | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| TSLA | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| CAT | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| GE | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| XOM | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |
| CVX | 2516 | 2515 | 2496 | 2496 | 2496 | 2496 | 2495 |

Each stock contributes 20 saved ACF values; ACF estimates retain the original
2015–2022 training-return sample, not the 2015–2024 development observations.
The descriptive statistics CSV has exactly twelve rows in configured order.

## Data and interpretation safeguards

No unexpected missingness, infinite value, duplicate ticker/date, source-key
misalignment or target mismatch was accepted. The original BRK.B observations
on 2021-03-16 and 2021-03-17 (Volume 67,200 and 67,700) remain in every applicable
activity and pair sample. They are known suspicious source observations, not
corrected values; their extreme abnormalities remain visible. Only documented
initial missingness and the explicit next-day date boundary remove plotted rows.
Stored AbsLastReturn and next-day targets are checked against the development
market returns without regenerating features or storing alternative definitions.

Yahoo Close/Volume are preserved vendor fields, not independently reconstructed
historically unadjusted tape data. DollarVolume = Close × Volume is the stored
activity representation. Adjusted-price trajectories are not an independently
reconstructed reinvestment strategy. See [PROJECT.md](../../PROJECT.md)
for source-vintage, corporate-action, retrospective-universe and historical
holdout-exposure qualifications. This post-completion descriptive
package cannot establish earlier holdout blindness.

## Current stock-specific final results — links only, not recomputed

Model B is selected per stock among B1–B3 using saved 2023–2024
validation MSE; robustness Model S is selected independently among S1–S3.
AAPL selects B2/S2, TSLA B3/S3, and the other ten stocks B1/S1.
This stock-specific comparison followed an earlier evaluation; that earlier evidence is HISTORICAL.

- [Current numerical report](../final_2025_stock_specific_volume_report.md)

Figures:

- [Primary B improvement, 2025](../figures/final_2025_stock_specific_volume/model_b_mse_improvement.png)
- [Share-volume S robustness, 2025](../figures/final_2025_stock_specific_volume/model_s_mse_improvement.png)
- [B/S validation versus final test](../figures/final_2025_stock_specific_volume/validation_vs_test.png)

Tables:

- [Selected models and validation evidence](../tables/stock_specific_volume_model_selection.csv)
- [All stock/model metrics](../tables/final_2025_stock_specific_volume_metrics.csv)
- [Primary A/B and robustness A/S comparisons](../tables/final_2025_stock_specific_volume_comparison.csv)
- [Equal-stock summary](../tables/final_2025_stock_specific_volume_cross_stock_summary.csv)
- [Common training histories](../tables/final_2025_stock_specific_volume_training_counts.csv)

The validation/test scatter is descriptive: saved validation used complete
B3/S3 estimation histories, while final evaluation uses common eligible histories
for each stock's selected A/B/S predictors. TSLA's persistence features impose
additional warm-up. It does not isolate only a change of market period. These
EDA plots neither revise the selections nor calculate a competing final result.

## Compact development robustness evidence

The [BRK.B source-anomaly sensitivity chart](../figures/model_b_robustness/brkb_sensitivity.png)
and [compact summary](../tables/model_b_brkb_sensitivity_summary.csv) preserve
the robustness evidence discussed in PROJECT. This masks
dependencies rather than inventing source corrections; it is not a new final
2025 result. Other development plots are regenerated by their stage scripts.

## Descriptive reading of the figures

These are visual observations, not fitted effects or new predictive evidence.

| Figure | Descriptive observation |
| --- | --- |
| Indexed adjusted prices | Paths diverge substantially: NVDA and TSLA show large changes from their starting index; GE shows a prolonged decline followed by recovery. Within-panel stocks do not move identically. These are historical paths, not performance recommendations. |
| Return ECDFs | Most returns cluster near zero, with visible tails on both sides. TSLA and NVDA have broader distributions than JNJ and BRK.B on the common return axis. |
| Existing volatility | A marked 2020 volatility spike is visible across the panels; volatility also varies outside that episode. TSLA and NVDA often have larger daily fluctuation scales than their panel companions. |
| Activity distributions | Log activity levels differ substantially by company. Abnormal-activity boxes occupy more comparable scales while retaining long tails; BRK.B has conspicuous negative source-anomaly points. Dollar/share distributions look similar but are not identical. |
| Same-day magnitude | Many panels show a wider range of absolute moves at higher abnormal activity. Large and small moves coexist, and the original low-activity BRK.B observations remain separate from the central cloud. This is contemporaneous description only. |
| Next-day signed return | Clouds remain concentrated around zero with both signs represented at unusual activity levels. They do not visually reproduce the same-day magnitude pattern as a clear signed direction; this unconditional view neither establishes nor rules out predictability. |
| Saved ACF appendix | Most stored values are modest, with positive and negative values varying across lags and stocks. Color differences carry no significance assessment and do not reopen lag selection. |

## Report and oral-exam use

The price/volatility context, the same-day versus next-day contrast and the
existing primary final-result chart form the shortest oral narrative. Use the
statistics CSV and ECDFs for dispersion/tail context; the activity boxplots explain
scale and abnormality. The ACF heatmap is optional appendix material because
training dependence already has a documented role in the historical model search.
The two relationship figures deliberately answer different questions; one does
not provide forecasting evidence for the other. No sector inference is intended.
