r"""Development-only descriptive report assets from stored data; no model fitting.

Run from the repository root: .\.venv\Scripts\python.exe -B src/analysis/report_eda.py
Only outputs/report/ is written. Existing analytical artifacts are never published
or regenerated. Numeric target conversion happens AFTER both date cutoffs.
"""

from pathlib import Path
import os
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
START, END = "2015-01-01", "2024-12-31"
VALIDATION = pd.Timestamp("2023-01-01")
COLORS = ("#355f7c", "#9b6747")
MARKET = "data/processed/market_daily.csv"
RETURN_FEATURES = "data/processed/model_a_features.csv"
DOLLAR_FEATURES = "data/processed/model_b_features.csv"
SHARE_FEATURES = "data/processed/model_b_share_volume_features.csv"
ACF = "outputs/tables/model_a_acf_pacf.csv"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_development(root, path, numeric, metadata=(), next_day=False):
    """Select source strings before interpreting any development-only values."""
    columns = list(dict.fromkeys(["Date", "display_ticker", *metadata, *numeric]))
    if next_day:
        columns.append("TargetDate")
    frame = pd.read_csv(root / path, usecols=columns, dtype=str, keep_default_na=False)
    mask = frame.Date.between(START, END)
    if next_day:
        mask &= frame.TargetDate.between(START, END)
    frame = frame.loc[mask].copy()
    for column in ["Date"] + (["TargetDate"] if next_day else []):
        dates = pd.to_datetime(frame[column], format="%Y-%m-%d", errors="raise")
        require(dates.dt.strftime("%Y-%m-%d").eq(frame[column]).all(), f"Invalid {column}")
        require(frame[column].between(START, END).all(), f"Outside development: {column}")
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
        require(not np.isinf(frame[column]).any(), f"Infinite {path}: {column}")
    require(not frame.duplicated(["display_ticker", "Date"]).any(), f"Duplicate rows: {path}")
    if next_day:
        require(frame.Date.lt(frame.TargetDate).all(), "Next-day target must follow Date")
    return frame.reset_index(drop=True)


def check_missingness(frame, tickers, expected):
    """Only the saved variables' documented initial warm-up may be omitted."""
    require(set(frame.display_ticker) == set(tickers), "Incomplete twelve-stock universe")
    for ticker in tickers:
        stock = frame.loc[frame.display_ticker.eq(ticker)].reset_index(drop=True)
        require(stock.Date.is_monotonic_increasing, f"Unordered dates: {ticker}")
        for column, leading_rows in expected.items():
            np.testing.assert_array_equal(stock[column].isna().to_numpy(),
                                          np.arange(len(stock)) < leading_rows,
                                          err_msg=f"Unexpected missingness: {ticker}/{column}")


def load_inputs(root=ROOT):
    universe = pd.read_csv(root / "config/universe.csv", dtype=str, keep_default_na=False)
    tickers = universe.display_ticker.tolist()
    require(len(tickers) == len(set(tickers)) == 12, "Expected twelve unique configured stocks")
    require(universe.groupby("sector", sort=False).size().tolist() == [2] * 6,
            "Expected two stocks in each of six configured sectors")
    market = read_development(root, MARKET, ["Adj Close", "LogReturn", "Volume", "DollarVolume"],
                              ["company", "sector"])
    vol = read_development(root, RETURN_FEATURES, ["RealizedVol20"], ["sector"])
    dollar = read_development(root, DOLLAR_FEATURES,
                              ["LogDollarVolume", "AbnormalDollarVolume20", "AbsLastReturn"])
    share = read_development(root, SHARE_FEATURES, ["LogShareVolume", "AbnormalShareVolume20"])
    # Never parse the late-2024 information row's 2025 outcome.
    following = read_development(root, DOLLAR_FEATURES,
                                 ["AbnormalDollarVolume20", "TargetNextReturn"], next_day=True)
    check_missingness(market, tickers, {"Adj Close": 0, "Volume": 0, "DollarVolume": 0, "LogReturn": 1})
    check_missingness(vol, tickers, {"RealizedVol20": 20})
    check_missingness(dollar, tickers,
                       {"LogDollarVolume": 0, "AbnormalDollarVolume20": 20, "AbsLastReturn": 1})
    check_missingness(share, tickers, {"LogShareVolume": 0, "AbnormalShareVolume20": 20})
    check_missingness(following, tickers, {"AbnormalDollarVolume20": 20, "TargetNextReturn": 0})
    keys = ["display_ticker", "Date"]
    for frame in [vol, dollar, share]:
        pd.testing.assert_frame_equal(frame[keys], market[keys], check_exact=True)
    require(market[["Adj Close", "Volume", "DollarVolume"]].gt(0).all().all(),
            "Nonpositive source price/activity")
    np.testing.assert_allclose(dollar.AbsLastReturn, market.LogReturn.abs(),
                               rtol=1e-12, atol=1e-14, equal_nan=True)
    for row in universe.itertuples(index=False):
        stock = market.loc[market.display_ticker.eq(row.display_ticker)]
        require(stock.company.eq(row.company).all() and stock.sector.eq(row.sector).all(),
                f"Identifier mismatch: {row.display_ticker}")
        require(vol.loc[vol.display_ticker.eq(row.display_ticker), "sector"].eq(row.sector).all(),
                "Volatility sector mismatch")
        expected = stock[["Date", "LogReturn"]].copy()
        expected["TargetDate"] = expected.Date.shift(-1)
        expected["TargetNextReturn"] = expected.LogReturn.shift(-1)
        expected = expected.iloc[:-1].reset_index(drop=True)
        stored = following.loc[following.display_ticker.eq(row.display_ticker)].reset_index(drop=True)
        pd.testing.assert_frame_equal(stored[["Date", "TargetDate"]], expected[["Date", "TargetDate"]])
        np.testing.assert_allclose(stored.TargetNextReturn, expected.TargetNextReturn, rtol=1e-12, atol=1e-14)
    # These original observations must remain visible in the descriptive samples.
    suspicious = ["2021-03-16", "2021-03-17"]
    brk = market.loc[market.display_ticker.eq("BRK.B") & market.Date.isin(suspicious)]
    require(brk.Date.tolist() == suspicious and brk.Volume.tolist() == [67200, 67700],
            "Original BRK.B suspicious observations not present")
    for frame in [dollar, share, following]:
        subset = frame.loc[frame.display_ticker.eq("BRK.B") & frame.Date.isin(suspicious)]
        require(len(subset) == 2 and subset.notna().all().all(), "BRK.B rows unavailable")
    acf = pd.read_csv(root / ACF, dtype=str, keep_default_na=False)
    acf = acf.loc[acf.diagnostic_type.eq("ACF")].copy()
    acf["lag"] = pd.to_numeric(acf.lag, errors="raise")
    acf = acf.loc[acf.lag.between(1, 20)].copy()
    acf["value"] = pd.to_numeric(acf.value, errors="raise")
    require(not acf.duplicated(["display_ticker", "lag"]).any(), "Duplicate stored ACF")
    matrix = acf.pivot(index="display_ticker", columns="lag", values="value").reindex(tickers)
    require(matrix.columns.tolist() == list(range(1, 21)) and np.isfinite(matrix).all().all(),
            "Incomplete stored training ACF")
    diagnostics = pd.read_csv(root / "outputs/tables/model_a_return_diagnostics.csv", dtype=str)
    require(set(diagnostics.display_ticker) == set(tickers) and len(diagnostics) == 12 and
            diagnostics.training_last_date.le("2022-12-31").all() and
            diagnostics.training_first_date.ge(START).all(), "Unexpected stored ACF training period")
    return universe, market, vol, dollar, share, following, matrix


def descriptive_statistics(universe, market):
    rows = []
    for security in universe.to_dict("records"):
        stock = market.loc[market.display_ticker.eq(security["display_ticker"])]
        returns = stock.LogReturn.dropna()
        rows.append({"display_ticker": security["display_ticker"], "company": security["company"],
                     "sector": security["sector"], "first_date": stock.Date.iloc[0],
                     "last_date": stock.Date.iloc[-1], "n_return_observations": len(returns),
                     "mean_daily_log_return": returns.mean(), "median_daily_log_return": returns.median(),
                     "std_daily_log_return": returns.std(ddof=1), "q01_daily_log_return": returns.quantile(.01),
                     "q99_daily_log_return": returns.quantile(.99), "median_share_volume": stock.Volume.median(),
                     "median_dollar_volume": stock.DollarVolume.median()})
    return pd.DataFrame(rows)


def row_counts(universe, market, vol, dollar, share, following):
    rows = []
    for ticker in universe.display_ticker:
        parts = [f.loc[f.display_ticker.eq(ticker)] for f in [market, vol, dollar, share, following]]
        m, v, d, s, f = parts
        rows.append({"stock": ticker, "price / activity levels": len(m), "returns / ECDF": m.LogReturn.notna().sum(),
                     "volatility20": v.RealizedVol20.notna().sum(),
                     "abnormal dollar": d.AbnormalDollarVolume20.notna().sum(),
                     "abnormal share": s.AbnormalShareVolume20.notna().sum(),
                     "same-day pair": d[["AbnormalDollarVolume20", "AbsLastReturn"]].notna().all(axis=1).sum(),
                     "next-day pair": f[["AbnormalDollarVolume20", "TargetNextReturn"]].notna().all(axis=1).sum()})
    return pd.DataFrame(rows)


def finish(fig, directory, name, title, caption, bottom=.12):
    import matplotlib.pyplot as plt
    fig.suptitle(title, fontsize=12, y=.99)
    fig.text(.02, .015, caption, ha="left", va="bottom", fontsize=8.5, linespacing=1.4)
    fig.tight_layout(rect=(0, bottom, 1, .955), h_pad=1.6, w_pad=1.2)
    fig.savefig(directory / name, dpi=240, facecolor="white")
    plt.close(fig)


def time_panels(universe, frame, column, indexed, directory):
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    fig, axes = plt.subplots(3, 2, figsize=(11, 9), sharex=True, sharey=True)
    for ax, (sector, securities) in zip(axes.flat, universe.groupby("sector", sort=False)):
        for color, ticker in zip(COLORS, securities.display_ticker):
            stock = frame.loc[frame.display_ticker.eq(ticker)]
            values = stock[column]
            shown = 100 * values / values.iloc[0] if indexed else 100 * values
            ax.plot(pd.to_datetime(stock.Date), shown, label=ticker, color=color, linewidth=.85)
        if indexed:
            ax.set_yscale("log")
            ax.axhline(100, color=".75", linewidth=.6)
        ax.axvline(VALIDATION, color=".35", linestyle="--", linewidth=.9)
        ax.set_title(sector, fontsize=10)
        ax.set_xlim(pd.Timestamp(START), pd.Timestamp(END))
        ax.xaxis.set_major_locator(mdates.YearLocator(2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.legend(loc="upper left", frameon=False, fontsize=9, ncols=2)
    fig.supylabel("Adjusted-price index (start = 100; log scale)" if indexed else
                  "Daily 20-return sample std (100 × log-return units)", fontsize=10)
    if indexed:
        finish(fig, directory, "eda_indexed_adjusted_close.png", "Adjusted-price paths · development period, 2015–2024",
               "Rebasing is for visual comparability only; sector grouping is organizational, not sector-performance analysis.\n"
               "These vendor-adjusted paths are not an investment strategy. Dashed line: start of validation, 1 January 2023.")
    else:
        finish(fig, directory, "eda_realized_vol20.png", "Existing 20-day volatility · development period, 2015–2024",
               "Stored RealizedVol20: sample std (ddof=1) of r[t] through r[t−19], in daily units; not annualized.\n"
               "Historical Model A candidate shown descriptively, not a new final predictor. Dashed line: 1 January 2023.")


def return_distributions(universe, market, directory):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 4, figsize=(12, 8), sharex=True, sharey=True)
    for ax, ticker in zip(axes.flat, universe.display_ticker):
        values = np.sort(market.loc[market.display_ticker.eq(ticker), "LogReturn"].dropna().to_numpy()) * 100
        ax.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values) + 1) / len(values)],
                where="post", linewidth=1.2, color=COLORS[0])
        ax.axvline(0, color=".55", linewidth=.7)
        ax.set_title(ticker, fontsize=10)
        ax.set_ylim(-.02, 1.02)
    fig.supxlabel("Daily signed log return (100 × log-return units)", y=.11, fontsize=10)
    fig.supylabel("Empirical cumulative probability", fontsize=10)
    finish(fig, directory, "eda_return_distributions.png", "Daily return distributions · 2015–2024",
           "Every defined development return is retained, including extremes; only the initial undefined return is omitted.\n"
           "Common return axes; empirical CDFs only. No fitted distribution, normality test, clipping or winsorization.", bottom=.17)


def activity_distributions(universe, dollar, share, directory):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    items = [(dollar, "LogDollarVolume", "Log dollar activity", "ln(Yahoo Close × Volume)"),
             (share, "LogShareVolume", "Log share volume", "ln(Yahoo share Volume)"),
             (dollar, "AbnormalDollarVolume20", "Abnormal dollar activity", "Prior-20 log-activity std units"),
             (share, "AbnormalShareVolume20", "Abnormal share volume", "Prior-20 log-activity std units")]
    for ax, (frame, column, title, label) in zip(axes.flat, items):
        values = [frame.loc[frame.display_ticker.eq(t), column].dropna().to_numpy() for t in universe.display_ticker]
        ax.boxplot(values, tick_labels=universe.display_ticker.tolist(), widths=.55, showfliers=True,
                   flierprops={"marker": ".", "markersize": 2, "alpha": .35, "markeredgecolor": COLORS[0]},
                   medianprops={"color": COLORS[1], "linewidth": 1.2},
                   boxprops={"color": COLORS[0]}, whiskerprops={"color": COLORS[0]}, capprops={"color": COLORS[0]})
        ax.set(title=title, ylabel=label)
        ax.tick_params(axis="x", labelrotation=55)
        if column.startswith("Abnormal"):
            ax.axhline(0, color=".65", linewidth=.6)
    # Matched abnormality scale for comparing representations; keep every outlier.
    lower = min(axes[1, 0].get_ylim()[0], axes[1, 1].get_ylim()[0])
    upper = max(axes[1, 0].get_ylim()[1], axes[1, 1].get_ylim()[1])
    for ax in axes[1]:
        ax.set_ylim(lower, upper)
    finish(fig, directory, "eda_activity_distributions.png", "Trading-activity distributions · 2015–2024",
           "Stored log/abnormal activity only: today's log activity relative to the preceding 20 observations (excluding today; ddof=1).\n"
           "Boxes show Q1–Q3 and medians; whiskers extend to observed values within 1.5 IQR. All outliers, including BRK.B, remain visible.\n"
           "Abnormality uses a recent-history scale; it is not assumed normally distributed. No new correlation is calculated.", bottom=.16)


def activity_return_panels(universe, frame, next_day, directory):
    import matplotlib.pyplot as plt
    ycolumn = "TargetNextReturn" if next_day else "AbsLastReturn"
    fig, axes = plt.subplots(3, 4, figsize=(12, 8.5), sharex=True, sharey=True)
    for ax, ticker in zip(axes.flat, universe.display_ticker):
        stock = frame.loc[frame.display_ticker.eq(ticker)].dropna(subset=["AbnormalDollarVolume20", ycolumn])
        ax.scatter(stock.AbnormalDollarVolume20, 100 * stock[ycolumn], s=4, alpha=.18,
                   color=COLORS[0], edgecolors="none")
        ax.axhline(0, color=".6", linewidth=.6)
        ax.set_title(f"{ticker}  (n={len(stock):,})", fontsize=10)
    if not next_day:
        for ax in axes.flat:
            ax.set_ylim(bottom=0)
    fig.supxlabel("AbnormalDollarVolume20 (prior-20 log-activity std units)", y=.12, fontsize=10)
    fig.supylabel(("Next-day signed" if next_day else "Same-day absolute") +
                  " log return (100 × log-return units)", fontsize=10)
    if next_day:
        finish(fig, directory, "eda_activity_vs_next_day_signed_return.png", "Abnormal activity and next-day signed return · 2015–2024",
               "Unconditional descriptive development-period visualization, not an out-of-sample predictive test.\n"
               "Both information Date and TargetDate are within 2015–2024; no 2025 outcome enters.\n"
               "Not a basis for changing the frozen model. All valid pairs shown; no fit, smoother, correlation or significance test.", bottom=.19)
    else:
        finish(fig, directory, "eda_activity_vs_same_day_magnitude.png", "Abnormal activity and same-day return magnitude · 2015–2024",
               "Contemporaneous descriptive relationship: it does not imply causality and is not a forecasting result.\n"
               "Dollar activity includes the current Yahoo Close component. All valid pairs and original BRK.B observations retained.\n"
               "AbsLastReturn has its existing meaning |r[t]|; no regression, smoother, correlation or significance test.", bottom=.19)


def acf_heatmap(matrix, directory):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5))
    limit = float(np.abs(matrix.to_numpy()).max())
    picture = ax.imshow(matrix, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
    ax.set(xticks=range(20), xticklabels=range(1, 21), yticks=range(12), yticklabels=matrix.index,
           xlabel="Return lag (trading observations)")
    ax.grid(False)
    fig.colorbar(picture, ax=ax, label="Stored sample ACF", fraction=.035, pad=.025)
    finish(fig, directory, "eda_acf_heatmap.png", "Existing training-return autocorrelation · 2015–2022",
           "Uses only saved ACF estimates at lags 1–20; training period only. No ACF/PACF recomputation or significance bands.\n"
           "Descriptive appendix: values do not select lags, change frozen models or establish out-of-sample predictability.", bottom=.17)


def markdown_table(frame):
    columns = frame.columns.tolist()
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(str(v) for v in row) + " |")
    return "\n".join(lines)


def write_index(output, counts):
    text = """# Descriptive EDA for the written report

This presentation package uses all twelve frozen stocks and stored data only.
It is descriptive and did not alter model selection, features, windows, frozen
Model A/B/S, validation results or 2025 results. It contains no predictive fit,
new statistical test, download or dependency. Sector panels organize the stocks;
they do not estimate sector effects. Stock order follows config/universe.csv.

## Reproduce

From the repository root, using the existing environment:

```powershell
.\\.venv\\Scripts\\python.exe -B src/analysis/report_eda.py
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

"""
    text += markdown_table(counts)
    text += """

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
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def run(root=ROOT):
    universe, market, vol, dollar, share, following, matrix = load_inputs(root)
    output = root / "outputs/report"
    figures, tables = output / "figures", output / "tables"
    figures.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    statistics = descriptive_statistics(universe, market)
    counts = row_counts(universe, market, vol, dollar, share, following)
    # Private temporary font configuration prevents mutation of existing caches.
    old_config = os.environ.get("MPLCONFIGDIR")
    with TemporaryDirectory(prefix="report-eda-mpl-") as mpl_config:
        os.environ["MPLCONFIGDIR"] = mpl_config
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10,
                                 "axes.titlesize": 10, "axes.labelsize": 9,
                                 "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                                 "axes.spines.top": False, "axes.spines.right": False,
                                 "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .5,
                                 "legend.fontsize": 9, "savefig.facecolor": "white"}):
                time_panels(universe, market, "Adj Close", True, figures)
                return_distributions(universe, market, figures)
                time_panels(universe, vol, "RealizedVol20", False, figures)
                activity_distributions(universe, dollar, share, figures)
                activity_return_panels(universe, dollar, False, figures)
                activity_return_panels(universe, following, True, figures)
                acf_heatmap(matrix, figures)
            # Decode each PNG before declaring success; no GUI launch needed.
            for path in figures.glob("eda_*.png"):
                pixels = plt.imread(path)
                require(pixels.ndim == 3 and pixels.shape[0] > 0, f"Unreadable PNG: {path}")
        finally:
            if old_config is None:
                os.environ.pop("MPLCONFIGDIR", None)
            else:
                os.environ["MPLCONFIGDIR"] = old_config
    statistics.to_csv(tables / "eda_stock_descriptive_statistics.csv", index=False)
    write_index(output, counts)
    print(counts.to_string(index=False))
    print("\nAll twelve stocks; 2025 excluded before numeric conversion, including next-day targets.")
    print("BRK.B suspicious observations retained; only documented structural missingness omitted.")
    print("Created seven consolidated PNGs, one statistics CSV and README; no models or tests run.")
    return counts


if __name__ == "__main__":
    run()
