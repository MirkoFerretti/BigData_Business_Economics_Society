"""Phase 2A: return-only evidence for human lag review; never select a final model.

IC fits use conditional Gaussian likelihood/OLS, an intercept only, and
hold_back=10 for ALL p, hence identical training response dates. Counts for
AIC/BIC include p+1 regression coefficients and one innovation variance.
Validation uses all prior data with only each candidate's p-row lag loss;
AR(0) is therefore the full expanding historical mean, not the IC-fit mean.
ACF uses demeaned, unadjusted autocovariances; PACF uses Yule-Walker MLE.
Ljung-Box is applied to training returns, not fitted residuals (model_df=0).
Figures omit significance bands; no significance cutoff selects a lag.
"""

import os
from pathlib import Path
import re
from shutil import copyfile
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.stattools import acf, pacf


ROOT = Path(__file__).resolve().parents[2]
TRAIN_START, TRAIN_END = "2015-01-02", "2022-12-31"
VALID_START, VALID_END = "2023-01-01", "2024-12-31"
MAX_P, DIAGNOSTIC_LAGS = 10, 20
LB_LAGS = [5, 10, 20]
TABLE_NAMES = [
    "return_diagnostics", "acf_pacf", "ar_information_criteria",
    "validation_metrics", "lag_summary", "validation_forecasts",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_development(root: Path = ROOT) -> dict[str, pd.Series]:
    universe = pd.read_csv(root / "config/universe.csv", dtype=str)
    tickers = universe["display_ticker"].tolist()
    require(len(tickers) == len(set(tickers)) == 12, "Expected the frozen 12-stock universe")
    require(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", t) for t in tickers),
            "Unsafe or missing display ticker in universe")
    # Read only these three fields. Test returns remain uninterpreted strings:
    # even missing/infinite/nonnumeric 2025 values cannot enter a calculation.
    source = pd.read_csv(root / "data/processed/market_daily.csv",
                         usecols=["Date", "display_ticker", "LogReturn"],
                         dtype=str, keep_default_na=False)
    require(set(source["display_ticker"]) == set(tickers), "Processed universe mismatch")
    test_present = set(source.loc[source["Date"].between("2025-01-01", "2025-12-31"),
                                  "display_ticker"])
    require(test_present == set(tickers), "Expected final-test observations for every stock")
    development = source.loc[source["Date"].between(TRAIN_START, VALID_END)].copy()
    del source  # Nothing downstream can access the test rows.
    development["Date"] = pd.to_datetime(development["Date"], format="%Y-%m-%d", errors="raise")
    development["LogReturn"] = pd.to_numeric(
        development["LogReturn"].replace("", np.nan), errors="raise")
    stocks = {}
    for ticker in tickers:
        stock = development.loc[development["display_ticker"].eq(ticker)]
        dates = pd.DatetimeIndex(stock["Date"])
        require(not stock.empty and dates.is_monotonic_increasing and dates.is_unique,
                f"{ticker}: missing stock, duplicate dates, or nonchronological dates")
        returns = pd.Series(stock["LogReturn"].to_numpy(dtype=float), index=dates, name=ticker)
        require(dates[0] == pd.Timestamp(TRAIN_START), f"{ticker}: initial source date changed")
        require(returns.isna().sum() == 1 and pd.isna(returns.iloc[0]),
                f"{ticker}: expected only the structural initial missing LogReturn")
        returns = returns.iloc[1:]  # Remove only the known first-return structural NaN.
        check_development(returns)
        stocks[ticker] = returns
    return stocks


def check_development(returns: pd.Series) -> None:
    require(isinstance(returns.index, pd.DatetimeIndex) and len(returns) > 0,
            "Expected nonempty dated development returns")
    dates = returns.index
    require(dates.is_monotonic_increasing and dates.is_unique, "Nonchronological or duplicate dates")
    require(dates.min() >= pd.Timestamp(TRAIN_START) and dates.max() <= pd.Timestamp(VALID_END),
            "Temporal leakage: data outside the authorized development period")
    require(np.isfinite(returns.to_numpy()).all(), "Unexpected missing or infinite development return")
    require((dates <= TRAIN_END).sum() > 2 * DIAGNOSTIC_LAGS + MAX_P,
            "Insufficient initial training returns")
    require((dates >= VALID_START).any(), "No validation observations")


def training_diagnostics(ticker: str, train: pd.Series) -> tuple[dict, pd.DataFrame]:
    require(train.index.min() >= pd.Timestamp(TRAIN_START) and train.index.max() <= pd.Timestamp(TRAIN_END),
            "Temporal leakage: diagnostics must use initial training only")
    values = train.to_numpy()
    lb = acorr_ljungbox(values, lags=LB_LAGS, model_df=0, return_df=True)
    row = {
        "display_ticker": ticker, "training_observations": len(train),
        "training_first_date": train.index[0].strftime("%Y-%m-%d"),
        "training_last_date": train.index[-1].strftime("%Y-%m-%d"),
        "mean_return": values.mean(), "std_return": values.std(ddof=1),
        "std_ddof": 1, "pacf_method": "ywm", "ljung_box_model_df": 0,
    }
    for lag in LB_LAGS:
        row[f"ljung_box_stat_lag_{lag}"] = lb.loc[lag, "lb_stat"]
        row[f"ljung_box_pvalue_lag_{lag}"] = lb.loc[lag, "lb_pvalue"]
    diagnostic_values = {
        "ACF": acf(values, nlags=DIAGNOSTIC_LAGS, adjusted=False, fft=False, missing="raise"),
        "PACF": pacf(values, nlags=DIAGNOSTIC_LAGS, method="ywm"),
    }
    tidy = pd.DataFrame([
        {"display_ticker": ticker, "diagnostic_type": kind, "lag": lag, "value": value}
        for kind, estimates in diagnostic_values.items() for lag, value in enumerate(estimates)
    ])
    return row, tidy


def training_candidates(ticker: str, train: pd.Series) -> pd.DataFrame:
    require(train.index.min() >= pd.Timestamp(TRAIN_START) and train.index.max() <= pd.Timestamp(TRAIN_END),
            "Temporal leakage: information criteria must use initial training only")
    rows = []
    for p in range(MAX_P + 1):
        result = AutoReg(train.to_numpy(), lags=p, trend="c", seasonal=False,
                         hold_back=MAX_P, missing="raise").fit()
        require(int(result.nobs) == len(train) - MAX_P, "Incomparable IC response samples")
        rows.append({
            "display_ticker": ticker, "p": p, "training_observations": int(result.nobs),
            "training_returns_available": len(train), "comparison_hold_back": MAX_P,
            "training_first_response_date": train.index[MAX_P].strftime("%Y-%m-%d"),
            "training_last_response_date": train.index[-1].strftime("%Y-%m-%d"),
            "coefficient_count": len(result.params), "variance_parameter_count": 1,
            "parameter_count": len(result.params) + 1,
            "log_likelihood": result.llf, "AIC": result.aic, "BIC": result.bic,
        })
    return pd.DataFrame(rows)


def forecast_from_history(history: np.ndarray, p: int) -> float:
    """Refit conditional OLS from scratch; the caller passes strictly past data."""
    require(0 <= p <= MAX_P and len(history) > 2 * p + 1, "Invalid AR estimation history")
    require(np.isfinite(history).all(), "Invalid historical return")
    if p == 0:
        return float(history.mean())
    # Target r_i uses r_(i-1), ..., r_(i-p); no trend, seasonal or other regressors.
    design = np.column_stack([
        np.ones(len(history) - p),
        *(history[p - lag:len(history) - lag] for lag in range(1, p + 1)),
    ])
    coefficients, _, rank, _ = np.linalg.lstsq(design, history[p:], rcond=None)
    require(rank == p + 1, f"Rank-deficient AR({p}) estimation sample")
    forecast_row = np.r_[1.0, history[-p:][::-1]]
    return float(forecast_row @ coefficients)


def expanding_forecasts(ticker: str, returns: pd.Series, p: int) -> pd.DataFrame:
    check_development(returns)  # Reject any 2025 row before extracting numeric arrays.
    dates, values = returns.index, returns.to_numpy()
    rows = []
    for position in np.flatnonzero(dates >= VALID_START):
        date = dates[position]
        require(pd.Timestamp(VALID_START) <= date <= pd.Timestamp(VALID_END),
                "Temporal leakage: forecast outside validation")
        past_dates = dates[:position]
        require(len(past_dates) > 0 and past_dates[-1] < date,
                "Temporal leakage: forecast estimation includes current/future date")
        predicted = forecast_from_history(values[:position], p)
        require(np.isfinite(predicted), f"{ticker} AR({p}): nonfinite prediction")
        # Access the realized outcome only after computing its prediction.
        rows.append({
            "Date": date.strftime("%Y-%m-%d"), "display_ticker": ticker, "p": p,
            "predicted_return": predicted, "realized_return": float(values[position]),
            "training_start_date": past_dates[p].strftime("%Y-%m-%d"),
            "training_end_date": past_dates[-1].strftime("%Y-%m-%d"),
            "history_observations": position, "training_observations": position - p,
        })
    return pd.DataFrame(rows)


def validation_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    require(forecasts["Date"].between(VALID_START, VALID_END).all(),
            "Temporal leakage: metrics outside validation")
    rows = []
    for (ticker, p), group in forecasts.groupby(["display_ticker", "p"], sort=False):
        error = group["realized_return"].to_numpy() - group["predicted_return"].to_numpy()
        mse = np.mean(error ** 2)
        rows.append({"display_ticker": ticker, "p": p, "validation_observations": len(group),
                     "MSE": mse, "RMSE": np.sqrt(mse), "MAE": np.mean(np.abs(error))})
    return pd.DataFrame(rows)


def lag_summary(criteria: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    """Report descriptive minima only; never configure or refit a chosen model."""
    rows = []
    for ticker in criteria["display_ticker"].unique():
        ic = criteria.loc[criteria["display_ticker"].eq(ticker)].sort_values("p")
        validation = metrics.loc[metrics["display_ticker"].eq(ticker)].sort_values("p")
        row = {"display_ticker": ticker}
        for label, frame, column in [("training_BIC", ic, "BIC"), ("training_AIC", ic, "AIC"),
                                     ("validation_MSE", validation, "MSE"),
                                     ("validation_MAE", validation, "MAE")]:
            # In an exact tie, report the smallest p; no final selection is made.
            row[f"p_min_{label}"] = int(frame.loc[frame[column].idxmin(), "p"])
            row[f"number_of_minima_{label}"] = int(frame[column].eq(frame[column].min()).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def validate_outputs(tables: dict[str, pd.DataFrame], stocks: dict[str, pd.Series]) -> None:
    for name, table in tables.items():
        require(set(table["display_ticker"]) == set(stocks), f"{name}: universe mismatch")
        numeric = table.select_dtypes(include="number")
        require(np.isfinite(numeric.to_numpy()).all(), f"{name}: missing/infinite numeric output")
    ic, forecasts = tables["ar_information_criteria"], tables["validation_forecasts"]
    require(not forecasts.duplicated(["display_ticker", "p", "Date"]).any(), "Duplicate forecasts")
    require(forecasts["Date"].between(VALID_START, VALID_END).all(), "Temporal leakage in forecast dates")
    require(forecasts["training_end_date"].lt(forecasts["Date"]).all(), "Temporal leakage in fitting dates")
    require(ic["training_last_response_date"].le(TRAIN_END).all(), "Temporal leakage in IC fits")
    for ticker, returns in stocks.items():
        train, validation = returns.loc[:TRAIN_END], returns.loc[VALID_START:VALID_END]
        ticker_ic = ic.loc[ic["display_ticker"].eq(ticker)]
        require(ticker_ic["p"].tolist() == list(range(MAX_P + 1)), f"{ticker}: missing IC candidates")
        require(ticker_ic["training_observations"].eq(len(train) - MAX_P).all(), "Unequal IC samples")
        metric = tables["validation_metrics"].loc[tables["validation_metrics"]["display_ticker"].eq(ticker)]
        require(metric["p"].tolist() == list(range(MAX_P + 1)) and
                metric["validation_observations"].eq(len(validation)).all(), f"{ticker}: missing metrics")
        for p in range(MAX_P + 1):
            group = forecasts.loc[forecasts["display_ticker"].eq(ticker) & forecasts["p"].eq(p)]
            require(group["Date"].tolist() == validation.index.strftime("%Y-%m-%d").tolist(),
                    f"{ticker} AR({p}): missing or unordered validation predictions")
            np.testing.assert_array_equal(group["realized_return"], validation.to_numpy())
            positions = returns.index.get_indexer(validation.index)
            require(group["history_observations"].eq(positions).all() and
                    group["training_observations"].eq(positions - p).all(), "Window did not expand")
            require(group["training_end_date"].tolist() ==
                    returns.index[positions - 1].strftime("%Y-%m-%d").tolist(),
                    "Forecast origin is not the preceding observed trading day")


def save_figures(acf_pacf: pd.DataFrame, diagnostics: pd.DataFrame, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for ticker in diagnostics["display_ticker"]:
        dates = diagnostics.loc[diagnostics["display_ticker"].eq(ticker)].iloc[0]
        for kind in ["ACF", "PACF"]:
            values = acf_pacf.loc[acf_pacf["display_ticker"].eq(ticker) &
                                  acf_pacf["diagnostic_type"].eq(kind) & acf_pacf["lag"].gt(0)]
            fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
            ax.stem(values["lag"], values["value"], basefmt=" ")
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set(title=f"{ticker}: training return {kind}\n"
                         f"{dates['training_first_date']} to {dates['training_last_date']}",
                   xlabel="Lag (trading observations)", ylabel=kind, xticks=range(1, 21, 2))
            ax.set_xlim(0.5, 20.5)
            fig.savefig(directory / f"{ticker}_{kind.lower()}.png", dpi=150)
            plt.close(fig)


def publish_output(source: Path, destination: Path) -> None:
    """Publish validated bytes atomically using the destination folder's permissions."""
    # Moving directly out of a private TemporaryDirectory preserves its restrictive
    # Windows ACL. A new sibling file inherits the public output folder's ACL;
    # copying bytes (not metadata) then replacing within that folder preserves it.
    with NamedTemporaryFile(dir=destination.parent, prefix=f".{destination.name}.",
                            suffix=".tmp", delete=False) as handle:
        sibling = Path(handle.name)
    try:
        copyfile(source, sibling)
        sibling.replace(destination)
    finally:
        sibling.unlink(missing_ok=True)


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    stocks = load_development(root)
    diagnostics, correlations, criteria, forecasts = [], [], [], []
    for ticker, returns in stocks.items():
        print(f"{ticker}: training diagnostics and expanding validation for AR(0)..AR(10)", flush=True)
        train = returns.loc[:TRAIN_END]
        row, tidy = training_diagnostics(ticker, train)
        diagnostics.append(row)
        correlations.append(tidy)
        criteria.append(training_candidates(ticker, train))
        for p in range(MAX_P + 1):
            forecasts.append(expanding_forecasts(ticker, returns, p))
    tables = {
        "return_diagnostics": pd.DataFrame(diagnostics),
        "acf_pacf": pd.concat(correlations, ignore_index=True),
        "ar_information_criteria": pd.concat(criteria, ignore_index=True),
        "validation_forecasts": pd.concat(forecasts, ignore_index=True),
    }
    tables["validation_metrics"] = validation_metrics(tables["validation_forecasts"])
    tables["lag_summary"] = lag_summary(tables["ar_information_criteria"], tables["validation_metrics"])
    validate_outputs(tables, stocks)
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model_a-", dir=outputs) as temporary:
        stage = Path(temporary)
        for name in TABLE_NAMES:
            tables[name].to_csv(stage / f"model_a_{name}.csv", index=False)
        save_figures(tables["acf_pacf"], tables["return_diagnostics"], stage / "figures")
        (outputs / "tables").mkdir(exist_ok=True)
        figure_directory = outputs / "figures/model_a"
        figure_directory.mkdir(parents=True, exist_ok=True)
        for name in TABLE_NAMES:
            publish_output(stage / f"model_a_{name}.csv", outputs / f"tables/model_a_{name}.csv")
        for figure in (stage / "figures").glob("*.png"):
            publish_output(figure, figure_directory / figure.name)
    return tables


def main() -> None:
    tables = run()
    print("\nDescriptive lag minima for human review; no final specification selected:")
    print(tables["lag_summary"].to_string(index=False))
    print("\nTraining diagnostics (returns only, ending by 2022-12-31):")
    print(tables["return_diagnostics"].to_string(index=False))
    print(f"\nSaved six tables, {len(tables['validation_forecasts']):,} validation forecast rows, "
          "and 24 ACF/PACF figures. All development stops at 2024-12-31; 2025 returns untouched.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
