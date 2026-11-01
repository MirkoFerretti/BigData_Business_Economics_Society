"""D014–D015 ticker-local deterministic features; no test-period analysis."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.analysis.model_a_return_diagnostics import publish_output, require
from src.features.build_model_a_features import IDENTIFIERS, build_stock_features as build_return_features
from src.features.build_model_a_features import read_returns, read_universe

RETURN_FEATURES = ["LastReturn", "Prior4DayReturn", "Prior15DayReturn", "AbsLastReturn"]
AR_FEATURES = [f"ReturnLag{lag}" for lag in range(1, 10)]
VOLUME_FEATURES = ["AbnormalDollarVolume20", "DollarVolumeChange1", "DollarVolumePersistence5"]
COMPONENTS = ["LogDollarVolume", "PriorLogDollarVolumeMean20", "PriorLogDollarVolumeStd20"]
NUMERIC_COLUMNS = RETURN_FEATURES + AR_FEATURES + COMPONENTS + VOLUME_FEATURES + ["TargetNextReturn"]
BASELINE_PREDICTORS = {"A0": [], "AR(9)": AR_FEATURES, "A3": RETURN_FEATURES}


def read_frozen(root: Path = ROOT) -> dict[str, str]:
    universe = read_universe(root)
    frame = pd.read_csv(root / "config/model_a_frozen.csv", dtype=str, keep_default_na=False)
    require(list(frame.columns) == ["display_ticker", "frozen_model_a"], "Invalid frozen baseline schema")
    require(frame["display_ticker"].tolist() == universe["display_ticker"].tolist(),
            "Frozen baseline mapping must match the ordered universe exactly")
    require(frame["frozen_model_a"].isin(BASELINE_PREDICTORS).all(), "Unknown frozen return specification")
    return dict(zip(frame["display_ticker"], frame["frozen_model_a"]))


def read_sources(root: Path = ROOT, development_only: bool = False) -> dict[str, pd.DataFrame]:
    stocks = read_returns(root, development_only=development_only)
    activity = pd.read_csv(root / "data/processed/market_daily.csv",
                           usecols=["Date", "display_ticker", "DollarVolume"], dtype=str, keep_default_na=False)
    if development_only:
        activity = activity.loc[activity["Date"].between("2015-01-02", "2024-12-31")].copy()
    activity["DollarVolume"] = pd.to_numeric(activity["DollarVolume"], errors="raise")
    for ticker, stock in stocks.items():
        dollar = activity.loc[activity["display_ticker"].eq(ticker)]
        require(dollar["Date"].tolist() == stock["Date"].tolist(), f"{ticker}: DollarVolume date mismatch")
        require(np.isfinite(dollar["DollarVolume"]).all() and dollar["DollarVolume"].gt(0).all(),
                f"{ticker}: DollarVolume must be finite and strictly positive before taking logarithms")
        stocks[ticker] = stock.assign(DollarVolume=dollar["DollarVolume"].to_numpy())
    return stocks


def build_stock_features(stock: pd.DataFrame, frozen_model_a: str) -> pd.DataFrame:
    stock = stock.reset_index(drop=True)
    require(frozen_model_a in BASELINE_PREDICTORS, "Unknown frozen Model A")
    dollar = stock["DollarVolume"]
    require(np.isfinite(dollar).all() and dollar.gt(0).all(), "DollarVolume must be finite and positive")
    existing = build_return_features(stock)  # Same D012 formulas and next-trading-row target.
    result = existing[["Date", "TargetDate"] + IDENTIFIERS + RETURN_FEATURES + ["TargetNextReturn"]].copy()
    result["frozen_model_a"] = frozen_model_a
    for lag, column in enumerate(AR_FEATURES):
        # Row t predicts r_(t+1): target lag 1 is r_t, lag 9 is r_(t-8).
        result[column] = stock["LogReturn"].shift(lag)
    log_dollar = np.log(dollar)
    prior = log_dollar.shift(1).rolling(20, min_periods=20)
    prior_mean, prior_std = prior.mean(), prior.std(ddof=1)
    require(not prior_std.eq(0).any(), "Zero prior-20 log-DollarVolume standard deviation; no correction applied")
    result["LogDollarVolume"] = log_dollar
    result["PriorLogDollarVolumeMean20"] = prior_mean
    result["PriorLogDollarVolumeStd20"] = prior_std
    result["AbnormalDollarVolume20"] = (log_dollar - prior_mean) / prior_std
    result["DollarVolumeChange1"] = log_dollar.diff()
    result["DollarVolumePersistence5"] = result["AbnormalDollarVolume20"].rolling(5, min_periods=5).mean()
    return result


def validate_stock_features(frame: pd.DataFrame, source: pd.DataFrame, frozen: str) -> None:
    frame, source = frame.reset_index(drop=True), source.reset_index(drop=True)
    require(len(frame) == len(source) and frame["frozen_model_a"].eq(frozen).all(), "Rows or frozen baseline changed")
    expected_returns = build_return_features(source)
    pd.testing.assert_frame_equal(frame[["Date"] + IDENTIFIERS], source[["Date"] + IDENTIFIERS], check_exact=True)
    require(frame["TargetDate"].fillna("").equals(expected_returns["TargetDate"].fillna("")), "Misaligned TargetDate")
    for column in RETURN_FEATURES + ["TargetNextReturn"]:
        np.testing.assert_allclose(frame[column], expected_returns[column], rtol=1e-12, atol=1e-14, equal_nan=True)
    returns, logs = source["LogReturn"].to_numpy(), np.log(source["DollarVolume"].to_numpy())
    expected = {column: np.full(len(frame), np.nan) for column in AR_FEATURES + COMPONENTS + VOLUME_FEATURES}
    expected["LogDollarVolume"] = logs
    expected["DollarVolumeChange1"][1:] = np.diff(logs)
    for lag, column in enumerate(AR_FEATURES):
        expected[column][lag:] = returns[:len(returns) - lag] if lag else returns
    for i in range(20, len(frame)):
        # Explicit NumPy slices independently audit the pandas rolling windows.
        window = logs[i - 20:i]
        expected["PriorLogDollarVolumeMean20"][i] = window.mean()
        expected["PriorLogDollarVolumeStd20"][i] = window.std(ddof=1)
        expected["AbnormalDollarVolume20"][i] = (logs[i] - window.mean()) / window.std(ddof=1)
    for i in range(24, len(frame)):
        expected["DollarVolumePersistence5"][i] = expected["AbnormalDollarVolume20"][i - 4:i + 1].mean()
    for column, values in expected.items():
        np.testing.assert_array_equal(frame[column].isna(), np.isnan(values), err_msg=f"Missingness: {column}")
        np.testing.assert_allclose(frame[column], values, rtol=1e-11, atol=1e-12, equal_nan=True,
                                    err_msg=f"Causal alignment: {column}")
    require(not np.isinf(frame[NUMERIC_COLUMNS].to_numpy(float)).any(), "Infinite feature/target value")


def build(root: Path = ROOT) -> None:
    frozen, sources = read_frozen(root), read_sources(root)
    frames = []
    for ticker, source in sources.items():
        frame = build_stock_features(source, frozen[ticker])
        validate_stock_features(frame, source, frozen[ticker])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    destination = root / "data/processed/model_b_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-b-features-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        combined.to_csv(staged, index=False)
        saved = pd.read_csv(staged, dtype={c: str for c in ["Date", "TargetDate"] + IDENTIFIERS + ["frozen_model_a"]})
        for ticker, source in sources.items():
            validate_stock_features(saved.loc[saved["display_ticker"].eq(ticker)], source, frozen[ticker])
        publish_output(staged, destination)
    print(f"Built {destination}; ticker-local return/volume/target checks passed.")
    print("Deterministic feature construction only; no 2025 diagnostics or model evaluation.")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
