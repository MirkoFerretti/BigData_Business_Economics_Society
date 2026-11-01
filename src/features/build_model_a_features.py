"""Deterministic, ticker-local return features from D012; no model or test analysis."""

from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.analysis.model_a_return_diagnostics import publish_output, require

IDENTIFIERS = ["display_ticker", "yahoo_ticker", "company", "sector"]
FEATURES = ["LastReturn", "Prior4DayReturn", "Prior15DayReturn", "AbsLastReturn", "RealizedVol20"]
NUMERIC_COLUMNS = FEATURES + ["TargetNextReturn"]


def read_universe(root: Path) -> pd.DataFrame:
    universe = pd.read_csv(root / "config/universe.csv", dtype=str, keep_default_na=False)
    require(list(universe.columns) == IDENTIFIERS and len(universe) == 12,
            "Expected the frozen 12-stock universe and its identifying columns")
    require(not universe["display_ticker"].duplicated().any(), "Duplicate configured ticker")
    require(not universe.apply(lambda column: column.str.strip().eq("")).any().any(),
            "Blank universe identifier")
    require(all(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ticker)
                for ticker in universe["display_ticker"]), "Unsafe ticker")
    return universe


def read_returns(root: Path, development_only: bool = False) -> dict[str, pd.DataFrame]:
    universe = read_universe(root)
    source = pd.read_csv(root / "data/processed/market_daily.csv",
                         usecols=["Date"] + IDENTIFIERS + ["LogReturn"],
                         dtype=str, keep_default_na=False)
    require(set(source["display_ticker"]) == set(universe["display_ticker"]), "Source universe mismatch")
    if development_only:
        # Filter strings before numeric conversion: no test value is inspected.
        source = source.loc[source["Date"].between("2015-01-02", "2024-12-31")].copy()
    source["LogReturn"] = pd.to_numeric(source["LogReturn"].replace("", np.nan), errors="raise")
    stocks = {}
    for security in universe.to_dict("records"):
        ticker = security["display_ticker"]
        stock = source.loc[source["display_ticker"].eq(ticker)].reset_index(drop=True)
        require(not stock.empty, f"{ticker}: no source observations")
        dates = pd.to_datetime(stock["Date"], format="%Y-%m-%d", errors="raise")
        require(dates.notna().all() and dates.is_monotonic_increasing and dates.is_unique,
                f"{ticker}: missing, duplicate or unordered dates")
        require(dates.dt.strftime("%Y-%m-%d").eq(stock["Date"]).all(), f"{ticker}: nonstandard dates")
        for column in IDENTIFIERS:
            require(stock[column].eq(security[column]).all(), f"{ticker}: {column} mismatch")
        require(stock["LogReturn"].isna().sum() == 1 and pd.isna(stock["LogReturn"].iloc[0]),
                f"{ticker}: unexpected source missingness")
        require(np.isfinite(stock["LogReturn"].iloc[1:]).all(), f"{ticker}: nonfinite source return")
        stocks[ticker] = stock
    return stocks


def build_stock_features(stock: pd.DataFrame) -> pd.DataFrame:
    stock = stock.reset_index(drop=True)
    returns = stock["LogReturn"]
    features = stock[["Date"] + IDENTIFIERS].copy()
    features["TargetDate"] = stock["Date"].shift(-1)
    features["LastReturn"] = returns
    features["Prior4DayReturn"] = returns.shift(1).rolling(4, min_periods=4).sum()
    features["Prior15DayReturn"] = returns.shift(5).rolling(15, min_periods=15).sum()
    features["AbsLastReturn"] = returns.abs()
    features["RealizedVol20"] = returns.rolling(20, min_periods=20).std(ddof=1)
    features["TargetNextReturn"] = returns.shift(-1)
    return features


def validate_stock_features(features: pd.DataFrame, source: pd.DataFrame) -> None:
    """Independent window calculations check alignment, not predictive test performance."""
    features, source = features.reset_index(drop=True), source.reset_index(drop=True)
    require(len(features) == len(source), "Feature construction lost source rows")
    pd.testing.assert_frame_equal(features[["Date"] + IDENTIFIERS], source[["Date"] + IDENTIFIERS],
                                  check_exact=True)
    require(features["TargetDate"].fillna("").equals(source["Date"].shift(-1).fillna("")),
            "TargetDate is not the following ticker-specific trading observation")
    values = source["LogReturn"].to_numpy()
    expected = {column: np.full(len(source), np.nan) for column in NUMERIC_COLUMNS}
    expected["LastReturn"] = values.copy()
    expected["AbsLastReturn"] = np.abs(values)
    expected["TargetNextReturn"][:-1] = values[1:]
    for position in range(len(source)):
        if position >= 4 and np.isfinite(values[position - 4:position]).all():
            expected["Prior4DayReturn"][position] = values[position - 4:position].sum()
        if position >= 19 and np.isfinite(values[position - 19:position - 4]).all():
            expected["Prior15DayReturn"][position] = values[position - 19:position - 4].sum()
        if position >= 19 and np.isfinite(values[position - 19:position + 1]).all():
            expected["RealizedVol20"][position] = values[position - 19:position + 1].std(ddof=1)
    for column in NUMERIC_COLUMNS:
        actual = features[column].to_numpy(dtype=float)
        require(not np.isinf(actual).any(), f"Infinite {column}")
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected[column]),
                                      err_msg=f"Incorrect structural missingness in {column}")
        np.testing.assert_allclose(actual, expected[column], rtol=1e-12, atol=1e-14,
                                    equal_nan=True, err_msg=f"Incorrect alignment in {column}")


def build(root: Path = ROOT) -> None:
    sources = read_returns(root)
    stocks = []
    for source in sources.values():
        features = build_stock_features(source)
        validate_stock_features(features, source)
        stocks.append(features)
    combined = pd.concat(stocks, ignore_index=True)
    destination = root / "data/processed/model_a_features.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-a-features-", dir=destination.parent) as temporary:
        staged = Path(temporary) / destination.name
        combined.to_csv(staged, index=False)
        saved = pd.read_csv(staged, dtype={c: str for c in ["Date", "TargetDate"] + IDENTIFIERS})
        for ticker, source in sources.items():
            validate_stock_features(saved.loc[saved["display_ticker"].eq(ticker)], source)
        publish_output(staged, destination)
    print(f"Built {destination}; independent ticker/window/target alignment checks passed.")
    print("Structural missing values retained. Deterministic construction only; no 2025 analysis.")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
