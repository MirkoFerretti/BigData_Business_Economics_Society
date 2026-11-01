"""Build the Phase 1B daily base dataset from existing raw CSVs, without network I/O."""

from pathlib import Path
import re
from shutil import copyfile
import sys
from tempfile import NamedTemporaryFile, TemporaryDirectory

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
IDENTIFIERS = ["display_ticker", "yahoo_ticker", "company", "sector"]
MARKET_COLUMNS = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Dividends", "Stock Splits",
]
SOURCE_COLUMNS = ["Date"] + IDENTIFIERS + MARKET_COLUMNS
DERIVED_COLUMNS = ["DollarVolume", "LogReturn"]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_csv(path: Path) -> pd.DataFrame:
    # Use the pinned parser consistently; verify exact source values after writing.
    return pd.read_csv(path, dtype={column: str for column in ["Date"] + IDENTIFIERS})


def read_sources(root: Path) -> dict[str, pd.DataFrame]:
    universe_path = root / "config" / "universe.csv"
    universe = pd.read_csv(universe_path, dtype=str, keep_default_na=False)
    require(list(universe.columns) == IDENTIFIERS,
            f"{universe_path}: expected columns {IDENTIFIERS}")
    require(len(universe) == 12, f"{universe_path}: expected the frozen 12-stock universe")
    require(not universe.apply(lambda col: col.str.strip().eq("")).any().any(),
            f"{universe_path}: blank identifiers")
    for column in ["display_ticker", "yahoo_ticker"]:
        require(not universe[column].duplicated().any(),
                f"{universe_path}: duplicate {column}")

    sources = {}
    for security in universe.to_dict(orient="records"):
        ticker = security["display_ticker"]
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ticker) is not None,
                f"Unsafe display ticker: {ticker!r}")
        path = root / "data" / "raw" / f"{ticker}.csv"
        require(path.is_file(), f"{ticker}: missing raw file {path}; no download attempted")
        raw = read_csv(path)
        require(not raw.empty, f"{ticker}: raw file contains no observations")
        require(set(SOURCE_COLUMNS).issubset(raw.columns),
                f"{ticker}: missing source columns {sorted(set(SOURCE_COLUMNS) - set(raw.columns))}")
        require(not set(DERIVED_COLUMNS).intersection(raw.columns),
                f"{ticker}: raw file already contains Phase 1B derived columns")
        for column in IDENTIFIERS:
            require(raw[column].eq(security[column]).all(),
                    f"{ticker}: raw {column} does not match the universe configuration")
        validate_source(raw, ticker)
        sources[ticker] = raw
    return sources


def validate_source(raw: pd.DataFrame, ticker: str) -> None:
    dates = pd.to_datetime(raw["Date"], format="%Y-%m-%d", errors="raise")
    require(dates.notna().all() and dates.dt.strftime("%Y-%m-%d").eq(raw["Date"]).all(),
            f"{ticker}: missing or nonstandard Date values")
    require(dates.is_monotonic_increasing and not dates.duplicated().any(),
            f"{ticker}: source dates must be strictly increasing without duplicates")
    require(raw["Date"].between("2015-01-01", "2025-12-31").all(),
            f"{ticker}: source dates fall outside the accepted study period")
    for column in MARKET_COLUMNS:
        require(pd.api.types.is_numeric_dtype(raw[column]),
                f"{ticker}: nonnumeric source field {column}")
        require(np.isfinite(raw[column]).all(),
                f"{ticker}: missing or infinite {column}; investigate source data, no correction made")
    for column in ["Close", "Adj Close"]:
        require(raw[column].gt(0).all(),
                f"{ticker}: nonpositive {column}; logarithms require positive prices")
    require(raw["Volume"].ge(0).all(), f"{ticker}: negative source Volume")


def transform_stock(raw: pd.DataFrame) -> pd.DataFrame:
    """Add only the two accepted contemporaneous base transformations."""
    processed = raw.copy(deep=True)
    processed["DollarVolume"] = raw["Close"] * raw["Volume"]
    processed["LogReturn"] = np.log(raw["Adj Close"] / raw["Adj Close"].shift(1))
    return processed


def corporate_action_audit(stock: pd.DataFrame) -> pd.DataFrame:
    """Compute previous prices before selecting event rows, within one stock."""
    audit = stock[["Date", "display_ticker", "Dividends", "Stock Splits"]].copy()
    audit["PreviousClose"] = stock["Close"].shift(1)
    audit["Close"] = stock["Close"]
    audit["RawCloseLogReturn"] = np.log(stock["Close"] / audit["PreviousClose"])
    audit["PreviousAdjClose"] = stock["Adj Close"].shift(1)
    audit["Adj Close"] = stock["Adj Close"]
    audit["LogReturn"] = stock["LogReturn"]
    audit["Volume"] = stock["Volume"]
    audit["DollarVolume"] = stock["DollarVolume"]
    events = stock["Dividends"].ne(0) | stock["Stock Splits"].ne(0)
    return audit.loc[events].reset_index(drop=True)


def validate_processed(processed: pd.DataFrame, sources: dict[str, pd.DataFrame]) -> None:
    require(set(processed["display_ticker"]) == set(sources),
            "Processed dataset does not contain exactly the configured universe")
    require(len(processed) == sum(len(raw) for raw in sources.values()),
            "Processed dataset row count differs from raw inputs")
    require(not processed.duplicated(["display_ticker", "Date"]).any(),
            "Duplicate ticker/date rows in the processed dataset")
    for ticker, raw in sources.items():
        stock = processed.loc[processed["display_ticker"].eq(ticker)].reset_index(drop=True)
        require(len(stock) == len(raw), f"{ticker}: row count changed")
        # Includes identifiers, all market fields, and any extra source columns.
        pd.testing.assert_frame_equal(
            stock[raw.columns], raw.reset_index(drop=True), check_exact=True,
            obj=f"{ticker}: source-field preservation",
        )
        dates = stock["Date"]
        require(dates.is_monotonic_increasing and not dates.duplicated().any(),
                f"{ticker}: processed dates are not strictly increasing")
        require(stock["LogReturn"].isna().sum() == 1 and pd.isna(stock["LogReturn"].iloc[0]),
                f"{ticker}: expected exactly one initial missing LogReturn")
        require(np.isfinite(stock["LogReturn"].iloc[1:]).all(),
                f"{ticker}: infinite or unexpectedly missing LogReturn")
        positive_inputs = stock["Close"].gt(0) & stock["Volume"].gt(0)
        require(stock.loc[positive_inputs, "DollarVolume"].gt(0).all(),
                f"{ticker}: nonpositive DollarVolume despite positive inputs")
        require(np.isfinite(stock["DollarVolume"]).all(),
                f"{ticker}: nonfinite DollarVolume")


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


def build(root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    sources = read_sources(root)
    stocks = [transform_stock(raw) for raw in sources.values()]
    processed = pd.concat(stocks, ignore_index=True, sort=False)
    # Use a consistent core schema, retaining additional source columns if any.
    extra_columns = [c for c in processed if c not in SOURCE_COLUMNS + DERIVED_COLUMNS]
    processed = processed[SOURCE_COLUMNS + extra_columns + DERIVED_COLUMNS]
    audit = pd.concat([corporate_action_audit(stock) for stock in stocks], ignore_index=True)
    validate_processed(processed, sources)
    expected_events = sum(
        (raw["Dividends"].ne(0) | raw["Stock Splits"].ne(0)).sum()
        for raw in sources.values()
    )
    require(len(audit) == expected_events, "Corporate-action audit omitted or duplicated event rows")

    processed_path = root / "data" / "processed" / "market_daily.csv"
    audit_path = root / "outputs" / "tables" / "corporate_action_return_audit.csv"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    # Check serialized values before replacing either generated output.
    with TemporaryDirectory(prefix=".phase1b-", dir=processed_path.parent) as temporary:
        staged_processed = Path(temporary) / processed_path.name
        staged_audit = Path(temporary) / audit_path.name
        processed.to_csv(staged_processed, index=False)
        audit.to_csv(staged_audit, index=False)
        validate_processed(read_csv(staged_processed), sources)
        pd.testing.assert_frame_equal(read_csv(staged_audit), audit, check_exact=True,
                                      obj="Corporate-action audit CSV preservation")
        publish_output(staged_processed, processed_path)
        publish_output(staged_audit, audit_path)
    return processed, audit


def main() -> None:
    processed, audit = build()
    summary = processed.groupby("display_ticker", sort=False).agg(
        rows=("Date", "size"), first_date=("Date", "min"), last_date=("Date", "max"),
        missing_LogReturn=("LogReturn", lambda values: values.isna().sum()),
        min_LogReturn=("LogReturn", "min"), max_LogReturn=("LogReturn", "max"),
    )
    print(f"Processed {len(processed):,} rows across {len(summary)} stocks; "
          f"{processed['Date'].min()} through {processed['Date'].max()}")
    print(summary.to_string(float_format=lambda value: f"{value:.8f}"))
    print(f"\nCorporate-action audit rows: {len(audit)}")
    print("Validation passed, including exact source-field preservation after CSV serialization.")
    print(f"Saved {ROOT / 'data/processed/market_daily.csv'}")
    print(f"Saved {ROOT / 'outputs/tables/corporate_action_return_audit.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
