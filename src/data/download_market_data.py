"""Download source daily data and produce a descriptive audit; no transformations."""

from pathlib import Path
import re
import sys
import warnings

import pandas as pd
import yfinance as yf


ROOT = Path(__file__).resolve().parents[2]
START = "2015-01-01"
END = "2026-01-01"  # Yahoo's end date is exclusive.
IDENTIFIERS = ["display_ticker", "yahoo_ticker", "company", "sector"]
MARKET_COLUMNS = [
    "Open", "High", "Low", "Close", "Adj Close", "Volume",
    "Dividends", "Stock Splits",
]


def read_universe(path: Path) -> pd.DataFrame:
    """Validate configuration before making network requests or writing data."""
    universe = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(universe.columns) != IDENTIFIERS or universe.empty:
        raise ValueError(f"{path}: expected a nonempty CSV with columns {IDENTIFIERS}")
    if universe.apply(lambda column: column.str.strip().eq("")).any().any():
        raise ValueError(f"{path}: identifiers must not be blank")
    for column in ("display_ticker", "yahoo_ticker"):
        if universe[column].duplicated().any():
            raise ValueError(f"{path}: duplicate {column}")
    for ticker in universe["display_ticker"]:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", ticker):
            raise ValueError(f"Unsafe display ticker for CSV filename: {ticker!r}")
    return universe


def warn(ticker: str, message: str) -> None:
    warnings.warn(f"{ticker}: {message}", UserWarning, stacklevel=2)


def prepare_raw(history: pd.DataFrame, security: dict) -> pd.DataFrame:
    """Validate without changing market values or removing observations."""
    ticker = security["display_ticker"]
    if history.empty:
        raise ValueError(f"{ticker}: Yahoo returned no observations")
    for column in ("Close", "Volume"):
        if column not in history.columns:
            raise ValueError(f"{ticker}: essential column {column!r} is absent")
    if not isinstance(history.index, pd.DatetimeIndex):
        raise ValueError(f"{ticker}: expected a daily DatetimeIndex from Yahoo")

    raw = history.rename_axis("Date").reset_index()
    # Use local exchange dates, without converting timestamps through UTC.
    raw["Date"] = raw["Date"].dt.strftime("%Y-%m-%d")
    if raw["Date"].isna().any():
        raise ValueError(f"{ticker}: invalid source dates")
    if ((raw["Date"] < START) | (raw["Date"] >= END)).any():
        warn(ticker, "source includes dates outside the requested interval; retained")
    if raw["Date"].duplicated().any():
        warn(ticker, "duplicate source dates; retained")
    if not raw["Date"].is_monotonic_increasing:
        warn(ticker, "source dates are not chronological; retained in source order")

    for column in MARKET_COLUMNS:
        if column not in raw:
            warn(ticker, f"source column {column!r} is absent")
        elif raw[column].isna().any():
            warn(ticker, f"{column}: {raw[column].isna().sum()} missing values; retained")
    if raw["Volume"].eq(0).any():
        warn(ticker, f"{raw['Volume'].eq(0).sum()} zero-volume observations; retained")

    expected_months = set(pd.period_range(START, "2025-12", freq="M").astype(str))
    missing_months = sorted(expected_months - set(raw["Date"].str[:7]))
    if missing_months:
        warn(ticker, f"months without observations: {', '.join(missing_months)}")

    for column in IDENTIFIERS:
        raw[column] = security[column]
    ordered = ["Date"] + IDENTIFIERS + [c for c in MARKET_COLUMNS if c in raw]
    # Retain additional source fields as well, rather than silently discarding them.
    return raw[ordered + [c for c in raw.columns if c not in ordered]]


def audit_security(raw: pd.DataFrame) -> dict:
    def missing(column: str) -> int:
        return int(raw[column].isna().sum()) if column in raw else len(raw)

    def events(column: str):
        if column not in raw or raw[column].isna().all():
            return pd.NA  # Unknown is different from no events.
        return int((raw[column].notna() & raw[column].ne(0)).sum())

    return {
        "display_ticker": raw["display_ticker"].iloc[0],
        "first_available_date": raw["Date"].min(),
        "last_available_date": raw["Date"].max(),
        "number_of_rows": len(raw),
        "missing_adj_close": missing("Adj Close"),
        "missing_volume": missing("Volume"),
        "zero_volume_days": int(raw["Volume"].eq(0).sum()),
        "dividend_event_count": events("Dividends"),
        "stock_split_event_count": events("Stock Splits"),
    }


def main() -> None:
    universe = read_universe(ROOT / "config" / "universe.csv")
    raw_dir = ROOT / "data" / "raw"
    table_dir = ROOT / "outputs" / "tables"
    raw_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)
    # Keep yfinance's local timezone/cookie caches inside ignored generated data.
    yf.set_tz_cache_location(str(raw_dir / ".yfinance_cache"))
    downloaded = {}
    for security in universe.to_dict(orient="records"):
        ticker = security["display_ticker"]
        print(f"Downloading {ticker} ({security['yahoo_ticker']})...", flush=True)
        try:
            history = yf.Ticker(security["yahoo_ticker"]).history(
                start=START, end=END, interval="1d", auto_adjust=False,
                back_adjust=False, actions=True, repair=False, keepna=True,
                rounding=False, timeout=30,
            )
            downloaded[ticker] = prepare_raw(history, security)
        except Exception as exc:
            raise RuntimeError(f"Download/validation failed for {ticker}: {exc}") from exc

    # Compare observed dates across the frozen universe, without inventing an
    # exchange holiday calendar or treating ordinary weekends as missing data.
    all_dates = set().union(*(set(raw["Date"]) for raw in downloaded.values()))
    for ticker, raw in downloaded.items():
        missing_dates = sorted(all_dates - set(raw["Date"]))
        if missing_dates:
            warn(ticker, f"{len(missing_dates)} dates observed in other securities are absent: "
                 + ", ".join(missing_dates))

    # Validate all downloads before replacing any generated CSVs.
    for ticker, raw in downloaded.items():
        raw.to_csv(raw_dir / f"{ticker}.csv", index=False)
    audit = pd.DataFrame([audit_security(raw) for raw in downloaded.values()])
    audit.to_csv(table_dir / "raw_data_audit.csv", index=False)
    print("\nRaw-data audit (requested 2015-01-01 through 2025-12-31):")
    print(audit.to_string(index=False, na_rep="unknown"))
    print("\nCoverage checks compare observed dates across stocks and flag absent months; "
          "they do not certify completeness against an exchange calendar.")
    print(f"\nSaved {len(downloaded)} raw CSVs and {table_dir / 'raw_data_audit.csv'}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
