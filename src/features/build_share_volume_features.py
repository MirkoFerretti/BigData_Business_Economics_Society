"""D018 deterministic share-volume features, reusing the unchanged D015 arithmetic."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.analysis.model_a_return_diagnostics import publish_output, require
from src.features import build_model_b_features as dollar_features
from src.features.build_model_a_features import IDENTIFIERS, read_returns

DOLLAR_TO_SHARE = {column: column.replace("DollarVolume", "ShareVolume")
                   for column in dollar_features.COMPONENTS + dollar_features.VOLUME_FEATURES}
SHARE_TO_DOLLAR = {value: key for key, value in DOLLAR_TO_SHARE.items()}
VOLUME_FEATURES = [DOLLAR_TO_SHARE[column] for column in dollar_features.VOLUME_FEATURES]
NUMERIC_COLUMNS = [DOLLAR_TO_SHARE.get(column, column) for column in dollar_features.NUMERIC_COLUMNS]


def read_sources(root: Path = ROOT, development_only: bool = False) -> dict[str, pd.DataFrame]:
    sources = read_returns(root, development_only=development_only)
    activity = pd.read_csv(root / "data/processed/market_daily.csv",
                           usecols=["Date", "display_ticker", "Volume", "Stock Splits"], dtype=str, keep_default_na=False)
    if development_only:
        activity = activity.loc[activity["Date"].between("2015-01-02", "2024-12-31")].copy()
    activity["Volume"] = pd.to_numeric(activity["Volume"], errors="raise")
    # Stock Splits remains an uninterpreted source string here; the audit parses
    # that field only after the development cutoff, never for test observations.
    for ticker, source in sources.items():
        rows = activity.loc[activity["display_ticker"].eq(ticker)]
        require(rows["Date"].tolist() == source["Date"].tolist(), f"{ticker}: share-volume date mismatch")
        require(np.isfinite(rows["Volume"]).all() and rows["Volume"].gt(0).all(),
                f"{ticker}: Volume must be finite and positive before logarithms")
        sources[ticker] = source.assign(Volume=rows["Volume"].to_numpy(),
                                        **{"Stock Splits": rows["Stock Splits"].to_numpy()})
    return sources


def build_stock_features(source: pd.DataFrame, frozen: str) -> pd.DataFrame:
    # This in-memory adapter supplies share activity to the same exact 20/1/5
    # calculations. It does not read or change any primary DollarVolume value.
    supplied = source.assign(DollarVolume=source["Volume"])
    return dollar_features.build_stock_features(supplied, frozen).rename(columns=DOLLAR_TO_SHARE)


def validate_stock_features(frame: pd.DataFrame, source: pd.DataFrame, frozen: str) -> None:
    dollar_features.validate_stock_features(frame.rename(columns=SHARE_TO_DOLLAR),
                                            source.assign(DollarVolume=source["Volume"]), frozen)


def split_audit(root: Path = ROOT) -> pd.DataFrame:
    frozen = dollar_features.read_frozen(root)
    rows = []
    # All neighborhoods are restricted to the development prefix. A missing
    # next neighbor at the cutoff stays missing, rather than reading 2025.
    for ticker, source in read_sources(root, development_only=True).items():
        split_values = pd.to_numeric(source["Stock Splits"], errors="raise")
        frame = build_stock_features(source, frozen[ticker])
        for position in np.flatnonzero(split_values.ne(0)):
            rows.append({
                "display_ticker": ticker, "Date": source["Date"].iloc[position],
                "Stock Splits": split_values.iloc[position],
                "previous_date": source["Date"].iloc[position - 1] if position else "",
                "previous_Volume": source["Volume"].iloc[position - 1] if position else np.nan,
                "Volume": source["Volume"].iloc[position],
                "next_date": source["Date"].iloc[position + 1] if position + 1 < len(source) else "",
                "next_available_Volume": source["Volume"].iloc[position + 1] if position + 1 < len(source) else np.nan,
                "ShareVolumeChange1": frame["ShareVolumeChange1"].iloc[position],
                "AbnormalShareVolume20": frame["AbnormalShareVolume20"].iloc[position],
            })
    return pd.DataFrame(rows, columns=["display_ticker", "Date", "Stock Splits", "previous_date", "previous_Volume",
                                       "Volume", "next_date", "next_available_Volume", "ShareVolumeChange1", "AbnormalShareVolume20"])


def build(root: Path = ROOT) -> None:
    frozen = dollar_features.read_frozen(root)
    sources = read_sources(root)
    frames = []
    for ticker, source in sources.items():
        frame = build_stock_features(source, frozen[ticker])
        validate_stock_features(frame, source, frozen[ticker])
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    audit = split_audit(root)
    destination = root / "data/processed/model_b_share_volume_features.csv"
    audit_path = root / "outputs/tables/share_volume_split_audit.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".share-volume-features-", dir=destination.parent) as temporary:
        stage = Path(temporary)
        combined.to_csv(stage / destination.name, index=False)
        saved = pd.read_csv(stage / destination.name,
                             dtype={column: str for column in ["Date", "TargetDate"] + IDENTIFIERS + ["frozen_model_a"]})
        for ticker, source in sources.items():
            validate_stock_features(saved.loc[saved["display_ticker"].eq(ticker)], source, frozen[ticker])
        audit.to_csv(stage / audit_path.name, index=False)
        publish_output(stage / destination.name, destination)
        publish_output(stage / audit_path.name, audit_path)
    print(f"Built {destination}; positive-input, causal-window, ticker and target checks passed.")
    print(f"Saved {len(audit)} pre-2025 Yahoo corporate-action neighborhoods; no split interpretation or corrections applied.")
    print("No 2025 return, feature distribution or model performance analyzed.")


if __name__ == "__main__":
    try:
        build()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
