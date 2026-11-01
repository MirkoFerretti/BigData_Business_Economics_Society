"""D018 robustness replication using the unchanged Phase 3A fitting machinery.

An explicit field-name adapter supplies share-volume predictors to the same
common-case, target-aware OLS routines. No primary module state or artifact
is modified. Output labels are translated back to S0–S3 and share names.
"""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.analysis import model_b_volume_validation as primary
from src.analysis.model_a_return_diagnostics import publish_output, require
from src.features.build_model_a_features import IDENTIFIERS
from src.features import build_share_volume_features as features
from src.features.build_model_b_features import read_frozen

SPECIFICATIONS = ["S0", "S1", "S2", "S3"]
B_TO_S = dict(zip(primary.SPECIFICATIONS, SPECIFICATIONS))
S_TO_B = {value: key for key, value in B_TO_S.items()}
RENAME_COLUMNS = {
    "MSE_improvement_pct_vs_B0": "MSE_improvement_pct_vs_S0",
    "absolute_MSE_difference_vs_B0": "absolute_MSE_difference_vs_S0",
    "best_validation_MSE_improvement_pct_vs_B0": "best_validation_MSE_improvement_pct_vs_S0",
}


def to_primary(stock: pd.DataFrame) -> pd.DataFrame:
    """Rename only; values, dates, targets and frozen return predictors are unchanged."""
    return stock.rename(columns=features.SHARE_TO_DOLLAR)


def translate_table(table: pd.DataFrame) -> pd.DataFrame:
    result = table.rename(columns=RENAME_COLUMNS).copy()
    replacements = {**features.DOLLAR_TO_SHARE, **B_TO_S, "common_B3": "common_S3"}
    for column in result:
        if column in ["specification", "feature", "feature_1", "feature_2", "sample_alignment"] or (
                column.startswith("minimum_") and column.endswith("_specification")):
            result[column] = result[column].replace(replacements)
    return result


def load_development(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    frozen = read_frozen(root)
    source = pd.read_csv(root / "data/processed/model_b_share_volume_features.csv", dtype=str, keep_default_na=False)
    metadata = ["Date", "TargetDate"] + IDENTIFIERS + ["frozen_model_a"]
    require(set(metadata + features.NUMERIC_COLUMNS).issubset(source.columns), "Incomplete share-volume schema")
    require(set(source["display_ticker"]) == set(frozen), "Share-volume universe mismatch")
    development = source.loc[source["Date"].between(primary.TRAIN_START, primary.VALID_END) &
                              source["TargetDate"].between(primary.TRAIN_START, primary.VALID_END),
                              metadata + features.NUMERIC_COLUMNS].copy()
    del source  # Test information rows AND boundary test targets never get parsed.
    for column in features.NUMERIC_COLUMNS:
        development[column] = pd.to_numeric(development[column].replace("", np.nan), errors="raise")
    sources = features.read_sources(root, development_only=True)
    stocks = {}
    for ticker, source in sources.items():
        stock = development.loc[development["display_ticker"].eq(ticker)].reset_index(drop=True)
        require(primary.guard_development(to_primary(stock)) == frozen[ticker], "Frozen return mapping mismatch")
        expected = features.build_stock_features(source, frozen[ticker])
        expected = expected.loc[expected["TargetDate"].notna() & expected["TargetDate"].le(primary.VALID_END)].reset_index(drop=True)
        require(len(stock) == len(expected), "Missing share-volume development rows")
        pd.testing.assert_frame_equal(stock[metadata], expected[metadata], check_exact=True)
        np.testing.assert_allclose(stock[features.NUMERIC_COLUMNS], expected[features.NUMERIC_COLUMNS],
                                    rtol=1e-11, atol=1e-12, equal_nan=True,
                                    err_msg=f"{ticker}: share features/targets disagree with development source")
        stocks[ticker] = stock
    return stocks


def expanding_forecasts(ticker: str, stock: pd.DataFrame, specification: str) -> pd.DataFrame:
    require(specification in S_TO_B, "Unknown share-volume candidate")
    return translate_table(primary.expanding_forecasts(ticker, to_primary(stock), S_TO_B[specification]))


def representation_comparison(metrics: pd.DataFrame, root: Path = ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    dollar = pd.read_csv(root / "outputs/tables/model_b_validation_metrics.csv")
    require(set(dollar["display_ticker"]) == set(metrics["display_ticker"]), "Representation universes differ")
    rows = []
    for ticker in metrics["display_ticker"].unique():
        b = dollar.loc[dollar["display_ticker"].eq(ticker)].set_index("specification")
        s = metrics.loc[metrics["display_ticker"].eq(ticker)].set_index("specification")
        require(b.index.tolist() == primary.SPECIFICATIONS and s.index.tolist() == SPECIFICATIONS,
                "Incomplete representation comparison")
        np.testing.assert_allclose(s.loc["S0", ["MSE", "RMSE", "MAE"]].to_numpy(float),
                                    b.loc["B0", ["MSE", "RMSE", "MAE"]].to_numpy(float), rtol=1e-10, atol=1e-14,
                                    err_msg=f"{ticker}: frozen S0 does not reproduce common-sample B0")
        require(s["validation_observations"].eq(b.loc["B0", "validation_observations"]).all(), "Unequal validation counts")
        for i in range(1, 4):
            d, share = b.loc[f"B{i}"], s.loc[f"S{i}"]
            require(d["first_forecast_date"] == share["first_forecast_date"] and
                    d["last_forecast_date"] == share["last_forecast_date"], "Representation scoring dates differ")
            dv, sv = d["MSE_improvement_pct_vs_B0"], share["MSE_improvement_pct_vs_S0"]
            rows.append({"display_ticker": ticker, "frozen_model_a": share["frozen_model_a"],
                         "dollar_specification": f"B{i}", "share_specification": f"S{i}",
                         "validation_observations": int(share["validation_observations"]),
                         "dollar_MSE": d["MSE"], "share_MSE": share["MSE"],
                         "dollar_MSE_improvement_pct": dv, "share_MSE_improvement_pct": sv,
                         "difference_share_minus_dollar_pct_points": sv - dv,
                         "sign_agrees": bool(np.sign(dv) == np.sign(sv))})
    comparison = pd.DataFrame(rows)
    summaries = []
    for i in range(1, 4):
        part = comparison.loc[comparison["share_specification"].eq(f"S{i}")]
        d, s = part["dollar_MSE_improvement_pct"], part["share_MSE_improvement_pct"]
        summaries.append({"dollar_specification": f"B{i}", "share_specification": f"S{i}", "stocks": len(part),
                          "stocks_improved_dollar": int(d.gt(0).sum()), "stocks_improved_share": int(s.gt(0).sum()),
                          "stocks_improved_both": int((d.gt(0) & s.gt(0)).sum()), "sign_agreement_count": int(part["sign_agrees"].sum()),
                          "mean_dollar_improvement_pct": d.mean(), "median_dollar_improvement_pct": d.median(),
                          "mean_share_improvement_pct": s.mean(), "median_share_improvement_pct": s.median()})
    return comparison, pd.DataFrame(summaries)


def representation_correlations(stocks: dict[str, pd.DataFrame], root: Path = ROOT) -> pd.DataFrame:
    dollar_stocks = primary.load_development(root)
    rows = []
    for ticker, share in stocks.items():
        dollar, adapted = dollar_stocks[ticker], to_primary(share)
        require(share["Date"].tolist() == dollar["Date"].tolist() and
                share["TargetDate"].tolist() == dollar["TargetDate"].tolist(), "Misaligned correlation dates")
        frozen = primary.guard_development(adapted)
        common = primary.common_mask(dollar, frozen) & primary.common_mask(adapted, frozen) & share["TargetDate"].le(primary.TRAIN_END)
        rows.append({"display_ticker": ticker, "frozen_model_a": frozen, "training_observations": int(common.sum()),
                     "first_feature_date": share.loc[common, "Date"].iloc[0], "last_feature_date": share.loc[common, "Date"].iloc[-1],
                     "last_target_date": share.loc[common, "TargetDate"].iloc[-1],
                     "abnormal_dollar_share_correlation": dollar.loc[common, "AbnormalDollarVolume20"].corr(
                         share.loc[common, "AbnormalShareVolume20"])})
    return pd.DataFrame(rows)


def baseline_reproduction(forecasts: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    previous = pd.read_csv(root / "outputs/tables/model_b_validation_forecasts.csv")
    rows = []
    for ticker in forecasts["display_ticker"].unique():
        s0 = forecasts.loc[forecasts["display_ticker"].eq(ticker) & forecasts["specification"].eq("S0")].reset_index(drop=True)
        b0 = previous.loc[previous["display_ticker"].eq(ticker) & previous["specification"].eq("B0")].reset_index(drop=True)
        for column in ["forecast_date", "feature_date", "frozen_model_a", "training_observations",
                       "training_first_target_date", "training_last_target_date"]:
            pd.testing.assert_series_equal(s0[column], b0[column], check_exact=True)
        np.testing.assert_allclose(s0["predicted_return"], b0["predicted_return"], rtol=1e-10, atol=1e-14)
        np.testing.assert_array_equal(s0["realized_return"], b0["realized_return"])
        rows.append({"display_ticker": ticker, "frozen_model_a": s0["frozen_model_a"].iloc[0],
                     "validation_observations": len(s0), "initial_training_observations": int(s0["training_observations"].iloc[0]),
                     "maximum_absolute_prediction_difference_S0_vs_B0": (s0["predicted_return"] - b0["predicted_return"]).abs().max()})
    return pd.DataFrame(rows)


def save_figures(metrics: pd.DataFrame, comparison: pd.DataFrame, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    tickers = metrics["display_ticker"].unique()
    matrix = metrics.pivot(index="display_ticker", columns="specification", values="MSE_improvement_pct_vs_S0")
    matrix = matrix.loc[tickers, SPECIFICATIONS[1:]]
    limit = max(float(np.abs(matrix.to_numpy()).max()), .01)
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    heatmap = ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(matrix)):
        for j in range(3):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > limit * .6 else "black")
    ax.set(xticks=range(3), xticklabels=matrix.columns, yticks=range(len(matrix)), yticklabels=matrix.index,
           title="Share-volume validation MSE improvement versus S0 (%)\n2023–2024; positive values mean lower MSE")
    fig.colorbar(heatmap, ax=ax, label="MSE improvement (%)")
    fig.savefig(directory / "share_volume_mse_improvement.png", dpi=150)
    plt.close(fig)
    first = comparison.loc[comparison["share_specification"].eq("S1")].set_index("display_ticker").loc[tickers]
    x = np.arange(len(tickers))
    fig, ax = plt.subplots(figsize=(11, 4), layout="constrained")
    ax.bar(x - .18, first["dollar_MSE_improvement_pct"], width=.36, label="Dollar volume: B1")
    ax.bar(x + .18, first["share_MSE_improvement_pct"], width=.36, label="Share volume: S1")
    ax.axhline(0, color="black", linewidth=.8)
    ax.set(xticks=x, xticklabels=tickers, ylabel="MSE improvement vs baseline (%)",
           title="Activity representation robustness, 2023–2024")
    ax.legend()
    fig.savefig(directory / "dollar_vs_share_volume.png", dpi=150)
    plt.close(fig)


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    stocks = load_development(root)
    adapted_stocks = {ticker: to_primary(stock) for ticker, stock in stocks.items()}
    names = ["information_criteria", "coefficients", "volume_correlations", "vif", "volume_diagnostics"]
    collections = {name: [] for name in names + ["validation_forecasts"]}
    for ticker, stock in adapted_stocks.items():
        print(f"{ticker}: unchanged frozen return baseline, common-sample S0..S3", flush=True)
        for name, frame in zip(names, primary.initial_training(ticker, stock)):
            collections[name].append(frame)
        for specification in primary.SPECIFICATIONS:
            collections["validation_forecasts"].append(primary.expanding_forecasts(ticker, stock, specification))
    internal = {name: pd.concat(frames, ignore_index=True) for name, frames in collections.items()}
    internal["validation_metrics"] = primary.validation_metrics(internal["validation_forecasts"])
    internal["summary"] = primary.descriptive_summary(internal["information_criteria"], internal["validation_metrics"])
    primary.validate_outputs(internal, adapted_stocks)
    tables = {f"model_b_share_volume_{name}.csv": translate_table(internal[name]) for name in [
        "information_criteria", "coefficients", "validation_metrics", "validation_forecasts", "summary"]}
    metrics = tables["model_b_share_volume_validation_metrics.csv"]
    comparison, cross = representation_comparison(metrics, root)
    tables["model_b_volume_representation_comparison.csv"] = comparison
    tables["model_b_volume_representation_cross_stock_summary.csv"] = cross
    tables["model_b_abnormal_volume_correlations.csv"] = representation_correlations(stocks, root)
    tables["model_b_share_volume_baseline_reproduction.csv"] = baseline_reproduction(
        tables["model_b_share_volume_validation_forecasts.csv"], root)
    for filename, table in tables.items():
        require(np.isfinite(table.select_dtypes(include="number")).all().all(), f"Nonfinite robustness result: {filename}")
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    (outputs / "tables").mkdir(exist_ok=True)
    directory = outputs / "figures/model_b_robustness"
    directory.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".share-volume-validation-", dir=outputs) as temporary:
        stage = Path(temporary)
        for filename, table in tables.items():
            table.to_csv(stage / filename, index=False)
        save_figures(metrics, comparison, stage / "figures")
        for filename in tables:
            publish_output(stage / filename, outputs / "tables" / filename)
        for figure in (stage / "figures").glob("*.png"):
            publish_output(figure, directory / figure.name)
    print("\nDescriptive share-volume minima; no final model selected:")
    print(tables["model_b_share_volume_summary.csv"].to_string(index=False))
    print("\nShare-volume MSE improvements (%):")
    print(metrics.pivot(index="display_ticker", columns="specification", values="MSE_improvement_pct_vs_S0").to_string(
        float_format=lambda value: f"{value:.4f}"))
    print("\nCross-stock representation comparison:")
    print(cross.to_string(index=False))
    print("S0 reproduces all frozen B0 predictions; primary artifacts unchanged. No 2025 analysis.")
    return tables


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
