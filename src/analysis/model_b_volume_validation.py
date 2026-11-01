"""Phase 3A: frozen stock-specific return baselines plus causal volume information.

B0–B3 share the complete B3 estimation rows at EVERY forecast origin, not
just in initial fits. Eligible target dates are identical. Targets must be
observed by the information date. A separate original-sample B0 audit matches
Phase 2 and isolates changes caused solely by common-sample alignment.
IC counts include innovation variance, consistently with earlier phases.
SE/p-values are conventional nonrobust OLS diagnostics; no model is selected.
"""

import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

from src.analysis.model_a_return_diagnostics import TRAIN_START, TRAIN_END, VALID_START, VALID_END
from src.analysis.model_a_return_diagnostics import publish_output, require
from src.analysis.model_a_feature_validation import design
from src.features.build_model_a_features import IDENTIFIERS
from src.features.build_model_b_features import BASELINE_PREDICTORS, NUMERIC_COLUMNS, VOLUME_FEATURES
from src.features.build_model_b_features import build_stock_features, read_frozen, read_sources

SPECIFICATIONS = ["B0", "B1", "B2", "B3"]
TABLE_NAMES = [
    "information_criteria", "coefficients", "validation_metrics", "validation_forecasts",
    "volume_correlations", "vif", "summary", "volume_diagnostics", "baseline_reproduction",
    "cross_stock_summary",
]


def predictors(frozen: str, specification: str) -> list[str]:
    require(frozen in BASELINE_PREDICTORS and specification in SPECIFICATIONS, "Unknown frozen baseline or B model")
    return BASELINE_PREDICTORS[frozen] + VOLUME_FEATURES[:SPECIFICATIONS.index(specification)]


def guard_development(stock: pd.DataFrame) -> str:
    require(not stock.empty, "Empty development stock")
    require(stock["Date"].between(TRAIN_START, VALID_END).all() and
            stock["TargetDate"].between(TRAIN_START, VALID_END).all(),
            "Temporal leakage: unauthorized feature or outcome date")
    for column in ["Date", "TargetDate"]:
        require(stock[column].is_monotonic_increasing and stock[column].is_unique,
                "Duplicate or nonchronological dates")
    require(stock["Date"].lt(stock["TargetDate"]).all(), "Target must follow the information date")
    require(stock["frozen_model_a"].nunique() == 1, "Frozen baseline changed within a stock")
    frozen = stock["frozen_model_a"].iloc[0]
    require(frozen in BASELINE_PREDICTORS, "Unknown frozen Model A")
    require(not np.isinf(stock[NUMERIC_COLUMNS].to_numpy(float)).any(), "Infinite development value")
    require(np.isfinite(stock["TargetNextReturn"]).all(), "Unexpected missing development outcome")
    return frozen


def load_development(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    frozen = read_frozen(root)
    source = pd.read_csv(root / "data/processed/model_b_features.csv", dtype=str, keep_default_na=False)
    metadata = ["Date", "TargetDate"] + IDENTIFIERS + ["frozen_model_a"]
    require(set(metadata + NUMERIC_COLUMNS).issubset(source.columns), "Incomplete Model B feature schema")
    require(set(source["display_ticker"]) == set(frozen), "Feature universe mismatch")
    # Remove both 2025 information dates and late-2024 rows targeting 2025
    # BEFORE any numeric conversion, including before parsing diagnostic fields.
    development = source.loc[source["Date"].between(TRAIN_START, VALID_END) &
                              source["TargetDate"].between(TRAIN_START, VALID_END), metadata + NUMERIC_COLUMNS].copy()
    del source
    for column in NUMERIC_COLUMNS:
        development[column] = pd.to_numeric(development[column].replace("", np.nan), errors="raise")
    sources = read_sources(root, development_only=True)
    stocks = {}
    for ticker, raw in sources.items():
        stock = development.loc[development["display_ticker"].eq(ticker)].reset_index(drop=True)
        require(guard_development(stock) == frozen[ticker], f"{ticker}: frozen mapping mismatch")
        expected = build_stock_features(raw, frozen[ticker])
        expected = expected.loc[expected["TargetDate"].notna() & expected["TargetDate"].le(VALID_END)].reset_index(drop=True)
        require(len(stock) == len(expected), f"{ticker}: missing development feature rows")
        pd.testing.assert_frame_equal(stock[metadata], expected[metadata], check_exact=True)
        np.testing.assert_allclose(stock[NUMERIC_COLUMNS], expected[NUMERIC_COLUMNS],
                                    rtol=1e-11, atol=1e-12, equal_nan=True,
                                    err_msg=f"{ticker}: feature/target disagreement with development source")
        require(stock["TargetDate"].between(VALID_START, VALID_END).any(), "No validation targets")
        stocks[ticker] = stock
    return stocks


def common_mask(stock: pd.DataFrame, frozen: str) -> np.ndarray:
    # Complete B3 cases are a single fixed causal warm-up rule, shared by all
    # B candidates. No validation performance is consulted to define this mask.
    return stock[predictors(frozen, "B3")].notna().all(axis=1).to_numpy()


def initial_training(ticker: str, stock: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    frozen = guard_development(stock)
    common = stock.loc[common_mask(stock, frozen) & stock["TargetDate"].le(TRAIN_END)]
    require(len(common) > len(predictors(frozen, "B3")) + 2, "Insufficient common training observations")
    metadata = {"display_ticker": ticker, "frozen_model_a": frozen, "training_observations": len(common),
                "first_feature_date": common["Date"].iloc[0], "last_feature_date": common["Date"].iloc[-1],
                "first_target_date": common["TargetDate"].iloc[0], "last_target_date": common["TargetDate"].iloc[-1]}
    criteria, coefficients = [], []
    for specification in SPECIFICATIONS:
        x = design(common, predictors(frozen, specification))
        require(np.linalg.matrix_rank(x) == len(x.columns), f"{ticker} {specification}: rank-deficient initial design")
        result = OLS(common["TargetNextReturn"], x, missing="raise").fit()
        k = len(result.params) + 1
        criteria.append({**metadata, "specification": specification, "parameter_count": k,
                         "coefficient_count": len(result.params), "variance_parameter_count": 1,
                         "log_likelihood": result.llf, "AIC": -2 * result.llf + 2 * k,
                         "BIC": -2 * result.llf + np.log(len(common)) * k})
        for feature in x.columns:
            coefficients.append({**metadata, "specification": specification, "feature": feature,
                                 "estimate": result.params[feature], "standard_error": result.bse[feature],
                                 "p_value": result.pvalues[feature], "covariance_type": "nonrobust"})
    b3 = predictors(frozen, "B3")
    matrix = common[b3].corr()
    correlations = [{**metadata, "feature_1": a, "feature_2": b, "correlation": matrix.loc[a, b]}
                    for a in b3 for b in b3]
    x = design(common, b3).to_numpy()
    vif = [{**metadata, "specification": "B3", "feature": feature,
            "VIF": variance_inflation_factor(x, position + 1)} for position, feature in enumerate(b3)]
    diagnostics = []
    for feature in VOLUME_FEATURES:
        values = common[feature]
        row = {**metadata, "feature": feature, "mean": values.mean(), "std": values.std(ddof=1),
               "minimum": values.min(), "maximum": values.max(),
               "minimum_reference_std20": common["PriorLogDollarVolumeStd20"].min()}
        for label, quantile in [("q01", .01), ("q05", .05), ("q25", .25), ("median", .5),
                                 ("q75", .75), ("q95", .95), ("q99", .99)]:
            row[label] = values.quantile(quantile)
        diagnostics.append(row)
    return tuple(pd.DataFrame(rows) for rows in [criteria, coefficients, correlations, vif, diagnostics])


def expanding_forecasts(ticker: str, stock: pd.DataFrame, specification: str,
                         common_sample: bool = True) -> pd.DataFrame:
    frozen = guard_development(stock)
    require(common_sample or specification == "B0", "Original-sample reproduction is restricted to the frozen B0")
    columns = predictors(frozen, specification)
    x, y = design(stock, columns).to_numpy(), stock["TargetNextReturn"].to_numpy()
    origins, targets = stock["Date"].to_numpy(), stock["TargetDate"].to_numpy()
    usable = common_mask(stock, frozen) if common_sample else np.isfinite(x).all(axis=1)
    rows = []
    for position in np.flatnonzero((targets >= VALID_START) & (targets <= VALID_END)):
        origin, forecast_date = origins[position], targets[position]
        require(usable[position] and np.isfinite(x[position]).all(), "Unavailable validation predictors")
        fit_positions = np.flatnonzero(usable & (targets <= origin))
        require(len(fit_positions) > x.shape[1] and (fit_positions < position).all() and
                (targets[fit_positions] <= origin).all() and (targets[fit_positions] < forecast_date).all(),
                "Temporal leakage or insufficient known-target training rows")
        if not columns:
            predicted = float(y[fit_positions].mean())
        else:
            parameters, _, rank, _ = np.linalg.lstsq(x[fit_positions], y[fit_positions], rcond=None)
            require(rank == x.shape[1], f"{ticker} {specification}: rank-deficient expanding fit")
            predicted = float(x[position] @ parameters)
        require(np.isfinite(predicted), "Nonfinite validation prediction")
        rows.append({"forecast_date": forecast_date, "feature_date": origin, "display_ticker": ticker,
                     "frozen_model_a": frozen, "specification": specification, "predicted_return": predicted,
                     "realized_return": float(y[position]), "training_observations": len(fit_positions),
                     "training_first_target_date": targets[fit_positions[0]],
                     "training_last_target_date": targets[fit_positions[-1]],
                     "training_last_feature_date": origins[fit_positions[-1]],
                     "sample_alignment": "common_B3" if common_sample else "original_frozen_baseline"})
    return pd.DataFrame(rows)


def validation_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    require(forecasts["forecast_date"].between(VALID_START, VALID_END).all(), "Unauthorized scoring period")
    rows = []
    for (ticker, frozen, specification), group in forecasts.groupby(
            ["display_ticker", "frozen_model_a", "specification"], sort=False):
        errors = group["realized_return"].to_numpy() - group["predicted_return"].to_numpy()
        mse = np.mean(errors ** 2)
        rows.append({"display_ticker": ticker, "frozen_model_a": frozen, "specification": specification,
                     "validation_observations": len(group), "MSE": mse, "RMSE": np.sqrt(mse),
                     "MAE": np.mean(np.abs(errors)), "first_forecast_date": group["forecast_date"].iloc[0],
                     "last_forecast_date": group["forecast_date"].iloc[-1]})
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["specification"].eq("B0")].set_index("display_ticker")["MSE"]
    require(baseline.gt(0).all(), "B0 MSE must be positive")
    b0 = metrics["display_ticker"].map(baseline)
    # Absolute means expressed in MSE units, not abs(...): positive is a gain,
    # consistently with percentage improvement. Negative means deterioration.
    metrics["absolute_MSE_difference_vs_B0"] = b0 - metrics["MSE"]
    metrics["MSE_improvement_pct_vs_B0"] = 100 * (b0 - metrics["MSE"]) / b0
    return metrics


def baseline_reproduction(ticker: str, stock: pd.DataFrame, common_b0: pd.DataFrame,
                           root: Path = ROOT) -> dict:
    frozen = guard_development(stock)
    original = expanding_forecasts(ticker, stock, "B0", common_sample=False)
    original_metrics = validation_metrics(original).iloc[0]
    common_metrics = validation_metrics(common_b0).iloc[0]
    if frozen == "AR(9)":
        phase, filename, column, label = "2A", "model_a_validation_metrics.csv", "p", 9
    else:
        phase, filename, column, label = "2B", "model_a_feature_validation_metrics.csv", "specification", frozen
    prior = pd.read_csv(root / "outputs/tables" / filename)
    prior = prior.loc[prior["display_ticker"].eq(ticker) & prior[column].eq(label)]
    require(len(prior) == 1, f"{ticker}: missing/duplicate frozen baseline reference")
    reference = prior.iloc[0]
    require(len(original) == reference["validation_observations"], "Frozen baseline date-count mismatch")
    fields = ["MSE", "RMSE", "MAE"]
    np.testing.assert_allclose(original_metrics[fields].to_numpy(float), reference[fields].to_numpy(float),
                                rtol=1e-10, atol=1e-14, err_msg=f"{ticker}: frozen baseline reproduction failed")
    require(original["forecast_date"].tolist() == common_b0["forecast_date"].tolist(), "Alignment changed validation dates")
    return {"display_ticker": ticker, "frozen_model_a": frozen, "reference_phase": phase,
            "validation_observations": len(original), "prior_validation_MSE": reference["MSE"],
            "reproduced_original_sample_MSE": original_metrics["MSE"], "common_sample_B0_MSE": common_metrics["MSE"],
            "original_initial_training_observations": int(original["training_observations"].iloc[0]),
            "common_initial_training_observations": int(common_b0["training_observations"].iloc[0]),
            "initial_rows_lost_to_alignment": int(original["training_observations"].iloc[0] - common_b0["training_observations"].iloc[0]),
            "MSE_common_minus_prior": common_metrics["MSE"] - reference["MSE"],
            "MSE_change_pct_due_to_alignment": 100 * (common_metrics["MSE"] - reference["MSE"]) / reference["MSE"],
            "max_abs_original_metric_reproduction_error": float(np.max(np.abs(
                original_metrics[fields].to_numpy(float) - reference[fields].to_numpy(float))))}


def descriptive_summary(criteria: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in criteria["display_ticker"].unique():
        ic = criteria.loc[criteria["display_ticker"].eq(ticker)].sort_values("specification")
        validation = metrics.loc[metrics["display_ticker"].eq(ticker)].sort_values("specification")
        row = {"display_ticker": ticker, "frozen_model_a": ic["frozen_model_a"].iloc[0]}
        for label, frame, value in [("BIC", ic, "BIC"), ("AIC", ic, "AIC"),
                                    ("validation_MSE", validation, "MSE"), ("validation_MAE", validation, "MAE")]:
            row[f"minimum_{label}_specification"] = frame.loc[frame[value].idxmin(), "specification"]
            row[f"number_of_minima_{label}"] = int(frame[value].eq(frame[value].min()).sum())
        row["best_validation_MSE_improvement_pct_vs_B0"] = validation["MSE_improvement_pct_vs_B0"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def cross_stock_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for specification in SPECIFICATIONS[1:]:
        values = metrics.loc[metrics["specification"].eq(specification), "MSE_improvement_pct_vs_B0"]
        rows.append({"specification": specification, "stocks": len(values), "stocks_improved": int(values.gt(0).sum()),
                     "mean_MSE_improvement_pct": values.mean(), "median_MSE_improvement_pct": values.median()})
    # Any_volume summarizes each stock's best of B1–B3, even if all three worsen
    # its MSE. It is descriptive and does not configure or select a final model.
    any_values = metrics.loc[metrics["specification"].ne("B0")].groupby("display_ticker")["MSE_improvement_pct_vs_B0"].max()
    rows.append({"specification": "Any_volume", "stocks": len(any_values), "stocks_improved": int(any_values.gt(0).sum()),
                 "mean_MSE_improvement_pct": any_values.mean(), "median_MSE_improvement_pct": any_values.median()})
    return pd.DataFrame(rows)


def validate_outputs(tables: dict[str, pd.DataFrame], stocks: dict[str, pd.DataFrame]) -> None:
    for name, table in tables.items():
        require(np.isfinite(table.select_dtypes(include="number")).all().all(), f"{name}: nonfinite output")
        if name != "cross_stock_summary":
            require(set(table["display_ticker"]) == set(stocks), f"{name}: universe mismatch")
    forecasts, criteria = tables["validation_forecasts"], tables["information_criteria"]
    require(not forecasts.duplicated(["display_ticker", "specification", "forecast_date"]).any(), "Duplicate predictions")
    require(forecasts["forecast_date"].between(VALID_START, VALID_END).all() and
            forecasts["training_last_target_date"].le(forecasts["feature_date"]).all() and
            forecasts["feature_date"].lt(forecasts["forecast_date"]).all(), "Temporal leakage in prediction records")
    for name in ["information_criteria", "coefficients", "volume_correlations", "vif", "volume_diagnostics"]:
        require(tables[name]["last_target_date"].le(TRAIN_END).all(), "Validation outcome in training diagnostic")
    for ticker, stock in stocks.items():
        frozen = guard_development(stock)
        expected = stock.loc[stock["TargetDate"].between(VALID_START, VALID_END)]
        require(common_mask(stock, frozen)[expected.index].all(), "Unavailable validation common cases")
        ic = criteria.loc[criteria["display_ticker"].eq(ticker)]
        require(ic["training_observations"].nunique() == 1, "Unequal initial-fit samples")
        reference = None
        for name in ["information_criteria", "validation_metrics"]:
            part = tables[name].loc[tables[name]["display_ticker"].eq(ticker)]
            require(part["specification"].tolist() == SPECIFICATIONS and part["frozen_model_a"].eq(frozen).all(),
                    "Missing candidates or changed frozen baseline")
        for specification in SPECIFICATIONS:
            part = forecasts.loc[forecasts["display_ticker"].eq(ticker) & forecasts["specification"].eq(specification)]
            require(part["forecast_date"].tolist() == expected["TargetDate"].tolist() and
                    part["feature_date"].tolist() == expected["Date"].tolist(), "Unequal validation target dates")
            np.testing.assert_array_equal(part["realized_return"], expected["TargetNextReturn"])
            require(part["training_last_target_date"].tolist() == expected["Date"].tolist(), "Stale fitting window")
            require(part["training_observations"].diff().iloc[1:].eq(1).all(), "Window did not expand by one observation")
            signature = part[["training_observations", "training_first_target_date", "training_last_target_date"]].reset_index(drop=True)
            if reference is None:
                reference = signature
            else:
                pd.testing.assert_frame_equal(signature, reference, check_exact=True, obj="Common estimation sample")


def save_figures(tables: dict[str, pd.DataFrame], directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    metrics = tables["validation_metrics"]
    tickers = metrics["display_ticker"].unique()
    for ticker in tickers:
        part = metrics.loc[metrics["display_ticker"].eq(ticker)]
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        bars = ax.bar(part["specification"], part["MSE"], color="steelblue")
        ax.bar_label(bars, labels=[f"{value:.6g}" for value in part["MSE"]], fontsize=8, padding=3)
        ax.margins(y=.15)
        ax.set(title=f"{ticker}: validation MSE, 2023–2024\nFrozen return formula: {part['frozen_model_a'].iloc[0]}",
               xlabel="Volume candidate (common estimation sample)", ylabel="MSE")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.savefig(directory / f"{ticker}_validation_mse.png", dpi=150)
        plt.close(fig)
    matrix = metrics.pivot(index="display_ticker", columns="specification", values="MSE_improvement_pct_vs_B0")
    matrix = matrix.loc[tickers, SPECIFICATIONS[1:]]
    limit = max(float(np.abs(matrix.to_numpy()).max()), .01)
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    image = ax.imshow(matrix.to_numpy(), vmin=-limit, vmax=limit, cmap="RdBu", aspect="auto")
    for i in range(len(matrix)):
        for j in range(3):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(value) > limit * .6 else "black")
    ax.set(xticks=range(3), xticklabels=matrix.columns, yticks=range(len(matrix)), yticklabels=matrix.index,
           title="Volume validation MSE improvement versus B0 (%)\n2023–2024; positive values indicate lower MSE")
    fig.colorbar(image, ax=ax, label="MSE improvement (%)")
    fig.savefig(directory / "cross_stock_mse_improvement.png", dpi=150)
    plt.close(fig)
    coefficients = tables["coefficients"].loc[tables["coefficients"]["specification"].eq("B3")]
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), sharey=True, layout="constrained")
    for ax, feature in zip(axes, VOLUME_FEATURES):
        part = coefficients.loc[coefficients["feature"].eq(feature)].set_index("display_ticker").loc[tickers]
        ax.errorbar(part["estimate"], np.arange(len(tickers)), xerr=part["standard_error"], fmt="o", capsize=3)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=feature.replace("DollarVolume", "DollarVolume\n"), xlabel="Log return per feature unit")
        ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    axes[0].set(yticks=range(len(tickers)), yticklabels=tickers)
    axes[0].invert_yaxis()
    fig.suptitle("B3 volume coefficients ± one conventional OLS standard error\nInitial training only; predictors retain their own units")
    fig.savefig(directory / "volume_coefficients.png", dpi=150)
    plt.close(fig)


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    stocks = load_development(root)
    initial_names = ["information_criteria", "coefficients", "volume_correlations", "vif", "volume_diagnostics"]
    collections = {name: [] for name in initial_names + ["validation_forecasts"]}
    reproductions = []
    for ticker, stock in stocks.items():
        print(f"{ticker}: frozen {guard_development(stock)}, common-sample B0..B3 and baseline reproduction", flush=True)
        for name, table in zip(initial_names, initial_training(ticker, stock)):
            collections[name].append(table)
        for specification in SPECIFICATIONS:
            path = expanding_forecasts(ticker, stock, specification)
            collections["validation_forecasts"].append(path)
            if specification == "B0":
                reproductions.append(baseline_reproduction(ticker, stock, path, root))
    tables = {name: pd.concat(parts, ignore_index=True) for name, parts in collections.items()}
    tables["validation_metrics"] = validation_metrics(tables["validation_forecasts"])
    tables["summary"] = descriptive_summary(tables["information_criteria"], tables["validation_metrics"])
    tables["baseline_reproduction"] = pd.DataFrame(reproductions)
    tables["cross_stock_summary"] = cross_stock_summary(tables["validation_metrics"])
    validate_outputs(tables, stocks)
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-b-validation-", dir=outputs) as temporary:
        stage = Path(temporary)
        for name in TABLE_NAMES:
            tables[name].to_csv(stage / f"model_b_{name}.csv", index=False)
        save_figures(tables, stage / "figures")
        (outputs / "tables").mkdir(exist_ok=True)
        figure_directory = outputs / "figures/model_b"
        figure_directory.mkdir(parents=True, exist_ok=True)
        for name in TABLE_NAMES:
            publish_output(stage / f"model_b_{name}.csv", outputs / f"tables/model_b_{name}.csv")
        for figure in (stage / "figures").glob("*.png"):
            publish_output(figure, figure_directory / figure.name)
    return tables


def main() -> None:
    tables = run()
    print("\nDescriptive B minima; no final Model B specification:")
    print(tables["summary"].to_string(index=False))
    print("\nValidation MSE improvement versus common-sample B0 (%):")
    print(tables["validation_metrics"].pivot(index="display_ticker", columns="specification",
                                            values="MSE_improvement_pct_vs_B0").to_string(float_format=lambda x: f"{x:.4f}"))
    print("\nCross-stock descriptive summary:")
    print(tables["cross_stock_summary"].to_string(index=False))
    print(f"\nSaved ten tables, {len(tables['validation_forecasts']):,} forecasts and 14 figures.")
    print("All original-sample frozen baselines reproduced. All reported outcomes stop by 2024-12-31; 2025 untouched.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
