"""Phase 2B return-only candidate evidence; no automatic final specification.

Date is the end-of-day information/feature date; TargetDate (forecast_date
in outputs) is the next trading observation. A row enters fitting only once
its target is observed, i.e. TargetDate <= the current information date.
Initial fits/correlations/VIF share A4's complete training response sample
for comparable IC. Validation instead uses all usable past rows per model,
so A0 is the historical mean and A1 is equivalent to Phase 2A AR(1).
IC parameter counts include the innovation variance, as in Phase 2A;
coefficient SE/p-values are conventional, nonrobust OLS t-test diagnostics.
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

from src.analysis.model_a_return_diagnostics import TRAIN_END, TRAIN_START, VALID_END, VALID_START
from src.analysis.model_a_return_diagnostics import publish_output, require
from src.features.build_model_a_features import FEATURES, IDENTIFIERS, NUMERIC_COLUMNS
from src.features.build_model_a_features import build_stock_features, read_returns, read_universe

SPECIFICATIONS = {
    "A0": [], "A1": FEATURES[:1], "A2": FEATURES[:3],
    "A3": FEATURES[:4], "A4": FEATURES[:5],
}
TABLE_FILENAMES = {
    name: f"model_a_feature_{name}.csv" for name in [
        "information_criteria", "coefficients", "validation_metrics", "validation_forecasts",
        "correlations", "vif", "summary",
    ]
}
TABLE_FILENAMES["phase2_comparison"] = "model_a_phase2_comparison.csv"


def guard_development(stock: pd.DataFrame) -> None:
    require(not stock.empty, "Empty development stock")
    require(stock["Date"].between(TRAIN_START, VALID_END).all() and
            stock["TargetDate"].between(TRAIN_START, VALID_END).all(),
            "Temporal leakage: development contains an unauthorized information or target date")
    require(stock["Date"].is_monotonic_increasing and stock["Date"].is_unique and
            stock["TargetDate"].is_monotonic_increasing and stock["TargetDate"].is_unique,
            "Duplicate or nonchronological feature/target dates")
    require(stock["Date"].lt(stock["TargetDate"]).all(), "Target does not follow the information date")
    require(not np.isinf(stock[NUMERIC_COLUMNS].to_numpy(dtype=float)).any(), "Infinite development value")
    require(np.isfinite(stock["TargetNextReturn"]).all(), "Unexpected missing development target")


def load_development(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    universe = read_universe(root)
    source = pd.read_csv(root / "data/processed/model_a_features.csv", dtype=str, keep_default_na=False)
    required = ["Date", "TargetDate"] + IDENTIFIERS + NUMERIC_COLUMNS
    require(set(required).issubset(source.columns), "Incomplete Model A feature schema")
    require(set(source["display_ticker"]) == set(universe["display_ticker"]), "Feature universe mismatch")
    # Exclude BOTH test information rows and the 2024 row targeting 2025 before
    # numeric conversion. Thus even poisoned test/boundary targets are never parsed.
    development = source.loc[source["Date"].between(TRAIN_START, VALID_END) &
                              source["TargetDate"].between(TRAIN_START, VALID_END), required].copy()
    del source
    for column in NUMERIC_COLUMNS:
        development[column] = pd.to_numeric(development[column].replace("", np.nan), errors="raise")
    raw_development = read_returns(root, development_only=True)
    stocks = {}
    for ticker, raw in raw_development.items():
        stock = development.loc[development["display_ticker"].eq(ticker)].reset_index(drop=True)
        guard_development(stock)
        # Independent source verification uses only <=2024 returns. Rebuilding the
        # development prefix cannot access or require the first return of 2025.
        expected = build_stock_features(raw)
        expected = expected.loc[expected["TargetDate"].notna() &
                                expected["TargetDate"].le(VALID_END)].reset_index(drop=True)
        require(len(stock) == len(expected), f"{ticker}: feature dates/rows missing")
        pd.testing.assert_frame_equal(stock[["Date", "TargetDate"] + IDENTIFIERS],
                                      expected[["Date", "TargetDate"] + IDENTIFIERS], check_exact=True)
        np.testing.assert_allclose(stock[NUMERIC_COLUMNS], expected[NUMERIC_COLUMNS],
                                    rtol=1e-12, atol=1e-14, equal_nan=True,
                                    err_msg=f"{ticker}: features or target disagree with development returns")
        require(stock["TargetDate"].between(VALID_START, VALID_END).any(), f"{ticker}: no validation targets")
        stocks[ticker] = stock
    return stocks


def design(frame: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    result = pd.DataFrame({"Intercept": np.ones(len(frame))}, index=frame.index)
    for column in predictors:
        result[column] = frame[column]
    return result


def initial_training(ticker: str, stock: pd.DataFrame) -> tuple[pd.DataFrame, ...]:
    guard_development(stock)
    common = stock.loc[stock["TargetDate"].le(TRAIN_END) & stock[FEATURES].notna().all(axis=1)]
    require(len(common) > len(FEATURES) + 2, f"{ticker}: insufficient common training rows")
    require(common["Date"].lt(common["TargetDate"]).all() and common["TargetDate"].le(TRAIN_END).all(),
            "Temporal leakage in initial fitting")
    metadata = {"display_ticker": ticker, "training_observations": len(common),
                "first_feature_date": common["Date"].iloc[0], "last_feature_date": common["Date"].iloc[-1],
                "first_target_date": common["TargetDate"].iloc[0], "last_target_date": common["TargetDate"].iloc[-1]}
    criteria, coefficients = [], []
    for specification, predictors in SPECIFICATIONS.items():
        x = design(common, predictors)
        require(np.linalg.matrix_rank(x) == len(x.columns), f"{ticker} {specification}: rank-deficient design")
        result = OLS(common["TargetNextReturn"], x, missing="raise").fit()
        k = len(result.params) + 1  # Include innovation variance consistently with Phase 2A.
        criteria.append({**metadata, "specification": specification, "parameter_count": k,
                         "coefficient_count": len(result.params), "variance_parameter_count": 1,
                         "log_likelihood": result.llf, "AIC": -2 * result.llf + 2 * k,
                         "BIC": -2 * result.llf + np.log(len(common)) * k})
        for feature in x.columns:
            coefficients.append({"display_ticker": ticker, "specification": specification,
                                 "feature": feature, "estimate": result.params[feature],
                                 "standard_error": result.bse[feature], "p_value": result.pvalues[feature],
                                 "covariance_type": "nonrobust", "training_observations": len(common)})
    matrix = common[FEATURES].corr()
    correlations = [{**metadata, "feature_1": a, "feature_2": b, "correlation": matrix.loc[a, b]}
                    for a in FEATURES for b in FEATURES]
    x = design(common, FEATURES).to_numpy()
    vif = [{**metadata, "specification": "A4", "feature": feature,
            "VIF": variance_inflation_factor(x, index + 1)} for index, feature in enumerate(FEATURES)]
    return tuple(pd.DataFrame(rows) for rows in [criteria, coefficients, correlations, vif])


def expanding_forecasts(ticker: str, stock: pd.DataFrame, specification: str) -> pd.DataFrame:
    guard_development(stock)
    predictors = SPECIFICATIONS[specification]
    x, y = design(stock, predictors).to_numpy(), stock["TargetNextReturn"].to_numpy()
    origins, targets = stock["Date"].to_numpy(), stock["TargetDate"].to_numpy()
    usable = np.isfinite(x).all(axis=1) & np.isfinite(y)
    rows = []
    for position in np.flatnonzero((targets >= VALID_START) & (targets <= VALID_END)):
        origin, forecast_date = origins[position], targets[position]
        require(usable[position], f"{ticker} {specification}: missing validation predictor")
        # Current row's target is NOT yet observed. Earlier rows become eligible
        # by target date, not simply by whether their feature date is in the past.
        fit_positions = np.flatnonzero(usable & (targets <= origin))
        require(len(fit_positions) > x.shape[1], "Insufficient expanding estimation sample")
        require((fit_positions < position).all() and (targets[fit_positions] <= origin).all() and
                (targets[fit_positions] < forecast_date).all(), "Temporal leakage: unobserved training target")
        if specification == "A0":
            predicted = float(y[fit_positions].mean())
        else:
            parameters, _, rank, _ = np.linalg.lstsq(x[fit_positions], y[fit_positions], rcond=None)
            require(rank == x.shape[1], f"{ticker} {specification}: rank-deficient expanding fit")
            predicted = float(x[position] @ parameters)
        require(np.isfinite(predicted), f"{ticker} {specification}: infinite prediction")
        rows.append({"forecast_date": forecast_date, "feature_date": origin,
                     "display_ticker": ticker, "specification": specification,
                     "predicted_return": predicted, "realized_return": float(y[position]),
                     "training_observations": len(fit_positions),
                     "training_first_target_date": targets[fit_positions[0]],
                     "training_last_target_date": targets[fit_positions[-1]],
                     "training_last_feature_date": origins[fit_positions[-1]]})
    return pd.DataFrame(rows)


def validation_metrics(forecasts: pd.DataFrame) -> pd.DataFrame:
    require(forecasts["forecast_date"].between(VALID_START, VALID_END).all(), "Unauthorized metric period")
    rows = []
    for (ticker, specification), group in forecasts.groupby(["display_ticker", "specification"], sort=False):
        error = group["realized_return"].to_numpy() - group["predicted_return"].to_numpy()
        mse = np.mean(error ** 2)
        rows.append({"display_ticker": ticker, "specification": specification,
                     "validation_observations": len(group), "MSE": mse, "RMSE": np.sqrt(mse),
                     "MAE": np.mean(np.abs(error)), "first_forecast_date": group["forecast_date"].iloc[0],
                     "last_forecast_date": group["forecast_date"].iloc[-1]})
    metrics = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["specification"].eq("A0")].set_index("display_ticker")["MSE"]
    require(baseline.gt(0).all(), "A0 MSE must be positive for percentage comparisons")
    a0 = metrics["display_ticker"].map(baseline)
    metrics["MSE_improvement_pct_vs_A0"] = 100 * (a0 - metrics["MSE"]) / a0
    return metrics


def descriptive_summary(criteria: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ticker in criteria["display_ticker"].unique():
        ic = criteria.loc[criteria["display_ticker"].eq(ticker)].sort_values("specification")
        validation = metrics.loc[metrics["display_ticker"].eq(ticker)].sort_values("specification")
        row = {"display_ticker": ticker}
        for label, frame, value in [("BIC", ic, "BIC"), ("AIC", ic, "AIC"),
                                    ("validation_MSE", validation, "MSE"),
                                    ("validation_MAE", validation, "MAE")]:
            # Exact ties are reported using the first label; ties are counted.
            row[f"minimum_{label}_specification"] = frame.loc[frame[value].idxmin(), "specification"]
            row[f"number_of_minima_{label}"] = int(frame[value].eq(frame[value].min()).sum())
        row["best_validation_MSE_improvement_pct_vs_A0"] = validation["MSE_improvement_pct_vs_A0"].max()
        rows.append(row)
    return pd.DataFrame(rows)


def phase2_comparison(metrics: pd.DataFrame, root: Path = ROOT) -> pd.DataFrame:
    prior = pd.read_csv(root / "outputs/tables/model_a_validation_metrics.csv")
    require(set(prior["display_ticker"]) == set(metrics["display_ticker"]), "Phase 2A universe mismatch")
    require(not prior.duplicated(["display_ticker", "p"]).any(), "Duplicated Phase 2A metrics")
    require(np.isfinite(prior[["MSE", "RMSE", "MAE"]]).all().all(), "Invalid Phase 2A metrics")
    rows = []
    for ticker in metrics["display_ticker"].unique():
        current = metrics.loc[metrics["display_ticker"].eq(ticker)].set_index("specification")
        raw = prior.loc[prior["display_ticker"].eq(ticker)].set_index("p").sort_index()
        require(raw.index.tolist() == list(range(11)), f"{ticker}: incomplete Phase 2A candidates")
        require(raw["validation_observations"].eq(current.loc["A0", "validation_observations"]).all(),
                "Phase 2A and Phase 2B validation counts differ")
        # These numerical equivalences also audit target alignment without
        # recomputing any Phase 2A model, forecast or statistical result.
        for specification, p in [("A0", 0), ("A1", 1)]:
            np.testing.assert_allclose(current.loc[specification, ["MSE", "RMSE", "MAE"]].to_numpy(float),
                                        raw.loc[p, ["MSE", "RMSE", "MAE"]].to_numpy(float),
                                        rtol=1e-10, atol=1e-14,
                                        err_msg=f"{ticker}: {specification}/AR({p}) alignment mismatch")
        for specification, row in current.iterrows():
            rows.append({"display_ticker": ticker, "source_phase": "2B", "candidate": specification,
                         "validation_observations": row["validation_observations"],
                         "MSE": row["MSE"], "RMSE": row["RMSE"], "MAE": row["MAE"],
                         "MSE_improvement_pct_vs_A0": row["MSE_improvement_pct_vs_A0"]})
        p = int(raw["MSE"].idxmin())
        row = raw.loc[p]
        baseline = current.loc["A0", "MSE"]
        rows.append({"display_ticker": ticker, "source_phase": "2A", "candidate": f"AR({p})",
                     "validation_observations": int(row["validation_observations"]),
                     "MSE": row["MSE"], "RMSE": row["RMSE"], "MAE": row["MAE"],
                     "MSE_improvement_pct_vs_A0": 100 * (baseline - row["MSE"]) / baseline})
    return pd.DataFrame(rows)


def validate_outputs(tables: dict[str, pd.DataFrame], stocks: dict[str, pd.DataFrame]) -> None:
    for name, frame in tables.items():
        require(set(frame["display_ticker"]) == set(stocks), f"{name}: missing stock")
        require(np.isfinite(frame.select_dtypes(include="number")).all().all(), f"{name}: invalid numeric output")
    forecasts, metrics, criteria = (tables[name] for name in ["validation_forecasts", "validation_metrics", "information_criteria"])
    require(not forecasts.duplicated(["display_ticker", "specification", "forecast_date"]).any(), "Duplicate forecasts")
    require(forecasts["forecast_date"].between(VALID_START, VALID_END).all(), "Temporal leakage in forecasts")
    require(forecasts["training_last_target_date"].le(forecasts["feature_date"]).all() and
            forecasts["feature_date"].lt(forecasts["forecast_date"]).all(), "Unobserved target used in fit")
    require(criteria["last_target_date"].le(TRAIN_END).all(), "Validation outcome used in initial fit")
    for ticker, stock in stocks.items():
        expected = stock.loc[stock["TargetDate"].between(VALID_START, VALID_END)]
        for name in ["information_criteria", "validation_metrics"]:
            part = tables[name].loc[tables[name]["display_ticker"].eq(ticker)]
            require(part["specification"].tolist() == list(SPECIFICATIONS), f"{ticker}: incomplete A0–A4")
        ic = criteria.loc[criteria["display_ticker"].eq(ticker)]
        require(ic["training_observations"].nunique() == 1, "Incomparable initial-fit samples")
        for specification in SPECIFICATIONS:
            part = forecasts.loc[forecasts["display_ticker"].eq(ticker) &
                                 forecasts["specification"].eq(specification)]
            require(part["forecast_date"].tolist() == expected["TargetDate"].tolist() and
                    part["feature_date"].tolist() == expected["Date"].tolist(), "Missing or misaligned forecast dates")
            np.testing.assert_array_equal(part["realized_return"], expected["TargetNextReturn"])
            require(part["training_last_target_date"].tolist() == expected["Date"].tolist(),
                    "Fit did not expand through the last observed return")
            require(part["training_observations"].diff().iloc[1:].eq(1).all(), "Non-expanding estimation window")
        require(metrics.loc[metrics["display_ticker"].eq(ticker), "validation_observations"].eq(len(expected)).all(),
                "Validation sample mismatch")


def save_figures(metrics: pd.DataFrame, comparison: pd.DataFrame, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    tickers = metrics["display_ticker"].unique()
    for ticker in tickers:
        part = metrics.loc[metrics["display_ticker"].eq(ticker)]
        raw = comparison.loc[comparison["display_ticker"].eq(ticker) & comparison["source_phase"].eq("2A")].iloc[0]
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.bar(part["specification"], part["MSE"], color="steelblue")
        ax.axhline(raw["MSE"], color="black", linestyle="--", label=f"Phase 2A minimum: {raw['candidate']}")
        ax.set(title=f"{ticker}: validation MSE, 2023–2024", xlabel="Candidate specification", ylabel="MSE")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.legend(fontsize=8)
        fig.savefig(directory / f"{ticker}_validation_mse.png", dpi=150)
        plt.close(fig)
    matrix = metrics.pivot(index="display_ticker", columns="specification", values="MSE_improvement_pct_vs_A0")
    matrix = matrix.loc[tickers, ["A1", "A2", "A3", "A4"]]
    limit = max(float(np.abs(matrix.to_numpy()).max()), 0.01)
    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    heatmap = ax.imshow(matrix.to_numpy(), cmap="RdBu", vmin=-limit, vmax=limit, aspect="auto")
    for i in range(len(matrix)):
        for j in range(len(matrix.columns)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center",
                    color="white" if abs(value) > limit * 0.6 else "black", fontsize=9)
    ax.set(xticks=range(4), xticklabels=matrix.columns, yticks=range(len(matrix)), yticklabels=matrix.index,
           title="Validation MSE improvement versus A0 (%)\n2023–2024; positive values indicate lower MSE")
    fig.colorbar(heatmap, ax=ax, label="MSE improvement (%)")
    fig.savefig(directory / "cross_stock_mse_improvement.png", dpi=150)
    plt.close(fig)


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    stocks = load_development(root)
    collections = {name: [] for name in ["information_criteria", "coefficients", "correlations", "vif", "validation_forecasts"]}
    for ticker, stock in stocks.items():
        print(f"{ticker}: common-sample initial fits and target-aware expanding A0..A4 validation", flush=True)
        initial = initial_training(ticker, stock)
        for name, table in zip(["information_criteria", "coefficients", "correlations", "vif"], initial):
            collections[name].append(table)
        for specification in SPECIFICATIONS:
            collections["validation_forecasts"].append(expanding_forecasts(ticker, stock, specification))
    tables = {name: pd.concat(frames, ignore_index=True) for name, frames in collections.items()}
    tables["validation_metrics"] = validation_metrics(tables["validation_forecasts"])
    tables["summary"] = descriptive_summary(tables["information_criteria"], tables["validation_metrics"])
    tables["phase2_comparison"] = phase2_comparison(tables["validation_metrics"], root)
    validate_outputs(tables, stocks)
    outputs = root / "outputs"
    outputs.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".model-a-feature-validation-", dir=outputs) as temporary:
        stage = Path(temporary)
        for name, filename in TABLE_FILENAMES.items():
            tables[name].to_csv(stage / filename, index=False)
        save_figures(tables["validation_metrics"], tables["phase2_comparison"], stage / "figures")
        (outputs / "tables").mkdir(exist_ok=True)
        figure_directory = outputs / "figures/model_a_features"
        figure_directory.mkdir(parents=True, exist_ok=True)
        for filename in TABLE_FILENAMES.values():
            publish_output(stage / filename, outputs / "tables" / filename)
        for figure in (stage / "figures").glob("*.png"):
            publish_output(figure, figure_directory / figure.name)
    return tables


def main() -> None:
    tables = run()
    print("\nDescriptive minima for human review; no final Model A specification:")
    print(tables["summary"].to_string(index=False))
    print("\nValidation MSE improvements versus A0 (%):")
    print(tables["validation_metrics"].pivot(index="display_ticker", columns="specification",
                                            values="MSE_improvement_pct_vs_A0").to_string(float_format=lambda x: f"{x:.4f}"))
    print(f"\nSaved eight tables, {len(tables['validation_forecasts']):,} validation forecasts and 13 figures.")
    print("All diagnostics and outcomes stop at 2024-12-31. No volume information used; 2025 remains untouched.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
