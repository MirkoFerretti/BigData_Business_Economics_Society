"""Primary stock-specific validation-selected volume evaluation (D021).

Selection reads only saved 2023-2024 metrics, never 2025 outcomes. D021
supersedes the historical common-B1/S1 specification without claiming an
unseen holdout: this comparison followed the earlier 2025 evaluation.
"""

import hashlib
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.insert(0, str(ROOT))

# Reuse only verified feature loading and frozen constants from the historical
# evaluator. No historical evaluation entry point or result CSV is required.
from src.analysis import final_2025_evaluation as original
from src.analysis import model_b_volume_validation as validation
from src.analysis.model_a_feature_validation import design
from src.analysis.model_a_return_diagnostics import publish_output, require

PREFIX = "final_2025_stock_specific_volume_"
SELECTION = "stock_specific_volume_model_selection.csv"
REPORT = "final_2025_stock_specific_volume_report.md"
TABLES = ("forecasts", "metrics", "comparison", "cross_stock_summary", "training_counts")


def select_models(root: Path = ROOT) -> pd.DataFrame:
    """Argmin raw validation MSE; candidate order breaks exact ties simply."""
    mapping = original.dollar.read_frozen(root)
    require(mapping == original.FROZEN, "Frozen Model A mapping changed")
    rows = {t: {"display_ticker": t, "frozen_model_a": f} for t, f in mapping.items()}
    for family, filename in [
        ("B", "model_b_validation_metrics.csv"),
        ("S", "model_b_share_volume_validation_metrics.csv"),
    ]:
        # Ignore percentage improvements, forecasts, coefficients and all test files.
        saved = pd.read_csv(root / "outputs/tables" / filename, float_precision="round_trip",
                            usecols=["display_ticker", "frozen_model_a", "specification", "MSE",
                                     "first_forecast_date", "last_forecast_date", "validation_observations"])
        require(set(saved.display_ticker) == set(mapping), f"{family}: validation universe mismatch")
        require(not saved.duplicated(["display_ticker", "specification"]).any(), "Duplicate validation metrics")
        require(saved.first_forecast_date.eq("2023-01-03").all() and
                saved.last_forecast_date.eq("2024-12-31").all() and
                saved.validation_observations.eq(502).all(), "Unexpected validation period/coverage")
        require(np.isfinite(saved.MSE).all() and saved.MSE.gt(0).all(), "Invalid validation MSE")
        for ticker, frozen in mapping.items():
            part = saved.loc[saved.display_ticker.eq(ticker)].set_index("specification")
            require(set(part.index) == {family + str(i) for i in range(4)} and
                    part.frozen_model_a.eq(frozen).all(), f"{ticker}: incomplete family or changed A")
            candidates = [family + str(i) for i in (1, 2, 3)]
            winner = min(candidates, key=lambda label: part.loc[label, "MSE"])
            row, lower = rows[ticker], family.lower()
            row[f"selected_model_{lower}"] = winner
            for label in [family + str(i) for i in range(4)]:
                row[f"{label.lower()}_validation_mse"] = float(part.loc[label, "MSE"])
            baseline, selected = float(part.loc[family + "0", "MSE"]), float(part.loc[winner, "MSE"])
            row[f"selected_{lower}_validation_mse"] = selected
            row[f"selected_{lower}_validation_mse_difference_vs_{lower}0"] = baseline - selected
            row[f"selected_{lower}_validation_mse_improvement_pct"] = 100 * (baseline - selected) / baseline
    return pd.DataFrame(rows.values())


def check_expected(selection: pd.DataFrame) -> None:
    # Sanity assertion only: this mapping never supplies the selected models.
    discrepancies = []
    for row in selection.to_dict("records"):
        for family in ("B", "S"):
            expected = family + {"AAPL": "2", "TSLA": "3"}.get(row["display_ticker"], "1")
            actual = row[f"selected_model_{family.lower()}"]
            if actual != expected:
                values = {family + str(i): row[f"{family.lower()}{i}_validation_mse"] for i in (1, 2, 3)}
                discrepancies.append(f"{row['display_ticker']}: expected {expected}, derived {actual}; raw MSE={values}")
    require(not discrepancies, "Validation argmin differs from expected mapping: " + "; ".join(discrepancies))


def predictors(frozen: str, specification: str) -> list[str]:
    if specification == frozen:
        return original.dollar.BASELINE_PREDICTORS[frozen]
    if specification in ("B1", "B2", "B3"):
        return validation.predictors(frozen, specification)
    require(specification in ("S1", "S2", "S3"), "Unknown selected model")
    return [original.share.DOLLAR_TO_SHARE.get(c, c)
            for c in validation.predictors(frozen, "B" + specification[1])]


def load_stocks(root: Path = ROOT) -> dict:
    # Read-only reuse: verifies every stored numeric feature against its builder,
    # source chronology and frozen mapping. Never calls the original evaluator.
    stocks = original.load_stocks(root)
    shares = pd.read_csv(root / "data/processed/model_b_share_volume_features.csv")
    for ticker, stock in stocks.items():
        extra = shares.loc[shares.display_ticker.eq(ticker), ["Date", "TargetDate"] + original.share.VOLUME_FEATURES]
        stocks[ticker] = stock.drop(columns=["AbnormalShareVolume20"]).merge(
            extra, on=["Date", "TargetDate"], how="left", validate="one_to_one", sort=False)
    return stocks


def common_mask(stock: pd.DataFrame, specs: dict) -> np.ndarray:
    frozen = stock.frozen_model_a.iloc[0]
    columns = list(dict.fromkeys(c for spec in specs.values() for c in predictors(frozen, spec)))
    require(not np.isinf(stock[columns].to_numpy(float)).any(), "Infinite selected predictor")
    # Only predictor availability determines eligibility, never outcome/loss value.
    return np.isfinite(stock[columns].to_numpy(float)).all(axis=1)


def predict_at(stock: pd.DataFrame, position: int, specs: dict, model: str) -> tuple:
    frozen = stock.frozen_model_a.iloc[0]
    usable = common_mask(stock, specs)
    origin, target = stock.loc[position, ["Date", "TargetDate"]]
    fit = np.flatnonzero(usable & stock.TargetDate.le(origin).to_numpy())
    columns = predictors(frozen, specs[model])
    require(usable[position] and len(fit) > len(columns) + 1 and (fit < position).all() and
            stock.TargetDate.iloc[fit].lt(target).all(), "Unavailable predictors or invalid fit chronology")
    y = stock.TargetNextReturn.iloc[fit].to_numpy(float)
    require(np.isfinite(y).all(), "Invalid historical outcome; no outcome-based row deletion permitted")
    if columns:
        x = design(stock.iloc[fit], columns).to_numpy(float)
        parameters, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        require(rank == x.shape[1], "Rank-deficient selected model")
        prediction = float(design(stock.iloc[[position]], columns).to_numpy(float)[0] @ parameters)
    else:
        prediction = float(y.mean())
    require(np.isfinite(prediction), "Invalid prediction")
    return prediction, fit


def expanding_forecasts(ticker: str, stock: pd.DataFrame, specs: dict) -> pd.DataFrame:
    rows = []
    for position in np.flatnonzero(stock.TargetDate.between(original.TEST_START, original.TEST_END)):
        predictions, fits = {}, []
        for model in original.MODELS:
            predictions[model], fit = predict_at(stock, position, specs, model)
            fits.append(fit)
        for fit in fits[1:]:
            np.testing.assert_array_equal(fit, fits[0])
        realized = float(stock.TargetNextReturn.iloc[position])  # reveal after all predictions
        require(np.isfinite(realized), "Invalid test outcome")
        errors = {m: realized - p for m, p in predictions.items()}
        for model in original.MODELS:
            fit = fits[0]
            rows.append({"display_ticker": ticker, "frozen_model_a": specs["A"], "model": model,
                         "specification": specs[model], "forecast_date": stock.TargetDate.iloc[position],
                         "feature_date": stock.Date.iloc[position], "predicted_return": predictions[model],
                         "realized_return": realized, "forecast_error": errors[model],
                         "squared_error": errors[model] ** 2, "absolute_error": abs(errors[model]),
                         "training_observations": len(fit),
                         "training_first_target_date": stock.TargetDate.iloc[fit[0]],
                         "training_last_target_date": stock.TargetDate.iloc[fit[-1]],
                         "sample_alignment": "common_selected_A_B_S",
                         "loss_difference_B_vs_A": errors["A"] ** 2 - errors["B"] ** 2,
                         "loss_difference_S_vs_A": errors["A"] ** 2 - errors["S"] ** 2})
    return pd.DataFrame(rows)


def summarize(forecasts: pd.DataFrame, selection: pd.DataFrame) -> dict:
    """Summarize current forecasts and validation selection without filesystem IO."""
    metrics = forecasts.groupby(["display_ticker", "frozen_model_a", "model", "specification"], sort=False).agg(
        test_observations=("squared_error", "size"), MSE=("squared_error", "mean"),
        MAE=("absolute_error", "mean")).reset_index()
    metrics["RMSE"] = np.sqrt(metrics.MSE)
    comparisons, cross = [], []
    a = metrics.loc[metrics.model.eq("A")].set_index("display_ticker")
    chosen = selection.set_index("display_ticker")
    for model in ("B", "S"):
        b = metrics.loc[metrics.model.eq(model)].set_index("display_ticker").loc[a.index]
        part = a[["frozen_model_a"]].copy()
        part["model"] = model
        part["selected_model"] = b.specification
        for metric in ("MSE", "RMSE", "MAE"):
            part[f"Model_A_{metric}"] = a[metric]
            part[f"Model_volume_{metric}"] = b[metric]
        part["MSE_difference_vs_A"] = a.MSE - b.MSE
        part["MSE_improvement_pct_vs_A"] = 100 * (a.MSE - b.MSE) / a.MSE
        part["validation_MSE_improvement_pct"] = chosen[f"selected_{model.lower()}_validation_mse_improvement_pct"]
        part["validation_test_sign_agrees"] = np.sign(part.validation_MSE_improvement_pct).eq(np.sign(part.MSE_improvement_pct_vs_A))
        comparisons.append(part.reset_index())
        gain = part.MSE_improvement_pct_vs_A
        cross.append({"model": model, "stocks": len(gain), "stocks_improved": int(gain.gt(0).sum()),
                      "stocks_worsened": int(gain.lt(0).sum()), "stocks_unchanged": int(gain.eq(0).sum()),
                      "mean_MSE_improvement_pct": gain.mean(), "median_MSE_improvement_pct": gain.median(),
                      "minimum_MSE_improvement_pct": gain.min(), "maximum_MSE_improvement_pct": gain.max(),
                      "validation_test_sign_agreement_count": int(part.validation_test_sign_agrees.sum())})
    return {"forecasts": forecasts, "metrics": metrics, "comparison": pd.concat(comparisons, ignore_index=True),
            "cross_stock_summary": pd.DataFrame(cross)}


def audit_forecasts(tables: dict, stocks: dict, specs: dict) -> pd.DataFrame:
    forecasts = tables["forecasts"]
    require(len(forecasts) == 9000 and not forecasts.duplicated(["display_ticker", "model", "forecast_date"]).any(),
            "Missing or duplicate forecasts")
    coverage = []
    for ticker, stock in stocks.items():
        expected = stock.loc[stock.TargetDate.between(original.TEST_START, original.TEST_END)]
        require(len(expected) == 250, "Expected exactly 250 targets per stock")
        signatures = []
        for model in original.MODELS:
            part = forecasts.loc[forecasts.display_ticker.eq(ticker) & forecasts.model.eq(model)]
            require(part.forecast_date.tolist() == expected.TargetDate.tolist() and
                    part.feature_date.tolist() == expected.Date.tolist(), "Misaligned target dates")
            require(part.specification.eq(specs[ticker][model]).all() and
                    part.frozen_model_a.eq(original.FROZEN[ticker]).all(), "Changed frozen model")
            np.testing.assert_array_equal(part.realized_return, expected.TargetNextReturn)
            require(part.training_last_target_date.tolist() == expected.Date.tolist() and
                    part.training_observations.diff().iloc[1:].eq(1).all(), "Stale or future fit")
            signatures.append(part[["training_observations", "training_first_target_date", "training_last_target_date"]].reset_index(drop=True))
        for signature in signatures[1:]:
            pd.testing.assert_frame_equal(signatures[0], signature)
        # Hostile current/future outcome poisoning, prefix truncation, and next-fit inclusion.
        for position in expected.index[[0, 125, 249]]:
            poisoned = stock.copy()
            poisoned.loc[position:, "TargetNextReturn"] = np.inf
            for model in original.MODELS:
                before, fit = predict_at(stock, position, specs[ticker], model)
                require(before == predict_at(poisoned, position, specs[ticker], model)[0] ==
                        predict_at(stock.iloc[:position + 1], position, specs[ticker], model)[0], "Future leakage")
                if position + 1 < len(stock):
                    np.testing.assert_array_equal(predict_at(stock, position + 1, specs[ticker], model)[1], np.r_[fit, position])
        coverage.append({"display_ticker": ticker, "frozen_model_a": specs[ticker]["A"],
                         "selected_model_b": specs[ticker]["B"], "selected_model_s": specs[ticker]["S"],
                         "targets_per_model": len(expected),
                         "initial_training_observations": int(signatures[0].training_observations.iloc[0]),
                         "final_training_observations": int(signatures[0].training_observations.iloc[-1]),
                         "first_training_target_date": signatures[0].training_first_target_date.iloc[0]})
    for name, frame in tables.items():
        require(np.isfinite(frame.select_dtypes(include="number")).all().all(), f"Nonfinite {name}")
    return pd.DataFrame(coverage)


def protected_hashes(root: Path) -> dict:
    allowed = {SELECTION, REPORT} | {PREFIX + n + ".csv" for n in TABLES} | {PREFIX + "integrity.json"}
    paths = []
    for folder in ("data", "config", "src", "tests", "outputs"):
        for path in (root / folder).rglob("*"):
            if (path.is_file() and "__pycache__" not in path.parts and ".yfinance_cache" not in path.parts
                    and path.name not in allowed and "final_2025_stock_specific_volume" not in path.parts):
                paths.append(path)
    paths += [root / n for n in ("PROJECT.md", "README.md", "requirements.txt")]
    # Local provenance documents are optional in the published checkout.
    paths += [root / n for n in ("AGENTS.md", "DECISIONS.md") if (root / n).is_file()]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def save_figures(tables: dict, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
    for model, ax in zip(("B", "S"), axes):
        part = tables["comparison"].loc[lambda x: x.model.eq(model)]
        gain = part.MSE_improvement_pct_vs_A
        barfig, barax = plt.subplots(figsize=(10, 4), layout="constrained")
        barax.bar(part.display_ticker, gain, color=["#28789a" if v > 0 else "#c46c36" for v in gain])
        barax.axhline(0, color="black", linewidth=.8)
        barax.set(title=f"2025 validation-selected {model} versus frozen A", ylabel="MSE improvement (%)")
        barfig.savefig(directory / f"model_{model.lower()}_mse_improvement.png", dpi=150)
        plt.close(barfig)
        ax.scatter(part.validation_MSE_improvement_pct, gain)
        for row in part.itertuples():
            ax.annotate(row.display_ticker, (row.validation_MSE_improvement_pct, row.MSE_improvement_pct_vs_A), fontsize=8)
        ax.axhline(0, color="black", linewidth=.8)
        ax.axvline(0, color="black", linewidth=.8)
        ax.set(title=f"Selected {model}", xlabel="2023-2024 MSE improvement (%)", ylabel="2025 MSE improvement (%)")
    fig.savefig(directory / "validation_vs_test.png", dpi=150)
    plt.close(fig)


def report(tables: dict, selection: pd.DataFrame) -> str:
    # Avoid a new tabulate dependency.
    def markdown(frame):
        def fmt(v):
            return f"{v:.9g}" if isinstance(v, float) else str(v)
        return "| " + " | ".join(frame.columns) + " |\n| " + " | ".join(["---"] * len(frame.columns)) + " |\n" + "\n".join(
            "| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None))
    sections = ["# Final 2025 evaluation: stock-specific volume selection",
                "D021 supersedes the historical common-B1/S1 specification in D019 for final reporting. "
                "Model B is selected per stock among B1-B3 and robustness Model S among S1-S3 using only "
                "saved 2023-2024 raw validation MSE; exact ties favor simplicity. B0/S0 contain no volume "
                "information and are benchmarks only. AAPL selects B2/S2, TSLA B3/S3, and the other ten "
                "stocks B1/S1. This comparison followed the earlier 2025 evaluation; it does not establish "
                "preregistration or a newly unseen holdout.",
                "Positive improvement means 100*(MSE_A-MSE_volume)/MSE_A. Cross-stock summaries weight stocks equally. "
                "Validation uses the unchanged original common B3/S3 histories. Test comparisons use each stock's "
                "selected A/B/S predictor intersection, including TSLA's additional persistence warm-up. "
                "2025 was excluded from predictive model fitting, feature selection, validation and model selection "
                "during development. Final expanding refits may use already-realized earlier 2025 returns. "
                "Earlier full-period descriptive diagnostics exposed 2025 outcomes; selection here still reads only validation MSE.",
                "## Derived selection", markdown(selection), "## Training history", markdown(tables["training_counts"]),
                "## Per-stock 2025 MSE, RMSE and MAE", markdown(tables["metrics"]),
                "## Validation and 2025 comparison", markdown(tables["comparison"]),
                "## Cross-stock summaries", markdown(tables["cross_stock_summary"]),
                "## Audit", "All 250 targets per stock are forecast once per model (9,000 forecasts). "
                "Actual fit indices are identical across A/B/S at every origin. Targets enter fits only once known. "
                "Current/future outcome poisoning and prefix truncation pass at first, middle and final test origins. "
                "Full deterministic forecast recomputation passes. Stored feature/source checks pass. "
                "SHA-256 checks preserve existing sources, data, validation artifacts, project documents and historical outputs. "
                "See the separate integrity JSON for hashes and selected predictors. No outcome/loss filtering, new models, "
                "tuning, significance tests or sector interpretation is performed."]
    return "\n\n".join(sections) + "\n"


def run(root: Path = ROOT) -> dict:
    before = protected_hashes(root)
    selection = select_models(root)  # must precede loading any test data/results
    check_expected(selection)
    specs = {r.display_ticker: {"A": r.frozen_model_a, "B": r.selected_model_b, "S": r.selected_model_s}
             for r in selection.itertuples()}
    stocks = load_stocks(root)
    paths = []
    for ticker, stock in stocks.items():
        print(f"{ticker}: {specs[ticker]}", flush=True)
        paths.append(expanding_forecasts(ticker, stock, specs[ticker]))
    forecasts = pd.concat(paths, ignore_index=True)
    tables = summarize(forecasts, selection)
    tables["training_counts"] = audit_forecasts(tables, stocks, specs)
    repeated = pd.concat([expanding_forecasts(t, s, specs[t]) for t, s in stocks.items()], ignore_index=True)
    pd.testing.assert_frame_equal(forecasts, repeated, check_exact=True)
    audit = {"status": "primary stock-specific final evaluation under D021; historical D019 specification superseded",
             "chronology_note": "Stock-specific comparison followed an earlier 2025 evaluation; no claim of preregistration or an unseen holdout",
             "selection_inputs": ["outputs/tables/model_b_validation_metrics.csv", "outputs/tables/model_b_share_volume_validation_metrics.csv"],
             "selection_rule": "minimum saved 2023-2024 MSE among 1,2,3; exact ties prefer lower complexity",
             "selection_before_test_loading": True, "expected_mapping_matched": True,
             "frozen_model_a": original.FROZEN, "selected_specifications": specs,
             "predictors": {t: {m: predictors(s["A"], label) for m, label in s.items()} for t, s in specs.items()},
             "identical_actual_fit_indices_checked_at_every_origin": True,
             "outcome_poisoning_prefix_and_subsequent_fit_checks_passed": True,
             "exact_forecast_recomputation_passed": True, "stored_features_verified_against_original_builders": True,
             "protected_files_unchanged": True, "protected_file_sha256": before}
    output = root / "outputs"
    with TemporaryDirectory(prefix=".stock-specific-volume-", dir=output) as temporary:
        stage = Path(temporary)
        for name in TABLES:
            tables[name].to_csv(stage / (PREFIX + name + ".csv"), index=False)
        selection.to_csv(stage / SELECTION, index=False)
        (stage / (PREFIX + "integrity.json")).write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        (stage / REPORT).write_text(report(tables, selection), encoding="utf-8")
        save_figures(tables, stage / "figures")
        # Temporary staging files are excluded by checking the original path set.
        require(all((root / p).is_file() and hashlib.sha256((root / p).read_bytes()).hexdigest() == h
                    for p, h in before.items()), "Protected artifact changed")
        for path in stage.iterdir():
            if path.is_file():
                publish_output(path, output / ("" if path.name == REPORT else "tables") / path.name)
        destination = output / "figures/final_2025_stock_specific_volume"
        destination.mkdir(parents=True, exist_ok=True)
        for path in (stage / "figures").glob("*.png"):
            publish_output(path, destination / path.name)
    require(before == protected_hashes(root), "Protected files changed during evaluation/publication")
    print(tables["cross_stock_summary"].to_string(index=False))
    return tables


if __name__ == "__main__":
    run()
