"""Historical D019–D020 common-B1/S1 evaluation, superseded by D021.

The current stock-specific evaluator reuses load_stocks() and frozen constants
from this module; running this historical entry point is not required.

Dates label target returns, not forecast origins. All fits use the same
complete A+B1+S1 historical rows with TargetDate <= the information date.
The existing feature builders are used for read-only source verification.
No development selection, new inference or data-file mutation occurs here.
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

from src.analysis.model_a_feature_validation import design
from src.analysis.model_a_return_diagnostics import publish_output, require
from src.features import build_model_b_features as dollar
from src.features import build_share_volume_features as share

TEST_START, TEST_END = "2025-01-01", "2025-12-31"
FROZEN = dict(zip(
    ["NVDA", "AAPL", "BRK.B", "JPM", "LLY", "JNJ", "AMZN", "TSLA", "CAT", "GE", "XOM", "CVX"],
    ["A0", "A0", "A0", "AR(9)", "A0", "A0", "A3", "A3", "AR(9)", "AR(9)", "A0", "A0"],
))
MODELS = ("A", "B", "S")
EXTRA = {"A": [], "B": ["AbnormalDollarVolume20"], "S": ["AbnormalShareVolume20"]}
TABLE_FILES = {
    "forecasts": "final_2025_forecasts.csv",
    "metrics": "final_2025_metrics.csv",
    "model_b_comparison": "final_2025_model_b_comparison.csv",
    "share_volume_robustness": "final_2025_share_volume_robustness.csv",
    "validation_vs_test": "final_validation_vs_test_comparison.csv",
    "cross_stock_summary": "final_2025_cross_stock_summary.csv",
}


def predictors(frozen: str, model: str) -> list[str]:
    require(frozen in dollar.BASELINE_PREDICTORS and model in MODELS, "Unknown final specification")
    return dollar.BASELINE_PREDICTORS[frozen] + EXTRA[model]


def protected_hashes(root: Path) -> dict[str, str]:
    """Protect data, configuration, code, decisions and earlier outputs byte for byte."""
    paths = []
    for folder in ["data", "config", "src", "outputs/tables", "outputs/figures"]:
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root).as_posix()
            final_output = relative.startswith("outputs/") and (
                path.name.startswith("final_") or "/final_2025/" in relative)
            if path.is_file() and "__pycache__" not in path.parts and not final_output:
                paths.append(path)
    paths += [root / name for name in ["PROJECT.md", "README.md"]]
    # Local provenance documents are optional in the published checkout.
    paths += [root / name for name in ["AGENTS.md", "DECISIONS.md"] if (root / name).is_file()]
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def load_stocks(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    mapping = dollar.read_frozen(root)
    require(mapping == FROZEN, "D014 frozen stock mapping changed")
    dollar_sources, share_sources = dollar.read_sources(root), share.read_sources(root)
    stored = {
        "dollar": pd.read_csv(root / "data/processed/model_b_features.csv", keep_default_na=False, dtype=str),
        "share": pd.read_csv(root / "data/processed/model_b_share_volume_features.csv", keep_default_na=False, dtype=str),
    }
    for frame in stored.values():
        require(set(frame.display_ticker) == set(FROZEN), "Stored feature universe changed")
    stocks = {}
    metadata = ["Date", "TargetDate"] + dollar.IDENTIFIERS + ["frozen_model_a"]
    for ticker, frozen in mapping.items():
        verified = {}
        for label, module, sources in [("dollar", dollar, dollar_sources), ("share", share, share_sources)]:
            source = sources[ticker]
            frame = stored[label].loc[stored[label].display_ticker.eq(ticker)].reset_index(drop=True).copy()
            expected = module.build_stock_features(source, frozen)
            require(len(frame) == len(expected), f"{ticker}: missing stored {label} rows")
            pd.testing.assert_frame_equal(frame[metadata], expected[metadata].fillna(""), check_exact=True)
            for column in module.NUMERIC_COLUMNS:
                frame[column] = pd.to_numeric(frame[column].replace("", np.nan), errors="raise")
            np.testing.assert_allclose(frame[module.NUMERIC_COLUMNS], expected[module.NUMERIC_COLUMNS],
                                       rtol=1e-11, atol=1e-12, equal_nan=True)
            # Rebuilding prefixes confirms that future raw outcomes/activity
            # cannot enter present features. Do not publish rebuilt features.
            test_positions = np.flatnonzero(source.Date.between(TEST_START, TEST_END))
            require(len(test_positions) > 0, f"{ticker}: no 2025 source dates")
            for position in [test_positions[0] - 1, test_positions[len(test_positions) // 2], test_positions[-1]]:
                prefix = module.build_stock_features(source.iloc[:position + 1], frozen)
                columns = dollar.BASELINE_PREDICTORS[frozen] + module.VOLUME_FEATURES[:1]
                np.testing.assert_allclose(prefix[columns], expected.loc[:position, columns],
                                           rtol=1e-11, atol=1e-12, equal_nan=True)
            verified[label] = frame
        a, s = verified["dollar"], verified["share"]
        shared = metadata + dollar.RETURN_FEATURES + dollar.AR_FEATURES + ["TargetNextReturn"]
        pd.testing.assert_frame_equal(a[shared], s[shared], check_exact=True)
        stock = a.assign(AbnormalShareVolume20=s.AbnormalShareVolume20)
        stock = stock.loc[stock.TargetDate.between("2015-01-02", TEST_END)].reset_index(drop=True)
        require(stock.Date.is_unique and stock.Date.is_monotonic_increasing and
                stock.TargetDate.is_unique and stock.TargetDate.is_monotonic_increasing and
                stock.Date.lt(stock.TargetDate).all(), f"{ticker}: invalid chronology")
        needed = predictors(frozen, "B") + EXTRA["S"]
        require(not np.isinf(stock[needed].to_numpy(float)).any() and
                np.isfinite(stock.TargetNextReturn).all(), f"{ticker}: invalid predictors/outcomes")
        source = dollar_sources[ticker]
        expected_dates = source.loc[source.Date.between(TEST_START, TEST_END), "Date"].tolist()
        require(stock.loc[stock.TargetDate.between(TEST_START, TEST_END), "TargetDate"].tolist() == expected_dates,
                f"{ticker}: test coverage differs from stored market dates")
        stocks[ticker] = stock
    return stocks


def common_mask(stock: pd.DataFrame, frozen: str) -> np.ndarray:
    return np.isfinite(stock[predictors(frozen, "B") + EXTRA["S"]].to_numpy(float)).all(axis=1)


def predict_at(stock: pd.DataFrame, position: int, frozen: str, model: str) -> tuple[float, np.ndarray]:
    """Same conditional OLS/mean arithmetic as Phase 3A, with final common rows."""
    origin, target = stock.loc[position, ["Date", "TargetDate"]]
    usable = common_mask(stock, frozen)
    require(usable[position], "Unavailable common test predictors")
    fit = np.flatnonzero(usable & stock.TargetDate.le(origin).to_numpy())
    columns = predictors(frozen, model)
    require(len(fit) > len(columns) + 1 and (fit < position).all() and
            stock.TargetDate.iloc[fit].le(origin).all() and
            stock.TargetDate.iloc[fit].lt(target).all(), "Future outcome in fit or insufficient sample")
    # Read only historical outcomes; the current/future outcomes are not accessed.
    y = stock.TargetNextReturn.iloc[fit].to_numpy(float)
    require(np.isfinite(y).all(), "Nonfinite estimation outcome")
    if not columns:
        prediction = float(y.mean())
    else:
        x = design(stock.iloc[fit], columns).to_numpy(float)
        parameters, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        require(rank == x.shape[1], f"Rank-deficient {frozen} {model} fit")
        prediction = float(design(stock.iloc[[position]], columns).to_numpy(float)[0] @ parameters)
    require(np.isfinite(prediction), "Nonfinite prediction")
    return prediction, fit


def expanding_forecasts(ticker: str, stock: pd.DataFrame) -> pd.DataFrame:
    frozen = stock.frozen_model_a.iloc[0]
    require(stock.frozen_model_a.eq(frozen).all(), "Baseline changed within stock")
    positions = np.flatnonzero(stock.TargetDate.between(TEST_START, TEST_END))
    rows = []
    for position in positions:
        predictions = {}
        reference_fit = None
        for model in MODELS:
            predicted, fit = predict_at(stock, position, frozen, model)
            if reference_fit is not None:
                np.testing.assert_array_equal(fit, reference_fit)
            reference_fit = fit
            predictions[model] = predicted
        # Reveal the outcome only after all three predictions are generated.
        realized = float(stock.TargetNextReturn.iloc[position])
        errors = {model: realized - predicted for model, predicted in predictions.items()}
        for model in MODELS:
            fit = reference_fit
            rows.append({
                "forecast_date": stock.TargetDate.iloc[position], "feature_date": stock.Date.iloc[position],
                "display_ticker": ticker, "model": model, "frozen_model_a": frozen,
                "predicted_return": predictions[model], "realized_return": realized,
                "forecast_error": errors[model], "squared_error": errors[model] ** 2,
                "absolute_error": abs(errors[model]), "training_observations": len(fit),
                "training_first_target_date": stock.TargetDate.iloc[fit[0]],
                "training_last_target_date": stock.TargetDate.iloc[fit[-1]],
                "training_last_feature_date": stock.Date.iloc[fit[-1]],
                "sample_alignment": "common_A_B1_S1",
                "loss_difference_B_vs_A": errors["A"] ** 2 - errors["B"] ** 2,
                "loss_difference_S_vs_A": errors["A"] ** 2 - errors["S"] ** 2,
            })
    return pd.DataFrame(rows)


def summarize(forecasts: pd.DataFrame, root: Path) -> dict[str, pd.DataFrame]:
    rows = []
    for (ticker, frozen, model), group in forecasts.groupby(["display_ticker", "frozen_model_a", "model"], sort=False):
        mse = group.squared_error.mean()
        rows.append({"display_ticker": ticker, "frozen_model_a": frozen, "model": model,
                     "test_observations": len(group), "MSE": mse, "RMSE": np.sqrt(mse),
                     "MAE": group.absolute_error.mean()})
    metrics = pd.DataFrame(rows)
    tables = {"forecasts": forecasts, "metrics": metrics}
    cross = []
    validation = metrics.loc[metrics.model.eq("A"), ["display_ticker", "frozen_model_a"]].copy()
    for model, name, filename, spec, valcol in [
        ("B", "model_b_comparison", "model_b_validation_metrics.csv", "B1", "MSE_improvement_pct_vs_B0"),
        ("S", "share_volume_robustness", "model_b_share_volume_validation_metrics.csv", "S1", "MSE_improvement_pct_vs_S0"),
    ]:
        a = metrics.loc[metrics.model.eq("A")].set_index("display_ticker")
        b = metrics.loc[metrics.model.eq(model)].set_index("display_ticker").loc[a.index]
        require(a.MSE.gt(0).all(), "Nonpositive baseline MSE")
        comparison = a[["frozen_model_a"]].copy()
        for metric in ["MSE", "MAE"]:
            comparison[f"Model_A_{metric}"] = a[metric]
            comparison[f"Model_{model}_{metric}"] = b[metric]
        difference = a.MSE - b.MSE
        improvement = 100 * difference / a.MSE
        comparison[f"MSE_difference_{model}_vs_A"] = difference
        comparison[f"absolute_MSE_difference_{model}_vs_A"] = difference
        comparison[f"MSE_improvement_pct_{model}_vs_A"] = improvement
        tables[name] = comparison.reset_index()
        saved = pd.read_csv(root / "outputs/tables" / filename)
        saved = saved.loc[saved.specification.eq(spec)].set_index("display_ticker")
        require(saved.index.is_unique and set(saved.index) == set(a.index), "Incomplete saved validation")
        require(saved.loc[a.index, "frozen_model_a"].equals(a.frozen_model_a), "Validation baseline mismatch")
        validation[f"{spec}_validation_MSE_improvement_pct"] = validation.display_ticker.map(saved[valcol])
        validation[f"Model_{model}_2025_MSE_improvement_pct"] = validation.display_ticker.map(improvement)
        validation[f"{model}_sign_agrees"] = (
            np.sign(validation[f"{spec}_validation_MSE_improvement_pct"]) ==
            np.sign(validation[f"Model_{model}_2025_MSE_improvement_pct"]))
        cross.append({"comparison": f"Model {model} vs Model A", "stocks": len(a),
                      "stocks_improved": int(improvement.gt(0).sum()),
                      "stocks_worsened": int(improvement.lt(0).sum()),
                      "stocks_unchanged": int(improvement.eq(0).sum()),
                      "mean_MSE_improvement_pct": improvement.mean(),
                      "median_MSE_improvement_pct": improvement.median(),
                      "minimum_MSE_improvement_pct": improvement.min(),
                      "maximum_MSE_improvement_pct": improvement.max(),
                      "validation_test_sign_agreement_count": int(validation[f"{model}_sign_agrees"].sum())})
    tables["validation_vs_test"] = validation
    tables["cross_stock_summary"] = pd.DataFrame(cross)
    return tables


def validate_outputs(tables: dict, stocks: dict) -> list[dict]:
    for name, table in tables.items():
        require(np.isfinite(table.select_dtypes(include="number")).all().all(), f"{name}: nonfinite output")
        if name != "cross_stock_summary":
            require(set(table.display_ticker) == set(FROZEN), f"{name}: incomplete universe")
    forecasts = tables["forecasts"]
    require(not forecasts.duplicated(["display_ticker", "model", "forecast_date"]).any(), "Duplicate forecasts")
    require(forecasts.forecast_date.between(TEST_START, TEST_END).all(), "Wrong test period")
    coverage = []
    for ticker, stock in stocks.items():
        expected = stock.loc[stock.TargetDate.between(TEST_START, TEST_END)]
        require(len(expected) > 0 and common_mask(stock, FROZEN[ticker])[expected.index].all(), "Missing test features")
        signatures = []
        for model in MODELS:
            part = forecasts.loc[forecasts.display_ticker.eq(ticker) & forecasts.model.eq(model)]
            require(part.forecast_date.tolist() == expected.TargetDate.tolist() and
                    part.feature_date.tolist() == expected.Date.tolist(), "Missing/misaligned test target")
            require(part.frozen_model_a.eq(FROZEN[ticker]).all(), "Changed frozen baseline")
            np.testing.assert_array_equal(part.realized_return, expected.TargetNextReturn)
            require(part.training_last_target_date.le(part.feature_date).all() and
                    part.feature_date.lt(part.forecast_date).all(), "Leakage in forecast record")
            require(part.training_observations.diff().iloc[1:].eq(1).all() and
                    part.training_last_target_date.tolist() == expected.Date.tolist(), "Stale expanding window")
            require(part.iloc[0].feature_date <= "2024-12-31" and
                    part.iloc[0].training_last_target_date <= "2024-12-31", "First forecast uses 2025 information")
            signatures.append(part[["training_observations", "training_first_target_date",
                                    "training_last_target_date"]].reset_index(drop=True))
        for signature in signatures[1:]:
            pd.testing.assert_frame_equal(signatures[0], signature, check_exact=True)
        first = signatures[0].iloc[0]
        coverage.append({"display_ticker": ticker, "targets_per_model": len(expected),
                         "total_forecast_rows": 3 * len(expected),
                         "first_forecast_date": expected.TargetDate.iloc[0],
                         "last_forecast_date": expected.TargetDate.iloc[-1],
                         "first_information_date": expected.Date.iloc[0],
                         "initial_common_training_observations": int(first.training_observations),
                         "first_training_target_date": first.training_first_target_date,
                         "eligible_model_A_rows_excluded_initially": int((
                             np.isfinite(stock[predictors(FROZEN[ticker], "A")].to_numpy(float)).all(axis=1) &
                             stock.TargetDate.le(expected.Date.iloc[0]).to_numpy()).sum() - first.training_observations)})
    return coverage


def save_figures(tables: dict, directory: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for model, name in [("B", "model_b_comparison"), ("S", "share_volume_robustness")]:
        part = tables[name]
        values = part[f"MSE_improvement_pct_{model}_vs_A"]
        fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
        bars = ax.bar(part.display_ticker, values, color=["#28789a" if v > 0 else "#c46c36" for v in values])
        ax.bar_label(bars, labels=[f"{v:+.4f}" if abs(v) < .01 else f"{v:+.2f}" for v in values], padding=3, fontsize=8)
        ax.axhline(0, color="black", linewidth=.8)
        ax.margins(y=.2)
        ax.set(title=f"2025: Model {model} vs frozen Model A" + (" (share-volume robustness)" if model == "S" else ""),
               ylabel="MSE improvement (%)")
        fig.savefig(directory / f"model_{model.lower()}_mse_improvement.png", dpi=150)
        plt.close(fig)
    part = tables["validation_vs_test"]
    fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
    x, y = part.B1_validation_MSE_improvement_pct, part.Model_B_2025_MSE_improvement_pct
    ax.scatter(x, y)
    for ticker, vx, vy in zip(part.display_ticker, x, y):
        ax.annotate(ticker, (vx, vy), xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axhline(0, color="black", linewidth=.8)
    ax.axvline(0, color="black", linewidth=.8)
    ax.margins(.15)
    ax.set(xlabel="2023–2024 B1 validation MSE improvement (%)",
           ylabel="2025 Model B MSE improvement (%)", title="Abnormal dollar volume: validation to final test")
    fig.savefig(directory / "model_b_validation_vs_test.png", dpi=150)
    plt.close(fig)
    part = tables["model_b_comparison"]
    x = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(10, 4), layout="constrained")
    ax.bar(x - .19, part.Model_A_MSE, width=.38, label="Model A")
    ax.bar(x + .19, part.Model_B_MSE, width=.38, label="Model B")
    ax.set(xticks=x, xticklabels=part.display_ticker, ylabel="MSE (squared log-return units)",
           title="2025 final test MSE by stock")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend()
    fig.savefig(directory / "model_a_vs_b_mse.png", dpi=150)
    plt.close(fig)


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    before = protected_hashes(root)
    stocks = load_stocks(root)
    paths = []
    for ticker, stock in stocks.items():
        print(f"{ticker}: frozen {FROZEN[ticker]}, common-sample A/B/S final expanding forecasts", flush=True)
        paths.append(expanding_forecasts(ticker, stock))
    forecasts = pd.concat(paths, ignore_index=True)
    tables = summarize(forecasts, root)
    coverage = validate_outputs(tables, stocks)
    # Repeat every fit, not merely metric calculations. Identical predictions
    # imply identical derived errors/metrics; source hashes are checked again.
    repeated = pd.concat([expanding_forecasts(t, s) for t, s in stocks.items()], ignore_index=True)
    pd.testing.assert_frame_equal(forecasts, repeated, check_exact=True)
    audit = {"protocol": "D014, D019, D020", "test_start": TEST_START, "test_end": TEST_END,
             "frozen_model_a": FROZEN,
             "predictors": {t: {m: predictors(f, m) for m in MODELS} for t, f in FROZEN.items()},
             "sample_alignment": "Identical complete A+B1+S1 rows; TargetDate <= information date",
             "coverage_reference": "Every 2025 trading observation in the stored market data",
             "validation_note": "Previously saved B1/S1 results retain original common B3/S3 samples",
             "no_post_test_reselection": True, "exact_forecast_recomputation_passed": True,
             "prefix_feature_causality_checks_passed": True, "coverage": coverage,
             "protected_file_sha256": before}
    output = root / "outputs"
    with TemporaryDirectory(prefix=".final-2025-", dir=output) as temporary:
        stage = Path(temporary)
        for name, filename in TABLE_FILES.items():
            tables[name].to_csv(stage / filename, index=False)
        save_figures(tables, stage / "figures")
        require(before == protected_hashes(root), "Source or development artifacts changed during evaluation")
        audit["protected_files_unchanged"] = True
        (stage / "final_2025_integrity.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
        figures = output / "figures/final_2025"
        figures.mkdir(parents=True, exist_ok=True)
        for filename in list(TABLE_FILES.values()) + ["final_2025_integrity.json"]:
            publish_output(stage / filename, output / "tables" / filename)
        for figure in (stage / "figures").glob("*.png"):
            publish_output(figure, figures / figure.name)
    require(before == protected_hashes(root), "Protected files changed during publication")
    return tables


if __name__ == "__main__":
    try:
        result = run()
        print(result["metrics"].to_string(index=False))
        print(result["model_b_comparison"].to_string(index=False))
        print(result["share_volume_robustness"].to_string(index=False))
        print(result["cross_stock_summary"].to_string(index=False))
        print("All integrity checks passed. No model selection or new formal significance tests performed.")
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
