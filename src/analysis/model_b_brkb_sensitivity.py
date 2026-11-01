"""D017: isolated BRK.B dependency masking, without modifying any primary artifact."""

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
from src.features.build_model_b_features import AR_FEATURES, RETURN_FEATURES, VOLUME_FEATURES

SUSPICIOUS_DATES = ["2021-03-16", "2021-03-17"]


def dependency_masks(dates: pd.Series, suspicious_dates=SUSPICIOUS_DATES) -> pd.DataFrame:
    """Propagate source dependencies through the actual 20/1/5 feature graph.

    No fixed forward exclusion horizon is imposed: persistence inherits the
    union of five abnormality dependencies, each including its own prior 20.
    """
    dates = dates.reset_index(drop=True)
    require(set(suspicious_dates).issubset(set(dates)), "Suspicious source dates are missing")
    bad_log = dates.isin(suspicious_dates)
    bad_prior = bad_log.shift(1, fill_value=False).rolling(20, min_periods=1).max().astype(bool)
    bad_abnormal = bad_log | bad_prior
    return pd.DataFrame({
        "LogDollarVolume": bad_log,
        "PriorLogDollarVolumeMean20": bad_prior,
        "PriorLogDollarVolumeStd20": bad_prior,
        "AbnormalDollarVolume20": bad_abnormal,
        "DollarVolumeChange1": bad_log | bad_log.shift(1, fill_value=False),
        "DollarVolumePersistence5": bad_abnormal.rolling(5, min_periods=1).max().astype(bool),
    })


def mask_contaminated(stock: pd.DataFrame, suspicious_dates=SUSPICIOUS_DATES) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary.guard_development(stock)
    masks = dependency_masks(stock["Date"], suspicious_dates)
    sensitivity = stock.copy(deep=True).reset_index(drop=True)
    for feature in masks:
        sensitivity.loc[masks[feature], feature] = np.nan
    unchanged = ["Date", "TargetDate", "frozen_model_a"] + IDENTIFIERS + RETURN_FEATURES + AR_FEATURES + ["TargetNextReturn"]
    pd.testing.assert_frame_equal(sensitivity[unchanged], stock.reset_index(drop=True)[unchanged], check_exact=True)
    return sensitivity, masks


def run(root: Path = ROOT) -> dict[str, pd.DataFrame]:
    original = primary.load_development(root)["BRK.B"]
    sensitivity, masks = mask_contaminated(original)
    frozen = primary.guard_development(original)
    original_common = primary.common_mask(original, frozen)
    sensitivity_common = primary.common_mask(sensitivity, frozen)
    excluded = original_common & ~sensitivity_common
    training = original["TargetDate"].le(primary.TRAIN_END).to_numpy()
    validation = original["TargetDate"].between(primary.VALID_START, primary.VALID_END).to_numpy()
    require(not (excluded & validation).any(), "Unexpected contamination reached validation dates")
    require((excluded == (excluded & training)).all(), "Expected only initial-training exclusions")
    require(excluded.any(), "No contaminated usable training rows found")
    initial_names = ["information_criteria", "coefficients", "volume_correlations", "vif", "volume_diagnostics"]
    sensitivity_tables = dict(zip(initial_names, primary.initial_training("BRK.B", sensitivity)))
    paths = [primary.expanding_forecasts("BRK.B", sensitivity, specification) for specification in primary.SPECIFICATIONS]
    sensitivity_tables["validation_forecasts"] = pd.concat(paths, ignore_index=True)
    sensitivity_tables["validation_metrics"] = primary.validation_metrics(sensitivity_tables["validation_forecasts"])
    primary.validate_outputs(sensitivity_tables, {"BRK.B": sensitivity})

    def read_primary(name):
        frame = pd.read_csv(root / "outputs/tables" / f"model_b_{name}.csv")
        return frame.loc[frame["display_ticker"].eq("BRK.B")].copy()

    original_metrics = read_primary("validation_metrics")
    original_ic = read_primary("information_criteria")
    original_coefficients = read_primary("coefficients")
    original_forecasts = read_primary("validation_forecasts")
    # Validate that the original side is the unchanged Phase 3A result and
    # compares the same outcome dates, not a fresh alternative original fit.
    require(original_metrics["specification"].tolist() == primary.SPECIFICATIONS, "Incomplete original BRK.B results")
    for specification in primary.SPECIFICATIONS:
        before = original_forecasts.loc[original_forecasts["specification"].eq(specification)]
        after = sensitivity_tables["validation_forecasts"].loc[
            sensitivity_tables["validation_forecasts"]["specification"].eq(specification)]
        require(before["forecast_date"].tolist() == after["forecast_date"].tolist(), "Validation dates changed")
        np.testing.assert_array_equal(before["realized_return"], after["realized_return"])

    rows = []
    for version, metrics, criteria, coefficients in [
        ("original", original_metrics, original_ic, original_coefficients),
        ("sensitivity", sensitivity_tables["validation_metrics"], sensitivity_tables["information_criteria"],
         sensitivity_tables["coefficients"]),
    ]:
        for specification in primary.SPECIFICATIONS:
            metric = metrics.loc[metrics["specification"].eq(specification)].iloc[0]
            ic = criteria.loc[criteria["specification"].eq(specification)].iloc[0]
            row = {"version": version, **metric.to_dict(), "training_observations": int(ic["training_observations"]),
                   "AIC": ic["AIC"], "BIC": ic["BIC"]}
            coef = coefficients.loc[coefficients["specification"].eq(specification)].set_index("feature")
            for feature in VOLUME_FEATURES:
                for value in ["estimate", "standard_error", "p_value"]:
                    # Missing here means the feature is absent from this model,
                    # not a zero estimate or an unavailable fitted coefficient.
                    row[f"{feature}_{value}"] = coef.loc[feature, value] if feature in coef.index else np.nan
            rows.append(row)
    comparison = pd.DataFrame(rows)
    metadata = original[["Date", "TargetDate", "display_ticker"]].copy()
    mask_audit = pd.concat([metadata, masks.add_prefix("contaminated_")], axis=1)
    mask_audit["excluded_from_common_training"] = excluded & training
    mask_audit = mask_audit.loc[masks.any(axis=1)].reset_index(drop=True)
    affected = original.loc[excluded & training]
    before = original_metrics.set_index("specification")["MSE"]
    after = sensitivity_tables["validation_metrics"].set_index("specification")["MSE"]
    summary = pd.DataFrame([{
        "display_ticker": "BRK.B", "excluded_training_rows": int((excluded & training).sum()),
        "first_affected_feature_date": affected["Date"].iloc[0], "last_affected_feature_date": affected["Date"].iloc[-1],
        "first_affected_target_date": affected["TargetDate"].iloc[0], "last_affected_target_date": affected["TargetDate"].iloc[-1],
        "original_training_observations": int(original_ic["training_observations"].iloc[0]),
        "sensitivity_training_observations": int(sensitivity_tables["information_criteria"]["training_observations"].iloc[0]),
        "validation_observations": int(original_metrics["validation_observations"].iloc[0]), "validation_dates_removed": 0,
        "maximum_absolute_MSE_change": (after - before).abs().max(),
        "maximum_absolute_relative_MSE_change_pct": (100 * (after - before) / before).abs().max(),
        "original_any_volume_improves": bool(original_metrics.loc[original_metrics["specification"].ne("B0"), "MSE_improvement_pct_vs_B0"].gt(0).any()),
        "sensitivity_any_volume_improves": bool(sensitivity_tables["validation_metrics"].loc[
            sensitivity_tables["validation_metrics"]["specification"].ne("B0"), "MSE_improvement_pct_vs_B0"].gt(0).any()),
    }])
    forecasts = pd.concat([original_forecasts.assign(version="original"),
                           sensitivity_tables["validation_forecasts"].assign(version="sensitivity")], ignore_index=True)
    outputs = {
        "model_b_brkb_sensitivity.csv": comparison,
        "model_b_brkb_contamination_mask.csv": mask_audit,
        "model_b_brkb_sensitivity_summary.csv": summary,
        "model_b_brkb_sensitivity_forecasts.csv": forecasts,
    }
    directory = root / "outputs/figures/model_b_robustness"
    directory.mkdir(parents=True, exist_ok=True)
    (root / "outputs/tables").mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".brkb-sensitivity-", dir=root / "outputs") as temporary:
        stage = Path(temporary)
        for filename, frame in outputs.items():
            frame.to_csv(stage / filename, index=False)
        os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib"))
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
        x = np.arange(4)
        for offset, version in [(-.18, "original"), (.18, "sensitivity")]:
            values = comparison.loc[comparison["version"].eq(version)].set_index("specification").loc[primary.SPECIFICATIONS]
            ax.bar(x + offset, values["MSE_improvement_pct_vs_B0"], width=.36, label=version)
        ax.axhline(0, color="black", linewidth=.8)
        ax.set(xticks=x, xticklabels=primary.SPECIFICATIONS, ylabel="MSE improvement vs B0 (%)",
               title=f"BRK.B source-anomaly sensitivity, 2023–2024\n{len(affected)} contaminated training rows excluded")
        ax.legend()
        figure_name = "brkb_sensitivity.png"
        fig.savefig(stage / figure_name, dpi=150)
        plt.close(fig)
        for filename in outputs:
            publish_output(stage / filename, root / "outputs/tables" / filename)
        publish_output(stage / figure_name, directory / figure_name)
    print(summary.to_string(index=False))
    print(comparison[["version", "specification", "training_observations", "validation_observations", "MSE",
                      "MSE_improvement_pct_vs_B0"]].to_string(index=False))
    print("Primary observations/results unchanged; no replacements or 2025 analysis. Materiality remains for human review.")
    return outputs


if __name__ == "__main__":
    try:
        run()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
