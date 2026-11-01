"""Selection isolation, selected predictor histories, and future perturbations."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from src.analysis import final_2025_stock_specific_volume_evaluation as selected
from src.analysis.model_a_feature_validation import design
from src.features import build_model_b_features as dollar
from src.features import build_share_volume_features as share
from tests.test_final_2025_evaluation import source


def features(raw, frozen):
    a, s = dollar.build_stock_features(raw, frozen), share.build_stock_features(raw, frozen)
    return a.assign(**{c: s[c] for c in share.VOLUME_FEATURES}).iloc[:-1].reset_index(drop=True)


def selection_fixture(root):
    """Only config and synthetic validation metrics; no historical/test outputs."""
    (root / "config").mkdir()
    for name in ("model_a_frozen.csv", "universe.csv"):
        (root / "config" / name).write_bytes((selected.ROOT / "config" / name).read_bytes())
    directory = root / "outputs/tables"
    directory.mkdir(parents=True)
    for family, name in [("B", "model_b_validation_metrics.csv"), ("S", "model_b_share_volume_validation_metrics.csv")]:
        rows = []
        for ticker, frozen in selected.original.FROZEN.items():
            for i, mse in enumerate([.1, .4, .3, .3]):
                rows.append(dict(display_ticker=ticker, frozen_model_a=frozen, specification=family + str(i),
                                 MSE=mse, first_forecast_date="2023-01-03", last_forecast_date="2024-12-31",
                                 validation_observations=502, MSE_improvement_pct_vs_B0=999 * i))
        pd.DataFrame(rows).to_csv(directory / name, index=False)
    return selected.select_models(root)


class SelectedVolumeTests(unittest.TestCase):
    def test_selection_uses_raw_mse_excludes_zero_and_breaks_exact_ties(self):
        # Only selection inputs exist here: test data/results cannot be read.
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = selection_fixture(root)
            self.assertTrue(result.selected_model_b.eq("B2").all())
            self.assertTrue(result.selected_model_s.eq("S2").all())
            self.assertTrue(result.selected_b_validation_mse_improvement_pct.lt(0).all())
            with self.assertRaisesRegex(ValueError, "raw MSE"):
                selected.check_expected(result)

    def test_selected_histories_and_independent_ols(self):
        for frozen in ("A0", "A3", "AR(9)"):
            for b, s in (("B1", "S1"), ("B2", "S2"), ("B3", "S3"), ("B2", "S3")):
                specs = {"A": frozen, "B": b, "S": s}
                stock = features(source(), frozen)
                stock.loc[40, "DollarVolumeChange1"] = np.nan
                stock.loc[50, "ShareVolumePersistence5"] = np.nan
                position = stock.index[stock.TargetDate.eq("2025-01-02")][0]
                columns = dollar.BASELINE_PREDICTORS[frozen] + dollar.VOLUME_FEATURES[:int(b[1])] + share.VOLUME_FEATURES[:int(s[1])]
                eligible = stock.loc[stock[columns].notna().all(axis=1) & stock.TargetDate.le("2024-12-31")]
                for model in ("A", "B", "S"):
                    prediction, fit = selected.predict_at(stock, position, specs, model)
                    np.testing.assert_array_equal(fit, eligible.index)
                    expected = OLS(eligible.TargetNextReturn, design(eligible, selected.predictors(frozen, specs[model]))).fit().predict(
                        design(stock.iloc[[position]], selected.predictors(frozen, specs[model]))).iloc[0]
                    self.assertAlmostEqual(prediction, expected, places=14)
                self.assertEqual(40 in eligible.index, b == "B1")
                self.assertEqual(50 in eligible.index, s != "S3")

    def test_future_raw_data_current_outcomes_and_prefixes(self):
        raw = source()
        altered = raw.copy()
        future = altered.Date.ge("2025-01-02")
        altered.loc[future, "LogReturn"] += .5
        altered.loc[future, "Volume"] *= 15
        altered.loc[future, "DollarVolume"] *= 7
        for frozen in ("A0", "A3", "AR(9)"):
            stock, changed = features(raw, frozen), features(altered, frozen)
            position = stock.index[stock.TargetDate.eq("2025-01-02")][0]
            for complexity in (1, 2, 3):
                specs = {"A": frozen, "B": f"B{complexity}", "S": f"S{complexity}"}
                for model in ("A", "B", "S"):
                    before, fit = selected.predict_at(stock, position, specs, model)
                    self.assertEqual(before, selected.predict_at(changed, position, specs, model)[0])
                    poisoned = stock.copy()
                    poisoned.loc[position:, "TargetNextReturn"] = np.inf
                    self.assertEqual(before, selected.predict_at(poisoned, position, specs, model)[0])
                    self.assertEqual(before, selected.predict_at(stock.iloc[:position + 1], position, specs, model)[0])
                    np.testing.assert_array_equal(selected.predict_at(stock, position + 1, specs, model)[1], np.r_[fit, position])
                forecasts = selected.expanding_forecasts("TEST", stock, specs)
                self.assertEqual(len(forecasts), 24)
                self.assertFalse(forecasts.duplicated(["model", "forecast_date"]).any())

    def test_combined_summary_has_complete_metrics_for_both_families(self):
        stock = features(source(), "A0")
        forecasts = selected.expanding_forecasts("AAPL", stock, {"A": "A0", "B": "B2", "S": "S2"})
        with TemporaryDirectory() as temporary:
            selection = selection_fixture(Path(temporary))
            # Summarization must work without reading any CSV, even when old
            # common-B1/S1 results and real project outputs are unavailable.
            with patch.object(selected.pd, "read_csv", side_effect=AssertionError("Unexpected summary IO")):
                tables = selected.summarize(forecasts, selection)
        self.assertEqual(set(tables), {"forecasts", "metrics", "comparison", "cross_stock_summary"})
        self.assertEqual(tables["comparison"].selected_model.tolist(), ["B2", "S2"])
        for row in tables["cross_stock_summary"].itertuples():
            part = tables["comparison"].loc[tables["comparison"].model.eq(row.model)]
            self.assertEqual(row.stocks, 1)
            self.assertEqual(row.stocks_improved, int(part.MSE_improvement_pct_vs_A.gt(0).sum()))
            self.assertEqual(row.stocks_worsened, int(part.MSE_improvement_pct_vs_A.lt(0).sum()))
            self.assertEqual(row.validation_test_sign_agreement_count, int(part.validation_test_sign_agrees.sum()))
            self.assertAlmostEqual(row.mean_MSE_improvement_pct, part.MSE_improvement_pct_vs_A.mean())
        for table in tables.values():
            self.assertTrue(np.isfinite(table.select_dtypes(include="number")).all().all())
        for row in tables["metrics"].itertuples():
            errors = forecasts.loc[forecasts.model.eq(row.model), "forecast_error"].to_numpy()
            self.assertAlmostEqual(row.MSE, np.mean(errors ** 2), places=16)
            self.assertAlmostEqual(row.RMSE, np.sqrt(np.mean(errors ** 2)), places=16)
            self.assertAlmostEqual(row.MAE, np.mean(np.abs(errors)), places=16)


if __name__ == "__main__":
    unittest.main()
