"""Phase 2B arithmetic, trading-row alignment and target-availability regressions."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from src.features import build_model_a_features as features
from src.analysis import model_a_feature_validation as validation


def synthetic_source(ticker="TEST", seed=81):
    dates = pd.bdate_range(end="2022-12-30", periods=85).append(
        pd.bdate_range(start="2023-01-03", periods=7)).append(
        pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]))
    returns = np.random.default_rng(seed).normal(0.001, 0.02, len(dates))
    returns[0] = np.nan
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "display_ticker": ticker,
                         "yahoo_ticker": ticker, "company": ticker + " Company", "sector": "Test sector",
                         "LogReturn": returns})


def development(source):
    result = features.build_stock_features(source)
    return result.loc[result["Date"].le(validation.VALID_END) &
                       result["TargetDate"].le(validation.VALID_END)].reset_index(drop=True)


def fixture_repository(root):
    (root / "config").mkdir()
    (root / "data/processed").mkdir(parents=True)
    sources = [synthetic_source(f"TEST{i}", i + 1) for i in range(12)]
    pd.concat([source[features.IDENTIFIERS].iloc[:1] for source in sources]).to_csv(
        root / "config/universe.csv", index=False)
    pd.concat(sources).to_csv(root / "data/processed/market_daily.csv", index=False)
    return sources


class FeatureConstructionTests(unittest.TestCase):
    def test_exact_nonoverlapping_windows_and_sample_standard_deviation(self):
        source = synthetic_source()
        source["LogReturn"] = np.arange(len(source), dtype=float) / 100
        source.loc[0, "LogReturn"] = np.nan
        result = features.build_stock_features(source)
        features.validate_stock_features(result, source)
        row = result.iloc[25]
        self.assertEqual(row.LastReturn, 0.25)
        self.assertAlmostEqual(row.Prior4DayReturn, sum(range(21, 25)) / 100)
        self.assertAlmostEqual(row.Prior15DayReturn, sum(range(6, 21)) / 100)
        self.assertAlmostEqual(row.RealizedVol20, np.std(np.arange(6, 26) / 100, ddof=1))
        self.assertEqual(row.TargetNextReturn, 0.26)
        self.assertEqual(result[features.NUMERIC_COLUMNS].isna().sum().tolist(), [1, 5, 20, 1, 20, 1])

    def test_target_uses_next_trading_row_across_weekend_and_year_boundary(self):
        source = synthetic_source()
        result = features.build_stock_features(source)
        row = result.loc[result.Date.eq("2022-12-30")].iloc[0]
        self.assertEqual(row.TargetDate, "2023-01-03")
        self.assertEqual(row.TargetNextReturn, source.loc[source.Date.eq("2023-01-03"), "LogReturn"].item())
        self.assertTrue(pd.isna(result.TargetNextReturn.iloc[-1]))
        self.assertTrue(pd.isna(result.TargetDate.iloc[-1]))

    def test_future_returns_do_not_change_earlier_features(self):
        source = synthetic_source()
        changed = source.copy()
        changed.loc[35:, "LogReturn"] = 100.0
        before, after = (features.build_stock_features(frame) for frame in [source, changed])
        pd.testing.assert_frame_equal(before[features.FEATURES].iloc[:35], after[features.FEATURES].iloc[:35])

    def test_combined_file_keeps_tickers_independent(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            features.build(root)
            path = root / "data/processed/model_a_features.csv"
            original = pd.read_csv(path)
            raw_path = root / "data/processed/market_daily.csv"
            raw = pd.read_csv(raw_path)
            raw.loc[raw.display_ticker.eq("TEST0") & raw.LogReturn.notna(), "LogReturn"] *= 100
            raw.to_csv(raw_path, index=False)
            features.build(root)
            changed = pd.read_csv(path)
            pd.testing.assert_frame_equal(original.loc[original.display_ticker.ne("TEST0")],
                                          changed.loc[changed.display_ticker.ne("TEST0")], check_exact=True)


class TargetAwareValidationTests(unittest.TestCase):
    def test_forecasts_match_independent_ols_and_never_fit_unobserved_target(self):
        stock = development(synthetic_source())
        for name, predictors in validation.SPECIFICATIONS.items():
            forecasts = validation.expanding_forecasts("TEST", stock, name)
            first = forecasts.iloc[0]
            self.assertEqual(first.feature_date, "2022-12-30")
            self.assertEqual(first.forecast_date, "2023-01-03")
            for row in forecasts.itertuples(index=False):
                fit = stock.loc[stock.TargetDate.le(row.feature_date)].dropna(subset=predictors)
                model = OLS(fit.TargetNextReturn, validation.design(fit, predictors)).fit()
                current = stock.loc[stock.Date.eq(row.feature_date)]
                expected = model.predict(validation.design(current, predictors)).iloc[0]
                self.assertAlmostEqual(row.predicted_return, expected, places=12)
                self.assertEqual(row.training_observations, len(fit))
                self.assertEqual(row.training_last_target_date, row.feature_date)
                self.assertLess(row.training_last_target_date, row.forecast_date)
            criteria, _, _, _ = validation.initial_training("TEST", stock)
            self.assertEqual(criteria.training_observations.nunique(), 1)
            self.assertTrue(criteria.last_target_date.le("2022-12-31").all())

    def test_changed_future_outcome_cannot_affect_its_own_or_earlier_forecast(self):
        raw = synthetic_source()
        changed = raw.copy()
        changed.loc[changed.Date.ge("2023-01-05"), "LogReturn"] *= 20
        original, perturbed = development(raw), development(changed)
        for name in validation.SPECIFICATIONS:
            a = validation.expanding_forecasts("TEST", original, name)
            b = validation.expanding_forecasts("TEST", perturbed, name)
            np.testing.assert_array_equal(a.loc[a.forecast_date.le("2023-01-05"), "predicted_return"],
                                          b.loc[b.forecast_date.le("2023-01-05"), "predicted_return"])

    def test_test_rows_and_late_2024_targets_are_excluded_before_conversion(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            features.build(root)
            original = validation.load_development(root)
            path = root / "data/processed/model_a_features.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            test_rows = frame.Date.ge("2025-01-01")
            frame.loc[test_rows, features.NUMERIC_COLUMNS] = "TEST_VALUES_MUST_NOT_BE_PARSED"
            frame.loc[frame.TargetDate.ge("2025-01-01"), "TargetNextReturn"] = "UNOBSERVED_BOUNDARY_TARGET"
            frame.to_csv(path, index=False)
            source_path = root / "data/processed/market_daily.csv"
            raw = pd.read_csv(source_path, dtype=str, keep_default_na=False)
            raw.loc[raw.Date.ge("2025-01-01"), "LogReturn"] = "TEST_RETURNS_MUST_NOT_BE_PARSED"
            raw.to_csv(source_path, index=False)
            poisoned = validation.load_development(root)
            for ticker in original:
                pd.testing.assert_frame_equal(original[ticker], poisoned[ticker], check_exact=True)
                self.assertTrue(poisoned[ticker].Date.le("2024-12-30").all())
                self.assertTrue(poisoned[ticker].TargetDate.le("2024-12-31").all())

    def test_guard_rejects_any_test_information_or_target_date(self):
        stock = development(synthetic_source())
        for column in ["Date", "TargetDate"]:
            changed = stock.copy()
            changed.loc[changed.index[-1], column] = "2025-01-02"
            with self.assertRaisesRegex(ValueError, "Temporal leakage"):
                validation.expanding_forecasts("TEST", changed, "A4")


if __name__ == "__main__":
    unittest.main()
