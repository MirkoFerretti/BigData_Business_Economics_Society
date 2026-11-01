"""Phase 3A causal windows, frozen baselines, common samples and holdout isolation."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from src.analysis import model_a_return_diagnostics as raw_ar
from src.analysis import model_a_feature_validation as return_models
from src.analysis import model_b_volume_validation as validation
from src.features import build_model_a_features as return_features
from src.features import build_model_b_features as features


def synthetic_source(ticker="TEST", seed=81):
    dates = pd.bdate_range(end="2022-12-30", periods=85).append(
        pd.bdate_range(start="2023-01-03", periods=7)).append(
        pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]))
    generator = np.random.default_rng(seed)
    returns = generator.normal(.001, .02, len(dates))
    returns[0] = np.nan
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "display_ticker": ticker,
                         "yahoo_ticker": ticker, "company": ticker + " Company", "sector": "Test sector",
                         "LogReturn": returns, "DollarVolume": np.exp(generator.normal(20, .4, len(dates)))})


def development(source, frozen):
    frame = features.build_stock_features(source, frozen)
    return frame.loc[frame["Date"].le(validation.VALID_END) &
                      frame["TargetDate"].le(validation.VALID_END)].reset_index(drop=True)


def fixture_repository(root):
    (root / "config").mkdir()
    (root / "data/processed").mkdir(parents=True)
    sources = [synthetic_source(f"TEST{i}", i + 1) for i in range(12)]
    pd.concat([source[return_features.IDENTIFIERS].iloc[:1] for source in sources]).to_csv(
        root / "config/universe.csv", index=False)
    pd.DataFrame({"display_ticker": [f"TEST{i}" for i in range(12)],
                  "frozen_model_a": ["A0", "AR(9)", "A3"] * 4}).to_csv(root / "config/model_a_frozen.csv", index=False)
    pd.concat(sources).to_csv(root / "data/processed/market_daily.csv", index=False)


class VolumeFeatureTests(unittest.TestCase):
    def test_exact_prior20_change_and_five_day_persistence(self):
        source = synthetic_source()
        frame = features.build_stock_features(source, "AR(9)")
        features.validate_stock_features(frame, source, "AR(9)")
        logs = np.log(source.DollarVolume.to_numpy())
        i = 30
        self.assertAlmostEqual(frame.AbnormalDollarVolume20.iloc[i],
                               (logs[i] - logs[i-20:i].mean()) / logs[i-20:i].std(ddof=1))
        self.assertAlmostEqual(frame.DollarVolumeChange1.iloc[i], logs[i] - logs[i-1])
        expected = np.mean([(logs[j] - logs[j-20:j].mean()) / logs[j-20:j].std(ddof=1)
                            for j in range(i-4, i+1)])
        self.assertAlmostEqual(frame.DollarVolumePersistence5.iloc[i], expected)
        self.assertEqual(frame[features.VOLUME_FEATURES].isna().sum().tolist(), [20, 1, 24])
        changed = source.copy()
        changed.loc[i, "DollarVolume"] *= 100
        after = features.build_stock_features(changed, "AR(9)")
        self.assertEqual(frame.PriorLogDollarVolumeMean20.iloc[i], after.PriorLogDollarVolumeMean20.iloc[i])
        self.assertEqual(frame.PriorLogDollarVolumeStd20.iloc[i], after.PriorLogDollarVolumeStd20.iloc[i])

    def test_ar9_predictors_and_next_trading_target(self):
        source = synthetic_source()
        frame = features.build_stock_features(source, "AR(9)")
        for lag in range(1, 10):
            self.assertEqual(frame.loc[30, f"ReturnLag{lag}"], source.LogReturn.iloc[31-lag])
        row = frame.loc[frame.Date.eq("2022-12-30")].iloc[0]
        self.assertEqual(row.TargetDate, "2023-01-03")
        self.assertEqual(row.TargetNextReturn, source.loc[source.Date.eq("2023-01-03"), "LogReturn"].item())
        self.assertTrue(pd.isna(frame.TargetNextReturn.iloc[-1]))

    def test_future_changes_do_not_change_earlier_features(self):
        source = synthetic_source()
        changed = source.copy()
        changed.loc[45:, "DollarVolume"] *= 100
        changed.loc[45:, "LogReturn"] *= 20
        a, b = (features.build_stock_features(frame, "A3") for frame in [source, changed])
        columns = [column for column in features.NUMERIC_COLUMNS if column != "TargetNextReturn"]
        pd.testing.assert_frame_equal(a[columns].iloc[:45], b[columns].iloc[:45])

    def test_nonpositive_activity_and_zero_reference_deviation_fail(self):
        for value in [0., -1.]:
            with self.assertRaisesRegex(ValueError, "positive"):
                features.build_stock_features(synthetic_source().assign(DollarVolume=value), "A0")
        with self.assertRaisesRegex(ValueError, "Zero prior-20"):
            features.build_stock_features(synthetic_source().assign(DollarVolume=100.), "A0")

    def test_ticker_isolation_in_saved_combined_file(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            features.build(root)
            destination = root / "data/processed/model_b_features.csv"
            original = pd.read_csv(destination)
            path = root / "data/processed/market_daily.csv"
            source = pd.read_csv(path)
            source.loc[source.display_ticker.eq("TEST0"), "DollarVolume"] *= np.linspace(1., 10., source.display_ticker.eq("TEST0").sum())
            source.to_csv(path, index=False)
            features.build(root)
            changed = pd.read_csv(destination)
            pd.testing.assert_frame_equal(original.loc[original.display_ticker.ne("TEST0")],
                                          changed.loc[changed.display_ticker.ne("TEST0")], check_exact=True)


class FrozenBaselineValidationTests(unittest.TestCase):
    def test_actual_mapping_matches_d014_and_predictor_whitelist(self):
        self.assertEqual(features.read_frozen(), {
            "NVDA": "A0", "AAPL": "A0", "BRK.B": "A0", "JPM": "AR(9)",
            "LLY": "A0", "JNJ": "A0", "AMZN": "A3", "TSLA": "A3", "CAT": "AR(9)",
            "GE": "AR(9)", "XOM": "A0", "CVX": "A0"})
        for frozen in features.BASELINE_PREDICTORS:
            for i, specification in enumerate(validation.SPECIFICATIONS):
                self.assertEqual(validation.predictors(frozen, specification),
                                 features.BASELINE_PREDICTORS[frozen] + features.VOLUME_FEATURES[:i])

    def test_common_samples_and_independent_ols_refits(self):
        for frozen in features.BASELINE_PREDICTORS:
            stock = development(synthetic_source(), frozen)
            paths = []
            for specification in validation.SPECIFICATIONS:
                path = validation.expanding_forecasts("TEST", stock, specification)
                paths.append(path)
                columns = validation.predictors(frozen, specification)
                for row in [path.iloc[0], path.iloc[-1]]:
                    fit = stock.loc[validation.common_mask(stock, frozen) & stock.TargetDate.le(row.feature_date)]
                    reference = OLS(fit.TargetNextReturn, validation.design(fit, columns)).fit()
                    current = stock.loc[stock.Date.eq(row.feature_date)]
                    expected = reference.predict(validation.design(current, columns)).iloc[0]
                    self.assertAlmostEqual(row.predicted_return, expected, places=12)
                    self.assertEqual(row.training_observations, len(fit))
                    self.assertEqual(row.training_last_target_date, row.feature_date)
                    self.assertLess(row.training_last_target_date, row.forecast_date)
            for path in paths[1:]:
                pd.testing.assert_frame_equal(path[["forecast_date", "training_observations", "training_first_target_date"]],
                                              paths[0][["forecast_date", "training_observations", "training_first_target_date"]])
            original = validation.expanding_forecasts("TEST", stock, "B0", common_sample=False)
            expected_loss = {"A0": 24, "AR(9)": 15, "A3": 4}[frozen]
            self.assertEqual(original.training_observations.iloc[0] - paths[0].training_observations.iloc[0], expected_loss)

    def test_original_sample_reproduces_phase2_formulas(self):
        source = synthetic_source()
        for frozen in features.BASELINE_PREDICTORS:
            stock = development(source, frozen)
            path = validation.expanding_forecasts("TEST", stock, "B0", common_sample=False)
            if frozen == "A3":
                old = return_features.build_stock_features(source)
                old = old.loc[old.Date.le(validation.VALID_END) & old.TargetDate.le(validation.VALID_END)].reset_index(drop=True)
                reference = return_models.expanding_forecasts("TEST", old, "A3")
                np.testing.assert_allclose(path.predicted_return, reference.predicted_return, rtol=1e-12, atol=1e-14)
            else:
                for row in path.itertuples(index=False):
                    history = source.loc[source.Date.le(row.feature_date), "LogReturn"].dropna().to_numpy()
                    expected = raw_ar.forecast_from_history(history, 9 if frozen == "AR(9)" else 0)
                    self.assertAlmostEqual(row.predicted_return, expected, places=12)

    def test_future_validation_outcomes_cannot_change_earlier_predictions(self):
        source = synthetic_source()
        changed = source.copy()
        changed.loc[changed.Date.ge("2023-01-05"), "LogReturn"] *= 10
        changed.loc[changed.Date.ge("2023-01-05"), "DollarVolume"] *= 100
        for frozen in features.BASELINE_PREDICTORS:
            for specification in validation.SPECIFICATIONS:
                a = validation.expanding_forecasts("TEST", development(source, frozen), specification)
                b = validation.expanding_forecasts("TEST", development(changed, frozen), specification)
                np.testing.assert_array_equal(a.loc[a.forecast_date.le("2023-01-05"), "predicted_return"],
                                              b.loc[b.forecast_date.le("2023-01-05"), "predicted_return"])

    def test_test_and_boundary_values_are_never_parsed_by_analysis(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            features.build(root)
            original = validation.load_development(root)
            path = root / "data/processed/model_b_features.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.loc[frame.Date.ge("2025-01-01"), features.NUMERIC_COLUMNS] = "DO_NOT_PARSE_TEST_FEATURES"
            frame.loc[frame.TargetDate.ge("2025-01-01"), "TargetNextReturn"] = "UNOBSERVED_TARGET"
            frame.to_csv(path, index=False)
            path = root / "data/processed/market_daily.csv"
            frame = pd.read_csv(path, dtype=str, keep_default_na=False)
            frame.loc[frame.Date.ge("2025-01-01"), ["LogReturn", "DollarVolume"]] = "DO_NOT_PARSE_TEST_SOURCE"
            frame.to_csv(path, index=False)
            poisoned = validation.load_development(root)
            for ticker in original:
                pd.testing.assert_frame_equal(original[ticker], poisoned[ticker], check_exact=True)
            changed = poisoned["TEST0"].copy()
            changed.loc[changed.index[-1], "TargetDate"] = "2025-01-02"
            with self.assertRaisesRegex(ValueError, "Temporal leakage"):
                validation.expanding_forecasts("TEST0", changed, "B3")


if __name__ == "__main__":
    unittest.main()
