"""Numerical and temporal safeguards for Phase 2A; no live data or network needed."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from statsmodels.tsa.ar_model import AutoReg


MODULE_PATH = Path(__file__).resolve().parents[1] / "src/analysis/model_a_return_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("return_diagnostics", MODULE_PATH)
diagnostics = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostics)


class ReturnDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        train_dates = pd.bdate_range(end="2022-12-30", periods=80)
        validation_dates = pd.bdate_range(start="2023-01-03", periods=6)
        dates = train_dates.append(validation_dates)
        self.returns = pd.Series(np.random.default_rng(1729).normal(0.001, 0.02, len(dates)),
                                 index=dates, name="SYNTHETIC")

    def test_expanding_forecasts_match_independent_autoreg_refits(self):
        for p in range(11):
            forecasts = diagnostics.expanding_forecasts("SYNTHETIC", self.returns, p)
            for row in forecasts.itertuples(index=False):
                past = self.returns.loc[self.returns.index < row.Date].to_numpy()
                reference = AutoReg(past, lags=p, trend="c", hold_back=p).fit()
                expected = float(reference.predict(start=len(past), end=len(past))[0])
                self.assertAlmostEqual(row.predicted_return, expected, places=12)
                self.assertEqual(row.training_observations, len(past) - p)
                self.assertLess(row.training_end_date, row.Date)
                if p == 0:
                    self.assertAlmostEqual(row.predicted_return, past.mean(), places=15)

    def test_future_outcomes_cannot_change_earlier_predictions(self):
        changed = self.returns.copy()
        changed.iloc[-2:] = [100.0, -200.0]
        for p in [0, 1, 10]:
            original = diagnostics.expanding_forecasts("SYNTHETIC", self.returns, p)
            perturbed = diagnostics.expanding_forecasts("SYNTHETIC", changed, p)
            # Even the first changed outcome cannot enter its own prediction.
            np.testing.assert_array_equal(original.predicted_return.iloc[:-1],
                                          perturbed.predicted_return.iloc[:-1])

    def test_ic_uses_identical_response_dates_and_counts_variance(self):
        train = self.returns.loc[:diagnostics.TRAIN_END]
        candidates = diagnostics.training_candidates("SYNTHETIC", train)
        self.assertEqual(candidates.training_observations.unique().tolist(), [70])
        self.assertEqual(candidates.training_first_response_date.nunique(), 1)
        self.assertEqual(candidates.training_last_response_date.nunique(), 1)
        np.testing.assert_array_equal(candidates.parameter_count, candidates.p + 2)
        np.testing.assert_allclose(candidates.AIC,
                                   -2 * candidates.log_likelihood + 2 * candidates.parameter_count)
        np.testing.assert_allclose(candidates.BIC,
                                   -2 * candidates.log_likelihood + np.log(70) * candidates.parameter_count)

    def test_development_rejects_test_rows_and_unexpected_missingness(self):
        with_test = pd.concat([self.returns, pd.Series([999.0], index=pd.to_datetime(["2025-01-02"]))])
        with self.assertRaisesRegex(ValueError, "Temporal leakage"):
            diagnostics.expanding_forecasts("SYNTHETIC", with_test, 1)
        with self.assertRaisesRegex(ValueError, "Temporal leakage"):
            diagnostics.training_candidates("SYNTHETIC", self.returns)
        missing = self.returns.copy()
        missing.iloc[10] = np.nan
        with self.assertRaisesRegex(ValueError, "missing or infinite"):
            diagnostics.expanding_forecasts("SYNTHETIC", missing, 1)

    def test_test_returns_are_excluded_before_numeric_conversion(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config").mkdir()
            (root / "data/processed").mkdir(parents=True)
            tickers = [f"SYNTHETIC{i}" for i in range(12)]
            pd.DataFrame({"display_ticker": tickers}).to_csv(root / "config/universe.csv", index=False)
            rows = []
            for ticker in tickers:
                rows.append({"Date": "2015-01-02", "display_ticker": ticker, "LogReturn": ""})
                rows.extend({"Date": date.strftime("%Y-%m-%d"), "display_ticker": ticker, "LogReturn": value}
                            for date, value in self.returns.items())
                rows.append({"Date": "2025-01-02", "display_ticker": ticker, "LogReturn": "0.0"})
            source = pd.DataFrame(rows)
            path = root / "data/processed/market_daily.csv"
            source.to_csv(path, index=False)
            original = diagnostics.load_development(root)
            source.loc[source.Date.ge("2025-01-01"), "LogReturn"] = "DO_NOT_PARSE_TEST_RETURNS"
            source.to_csv(path, index=False)
            poisoned = diagnostics.load_development(root)
            for ticker in tickers:
                pd.testing.assert_series_equal(original[ticker], poisoned[ticker], check_exact=True)
                self.assertLessEqual(poisoned[ticker].index.max(), pd.Timestamp("2024-12-31"))


if __name__ == "__main__":
    unittest.main()
