"""Final-test temporal isolation and common-sample OLS, using synthetic data."""

import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from src.analysis import final_2025_evaluation as final
from src.analysis.model_a_feature_validation import design
from src.features import build_model_b_features as dollar
from src.features import build_share_volume_features as share


def source():
    rng = np.random.default_rng(2025)
    dates = pd.bdate_range(end="2024-12-31", periods=100).append(pd.bdate_range("2025-01-02", periods=8))
    returns = rng.normal(.001, .02, len(dates))
    returns[0] = np.nan
    volume = np.exp(rng.normal(14, .4, len(dates)))
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "LogReturn": returns,
                         "Volume": volume, "DollarVolume": volume * np.exp(rng.normal(5, .1, len(dates))),
                         "display_ticker": "TEST", "yahoo_ticker": "TEST", "company": "Test", "sector": "Test"})


def features(raw, frozen):
    a, s = dollar.build_stock_features(raw, frozen), share.build_stock_features(raw, frozen)
    return a.assign(AbnormalShareVolume20=s.AbnormalShareVolume20).iloc[:-1].reset_index(drop=True)


class FinalForecastTests(unittest.TestCase):
    def test_frozen_definitions_and_independent_ols_on_common_rows(self):
        self.assertEqual(final.FROZEN, {
            "NVDA": "A0", "AAPL": "A0", "BRK.B": "A0", "JPM": "AR(9)", "LLY": "A0", "JNJ": "A0",
            "AMZN": "A3", "TSLA": "A3", "CAT": "AR(9)", "GE": "AR(9)", "XOM": "A0", "CVX": "A0"})
        for frozen in ["A0", "AR(9)", "A3"]:
            stock = features(source(), frozen)
            # Deliberately differing feature availability must remove the same
            # historical rows for every final model, including intercept-only A.
            stock.loc[40, "AbnormalShareVolume20"] = np.nan
            stock.loc[50, "AbnormalDollarVolume20"] = np.nan
            position = stock.index[stock.TargetDate.eq("2025-01-02")][0]
            eligible = stock.loc[(stock.TargetDate <= stock.Date.iloc[position]) &
                                 stock[final.predictors(frozen, "B") + final.EXTRA["S"]].notna().all(axis=1)]
            fits = []
            for model in final.MODELS:
                predicted, fit = final.predict_at(stock, position, frozen, model)
                columns = final.predictors(frozen, model)
                self.assertEqual(columns, dollar.BASELINE_PREDICTORS[frozen] + final.EXTRA[model])
                expected = OLS(eligible.TargetNextReturn, design(eligible, columns)).fit().predict(
                    design(stock.iloc[[position]], columns)).iloc[0]
                self.assertAlmostEqual(predicted, expected, places=14)
                np.testing.assert_array_equal(fit, eligible.index)
                self.assertNotIn(40, fit)
                self.assertNotIn(50, fit)
                fits.append(fit)
            for fit in fits[1:]:
                np.testing.assert_array_equal(fits[0], fit)

    def test_future_raw_outcomes_and_activity_cannot_change_earlier_forecasts(self):
        raw = source()
        # Alter the first 2025 return and all future inputs. The first prediction
        # must be unchanged, but later predictions may respond to realized data.
        altered = raw.copy(deep=True)
        future = altered.Date.ge("2025-01-02")
        altered.loc[future, "LogReturn"] += .5
        altered.loc[future, "Volume"] *= 15
        altered.loc[future, "DollarVolume"] *= 7
        for frozen in ["A0", "AR(9)", "A3"]:
            stock, changed = features(raw, frozen), features(altered, frozen)
            position = stock.index[stock.TargetDate.eq("2025-01-02")][0]
            for model in final.MODELS:
                before, fit = final.predict_at(stock, position, frozen, model)
                after, changed_fit = final.predict_at(changed, position, frozen, model)
                self.assertEqual(before, after)
                np.testing.assert_array_equal(fit, changed_fit)
                # Even unparseable/nonfinite current and future target outcomes
                # cannot influence the current fit: none are accessed.
                poisoned = stock.copy()
                poisoned.loc[position:, "TargetNextReturn"] = np.inf
                self.assertEqual(before, final.predict_at(poisoned, position, frozen, model)[0])
                prefix = stock.iloc[:position + 1].copy()
                self.assertEqual(before, final.predict_at(prefix, position, frozen, model)[0])
                next_fit = final.predict_at(stock, position + 1, frozen, model)[1]
                np.testing.assert_array_equal(next_fit, np.r_[fit, position])

    def test_target_dates_year_boundary_and_daily_losses(self):
        for frozen in ["A0", "AR(9)", "A3"]:
            stock = features(source(), frozen)
            forecasts = final.expanding_forecasts("TEST", stock)
            self.assertEqual(len(forecasts), 8 * 3)
            self.assertEqual(forecasts.iloc[0].forecast_date, "2025-01-02")
            self.assertEqual(forecasts.iloc[0].feature_date, "2024-12-31")
            for _, day in forecasts.groupby("forecast_date"):
                day = day.set_index("model")
                self.assertEqual(day.training_observations.nunique(), 1)
                self.assertTrue(day.training_last_target_date.le(day.feature_date).all())
                for model in final.MODELS:
                    self.assertAlmostEqual(day.loc[model, "forecast_error"],
                                           day.loc[model, "realized_return"] - day.loc[model, "predicted_return"])
                self.assertAlmostEqual(day.loc["A", "loss_difference_B_vs_A"],
                                       day.loc["A", "squared_error"] - day.loc["B", "squared_error"])

    def test_rank_deficient_fit_fails_instead_of_changing_specification(self):
        stock = features(source(), "AR(9)")
        stock.loc[:, "ReturnLag2"] = stock.ReturnLag1
        position = stock.index[stock.TargetDate.eq("2025-01-02")][0]
        with self.assertRaisesRegex(ValueError, "Rank-deficient"):
            final.predict_at(stock, position, "AR(9)", "B")


if __name__ == "__main__":
    unittest.main()
