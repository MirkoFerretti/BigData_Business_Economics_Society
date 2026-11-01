"""Phase 3B dependency propagation, replication, boundaries and untouched holdout."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS

from src.analysis import model_b_brkb_sensitivity as sensitivity
from src.analysis import model_b_share_volume_validation as share_models
from src.analysis import model_b_volume_validation as primary
from src.features import build_model_b_features as dollar_features
from src.features import build_share_volume_features as share_features


def synthetic_source(ticker="TEST", seed=81):
    dates = pd.bdate_range(end="2022-12-30", periods=85).append(
        pd.bdate_range(start="2023-01-03", periods=7)).append(
        pd.to_datetime(["2024-12-30", "2024-12-31", "2025-01-02", "2025-01-03"]))
    random = np.random.default_rng(seed)
    returns = random.normal(.001, .02, len(dates))
    returns[0] = np.nan
    volume = np.exp(random.normal(14, .4, len(dates)))
    splits = np.zeros(len(dates))
    splits[dates.get_loc("2022-12-30")] = 1.25
    splits[dates.get_loc("2024-12-31")] = 2.
    return pd.DataFrame({"Date": dates.strftime("%Y-%m-%d"), "display_ticker": ticker,
                         "yahoo_ticker": ticker, "company": ticker + " Company", "sector": "Test sector",
                         "LogReturn": returns, "Volume": volume, "DollarVolume": volume * np.exp(random.normal(5,.1,len(dates))),
                         "Stock Splits": splits})


def development(source, frozen="A0", share=True):
    builder = share_features if share else dollar_features
    frame = builder.build_stock_features(source, frozen)
    return frame.loc[frame.Date.le(primary.VALID_END) & frame.TargetDate.le(primary.VALID_END)].reset_index(drop=True)


def fixture_repository(root):
    (root / "config").mkdir()
    (root / "data/processed").mkdir(parents=True)
    sources = [synthetic_source(f"TEST{i}", i + 1) for i in range(12)]
    pd.concat([source[dollar_features.IDENTIFIERS].iloc[:1] for source in sources]).to_csv(
        root / "config/universe.csv", index=False)
    pd.DataFrame({"display_ticker": [f"TEST{i}" for i in range(12)],
                  "frozen_model_a": ["A0", "AR(9)", "A3"] * 4}).to_csv(root / "config/model_a_frozen.csv", index=False)
    pd.concat(sources).to_csv(root / "data/processed/market_daily.csv", index=False)


class ContaminationTests(unittest.TestCase):
    def test_dependency_masks_match_explicit_source_sets(self):
        dates = pd.Series(pd.bdate_range("2021-01-01", periods=100).strftime("%Y-%m-%d"))
        for suspicious_positions in [{35, 36}, {30, 70}]:
            masks = sensitivity.dependency_masks(dates, dates.iloc[sorted(suspicious_positions)].tolist())
            for i in range(len(dates)):
                prior = set(range(max(0, i - 20), i))
                abnormal = prior | {i}
                persistent = set()
                for j in range(max(0, i - 4), i + 1):
                    persistent |= set(range(max(0, j - 20), j + 1))
                dependencies = {"LogDollarVolume": {i}, "PriorLogDollarVolumeMean20": prior,
                                "PriorLogDollarVolumeStd20": prior, "AbnormalDollarVolume20": abnormal,
                                "DollarVolumeChange1": {i} | ({i-1} if i else set()),
                                "DollarVolumePersistence5": persistent}
                for feature, inputs in dependencies.items():
                    self.assertEqual(masks.loc[i, feature], bool(inputs & suspicious_positions))
            if suspicious_positions == {35, 36}:
                self.assertEqual(masks.any(axis=1).sum(), 26)
                self.assertEqual(masks.DollarVolumeChange1.sum(), 3)
                self.assertEqual(masks.AbnormalDollarVolume20.sum(), 22)

    def test_masks_cover_actual_recalculation_changes(self):
        source = synthetic_source()
        changed = source.copy()
        changed.loc[[35, 36], "DollarVolume"] *= 20
        before, after = (dollar_features.build_stock_features(frame, "A0") for frame in [source, changed])
        masks = sensitivity.dependency_masks(source.Date, source.Date.iloc[[35,36]].tolist())
        for feature in masks:
            np.testing.assert_allclose(before.loc[~masks[feature], feature], after.loc[~masks[feature], feature],
                                        rtol=1e-10, atol=1e-11, equal_nan=True)

    def test_sensitivity_preserves_returns_and_uses_common_masked_rows(self):
        source = synthetic_source()
        stock = development(source, share=False)
        original = stock.copy(deep=True)
        masked, masks = sensitivity.mask_contaminated(stock, stock.Date.iloc[[35,36]].tolist())
        pd.testing.assert_frame_equal(stock, original, check_exact=True)
        pd.testing.assert_frame_equal(masked[["Date","TargetDate","TargetNextReturn"] + dollar_features.RETURN_FEATURES + dollar_features.AR_FEATURES],
                                      stock[["Date","TargetDate","TargetNextReturn"] + dollar_features.RETURN_FEATURES + dollar_features.AR_FEATURES])
        excluded = primary.common_mask(stock,"A0") & ~primary.common_mask(masked,"A0")
        self.assertEqual(excluded.sum(), 26)
        paths = []
        for specification in primary.SPECIFICATIONS:
            path = primary.expanding_forecasts("TEST", masked, specification)
            paths.append(path)
            for row in [path.iloc[0],path.iloc[-1]]:
                fit = masked.loc[primary.common_mask(masked,"A0") & masked.TargetDate.le(row.feature_date)]
                columns = primary.predictors("A0", specification)
                model = OLS(fit.TargetNextReturn,primary.design(fit,columns)).fit()
                expected = model.predict(primary.design(masked.loc[masked.Date.eq(row.feature_date)],columns)).iloc[0]
                self.assertAlmostEqual(expected,row.predicted_return,places=12)
                self.assertEqual(row.training_observations,len(fit))
        for path in paths[1:]:
            np.testing.assert_array_equal(path.training_observations,paths[0].training_observations)
            self.assertEqual(path.forecast_date.tolist(),paths[0].forecast_date.tolist())


class ShareVolumeTests(unittest.TestCase):
    def test_exact_windows_next_target_and_structural_missingness(self):
        source = synthetic_source()
        frame = share_features.build_stock_features(source,"AR(9)")
        share_features.validate_stock_features(frame,source,"AR(9)")
        values = np.log(source.Volume.to_numpy())
        i = 35
        self.assertAlmostEqual(frame.AbnormalShareVolume20.iloc[i],(values[i]-values[i-20:i].mean())/values[i-20:i].std(ddof=1))
        self.assertAlmostEqual(frame.ShareVolumeChange1.iloc[i],values[i]-values[i-1])
        self.assertAlmostEqual(frame.ShareVolumePersistence5.iloc[i],np.mean([
            (values[j]-values[j-20:j].mean())/values[j-20:j].std(ddof=1) for j in range(i-4,i+1)]))
        self.assertEqual(frame[share_features.VOLUME_FEATURES].isna().sum().tolist(),[20,1,24])
        row = frame.loc[frame.Date.eq("2022-12-30")].iloc[0]
        self.assertEqual(row.TargetDate,"2023-01-03")
        self.assertEqual(row.TargetNextReturn,source.loc[source.Date.eq("2023-01-03"),"LogReturn"].item())
        self.assertTrue(pd.isna(frame.TargetNextReturn.iloc[-1]))
        with self.assertRaisesRegex(ValueError,"positive"):
            share_features.build_stock_features(source.assign(Volume=0.),"A0")

    def test_frozen_S0_matches_B0_and_candidates_use_common_past_rows(self):
        for frozen in dollar_features.BASELINE_PREDICTORS:
            source = synthetic_source()
            stock = development(source,frozen)
            baseline = primary.expanding_forecasts("TEST",development(source,frozen,share=False),"B0")
            s0 = share_models.expanding_forecasts("TEST",stock,"S0")
            np.testing.assert_array_equal(s0.predicted_return,baseline.predicted_return)
            for specification in share_models.SPECIFICATIONS:
                path = share_models.expanding_forecasts("TEST",stock,specification)
                adapted = share_models.to_primary(stock)
                columns = primary.predictors(frozen,share_models.S_TO_B[specification])
                for row in [path.iloc[0],path.iloc[-1]]:
                    fit = adapted.loc[primary.common_mask(adapted,frozen)&adapted.TargetDate.le(row.feature_date)]
                    model = OLS(fit.TargetNextReturn,primary.design(fit,columns)).fit()
                    expected = model.predict(primary.design(adapted.loc[adapted.Date.eq(row.feature_date)],columns)).iloc[0]
                    self.assertAlmostEqual(expected,row.predicted_return,places=12)
                    self.assertEqual(row.training_last_target_date,row.feature_date)
                    self.assertLess(row.training_last_target_date,row.forecast_date)
                np.testing.assert_array_equal(path.training_observations,s0.training_observations)
                self.assertEqual(path.forecast_date.tolist(),s0.forecast_date.tolist())

    def test_future_outcomes_and_activity_cannot_change_earlier_predictions(self):
        source = synthetic_source()
        changed = source.copy()
        changed.loc[changed.Date.ge("2023-01-05"),"Volume"] *= 100
        changed.loc[changed.Date.ge("2023-01-05"),"LogReturn"] *= 20
        for specification in share_models.SPECIFICATIONS:
            a = share_models.expanding_forecasts("TEST",development(source,"A3"),specification)
            b = share_models.expanding_forecasts("TEST",development(changed,"A3"),specification)
            np.testing.assert_array_equal(a.loc[a.forecast_date.le("2023-01-05"),"predicted_return"],
                                          b.loc[b.forecast_date.le("2023-01-05"),"predicted_return"])

    def test_ticker_isolation_in_saved_share_features(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            share_features.build(root)
            destination = root / "data/processed/model_b_share_volume_features.csv"
            original = pd.read_csv(destination)
            path = root / "data/processed/market_daily.csv"
            source = pd.read_csv(path)
            mask = source.display_ticker.eq("TEST0")
            source.loc[mask,"Volume"] *= np.linspace(1.,5.,mask.sum())
            source.to_csv(path,index=False)
            share_features.build(root)
            changed = pd.read_csv(destination)
            pd.testing.assert_frame_equal(original.loc[original.display_ticker.ne("TEST0")],
                                          changed.loc[changed.display_ticker.ne("TEST0")],check_exact=True)

    def test_holdout_exclusion_and_split_audit_boundary(self):
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture_repository(root)
            share_features.build(root)
            original = share_models.load_development(root)
            path = root / "data/processed/model_b_share_volume_features.csv"
            frame = pd.read_csv(path,dtype=str,keep_default_na=False)
            frame.loc[frame.Date.ge("2025-01-01"),share_features.NUMERIC_COLUMNS] = "DO_NOT_PARSE_TEST_FEATURES"
            frame.loc[frame.TargetDate.ge("2025-01-01"),"TargetNextReturn"] = "UNOBSERVED_BOUNDARY_TARGET"
            frame.to_csv(path,index=False)
            path = root / "data/processed/market_daily.csv"
            frame = pd.read_csv(path,dtype=str,keep_default_na=False)
            frame.loc[frame.Date.ge("2025-01-01"),["LogReturn","Volume","Stock Splits"]] = "DO_NOT_PARSE_TEST_SOURCE"
            frame.to_csv(path,index=False)
            poisoned = share_models.load_development(root)
            for ticker in original:
                pd.testing.assert_frame_equal(original[ticker],poisoned[ticker],check_exact=True)
            audit = share_features.split_audit(root)
            boundary = audit.loc[audit.Date.eq("2024-12-31")]
            self.assertEqual(len(boundary),12)
            self.assertTrue(boundary.next_available_Volume.isna().all())
            self.assertTrue(boundary.next_date.eq("").all())
            self.assertTrue(audit.Date.le("2024-12-31").all())
            self.assertTrue(audit.loc[audit.Date.eq("2022-12-30"),"Stock Splits"].eq(1.25).all())
            bad = poisoned["TEST0"].copy()
            bad.loc[bad.index[-1],"TargetDate"] = "2025-01-02"
            with self.assertRaisesRegex(ValueError,"Temporal leakage"):
                share_models.expanding_forecasts("TEST0",bad,"S3")


if __name__ == "__main__":
    unittest.main()
