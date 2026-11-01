"""The generated reader index must point at the current final methodology."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.analysis import report_eda


class ReportIndexTests(unittest.TestCase):
    def test_generated_index_uses_current_results_without_historical_dependencies(self):
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            report_eda.write_index(output, pd.DataFrame({"stock": ["SYNTHETIC"]}))
            text = (output / "README.md").read_text(encoding="utf-8")
        for path in [
            "../final_2025_stock_specific_volume_report.md",
            "../tables/stock_specific_volume_model_selection.csv",
            *[f"../tables/final_2025_stock_specific_volume_{name}.csv" for name in
              ("metrics", "comparison", "cross_stock_summary", "training_counts")],
            *[f"../figures/final_2025_stock_specific_volume/{name}.png" for name in
              ("model_b_mse_improvement", "model_s_mse_improvement", "validation_vs_test")],
        ]:
            self.assertIn(f"]({path})", text)
        self.assertNotIn("](../figures/final_2025/", text)
        self.assertNotIn("](../project_full_audit.md)", text)
        self.assertIn("AAPL selects B2/S2, TSLA B3/S3", text)
        self.assertIn("HISTORICAL", text)


if __name__ == "__main__":
    unittest.main()
