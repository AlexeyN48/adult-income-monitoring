import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import yaml
from evidently import Dataset, Report, DataDefinition
from evidently.metrics import ValueDrift

from adult_monitoring import monitoring


def fixture_frame(n=12, age_shift=0, all_male=False):
    # Known binary confusion matrix: TP=3, FP=1, FN=1, TN=3 per 8-row base.
    truth = [1, 1, 1, 1, 0, 0, 0, 0] * (n // 8) + [1] * (n % 8)
    pred = [1, 1, 1, 0, 1, 0, 0, 0] * (n // 8) + [0] * (n % 8)
    sex = ["Male" if all_male else ("Female" if i % 2 == 0 else "Male") for i in range(n)]
    return pd.DataFrame({"age": [20 + (i % 8) + age_shift for i in range(n)],
                         "sex": sex, "income": [">50K" if x else "<=50K" for x in truth],
                         "prediction": pred})


class MonitoringTests(unittest.TestCase):
    def test_known_confusion_matrix_and_fairness_formulas(self):
        # 8 rows: TP=3, FP=1, FN=1, TN=3 => positive F1=.75, macro F1=.75.
        frame = pd.DataFrame({"income": [">50K", ">50K", "<=50K", "<=50K"] * 2,
                              "prediction": [1, 1, 1, 0, 1, 0, 0, 0],
                              "sex": ["Female"] * 4 + ["Male"] * 4})
        stats = monitoring._calc(frame, 2, 1)
        self.assertAlmostEqual(stats["f1_positive"], .75)
        self.assertAlmostEqual(stats["macro_f1"], .75)
        self.assertAlmostEqual(stats["groups"]["Female"]["selection_rate"], .75)
        self.assertAlmostEqual(stats["groups"]["Male"]["selection_rate"], .25)
        self.assertAlmostEqual(stats["fairness"]["demographic_parity_difference"], .5)
        self.assertAlmostEqual(stats["fairness"]["disparate_impact"], 3.0)
        self.assertAlmostEqual(stats["fairness"]["tpr_gap"], .5)

    def test_custom_fairness_metrics_tests_and_drift_are_evidently_snapshot_values(self):
        ref = fixture_frame(16)
        cur = fixture_frame(16, age_shift=6)
        report = Report([ValueDrift(column="age", method="wasserstein", threshold=.1),
                         monitoring.MonitoringScalar(key="fairness.demographic_parity_difference", title="DPD", min_group_rows=2, min_positive_labels=1),
                         monitoring.MonitoringScalar(key="fairness.abs_dpd", title="DPD bound", direction="lte", threshold=.1, min_group_rows=2, min_positive_labels=1),
                         monitoring.MonitoringScalar(key="fairness.validity.all_groups", title="Valid", direction="gt", threshold=.5, min_group_rows=2, min_positive_labels=1)],
                        include_tests=True)
        definition = DataDefinition(numerical_columns=["age"], categorical_columns=["sex", "income", "prediction"])
        snap = report.run(Dataset.from_pandas(cur, data_definition=definition), Dataset.from_pandas(ref, data_definition=definition))
        raw = snap.dict()
        by_key = {m.get("config", {}).get("key"): m for m in raw["metrics"] if m.get("config", {}).get("key")}
        self.assertAlmostEqual(by_key["fairness.demographic_parity_difference"]["value"], .5)
        age = next(m for m in raw["metrics"] if m["config"].get("column") == "age")
        self.assertIn("ValueDrift", age["metric_name"])
        self.assertGreater(age["value"], .1)
        age_test = next(t for t in raw["tests"] if t["metric_config"]["metric_id"] == age["id"])
        self.assertEqual(getattr(age_test["status"], "value", age_test["status"]), "FAIL")
        statuses = {(t["metric_config"]["params"].get("key"), getattr(t["status"], "value", str(t["status"]))) for t in raw["tests"]}
        self.assertIn(("fairness.validity.all_groups", "SUCCESS"), statuses)
        with tempfile.TemporaryDirectory() as td:
            html = Path(td) / "report.html"; snap.save_html(str(html))
            self.assertIn("DPD bound", html.read_text(encoding="utf-8"))

    def test_missing_group_and_zero_denominators_are_invalid_not_green(self):
        frame = pd.DataFrame({"income": ["<=50K"] * 8, "prediction": [0] * 8, "sex": ["Male"] * 8})
        stats = monitoring._calc(frame, 3, 2)
        self.assertIsNone(stats["fairness"]["demographic_parity_difference"])
        self.assertIsNone(stats["fairness"]["disparate_impact"])
        self.assertIsNone(stats["fairness"]["tpr_gap"])
        self.assertFalse(stats["fairness"]["validity"]["all_groups"])
        self.assertFalse(stats["fairness"]["validity"]["disparate_impact"])
        scalar = monitoring.MonitoringScalar(key="fairness.disparate_impact", title="DI", direction="gte", threshold=.8,
                                             min_group_rows=3, min_positive_labels=2)
        snap = Report([scalar, monitoring.MonitoringScalar(key="fairness.validity.disparate_impact", title="DI valid",
                      direction="gt", threshold=.5, min_group_rows=3, min_positive_labels=2)], include_tests=True).run(
                          Dataset.from_pandas(frame), Dataset.from_pandas(frame))
        values = [m["value"] for m in snap.dict()["metrics"] if m.get("config", {}).get("key") == "fairness.disparate_impact"]
        self.assertIsNone(monitoring._plain(values[0]))
        valid_test = next(t for t in snap.dict()["tests"] if t["metric_config"]["params"].get("key") == "fairness.validity.disparate_impact")
        self.assertEqual(valid_test["status"], "FAIL")
        custom = monitoring._extract_custom(snap, .8, {"di_min": .8})
        encoded = json.dumps(monitoring._plain(custom), allow_nan=False)
        self.assertIn('"fairness.disparate_impact": null', encoded)
        with tempfile.TemporaryDirectory() as td:
            html = Path(td) / "invalid.html"; snap.save_html(str(html))
            page = html.read_text(encoding="utf-8")
            self.assertIn("N/A", page)

    def test_checksum_mismatch_rejected_and_status_is_failed(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); (root / "data").mkdir(); (root / "scenarios").mkdir()
            (root / "reports").mkdir(); (root / "model.cbm").write_bytes(b"test model")
            (root / "scenarios" / "baseline.csv").write_text("synthetic", encoding="utf-8")
            (root / "data" / "reference.csv").write_text("synthetic", encoding="utf-8")
            features = list(monitoring.FEATURES)
            (root / "scenarios" / "manifest.json").write_text(json.dumps({"scenarios": {"baseline": {"file": "baseline.csv", "sha256": "expected", "rows": 1}},
              "model_sha256": "modelhash", "feature_schema": features, "reference_sha256": "refhash", "source_sha256": "sourcehash"}), encoding="utf-8")
            (root / "data" / "data_manifest.json").write_text(json.dumps({"model": {"sha256": "modelhash", "features": features}, "checksums": {"reference.csv": "refhash", "current.csv": "sourcehash"}}), encoding="utf-8")
            (root / "reports" / "training_summary.json").write_text(json.dumps({"model_sha256": "modelhash", "features": features}), encoding="utf-8")
            cfg = {"paths": {"processed_data": str(root / "data"), "reports": str(root / "reports"),
                              "model": str(root / "model.cbm")},
                   "scenarios": {"output_dir": str(root / "scenarios")}}
            cfg_path = Path(td) / "config.yaml"
            cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
            class FakeModel:
                feature_names_ = features
                def load_model(self, path): pass
            def fake_sha(path):
                path = Path(path)
                if path.name == "model.cbm": return "modelhash"
                if path.name == "baseline.csv": return "not-the-manifest-sha"
                return "refhash"
            with mock.patch.object(monitoring, "_sha", side_effect=fake_sha), mock.patch.object(monitoring, "CatBoostClassifier", FakeModel):
                with self.assertRaisesRegex(RuntimeError, "checksum mismatch"):
                    monitoring.calculate_scenario(cfg_path, "baseline")
            status = json.loads((Path(td) / "reports" / "evidently" / "baseline" / "status.json").read_text(encoding="utf-8"))
            self.assertEqual(status["status"], "failed")

    def test_configured_group_minimum_and_inclusive_bounds_are_applied_in_report(self):
        data = Dataset.from_pandas(fixture_frame(16))
        cfg = {"fairness_thresholds": {"min_group_rows": 2, "min_positive_labels": 1, "dpd_abs": .5}}
        snap = Report(monitoring._metrics({"alert_f1_threshold": .75}, cfg), include_tests=True).run(data, data)
        parsed = monitoring._extract_custom(snap, .75, {})
        self.assertEqual(parsed["evidently_tests"]["f1_positive"], ["success"])
        self.assertEqual(parsed["evidently_tests"]["fairness.abs_dpd"], ["success"])
        cfg["fairness_thresholds"]["min_group_rows"] = 100
        snap = Report(monitoring._metrics({"alert_f1_threshold": .75}, cfg), include_tests=True).run(data, data)
        parsed = monitoring._extract_custom(snap, .75, {})
        self.assertIsNone(parsed["evidently_metric_values"]["current"]["fairness.disparate_impact"])
        self.assertIsNone(parsed["evidently_metric_values"]["current"]["fairness.abs_dpd"])
        self.assertEqual(parsed["evidently_tests"]["fairness.validity.all_groups"], ["fail"])

    def test_zero_selection_denominator_preserves_available_reference_value(self):
        reference = fixture_frame(16)
        current = reference.copy()
        current.loc[current.sex.eq("Male"), "prediction"] = 0
        metric = monitoring.MonitoringScalar(key="fairness.disparate_impact", title="DI", direction="gte", threshold=.8,
                                             min_group_rows=2, min_positive_labels=1)
        snap = Report([metric], include_tests=True).run(Dataset.from_pandas(current), Dataset.from_pandas(reference))
        parsed = monitoring._extract_custom(snap, .5, {})
        self.assertIsNone(parsed["evidently_metric_values"]["current"]["fairness.disparate_impact"])
        self.assertAlmostEqual(parsed["evidently_metric_values"]["reference"]["fairness.disparate_impact"], 3)
        self.assertEqual(parsed["evidently_tests"]["fairness.disparate_impact"], ["fail"])
        widget = snap.context.get_metric_result(metric).widget[0]
        self.assertIn("N/A", str(widget.params))
        self.assertIn("3.000", str(widget.params))
        json.dumps(monitoring._plain(snap.dict()), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
