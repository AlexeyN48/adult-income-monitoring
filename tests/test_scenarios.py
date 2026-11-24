from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from catboost import CatBoostClassifier

from adult_monitoring.scenarios import _diagnostics, generate_scenarios


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ScenarioGenerationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Isolated, explicitly synthetic test data: tests never depend on ignored Adult artifacts.
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cls.processed = root / "processed"
        cls.processed.mkdir()
        cls.reports = root / "reports"
        cls.reports.mkdir()
        cls.model_path = root / "catboost.cbm"
        splits = {}
        for split, offset in (("train", 0), ("validation", 1000), ("reference", 2000), ("current", 3000)):
            rows = []
            # 100 Female + 100 Male, each with 50 positive labels. Nearby but
            # distinct ages keep splits disjoint while preserving the learnable pattern.
            for sex in ("Female", "Male"):
                for i in range(100):
                    rows.append({"row_id": f"{split}:{sex}:{i}", "sex": sex, "age": i + offset / 100000,
                                 "income": ">50K" if i < 50 else "<=50K"})
            splits[split] = pd.DataFrame(rows)
            splits[split].to_csv(cls.processed / f"{split}.csv", index=False)
        model = CatBoostClassifier(iterations=5, depth=2, verbose=False, allow_writing_files=False, random_seed=19)
        model.fit(splits["train"][["age"]], (splits["train"].income == ">50K").astype(int))
        model.save_model(str(cls.model_path))
        pred_sha = {}
        for split in ("reference", "current"):
            frame = splits[split].copy()
            p = model.predict_proba(frame[["age"]])[:, 1]
            frame["probability_gt50k"] = p
            frame["probability_le50k"] = 1 - p
            frame["prediction"] = (p >= .5).astype(int)
            path = cls.processed / f"{split}_predictions.csv"
            frame.to_csv(path, index=False)
            pred_sha[f"{split}_predictions.csv"] = sha(path)
        checksums = {f"{name}.csv": sha(cls.processed / f"{name}.csv") for name in splits}
        checksums.update(pred_sha)
        model_sha = sha(cls.model_path)
        (cls.processed / "data_manifest.json").write_text(json.dumps({
            "checksums": checksums,
            "model": {"sha256": model_sha, "features": ["age"]},
        }), encoding="utf-8")
        (cls.reports / "training_summary.json").write_text(json.dumps({
            "model_sha256": model_sha, "features": ["age"], "decision_threshold": .5,
            "predictions": {split: {"sha256": pred_sha[f"{split}_predictions.csv"]} for split in ("reference", "current")},
        }), encoding="utf-8")
        cls.config = root / "config.yaml"
        cls.output = root / "scenarios"
        cls.config_data = {
            "project": {"random_seed": 19},
            "paths": {"processed_data": str(cls.processed), "model": str(cls.model_path), "reports": str(cls.reports)},
            "scenarios": {"seed": 19, "output_dir": str(cls.output), "feature_drift": {"age_delta": 5},
                          "feature_drift_strong": {"age_delta": 15}, "target_drift": {"positive_rate": .60},
                          "fairness_shift": {"female_positive_rate": .40, "female_rows": 100},
                          "fairness_defect": {"female_probability_multiplier": .05}},
        }
        cls.config.write_text(yaml.safe_dump(cls.config_data), encoding="utf-8")
        cls.manifest = generate_scenarios(cls.config)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def read(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.output / f"{name}.csv", keep_default_na=False)

    def test_seed_reproduces_all_csvs_and_recovery_is_baseline(self):
        first = {name: entry["sha256"] for name, entry in self.manifest["scenarios"].items()}
        second_manifest = generate_scenarios(self.config)
        second = {name: entry["sha256"] for name, entry in second_manifest["scenarios"].items()}
        self.assertEqual(first, second)
        self.assertEqual(first["baseline"], first["recovery"])

    def test_feature_shifts_touch_only_age_and_labels_follow_source_ids(self):
        current = pd.read_csv(self.processed / "current.csv", keep_default_na=False)
        for scenario, delta in (("feature_drift", 5), ("feature_drift_strong", 15)):
            frame = self.read(scenario).set_index("row_id").loc[current.row_id].reset_index()
            np.testing.assert_allclose(pd.to_numeric(frame.age).to_numpy(), pd.to_numeric(current.age).to_numpy() + delta, rtol=0, atol=1e-12)
            self.assertEqual(frame.income.tolist(), current.income.tolist())
            self.assertEqual(frame.sex.tolist(), current.sex.tolist())

    def test_target_and_fairness_shifts_sample_real_labels_and_preserve_male(self):
        target = self.read("target_drift")
        self.assertAlmostEqual((target.income == ">50K").mean(), .60, delta=1 / len(target))
        self.assertTrue(target.groupby("row_id").income.nunique().max() == 1)
        originals = pd.read_csv(self.processed / "current.csv", keep_default_na=False).set_index("row_id")
        self.assertEqual(target.income.tolist(), originals.loc[target.row_id, "income"].tolist())
        self.assertTrue(target.sample_id.is_unique)
        shifted = self.read("fairness_shift")
        female = shifted[shifted.sex == "Female"]
        self.assertAlmostEqual((female.income == ">50K").mean(), .40, delta=1 / len(female))
        male_rows = shifted[shifted.sex == "Male"].sort_values("row_id").reset_index(drop=True)
        original_males = pd.read_csv(self.processed / "current.csv", keep_default_na=False)
        original_males = original_males[original_males.sex == "Male"].sort_values("row_id").reset_index(drop=True)
        pd.testing.assert_frame_equal(male_rows[original_males.columns], original_males)

    def test_defect_is_explicit_postprocessing(self):
        defect = self.read("fairness_defect")
        baseline = self.read("baseline").set_index("row_id").loc[defect.row_id]
        for col in ("age", "sex", "income"):
            self.assertEqual(defect[col].tolist(), baseline[col].tolist())
        self.assertEqual(defect.model_prediction.tolist(), baseline.model_prediction.tolist())
        self.assertEqual(defect.model_probability_gt50k.tolist(), baseline.model_probability_gt50k.tolist())
        female = defect.sex.eq("Female")
        actual = defect.loc[female, "probability_gt50k"].to_numpy()
        expected = defect.loc[female, "model_probability_gt50k"].to_numpy() * .05
        self.assertTrue(abs(actual - expected).max() < 1e-12)
        self.assertGreater(int(defect.loc[female, "model_prediction"].sum()), int(defect.loc[female, "prediction"].sum()))
        self.assertEqual(defect.loc[~female, "probability_gt50k"].tolist(), baseline.loc[baseline.sex.eq("Male"), "probability_gt50k"].tolist())
        self.assertTrue(defect.loc[defect.sex == "Female", "synthetic_postprocessing_applied"].astype(bool).all())
        self.assertFalse(defect.loc[defect.sex == "Male", "synthetic_postprocessing_applied"].astype(bool).any())
        self.assertFalse(self.manifest["scenarios"]["fairness_defect"]["parameters"]["is_real_catboost_behavior"])

    def test_zero_or_missing_fairness_denominator_is_not_valid(self):
        frame = self.read("baseline")
        frame.loc[frame.sex == "Male", "prediction"] = 0
        metrics = _diagnostics(frame)
        self.assertIsNone(metrics["disparate_impact_female_over_male"])
        self.assertFalse(metrics["metric_validity"]["disparate_impact"])
        empty_group = frame[frame.sex == "Female"]
        metrics = _diagnostics(empty_group)
        self.assertFalse(metrics["group_sizes_sufficient"])
        self.assertFalse(metrics["fairness_metrics_valid"])
        self.assertIsNone(metrics["tpr_gap_female_minus_male"])

    def test_input_checksum_mismatch_is_rejected(self):
        current = self.processed / "current.csv"
        original = current.read_bytes()
        try:
            current.write_bytes(original + b"\n")
            with self.assertRaisesRegex(RuntimeError, "Checksum current.csv"):
                generate_scenarios(self.config)
        finally:
            current.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
