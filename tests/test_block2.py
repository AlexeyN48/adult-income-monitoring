import unittest

import pandas as pd

from adult_monitoring.data import clean_frame, stratified_partition
from adult_monitoring.training import _binary_metrics, feature_columns


class DataPreparationTests(unittest.TestCase):
    def test_cleaning_trims_normalizes_test_label_and_marks_missing_categories(self):
        cleaned = clean_frame(pd.DataFrame({"income": [" >50K. ", " <=50K "], "workclass": [" ? ", " Private "], "sex": [" Female ", " Male "]}))
        self.assertEqual(cleaned.income.tolist(), [">50K", "<=50K"])
        self.assertEqual(cleaned.workclass.tolist(), ["__MISSING__", "Private"])
        self.assertEqual(cleaned.sex.tolist(), ["Female", "Male"])

    def test_unknown_target_label_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Неизвестные метки income"):
            clean_frame(pd.DataFrame({"income": ["maybe"]}))

    def test_stratified_splits_are_disjoint_and_reproducible(self):
        frame = pd.DataFrame({"row_id": [f"r{i}" for i in range(40)], "income": ["<=50K"] * 30 + [">50K"] * 10})
        first = stratified_partition(frame, {"train": .7, "validation": .3}, 42)
        again = stratified_partition(frame, {"train": .7, "validation": .3}, 42)
        self.assertFalse(set(first["train"].row_id) & set(first["validation"].row_id))
        self.assertEqual(first["train"].row_id.tolist(), again["train"].row_id.tolist())
        self.assertEqual(first["validation"].income.value_counts().to_dict(), {"<=50K": 9, ">50K": 3})

    def test_model_features_exclude_audit_id_and_target(self):
        frame = pd.DataFrame(columns=["age", "sex", "row_id", "income", "workclass"])
        self.assertEqual(feature_columns(frame), ["age", "workclass"])

    def test_f1_metrics_match_known_confusion_matrix(self):
        # TN=2, FP=1, FN=1, TP=2 -> each class has F1=2/3.
        metrics = _binary_metrics([0, 0, 0, 1, 1, 1], [0, 0, 1, 0, 1, 1])
        self.assertEqual(metrics["confusion_matrix"], {"tn": 2, "fp": 1, "fn": 1, "tp": 2})
        self.assertAlmostEqual(metrics["f1_positive_gt50k"], 2 / 3)
        self.assertAlmostEqual(metrics["macro_f1"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
