"""Reproducible, auditable monitoring scenarios built from immutable block-2 data."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier

TARGET = "income"
POSITIVE = ">50K"
LABELS = {"<=50K", POSITIVE}
PREDICTION_COLUMNS = ["model_probability_gt50k", "model_prediction", "probability_gt50k",
                      "probability_le50k", "prediction"]
MIN_EVALUATION_SIZE = 30


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_inputs(cfg: dict) -> tuple[dict, dict, dict[str, pd.DataFrame]]:
    paths = cfg["paths"]
    processed = Path(paths["processed_data"])
    model_path = Path(paths["model"])
    reports = Path(paths.get("reports", "reports"))
    data_manifest = _read_json(processed / "data_manifest.json")
    training = _read_json(reports / "training_summary.json")
    expected_model = data_manifest.get("model", {}).get("sha256")
    if not expected_model or expected_model != training.get("model_sha256") or _sha(model_path) != expected_model:
        raise RuntimeError("CatBoost checksum не соответствует data_manifest/training_summary; входы блока 2 изменены.")
    feature_names = training.get("features")
    if not isinstance(feature_names, list) or not feature_names or TARGET in feature_names or "sex" in feature_names or "row_id" in feature_names:
        raise RuntimeError("Некорректная feature schema в training_summary.json.")
    if feature_names != data_manifest.get("model", {}).get("features"):
        raise RuntimeError("Feature schema в training_summary.json и data_manifest.json различается.")
    frames = {}
    input_names = ("train", "validation", "reference", "current")
    for split in input_names:
        path = processed / f"{split}.csv"
        expected = data_manifest.get("checksums", {}).get(f"{split}.csv")
        if not expected or _sha(path) != expected:
            raise RuntimeError(f"Checksum {split}.csv не соответствует data_manifest.json.")
        frame = pd.read_csv(path, keep_default_na=False)
        required = {"row_id", "sex", TARGET, *feature_names}
        if not required.issubset(frame.columns) or not set(frame[TARGET]).issubset(LABELS):
            raise RuntimeError(f"{split}.csv не соответствует ожидаемой схеме или содержит неизвестные income labels.")
        if not set(frame["sex"]).issubset({"Female", "Male"}):
            raise RuntimeError(f"{split}.csv содержит неизвестное значение audit group sex.")
        if frame["row_id"].duplicated().any():
            raise RuntimeError(f"row_id повторяется в {split}.csv.")
        frames[split] = frame
    for split in ("reference", "current"):
        pred_path = processed / f"{split}_predictions.csv"
        expected = data_manifest.get("checksums", {}).get(f"{split}_predictions.csv")
        summary_expected = training.get("predictions", {}).get(split, {}).get("sha256")
        if not expected or expected != summary_expected or _sha(pred_path) != expected:
            raise RuntimeError(f"Checksum {split}_predictions.csv не соответствует manifest/training_summary.")
        pred = pd.read_csv(pred_path, keep_default_na=False)
        if list(pred["row_id"]) != list(frames[split]["row_id"]) or not set(PREDICTION_COLUMNS[-3:]).issubset(pred.columns):
            raise RuntimeError(f"{split}_predictions.csv не выровнен по row_id или не содержит прогнозы блока 2.")
    # Verify disjoint source records across all prepared partitions, including repeated full records.
    for i, left in enumerate(input_names):
        for right in input_names[i + 1:]:
            if set(frames[left].row_id) & set(frames[right].row_id):
                raise RuntimeError(f"Пересечение row_id между {left} и {right}.")
            common = set(map(tuple, frames[left][[ *feature_names, "sex", TARGET]].itertuples(index=False, name=None))) & \
                     set(map(tuple, frames[right][[ *feature_names, "sex", TARGET]].itertuples(index=False, name=None)))
            if common:
                raise RuntimeError(f"Пересечение исходных записей между {left} и {right}.")
    return data_manifest, training, frames


def _sample_group(group: pd.DataFrame, n: int, positives: int, rng: int) -> pd.DataFrame:
    if positives < 0 or positives > n:
        raise ValueError("Целевой размер positive class должен быть между 0 и размером выборки.")
    parts = []
    for label, count, offset in ((POSITIVE, positives, 1), ("<=50K", n - positives, 2)):
        pool = group[group[TARGET] == label]
        if count and pool.empty:
            raise ValueError(f"Нельзя сформировать сценарий: в исходной подгруппе нет класса {label}.")
        if count:
            parts.append(pool.sample(n=count, replace=count > len(pool), random_state=rng + offset))
    if not parts:
        raise ValueError("Сценарий не может быть пустым.")
    return pd.concat(parts).sample(frac=1, random_state=rng + 3).reset_index(drop=True)


def _sampling_ids(frame: pd.DataFrame) -> pd.DataFrame:
    counts: dict[str, int] = {}
    sample_ids = []
    for row_id in frame["row_id"].astype(str):
        draw = counts.get(row_id, 0)
        counts[row_id] = draw + 1
        sample_ids.append(f"{row_id}#draw={draw}")
    frame = frame.copy()
    frame.insert(1, "sample_id", sample_ids)
    return frame


def _diagnostics(frame: pd.DataFrame) -> dict:
    pred = frame["prediction"].astype(int)
    truth = (frame[TARGET] == POSITIVE).astype(int)
    tp = int(((pred == 1) & (truth == 1)).sum())
    fp = int(((pred == 1) & (truth == 0)).sum())
    fn = int(((pred == 0) & (truth == 1)).sum())
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None
    groups = {}
    for sex in ("Female", "Male"):
        subset = frame[frame["sex"] == sex]
        n, positives = len(subset), int((subset[TARGET] == POSITIVE).sum())
        group_tp = int(((subset["prediction"] == 1) & (subset[TARGET] == POSITIVE)).sum())
        groups[sex] = {"rows": n, "true_positive_labels": positives,
                       "selection_rate": float(subset["prediction"].mean()) if n else None,
                       "tpr": group_tp / positives if positives else None,
                       "valid_for_evaluation": n >= MIN_EVALUATION_SIZE and positives >= MIN_EVALUATION_SIZE}
    female_rate, male_rate = groups["Female"]["selection_rate"], groups["Male"]["selection_rate"]
    di = female_rate / male_rate if female_rate is not None and male_rate else None
    ftpr, mtpr = groups["Female"]["tpr"], groups["Male"]["tpr"]
    group_sizes_sufficient = all(groups[s]["valid_for_evaluation"] for s in groups)
    metric_validity = {"demographic_parity_difference": group_sizes_sufficient and female_rate is not None and male_rate is not None,
                       "disparate_impact": group_sizes_sufficient and female_rate is not None and male_rate is not None and male_rate > 0,
                       "tpr_gap": group_sizes_sufficient and ftpr is not None and mtpr is not None}
    return {"note": "Диагностика генератора; полноценный Evidently анализ будет добавлен в следующем блоке.",
            "rows": len(frame), "class_counts": frame[TARGET].value_counts().reindex(["<=50K", POSITIVE], fill_value=0).astype(int).to_dict(),
            "f1_positive_gt50k": f1, "groups": groups,
            "demographic_parity_difference_female_minus_male": female_rate - male_rate if female_rate is not None and male_rate is not None else None,
            "disparate_impact_female_over_male": di,
            "tpr_gap_female_minus_male": ftpr - mtpr if ftpr is not None and mtpr is not None else None,
            "group_sizes_sufficient": group_sizes_sufficient, "metric_validity": metric_validity,
            "fairness_metrics_valid": all(metric_validity.values())}


def generate_scenarios(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    _, training, inputs = _verify_inputs(cfg)
    settings = cfg.get("scenarios", {})
    seed = int(settings.get("seed", cfg["project"]["random_seed"]))
    threshold = float(training.get("decision_threshold", .5))
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise RuntimeError("Некорректный decision_threshold в training_summary.json.")
    current = inputs["current"].copy()
    model = CatBoostClassifier()
    model.load_model(str(Path(cfg["paths"]["model"])))
    features = training["features"]
    if list(model.feature_names_) != features:
        raise RuntimeError("Feature schema CatBoost CBM не совпадает с training_summary.json.")

    # Materialize every scenario from current independently, then run the actual saved CatBoost model.
    baseline_features = current.copy()
    scenarios: dict[str, tuple[pd.DataFrame, dict]] = {}
    scenarios["baseline"] = (baseline_features, {"kind": "unchanged_current", "seed": seed})
    age_delta = int(settings.get("feature_drift", {}).get("age_delta", 5))
    strong_delta = int(settings.get("feature_drift_strong", {}).get("age_delta", 15))
    if age_delta <= 0 or strong_delta <= 0:
        raise ValueError("Сдвиги возраста должны быть положительными целыми числами.")
    for name, delta in (("feature_drift", age_delta), ("feature_drift_strong", strong_delta)):
        changed = current.copy()
        changed["age"] = pd.to_numeric(changed["age"], errors="raise") + delta
        scenarios[name] = (changed, {"kind": "feature_shift", "feature": "age", "delta_years": delta,
                                     "all_rows_changed": True, "labels_preserved_by_row_id": True})

    n = len(current)
    target_cfg = settings.get("target_drift", {})
    target_rate = float(target_cfg.get("positive_rate", .60))
    if not 0 < target_rate < 1:
        raise ValueError("scenarios.target_drift.positive_rate должен быть между 0 и 1.")
    target_pos = round(n * target_rate)
    target_frame = _sample_group(current, n, target_pos, seed + 100)
    scenarios["target_drift"] = (target_frame, {"kind": "label_stratified_resampling", "target_positive_rate": target_rate,
                                               "requested_positive_rows": target_pos, "seed": seed + 100,
                                               "true_labels_rewritten": False})

    fairness_cfg = settings.get("fairness_shift", {})
    female_rate = float(fairness_cfg.get("female_positive_rate", .03))
    female_size = int(fairness_cfg.get("female_rows", int((current.sex == "Female").sum())))
    female = current[current.sex == "Female"]
    male = current[current.sex == "Male"].copy()
    if not 0 < female_rate < 1 or female_size < 1:
        raise ValueError("Некорректные scenarios.fairness_shift параметры.")
    female_pos = round(female_size * female_rate)
    if female_pos < MIN_EVALUATION_SIZE:
        raise ValueError(f"Fairness shift создаст только {female_pos} Female positive labels; минимум {MIN_EVALUATION_SIZE}.")
    female_sample = _sample_group(female, female_size, female_pos, seed + 200)
    shifted = pd.concat([male, female_sample]).sample(frac=1, random_state=seed + 201).reset_index(drop=True)
    scenarios["fairness_shift"] = (shifted, {"kind": "sex_specific_true_label_prevalence_resampling", "female_positive_rate_target": female_rate,
                                             "female_rows": female_size, "female_positive_rows": female_pos,
                                             "male_rows_unchanged": len(male), "male_records_unchanged": True,
                                             "seed": seed + 200, "true_labels_rewritten": False})
    scenarios["recovery"] = (baseline_features.copy(), {"kind": "exact_baseline_recovery", "seed": seed})

    # Defect is intentionally synthetic post-processing, never represented as a CatBoost output.
    defect_frame = current.copy()
    # Predictions are filled after model inference; effective probabilities are filled below.
    scenarios["fairness_defect"] = (defect_frame, {"kind": "synthetic_decision_postprocessing_defect", "affected_group": "Female",
                                                    "female_probability_multiplier": float(settings.get("fairness_defect", {}).get("female_probability_multiplier", .05)),
                                                    "effective_decision_threshold": threshold,
                                                    "features_and_true_labels_unchanged": True,
                                                    "is_real_catboost_behavior": False})
    multiplier = scenarios["fairness_defect"][1]["female_probability_multiplier"]
    if not math.isfinite(multiplier) or not 0 <= multiplier <= 1:
        raise ValueError("fairness_defect female_probability_multiplier должен быть в диапазоне [0,1].")

    out_dir = Path(settings.get("output_dir", str(Path(cfg["paths"]["processed_data"]) / "scenarios")))
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "source": "immutable current.csv", "source_sha256": _sha(Path(cfg["paths"]["processed_data"]) / "current.csv"),
                "reference_sha256": _sha(Path(cfg["paths"]["processed_data"]) / "reference.csv"),
                "model_sha256": training["model_sha256"], "feature_schema": features,
                "target": TARGET, "positive_label": POSITIVE, "decision_threshold": threshold,
                "probability_contract": {"model_probability_gt50k/model_prediction": "CatBoost predictions on scenario features before any synthetic decision post-processing",
                                         "probability_gt50k/probability_le50k/prediction": "effective values consumed by downstream monitoring; equal model values except in explicitly synthetic fairness_defect"},
                "sample_id_contract": "row_id identifies source; sample_id is unique per emitted draw and uses #draw=N for repeated source rows; draw 0 for unique rows.",
                "diagnostics_note": "Диагностика генератора; Evidently будет рассчитан в следующем блоке.", "scenarios": {}}
    for name, (frame, params) in scenarios.items():
        frame = _sampling_ids(frame.reset_index(drop=True))
        probs = model.predict_proba(frame[features])[:, 1]
        model_pred = (probs >= threshold).astype(int)
        effective_prob = probs.copy()
        effective_pred = model_pred.copy()
        if name == "fairness_defect":
            female_mask = frame.sex.eq("Female").to_numpy()
            effective_prob[female_mask] *= multiplier
            effective_pred = (effective_prob >= threshold).astype(int)
        frame["model_probability_gt50k"] = probs
        frame["model_prediction"] = model_pred
        frame["probability_gt50k"] = effective_prob
        frame["probability_le50k"] = 1.0 - effective_prob
        frame["prediction"] = effective_pred
        frame["synthetic_postprocessing_applied"] = name == "fairness_defect" and frame.sex.eq("Female")
        output = out_dir / f"{name}.csv"
        frame.to_csv(output, index=False)
        row_dupes = int(frame.row_id.duplicated(keep=False).sum())
        manifest["scenarios"][name] = {"file": output.name, "sha256": _sha(output), "rows": len(frame),
                                        "parameters": params, "duplicate_row_id_rows": row_dupes,
                                        "duplicate_row_id_excess_draws": int(len(frame) - frame.row_id.nunique()),
                                        "group_class_counts": {sex: frame[frame.sex == sex][TARGET].value_counts().reindex(["<=50K", POSITIVE], fill_value=0).astype(int).to_dict() for sex in ("Female", "Male")},
                                        "diagnostics": _diagnostics(frame)}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
