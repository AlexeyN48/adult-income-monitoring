"""Evidently based scenario monitoring and exporter friendly results."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import evidently
import pandas as pd
import yaml
from evidently import Dataset, Report, DataDefinition
from evidently.core.metric_types import SingleValueCalculation, SingleValueMetric
from evidently.presets import DataDriftPreset
from evidently.tests import lt, gt, lte, gte
from catboost import CatBoostClassifier
from evidently.legacy.renderers.html_widgets import counter, CounterData, WidgetSize

from .scenarios import TARGET, POSITIVE, _read_json, _sha

FEATURES = ["age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
            "occupation", "relationship", "race", "capital-gain", "capital-loss",
            "hours-per-week", "native-country"]


def _calc(frame: pd.DataFrame, min_rows: int = 30, min_positives: int = 30) -> dict:
    """All performance and fairness values are computed inside Evidently calculations."""
    y = frame[TARGET].eq(POSITIVE)
    p = frame["prediction"].astype(int).eq(1)
    tp = int((y & p).sum()); fp = int((~y & p).sum()); fn = int((y & ~p).sum())
    f1_den = 2 * tp + fp + fn
    f1 = 2 * tp / f1_den if f1_den else None
    # Binary macro-F1, explicitly averaging both labels.
    tn = int((~y & ~p).sum())
    f1_neg_den = 2 * tn + fp + fn
    f1_neg = 2 * tn / f1_neg_den if f1_neg_den else None
    result = {"rows": len(frame), "f1_positive": f1,
              "macro_f1": (f1 + f1_neg) / 2 if f1 is not None and f1_neg is not None else None,
              "f1_denominator": f1_den}
    groups = {}
    for sex in ("Female", "Male"):
        g = frame[frame["sex"] == sex]
        gy = g[TARGET].eq(POSITIVE)
        gp = g["prediction"].astype(int).eq(1)
        n, positives = len(g), int(gy.sum())
        selected = int(gp.sum()); true_positive = int((gy & gp).sum())
        sr = selected / n if n else None
        tpr = true_positive / positives if positives else None
        groups[sex] = {"rows": n, "positive_labels": positives, "predicted_positive": selected,
                       "true_positive": true_positive, "selection_rate": sr, "tpr": tpr,
                       "valid": n >= min_rows and positives >= min_positives}
    fr, mr = groups["Female"]["selection_rate"], groups["Male"]["selection_rate"]
    ft, mt = groups["Female"]["tpr"], groups["Male"]["tpr"]
    enough = all(g["valid"] for g in groups.values())
    result["groups"] = groups
    result["female_target_validity"] = len(frame[frame["sex"] == "Female"]) >= min_rows
    result["fairness"] = {
        "demographic_parity_difference": fr - mr if enough and fr is not None and mr is not None else None,
        "disparate_impact": fr / mr if enough and fr is not None and mr not in (None, 0) else None,
        "tpr_gap": ft - mt if enough and ft is not None and mt is not None else None,
        "abs_dpd": abs(fr - mr) if enough and fr is not None and mr is not None else None,
        "abs_tpr_gap": abs(ft - mt) if enough and ft is not None and mt is not None else None,
        "validity": {
            "demographic_parity_difference": enough and fr is not None and mr is not None,
            "disparate_impact": enough and fr is not None and mr not in (None, 0),
            "tpr_gap": enough and ft is not None and mt is not None,
        },
    }
    result["fairness"]["validity"]["all_groups"] = enough
    return result


class MonitoringScalar(SingleValueMetric):
    """Stable-key scalar metric; custom calculations remain in Evidently's report graph."""
    key: str
    title: str
    direction: Optional[str] = None
    threshold: Optional[float] = None
    min_group_rows: int = 30
    min_positive_labels: int = 30
    __calculation_type__ = None

    def get_bound_tests(self, context):
        if self.direction is None or self.threshold is None:
            return []
        test = lte(self.threshold) if self.direction == "lte" else gte(self.threshold) if self.direction == "gte" else lt(self.threshold) if self.direction == "lt" else gt(self.threshold)
        return [test.bind_single(self.get_fingerprint())]


class MonitoringCalculation(SingleValueCalculation[MonitoringScalar]):
    def display_name(self) -> str:
        return self.metric.title

    def calculate_value(self, dataset) -> float:
        data = dataset.as_dataframe()
        stats = _calc(data, self.metric.min_group_rows, self.metric.min_positive_labels)
        key = self.metric.key
        val = stats
        for part in key.split("."):
            val = val[part]
        return float(val) if val is not None else float("nan")

    def calculate(self, context, current_data, reference_data):
        current = self.calculate_value(current_data)
        cur = self.result(current)
        ref = None if reference_data is None else self.result(self.calculate_value(reference_data))
        if not math.isfinite(current) or (ref is not None and not math.isfinite(ref.value)):
            cur.widget = [counter(counters=[CounterData(label="Current", value=f"{current:.3f}" if math.isfinite(current) else "N/A"),
                                            CounterData(label="Reference", value=(f"{ref.value:.3f}" if math.isfinite(ref.value) else "N/A") if ref is not None else "unavailable")],
                                  title=f"{self.display_name()} — not assessable", size=WidgetSize.FULL)]
            if ref is not None: ref.widget = cur.widget
        return cur, ref


MonitoringScalar.__calculation_type__ = MonitoringCalculation


def _metrics(training: dict, cfg: dict) -> list:
    f1_threshold = float(training["alert_f1_threshold"])
    t = cfg.get("fairness_thresholds", {})
    common = {"min_group_rows": int(t.get("min_group_rows", 30)), "min_positive_labels": int(t.get("min_positive_labels", 30))}
    out = [MonitoringScalar(key="f1_positive", title="F1 >50K", direction="gte", threshold=f1_threshold, **common),
           MonitoringScalar(key="macro_f1", title="Macro-F1"),
           MonitoringScalar(key="rows", title="Sample rows")]
    for group in ("Female", "Male"):
        for key, title in (("rows", "rows"), ("positive_labels", "true positive labels"),
                           ("selection_rate", "selection rate"), ("tpr", "TPR")):
            out.append(MonitoringScalar(key=f"groups.{group}.{key}", title=f"{group}: {title}", **common))
    out += [MonitoringScalar(key="fairness.demographic_parity_difference", title="DPD Female − Male (signed)", **common),
            MonitoringScalar(key="fairness.abs_dpd", title="|DPD| check", direction="lte", threshold=float(t.get("dpd_abs", .10)), **common),
            MonitoringScalar(key="fairness.disparate_impact", title="DI Female / Male", direction="gte",
                             threshold=float(t.get("di_min", .80)), **common),
            MonitoringScalar(key="fairness.tpr_gap", title="TPR gap Female − Male (signed)", **common),
            MonitoringScalar(key="fairness.abs_tpr_gap", title="|TPR gap| check", direction="lte", threshold=float(t.get("tpr_gap_abs", .10)), **common)]
    out += [MonitoringScalar(key="fairness.validity.demographic_parity_difference", title="DPD valid", direction="gt", threshold=.5, **common),
            MonitoringScalar(key="fairness.validity.disparate_impact", title="DI valid", direction="gt", threshold=.5, **common),
            MonitoringScalar(key="fairness.validity.tpr_gap", title="TPR gap valid", direction="gt", threshold=.5, **common),
            MonitoringScalar(key="fairness.validity.all_groups", title="Fairness group-size validity", direction="gt", threshold=.5, **common)]
    out.append(MonitoringScalar(key="female_target_validity", title="Female target distribution validity", direction="gt", threshold=.5, **common))
    return out


def _plain(obj):
    if isinstance(obj, dict): return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, list): return [_plain(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj): return None
    if hasattr(obj, "value") and not isinstance(obj, (str, bytes)): return _plain(obj.value)
    if isinstance(obj, (str, int, float, bool)) or obj is None: return obj
    return str(obj)


def _extract_custom(snapshot, alert_f1: float, thresholds: dict) -> dict:
    """Read scalar values and Evidently test statuses via semantic aliases, never list offsets."""
    raw = snapshot.dict()
    metrics = {"reference": {}, "current": {}}
    tests = {}
    by_id = {}
    for item in raw.get("metrics", []):
        value = _plain(item.get("value"))
        config = item.get("config", {})
        key = config.get("key")
        if key:
            metrics["current"][key] = value
            by_id[item.get("id")] = key
            try:
                ref_result = snapshot.context.get_reference_metric_result(item.get("id"))
                metrics["reference"][key] = _plain(ref_result.value)
            except (ValueError, AttributeError, KeyError):
                metrics["reference"][key] = None
    for test in raw.get("tests", []):
        metric_id = test.get("metric_config", {}).get("metric_id")
        if metric_id in by_id:
            tests.setdefault(by_id[metric_id], []).append(str(test.get("status", "")).split(".")[-1].lower())
    return {"evidently_metric_values": metrics, "evidently_tests": tests,
            "thresholds": thresholds, "f1_alert_threshold_from_training": alert_f1}


def _calculate_scenario(config_path: str | Path, scenario: str) -> dict:
    """Validate immutable lineage/checksums, calculate one scenario, and atomically publish outputs."""
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    processed = Path(cfg["paths"]["processed_data"])
    training_path = Path(cfg["paths"].get("training_summary", Path(cfg["paths"].get("reports", "reports")) / "training_summary.json"))
    scenario_dir = Path(cfg.get("scenarios", {}).get("output_dir", str(processed / "scenarios")))
    manifest_path = scenario_dir / "manifest.json"
    manifest = _read_json(manifest_path)
    training = _read_json(training_path)
    data_manifest = _read_json(processed / "data_manifest.json")
    if manifest.get("source_sha256") != data_manifest.get("checksums", {}).get("current.csv"):
        raise RuntimeError("Scenario source checksum lineage mismatch")
    if manifest.get("reference_sha256") != data_manifest.get("checksums", {}).get("reference.csv"):
        raise RuntimeError("Scenario reference checksum lineage mismatch")
    if scenario not in manifest.get("scenarios", {}):
        raise ValueError(f"Unknown scenario: {scenario}")
    if manifest.get("model_sha256") != training.get("model_sha256") or manifest.get("model_sha256") != data_manifest["model"]["sha256"]:
        raise RuntimeError("Scenario/model checksum lineage mismatch")
    if manifest.get("feature_schema") != training.get("features") or training.get("features") != FEATURES:
        raise RuntimeError("Scenario/CBM feature schema lineage mismatch")
    model_path = Path(cfg["paths"]["model"])
    if _sha(model_path) != training.get("model_sha256"):
        raise RuntimeError("Actual CBM checksum mismatch")
    model = CatBoostClassifier(); model.load_model(str(model_path))
    if list(model.feature_names_) != training["features"]: raise RuntimeError("CBM feature schema mismatch")
    current_path = scenario_dir / manifest["scenarios"][scenario]["file"]
    out_dir = Path(cfg["paths"].get("reports", "reports")) / "evidently" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "status.json").write_text(json.dumps({"status": "calculating", "scenario": scenario,
              "attempted_at_utc": datetime.now(timezone.utc).isoformat()}, indent=2), encoding="utf-8")
    prior_path = out_dir / "monitoring.json"
    if prior_path.exists():
        try:
            old = json.loads(prior_path.read_text(encoding="utf-8"))
            old["status"] = "stale"
            old["last_attempt_at_utc"] = datetime.now(timezone.utc).isoformat()
            prior_path.write_text(json.dumps(_plain(old), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        except (OSError, ValueError):
            pass
    if _sha(current_path) != manifest["scenarios"][scenario]["sha256"]:
        raise RuntimeError("Scenario checksum mismatch; refusing monitoring")
    ref_path = processed / "reference.csv"
    if _sha(ref_path) != manifest.get("reference_sha256"):
        raise RuntimeError("Immutable reference checksum mismatch")
    current = pd.read_csv(current_path, keep_default_na=False)
    reference = pd.read_csv(ref_path, keep_default_na=False)
    ref_pred_path = processed / "reference_predictions.csv"
    expected_pred = data_manifest["checksums"].get("reference_predictions.csv")
    if not expected_pred or _sha(ref_pred_path) != expected_pred or expected_pred != training["predictions"]["reference"]["sha256"]:
        raise RuntimeError("Immutable reference_predictions checksum mismatch")
    ref_predictions = pd.read_csv(ref_pred_path, keep_default_na=False)
    if list(reference.row_id) != list(ref_predictions.row_id): raise RuntimeError("reference predictions are not aligned to reference row_id")
    reference["sample_id"] = reference.row_id
    for col in ("prediction", "probability_gt50k", "probability_le50k"):
        if col not in ref_predictions: raise RuntimeError(f"reference_predictions missing {col}")
        reference[col] = ref_predictions[col].values
    needed = set(FEATURES) | {"sex", TARGET, "prediction", "probability_gt50k", "probability_le50k", "row_id", "sample_id"}
    for name, frame in (("scenario", current), ("reference", reference)):
        if not needed.issubset(frame.columns): raise RuntimeError(f"{name} missing required schema columns")
        if frame.empty: raise RuntimeError(f"{name} dataset is empty")
        if frame["sample_id"].duplicated().any(): raise RuntimeError(f"{name} has duplicate sample_id values")
        if not set(frame[TARGET]).issubset({"<=50K", POSITIVE}): raise RuntimeError(f"{name} invalid labels")
        pred = pd.to_numeric(frame.prediction, errors="coerce")
        probs = pd.to_numeric(frame.probability_gt50k, errors="coerce")
        probs0 = pd.to_numeric(frame.probability_le50k, errors="coerce")
        if pred.isna().any() or not set(pred.unique()).issubset({0, 1}): raise RuntimeError(f"{name} nonbinary predictions")
        if probs.isna().any() or probs0.isna().any() or (~probs.between(0, 1)).any() or (~probs0.between(0, 1)).any(): raise RuntimeError(f"{name} invalid probabilities")
        if not ((probs + probs0 - 1).abs() < 1e-7).all(): raise RuntimeError(f"{name} probabilities do not sum to one")
        decision_threshold = float(training.get("decision_threshold", .5))
        if not (pred.astype(int).to_numpy() == (probs.to_numpy() >= decision_threshold).astype(int)).all():
            raise RuntimeError(f"{name} predictions disagree with effective positive-class probabilities")
        if not set(frame.sex).issubset({"Female", "Male"}): raise RuntimeError(f"{name} invalid/missing audit group")
        if not frame[FEATURES].notna().all().all(): raise RuntimeError(f"{name} feature missing values")
    if len(current) != int(manifest["scenarios"][scenario]["rows"]): raise RuntimeError("Scenario row-count mismatch")
    # monitoring columns for Evidently presets. No diagnostic manifest numbers enter calculations.
    ref = reference[FEATURES + ["sex", TARGET]].copy()
    cur = current[FEATURES + ["sex", TARGET]].copy()
    ref["prediction"] = reference.prediction.astype(int).astype(str)
    cur["prediction"] = current.prediction.astype(int).astype(str)
    cfg_monitor = cfg.get("monitoring", {})
    fairness_cfg = cfg_monitor.get("fairness_thresholds", {})
    thresholds = {"numeric_wasserstein": float(cfg_monitor.get("numeric_drift_threshold", .1)),
                  "categorical_js": float(cfg_monitor.get("categorical_drift_threshold", .1)),
                  "dpd_abs": float(fairness_cfg.get("dpd_abs", .10)),
                  "di_min": float(fairness_cfg.get("di_min", .80)),
                  "tpr_gap_abs": float(fairness_cfg.get("tpr_gap_abs", .10)),
                  "min_group_rows": int(fairness_cfg.get("min_group_rows", 30)),
                  "min_positive_labels": int(fairness_cfg.get("min_positive_labels", 30))}
    for key in ("numeric_wasserstein", "categorical_js", "dpd_abs", "di_min", "tpr_gap_abs"):
        value = thresholds[key]
        if not math.isfinite(value) or value < 0 or value > 1:
            raise ValueError(f"monitoring threshold {key} must be finite and in [0, 1]")
    if thresholds["min_group_rows"] < 1 or thresholds["min_positive_labels"] < 1:
        raise ValueError("fairness minimum group sizes must be positive integers")
    # Explicit feature-only drift: 13 training features, never sex/target/IDs/diagnostics.
    feature_report = DataDriftPreset(columns=FEATURES, num_method="wasserstein", cat_method="jensenshannon",
                                     num_threshold=thresholds["numeric_wasserstein"],
                                     cat_threshold=thresholds["categorical_js"], drift_share=.5)
    target_report = DataDriftPreset(columns=[TARGET], cat_method="jensenshannon",
                                   cat_threshold=thresholds["categorical_js"], drift_share=.5)
    # Include compact built-in group distributions (sex, target, prediction) to render visual comparisons.
    group_report = DataDriftPreset(columns=["sex", "prediction"], cat_method="jensenshannon",
                                   cat_threshold=thresholds["categorical_js"], drift_share=.5)
    metrics = [feature_report, target_report, group_report] + _metrics(training, cfg_monitor)
    report = Report(metrics, include_tests=True, metadata={"scenario": scenario, "purpose": "Adult Income monitoring"})
    categorical_features = training.get("categorical_features", data_manifest["model"].get("categorical_features", []))
    if not set(categorical_features).issubset(FEATURES): raise RuntimeError("Categorical feature lineage is invalid")
    numeric_features = [c for c in FEATURES if c not in categorical_features]
    data_definition = DataDefinition(numerical_columns=numeric_features,
                                     categorical_columns=categorical_features + ["sex", TARGET, "prediction"])
    snapshot = report.run(Dataset.from_pandas(cur, data_definition=data_definition),
                          Dataset.from_pandas(ref, data_definition=data_definition))
    female_current = current[current.sex.eq("Female")][[TARGET]].copy()
    female_reference = reference[reference.sex.eq("Female")][[TARGET]].copy()
    female_valid = len(female_current) >= thresholds["min_group_rows"] and len(female_reference) >= thresholds["min_group_rows"]
    female_snapshot = None
    if female_valid:
        female_snapshot = Report([DataDriftPreset(columns=[TARGET], cat_method="jensenshannon",
                                       cat_threshold=thresholds["categorical_js"], drift_share=.5)], include_tests=True).run(
            Dataset.from_pandas(female_current, data_definition=DataDefinition(categorical_columns=[TARGET])),
            Dataset.from_pandas(female_reference, data_definition=DataDefinition(categorical_columns=[TARGET])))
    custom = _extract_custom(snapshot, float(training["alert_f1_threshold"]), thresholds)
    # Age flag is independently explicit even when aggregate drift share is below 50%.
    ev = _plain(snapshot.dict())
    age_metric = next((m for m in ev.get("metrics", []) if "ValueDrift" in str(m.get("metric_name")) and m.get("config", {}).get("column") == "age"), None)
    test_by_id = {t.get("metric_config", {}).get("metric_id"): str(t.get("status", "")).split(".")[-1].lower()
                  for t in ev.get("tests", [])}
    feature_drift = {}
    for item in ev.get("metrics", []):
        if "ValueDrift" in str(item.get("metric_name")):
            col = item.get("config", {}).get("column")
            if col in FEATURES:
                feature_drift[col] = {"score": item.get("value"), "test_status": test_by_id.get(item.get("id"), "missing"),
                                      "method": item.get("config", {}).get("method"), "threshold": item.get("config", {}).get("threshold")}
    count_reports = [m for m in ev.get("metrics", []) if "DriftedColumnsCount" in str(m.get("metric_name"))]
    feature_aggregate = next((m for m in count_reports if m.get("config", {}).get("columns") == FEATURES), None)
    target_drift = next((m for m in ev.get("metrics", []) if "ValueDrift" in str(m.get("metric_name")) and m.get("config", {}).get("column") == TARGET), None)
    if len(feature_drift) != len(FEATURES): raise RuntimeError("Evidently feature drift report is missing training features")
    def drift_summary(report_snapshot: Optional[dict], column: str) -> dict:
        if report_snapshot is None: return {"status": "not_assessable", "validity": False, "score": None, "method": "jensenshannon", "threshold": thresholds["categorical_js"]}
        metric = next((m for m in report_snapshot.get("metrics", []) if "ValueDrift" in str(m.get("metric_name")) and m.get("config", {}).get("column") == column), None)
        if metric is None: return {"status": "not_assessable", "validity": False, "score": None, "method": "jensenshannon", "threshold": thresholds["categorical_js"]}
        status = next((str(t.get("status", "")).split(".")[-1].lower() for t in report_snapshot.get("tests", []) if t.get("metric_config", {}).get("metric_id") == metric.get("id")), "missing")
        return {"status": status, "validity": status != "missing", "score": metric.get("value"),
                "method": metric.get("config", {}).get("method"), "threshold": metric.get("config", {}).get("threshold")}
    aggregate_count = (feature_aggregate or {}).get("value", {}).get("count")
    aggregate_share = (feature_aggregate or {}).get("value", {}).get("share")
    aggregate_status = next((str(t.get("status", "")).split(".")[-1].lower() for t in ev.get("tests", [])
                             if t.get("metric_config", {}).get("metric_id") == (feature_aggregate or {}).get("id")), "missing")
    # snapshot data contains actual values as well; keep the full structured snapshot for future consumers.
    result = {"schema_version": 1, "status": "success", "scenario": scenario,
              "calculated_at_utc": datetime.now(timezone.utc).isoformat(),
              "source": {"scenario_file": current_path.as_posix(), "scenario_sha256": _sha(current_path),
                         "reference_file": ref_path.as_posix(), "reference_sha256": _sha(ref_path),
                         "scenario_manifest_sha256": _sha(processed / "scenarios" / "manifest.json"),
                         "model_sha256": training["model_sha256"], "feature_schema": FEATURES},
              "versions": {"python": platform.python_version(), "evidently": evidently.__version__,
                           "pandas": pd.__version__},
              "formulas": {"dpd": "selection_rate(Female) - selection_rate(Male)",
                           "di": "selection_rate(Female) / selection_rate(Male)",
                           "tpr_gap": "TPR(Female) - TPR(Male)",
                           "selection_rate": "predicted positive / group rows",
                           "TPR": "true positive predictions / true positive labels",
                           "undefined": "null plus validity failure; never interpreted as zero"},
              "thresholds": thresholds,
              "evidently": custom, "feature_drift": feature_drift,
              "feature_drift_aggregate": {"drifted_feature_count": aggregate_count, "drifted_feature_share": aggregate_share,
                                          "threshold_share": .5, "test_status": aggregate_status},
              "target_drift": drift_summary(ev, TARGET),
              "female_target_drift": {"status": "assessed" if female_valid else "not_assessable",
                                      "validity": female_valid,
                                      **drift_summary(_plain(female_snapshot.dict()) if female_snapshot is not None else None, TARGET)},
              "age_drift_metric": age_metric,
              "age_drift_status": test_by_id.get(age_metric.get("id"), "missing") if age_metric else "missing",
              "snapshot": ev}
    # Stage both files, then replace only after report and structured output are complete.
    stage = out_dir / ".staging"
    stage.mkdir(exist_ok=True)
    try:
        snapshot.save_html(str(stage / "report.html"))
        if female_snapshot is not None: female_snapshot.save_html(str(stage / "female_target_report.html"))
        else: (stage / "female_target_report.html").write_text("<html><body><h1>Female target drift: not assessable</h1><p>Female subgroup is missing or below configured minimum.</p></body></html>", encoding="utf-8")
        (stage / "snapshot.json").write_text(json.dumps(ev, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        (stage / "monitoring.json").write_text(json.dumps(_plain(result), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        for filename in ("report.html", "female_target_report.html", "snapshot.json", "monitoring.json"):
            (stage / filename).replace(out_dir / filename)
        status_tmp = stage / "status.json"
        status_tmp.write_text(json.dumps({"status": "success", "scenario": scenario,
                  "calculated_at_utc": result["calculated_at_utc"]}, indent=2), encoding="utf-8")
        status_tmp.replace(out_dir / "status.json")
    except Exception as exc:
        (out_dir / "status.json").write_text(json.dumps({"status": "failed", "scenario": scenario,
                  "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}, indent=2), encoding="utf-8")
        raise
    return {"scenario": scenario, "status": "success", "output_dir": str(out_dir),
            "f1_positive": custom["evidently_metric_values"]["current"].get("f1_positive"),
            "macro_f1": custom["evidently_metric_values"]["current"].get("macro_f1"),
            "rows": custom["evidently_metric_values"]["current"].get("rows")}


def calculate_scenario(config_path: str | Path, scenario: str) -> dict:
    """Public single-scenario entry point; failed attempts never leave a fresh success status."""
    if not isinstance(scenario, str) or not scenario.isidentifier():
        raise ValueError("Scenario must be a simple alias such as baseline or feature_drift")
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    out_dir = Path(cfg["paths"].get("reports", "reports")) / "evidently" / scenario
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        return _calculate_scenario(config_path, scenario)
    except Exception as exc:
        failed = {"status": "failed", "scenario": scenario,
                  "failed_at_utc": datetime.now(timezone.utc).isoformat(), "error": str(exc)}
        (out_dir / "status.json").write_text(json.dumps(failed, indent=2, ensure_ascii=False), encoding="utf-8")
        prior = out_dir / "monitoring.json"
        if prior.exists():
            try:
                old = json.loads(prior.read_text(encoding="utf-8")); old["status"] = "stale"
                old["last_attempt_at_utc"] = failed["failed_at_utc"]
                prior.write_text(json.dumps(_plain(old), indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
            except (OSError, ValueError): pass
        raise


def monitor(config_path: str | Path, scenario: Optional[str] = None) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if scenario not in (None, "all"):
        return calculate_scenario(config_path, scenario)
    scenario_dir = Path(cfg.get("scenarios", {}).get("output_dir", str(Path(cfg["paths"]["processed_data"]) / "scenarios")))
    manifest = _read_json(scenario_dir / "manifest.json")
    results = []
    for name in manifest["scenarios"]:
        results.append(calculate_scenario(config_path, name))
    return {"status": "success", "scenario_count": len(results), "results": results}
