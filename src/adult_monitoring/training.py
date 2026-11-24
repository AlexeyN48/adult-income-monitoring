"""Train and persist the block-2 CatBoost model and monitoring-ready predictions."""
from __future__ import annotations

import hashlib
import json
import math
import platform
from importlib.metadata import version
from pathlib import Path

import pandas as pd
import yaml
from catboost import CatBoostClassifier

from .data import CAT_COLUMNS, MISSING

TARGET = "income"
POSITIVE = ">50K"


def feature_columns(frame: pd.DataFrame, target: str = TARGET, audit: str = "sex") -> list[str]:
    excluded = {"row_id", target, audit}
    return [c for c in frame.columns if c not in excluded]


def _binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float]:
    tp = sum(a == 1 and b == 1 for a, b in zip(y_true, y_pred))
    fp = sum(a == 0 and b == 1 for a, b in zip(y_true, y_pred))
    fn = sum(a == 1 and b == 0 for a, b in zip(y_true, y_pred))
    tn = sum(a == 0 and b == 0 for a, b in zip(y_true, y_pred))
    f1_pos = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else 0.0
    f1_neg = (2 * tn / (2 * tn + fp + fn)) if (2 * tn + fp + fn) else 0.0
    return {"f1_positive_gt50k": f1_pos, "macro_f1": (f1_pos + f1_neg) / 2,
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}}


def train(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["data"].get("target_column", "income") != TARGET or cfg["data"].get("positive_label", POSITIVE) != POSITIVE:
        raise ValueError("Блок 2 поддерживает Adult target_column=income и positive_label='>50K'.")
    if cfg["model"].get("task_type", "CPU").upper() != "CPU":
        raise ValueError("В этом блоке CatBoost поддерживается только task_type=CPU.")
    if cfg["model"].get("name", "CatBoostClassifier") != "CatBoostClassifier":
        raise ValueError("В блоке 2 поддерживается только CatBoostClassifier.")
    if float(cfg["monitoring"].get("decision_threshold", .5)) != .5:
        raise ValueError("Блок 2 фиксирует decision_threshold=0.5 для предсказаний и validation F1.")
    alert_margin = float(cfg["monitoring"].get("alert_f1_margin", .05))
    if not math.isfinite(alert_margin) or not 0 <= alert_margin <= 1:
        raise ValueError("monitoring.alert_f1_margin должен быть конечным числом в диапазоне [0, 1].")
    paths = cfg["paths"]
    processed_dir, model_path = Path(paths["processed_data"]), Path(paths["model"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data_manifest_path = processed_dir / "data_manifest.json"
    if not data_manifest_path.exists():
        raise RuntimeError("Нет data_manifest.json. Сначала выполните команду prepare.")
    data_manifest = json.loads(data_manifest_path.read_text(encoding="utf-8"))
    for split in ("train", "validation", "reference", "current"):
        file = processed_dir / f"{split}.csv"
        if not file.exists():
            raise RuntimeError(f"Нет {file}. Сначала выполните команду prepare.")
        expected = data_manifest.get("checksums", {}).get(f"{split}.csv")
        actual = hashlib.sha256(file.read_bytes()).hexdigest()
        if not expected or expected != actual:
            raise RuntimeError(f"Checksum {split}.csv не соответствует data_manifest.json; повторите prepare.")
    params = dict(cfg["model"].get("params", {}))
    seed = int(cfg["project"]["random_seed"])
    params.update({"iterations": int(cfg["model"].get("iterations", params.get("iterations", 400))), "depth": int(cfg["model"].get("depth", params.get("depth", 6))),
                   "learning_rate": float(cfg["model"].get("learning_rate", params.get("learning_rate", .08))), "loss_function": "Logloss",
                   "eval_metric": "F1", "task_type": "CPU", "random_seed": seed,
                   "verbose": False, "allow_writing_files": False})
    if not 1 <= params["iterations"] <= 10000 or not 1 <= params["depth"] <= 12 or not 0 < params["learning_rate"] <= 1:
        raise ValueError("CatBoost params должны удовлетворять: iterations 1..10000, depth 1..12, learning_rate (0,1].")
    train_df = pd.read_csv(processed_dir / "train.csv", keep_default_na=False)
    val_df = pd.read_csv(processed_dir / "validation.csv", keep_default_na=False)
    for name, frame in (("train", train_df), ("validation", val_df)):
        if TARGET not in frame or not set(frame[TARGET]).issubset({"<=50K", POSITIVE}):
            raise RuntimeError(f"В {name} обнаружены отсутствующие или неизвестные метки income.")
    features = feature_columns(train_df, TARGET, cfg["data"].get("audit_column", "sex"))
    if "sex" in features or "row_id" in features or TARGET in features:
        raise RuntimeError("Leakage check failed: audit, row_id and target must not be model features.")
    cat_features = [c for c in CAT_COLUMNS if c in features]
    for frame in (train_df, val_df):
        for c in cat_features:
            frame[c] = frame[c].replace({"": MISSING, "?": MISSING}).fillna(MISSING).astype(str)
    y_train = (train_df[TARGET] == POSITIVE).astype(int)
    y_val = (val_df[TARGET] == POSITIVE).astype(int)
    model = CatBoostClassifier(**params)
    model.fit(train_df[features], y_train, cat_features=cat_features, eval_set=(val_df[features], y_val), early_stopping_rounds=int(cfg["model"].get("early_stopping_rounds", 60)))
    model.save_model(str(model_path))
    val_probs = model.predict_proba(val_df[features])[:, 1]
    val_pred = (val_probs >= .5).astype(int)
    val_metrics = _binary_metrics(y_val.tolist(), val_pred.tolist())
    alert_threshold = round(max(0.0, val_metrics["f1_positive_gt50k"] - alert_margin), 6)
    prediction_summaries = {}
    for split in ("reference", "current"):
        df = pd.read_csv(processed_dir / f"{split}.csv", keep_default_na=False)
        if TARGET not in df or not set(df[TARGET]).issubset({"<=50K", POSITIVE}):
            raise RuntimeError(f"В {split} обнаружены отсутствующие или неизвестные метки income.")
        for c in cat_features:
            df[c] = df[c].replace({"": MISSING, "?": MISSING}).fillna(MISSING).astype(str)
        probabilities = model.predict_proba(df[features])
        df["probability_le50k"] = probabilities[:, 0]
        df["probability_gt50k"] = probabilities[:, 1]
        df["prediction"] = (probabilities[:, 1] >= .5).astype(int)
        output = processed_dir / f"{split}_predictions.csv"
        df.to_csv(output, index=False)
        prediction_summaries[split] = {"rows": len(df), "predicted_positive": int(df["prediction"].sum()),
                                       "probability_columns": ["probability_le50k", "probability_gt50k"],
                                       "sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    result = {
        "model_file": str(model_path), "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "seed": seed, "features": features, "categorical_features": cat_features,
        "excluded_columns": ["sex", "row_id", TARGET], "target": TARGET, "positive_label": POSITIVE,
        "decision_threshold": .5, "alert_f1_threshold": alert_threshold,
        "alert_threshold_rule": f"max(0, validation positive-class F1 - {alert_margin}); selected once on validation, never tuned on current",
        "validation": {"rows": len(val_df), "class_balance": val_df[TARGET].value_counts().to_dict(), **val_metrics},
        "predictions": prediction_summaries, "model_params": params,
        "best_iteration": int(model.get_best_iteration()),
        "fitted_model_params": model.get_all_params(),
        "versions": {"python": platform.python_version(), "catboost": version("catboost"), "pandas": version("pandas"), "pyyaml": version("PyYAML"), "evidently": version("evidently")},
    }
    out = Path(paths.get("reports", "reports"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "training_summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    manifest_path = processed_dir / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["model"] = {"file": str(model_path), "sha256": result["model_sha256"], "features": features,
                         "categorical_features": cat_features, "excluded_columns": result["excluded_columns"],
                         "params": params, "alert_f1_threshold": alert_threshold,
                         "versions": result["versions"]}
    manifest["checksums"].update({f"{k}_predictions.csv": v["sha256"] for k, v in prediction_summaries.items()})
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return result
