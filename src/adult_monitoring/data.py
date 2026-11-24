"""Download, clean, deduplicate and reproducibly split the official UCI Adult data."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd
import yaml

UCI_ZIP = "https://archive.ics.uci.edu/static/public/2/adult.zip"
ATTRIBUTION = "Adult dataset, Barry Becker and Ronny Kohavi (1996); UCI Machine Learning Repository, https://doi.org/10.24432/C5XW20; CC BY 4.0. Modified: cleaning, exact deduplication and splitting."
RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num", "marital-status",
    "occupation", "relationship", "race", "sex", "capital-gain", "capital-loss",
    "hours-per-week", "native-country", "income",
]
CAT_COLUMNS = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]
MISSING = "__MISSING__"


def clean_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim all strings; normalize Adult missing values, labels and test-file dots."""
    result = frame.copy()
    for col in result.columns:
        if pd.api.types.is_object_dtype(result[col]) or pd.api.types.is_string_dtype(result[col]):
            result[col] = result[col].astype("string").str.strip()
    if "income" in result:
        result["income"] = result["income"].str.removesuffix(".").str.strip()
        result["income"] = result["income"].replace({"<=50K": "<=50K", ">50K": ">50K"})
        invalid = sorted(set(result["income"].dropna()) - {"<=50K", ">50K"})
        if invalid:
            raise ValueError(f"Неизвестные метки income: {invalid[:5]}")
    for col in CAT_COLUMNS:
        if col in result:
            result[col] = result[col].replace({"?": pd.NA, "": pd.NA}).fillna(MISSING).astype(str)
    return result


def stratified_partition(frame: pd.DataFrame, fractions: dict[str, float], seed: int) -> dict[str, pd.DataFrame]:
    """Split each class independently into disjoint partitions with exact rounded sizes."""
    if abs(sum(fractions.values()) - 1.0) > 1e-9 or any(v <= 0 for v in fractions.values()):
        raise ValueError("Split fractions must be positive and sum to 1.")
    labels = frame["income"]
    buckets: dict[str, list[pd.DataFrame]] = {name: [] for name in fractions}
    for cls, group in frame.groupby(labels, sort=True):
        shuffled = group.sample(frac=1, random_state=seed + (1 if cls == ">50K" else 0))
        sizes = [int(len(group) * f) for f in fractions.values()]
        sizes[-1] += len(group) - sum(sizes)
        offset = 0
        for (name, _), size in zip(fractions.items(), sizes):
            buckets[name].append(shuffled.iloc[offset:offset + size])
            offset += size
    return {name: pd.concat(parts).sample(frac=1, random_state=seed).reset_index(drop=True) for name, parts in buckets.items()}


def _read_raw(path: Path, source: str, skiprows: int = 0) -> pd.DataFrame:
    frame = pd.read_csv(path, header=None, names=RAW_COLUMNS, skiprows=skiprows, skipinitialspace=True, na_filter=False)
    # Preserve the physical source line number before any filtering or deduplication.
    frame.insert(0, "row_id", [f"{source}:{i + skiprows + 1}" for i in range(len(frame))])
    frame = clean_frame(frame)
    invalid = sorted(set(frame["income"]) - {"<=50K", ">50K"})
    if invalid:
        raise RuntimeError(f"Неизвестные метки income в {source}: {invalid[:5]}")
    for col in ("age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"):
        try:
            frame[col] = pd.to_numeric(frame[col], errors="raise")
        except (ValueError, TypeError) as exc:
            raise RuntimeError(f"Некорректное числовое поле {col} в {source}.") from exc
    return frame


def _download_archive(archive: Path) -> None:
    temp_name = None
    try:
        with urllib.request.urlopen(UCI_ZIP, timeout=45) as response, tempfile.NamedTemporaryFile(dir=archive.parent, suffix=".download", delete=False) as tmp:
            temp_name = tmp.name
            while chunk := response.read(1024 * 1024):
                tmp.write(chunk)
        # Validate before making the cache visible to later runs.
        with zipfile.ZipFile(temp_name) as zf:
            if not {"adult.data", "adult.test"}.issubset(zf.namelist()):
                raise RuntimeError("Официальный архив UCI не содержит ожидаемые adult.data и adult.test.")
        os.replace(temp_name, archive)
        temp_name = None
    except (urllib.error.URLError, OSError, zipfile.BadZipFile) as exc:
        raise RuntimeError(f"Не удалось загрузить/проверить официальный архив UCI Adult ({UCI_ZIP}): {exc}") from exc
    finally:
        if temp_name:
            Path(temp_name).unlink(missing_ok=True)


def prepare(config_path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    if cfg["data"].get("target_column", "income") != "income" or cfg["data"].get("positive_label", ">50K") != ">50K":
        raise ValueError("Блок 2 поддерживает Adult target_column=income и positive_label='>50K'.")
    if cfg["data"].get("source", "https://archive.ics.uci.edu/dataset/2/adult") != "https://archive.ics.uci.edu/dataset/2/adult":
        raise ValueError("В блоке 2 источник фиксирован: https://archive.ics.uci.edu/dataset/2/adult.")
    if cfg["data"].get("missing_category_marker", MISSING) != MISSING:
        raise ValueError(f"Блок 2 поддерживает маркер пропусков {MISSING}.")
    if cfg["data"].get("duplicate_policy", "exact_cleaned_15_columns_train_priority") != "exact_cleaned_15_columns_train_priority":
        raise ValueError("Блок 2 поддерживает duplicate_policy=exact_cleaned_15_columns_train_priority.")
    paths = cfg["paths"]
    raw_dir = Path(paths["raw_data"])
    processed_dir = Path(paths["processed_data"])
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "adult.zip"
    if not archive.exists():
        _download_archive(archive)
    try:
        zipfile.ZipFile(archive).close()
    except (OSError, zipfile.BadZipFile):
        archive.unlink(missing_ok=True)
        _download_archive(archive)
    raw_checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
        if not {"adult.data", "adult.test"}.issubset(names):
            raise RuntimeError("Архив UCI Adult не содержит adult.data и adult.test.")
        data_path, test_path = raw_dir / "adult.data", raw_dir / "adult.test"
        for name, target in [("adult.data", data_path), ("adult.test", test_path)]:
            target.write_bytes(zf.read(name))
    train_source = _read_raw(data_path, "adult.data")
    test_source = _read_raw(test_path, "adult.test", skiprows=1)
    # Train source wins across duplicate records; keep original IDs from source files.
    merged = pd.concat([train_source, test_source], ignore_index=True)
    original_count = len(merged)
    duplicates = merged.duplicated(subset=RAW_COLUMNS, keep="first")
    duplicate_rows = merged.loc[duplicates]
    kept = merged.loc[~duplicates].copy()
    train_src = kept[kept.row_id.str.startswith("adult.data:")].copy()
    test_src = kept[kept.row_id.str.startswith("adult.test:")].copy()
    seed = int(cfg["project"]["random_seed"])
    splits = stratified_partition(train_src, {"train": .8, "validation": .2}, seed)
    test_split = stratified_partition(test_src, {"reference": .5, "current": .5}, seed + 17)
    splits.update(test_split)
    id_sets = {key: set(part.row_id) for key, part in splits.items()}
    overlaps = {f"{a}:{b}": len(id_sets[a] & id_sets[b]) for a in id_sets for b in id_sets if a < b}
    if any(overlaps.values()):
        raise RuntimeError(f"Row ID overlap across splits: {overlaps}")
    # Independently verify full-record disjointness (includes sex and target).
    records = {key: set(map(tuple, part[RAW_COLUMNS].itertuples(index=False, name=None))) for key, part in splits.items()}
    content_overlaps = {f"{a}:{b}": len(records[a] & records[b]) for a in records for b in records if a < b}
    if any(content_overlaps.values()):
        raise RuntimeError(f"Full-record duplicate overlap across splits: {content_overlaps}")
    for key, frame in splits.items():
        frame.to_csv(processed_dir / f"{key}.csv", index=False)
    record = {
        "dataset": "UCI Adult", "source_url": "https://archive.ics.uci.edu/dataset/2/adult",
        "archive_url": UCI_ZIP, "attribution": ATTRIBUTION, "license": "CC BY 4.0",
        "raw_archive_sha256": raw_checksum, "row_counts": {k: len(v) for k, v in splits.items()},
        "raw_rows": {"adult.data": len(train_source), "adult.test": len(test_source), "combined_before_dedup": original_count,
                     "duplicate_records_removed_train_priority": len(duplicate_rows)},
        "duplicate_policy": "Exact duplicate on all 15 cleaned source columns; retain first adult.data row, then adult.test; source row_id remains file:physical-line.",
        "split_seed": seed, "split_fractions": {"adult.data": {"train": .8, "validation": .2}, "adult.test": {"reference": .5, "current": .5}},
        "row_id_overlap_counts": overlaps, "full_record_overlap_counts": content_overlaps,
        "class_balance": {k: v.income.value_counts().to_dict() for k, v in splits.items()},
        "categorical_columns": CAT_COLUMNS, "missing_category_marker": MISSING,
        "checksums": {f"{k}.csv": hashlib.sha256((processed_dir / f"{k}.csv").read_bytes()).hexdigest() for k in splits},
    }
    (processed_dir / "data_manifest.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record
