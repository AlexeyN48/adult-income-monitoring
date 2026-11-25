"""Atomic, low-cardinality Prometheus exporter for completed Evidently runs."""
from __future__ import annotations

import argparse
import json
import logging
import math
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from .monitoring import FEATURES, calculate_scenario

SCENARIOS = ("baseline", "feature_drift", "feature_drift_strong", "target_drift",
             "fairness_shift", "fairness_defect", "recovery")
GROUPS = ("Female", "Male")
TTL_SECONDS = 180
LOGGER = logging.getLogger(__name__)
MAX_FUTURE_SECONDS = 60


def finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def load_completed(report_dir: Path, scenario: str, status_path: Path | None = None) -> dict:
    """Accept only a fully matching successful status/report pair."""
    if scenario not in SCENARIOS:
        raise ValueError("unknown scenario")
    status_path = status_path or report_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    result = json.loads((report_dir / "monitoring.json").read_text(encoding="utf-8"))
    stamp = result.get("calculated_at_utc")
    if status.get("status") != "success" or result.get("status") != "success":
        raise ValueError("report/status is not successful")
    if status.get("scenario") != scenario or result.get("scenario") != scenario:
        raise ValueError("scenario mismatch")
    if not stamp or status.get("calculated_at_utc") != stamp:
        raise ValueError("report/status timestamp mismatch")
    parsed = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include timezone")
    if result.get("schema_version") != 1 or "evidently" not in result:
        raise ValueError("unsupported report schema")
    if parsed.timestamp() > time.time() + MAX_FUTURE_SECONDS:
        raise ValueError("report timestamp is in the future")
    return result


def metrics_from_result(result: dict, scenario: str) -> dict:
    values = result["evidently"]["evidently_metric_values"]
    current, reference = values["current"], values["reference"]
    features = result["feature_drift"]
    target, female_target = result["target_drift"], result["female_target_drift"]
    validity_keys = ("fairness.validity.all_groups", "fairness.validity.demographic_parity_difference",
                     "fairness.validity.disparate_impact", "fairness.validity.tpr_gap")
    fairness_validity = all(current.get(key) == 1.0 for key in validity_keys)
    if set(features) != set(FEATURES):
        raise ValueError("feature drift schema is incomplete or has unknown features")
    if any(features[f].get("test_status") not in ("success", "fail") for f in FEATURES):
        raise ValueError("unknown feature drift status")
    for key in validity_keys:
        if current.get(key) not in (0.0, 1.0):
            raise ValueError(f"missing fairness validity: {key}")
    f1 = finite(current.get("f1_positive")); threshold = finite(result["evidently"].get("f1_alert_threshold_from_training"))
    drift_share = finite(result["feature_drift_aggregate"].get("drifted_feature_share"))
    if f1 is None or not 0 <= f1 <= 1 or threshold is None or not 0 <= threshold <= 1:
        raise ValueError("F1 or its threshold is missing or out of range")
    if drift_share is None or not 0 <= drift_share <= 1:
        raise ValueError("drifted feature share is missing or out of range")
    for name, entry in features.items():
        score, threshold_value = finite(entry.get("score")), finite(entry.get("threshold"))
        if score is None or score < 0 or threshold_value is None or threshold_value < 0:
            raise ValueError(f"invalid drift values for {name}")
    if target.get("status") not in ("success", "fail") or female_target.get("status") not in ("success", "fail"):
        raise ValueError("unknown target drift status")
    thresholds = result.get("thresholds", {})
    for key in ("di_min", "dpd_abs", "tpr_gap_abs"):
        val = finite(thresholds.get(key))
        if val is None or val < 0:
            raise ValueError(f"missing or invalid fairness threshold: {key}")
    series: dict[str, Any] = {"scenario": scenario, "calculated_at_utc": result["calculated_at_utc"],
        "f1": f1, "f1_threshold": threshold,
        "drift_share": drift_share,
        "target_drift": (1.0 if target.get("status") == "fail" else 0.0) if target.get("validity") else None,
        "female_target_drift": (1.0 if female_target.get("status") == "fail" else 0.0) if female_target.get("validity") else None,
        "fairness_valid": 1.0 if fairness_validity else 0.0,
        "di": finite(current.get("fairness.disparate_impact")),
        "di_reference": finite(reference.get("fairness.disparate_impact")),
        "dpd": finite(current.get("fairness.abs_dpd")), "tpr_gap": finite(current.get("fairness.abs_tpr_gap")),
        "di_min": finite(result.get("thresholds", {}).get("di_min")),
        "dpd_max": finite(result.get("thresholds", {}).get("dpd_abs")),
        "tpr_gap_max": finite(result.get("thresholds", {}).get("tpr_gap_abs")),
        "feature": {}, "groups": {}}
    for feature in FEATURES:
        entry = features.get(feature, {})
        series["feature"][feature] = (finite(entry.get("score")), finite(entry.get("threshold")),
                                       1.0 if entry.get("test_status") == "fail" else 0.0 if entry.get("test_status") == "success" else None)
    for group in GROUPS:
        prefix = f"groups.{group}."
        series["groups"][group] = (finite(current.get(prefix + "selection_rate")), finite(current.get(prefix + "tpr")))
        if any(x is None or not 0 <= x <= 1 for x in series["groups"][group]):
            fairness_validity = False
    if fairness_validity:
        for metric in ("di", "dpd", "tpr_gap"):
            x = series[metric]
            if x is None or (metric != "di" and not 0 <= x <= 1) or (metric == "di" and x < 0):
                fairness_validity = False
    # Group and fairness metrics are undefined when validity fails; expose no misleading zeroes.
    if not fairness_validity:
        series["fairness_valid"] = 0.0
        series["di"] = series["dpd"] = series["tpr_gap"] = None
        series["groups"] = {g: (None, None) for g in GROUPS}
    return series


class Snapshot:
    """Copy-on-write metrics state. Readers see a complete old or new snapshot."""
    def __init__(self):
        self._lock = threading.Lock()
        self._update_lock = threading.Lock()
        self._value: dict | None = None
        self._errors = 0
        self._active = "baseline"
        self._failed = False

    def publish(self, value: dict) -> None:
        with self._lock:
            self._value = value
            self._active = value["scenario"]
            self._failed = False

    def fail(self) -> None:
        with self._lock:
            self._errors += 1
            self._failed = True

    def current(self) -> tuple[dict | None, int, str, bool, bool]:
        with self._lock:
            return self._value, self._errors, self._active, self._update_lock.locked(), self._failed

    def begin_update(self) -> bool:
        return self._update_lock.acquire(blocking=False)

    def end_update(self) -> None:
        self._update_lock.release()


def render(snapshot: Snapshot, now: float | None = None) -> str:
    value, errors, active, updating, failed = snapshot.current()
    now = time.time() if now is None else now
    stamp = 0.0
    fresh = False
    if value:
        try:
            stamp = datetime.fromisoformat(value["calculated_at_utc"].replace("Z", "+00:00")).timestamp()
            fresh = 0 <= now - stamp <= TTL_SECONDS and not failed
        except (ValueError, TypeError):
            stamp = 0.0
    lines = ["# HELP ml_monitoring_available Whether a valid, fresh completed report is loaded.",
             "# TYPE ml_monitoring_available gauge", f"ml_monitoring_available {int(bool(value) and fresh)}",
             "# HELP ml_monitoring_last_success_timestamp_seconds UTC timestamp of the last accepted completed report.",
             "# TYPE ml_monitoring_last_success_timestamp_seconds gauge", f"ml_monitoring_last_success_timestamp_seconds {stamp}",
             "# HELP ml_monitoring_errors_total Rejected report or failed scenario calculations.",
             "# TYPE ml_monitoring_errors_total counter", f"ml_monitoring_errors_total {errors}",
             "# HELP ml_scenario_update_in_progress Scenario calculation is running.",
             "# TYPE ml_scenario_update_in_progress gauge", f"ml_scenario_update_in_progress {int(updating)}",
             "# HELP ml_active_scenario_info Currently selected scenario.", "# TYPE ml_active_scenario_info gauge",
             f'ml_active_scenario_info{{scenario="{active}"}} 1']
    if value and fresh:
        for name, metric, helptext in (("ml_f1_score", "f1", "Positive-class F1 from Evidently."),
            ("ml_f1_threshold", "f1_threshold", "F1 alert threshold from training summary."),
            ("ml_drifted_feature_share", "drift_share", "Share of drifted features."),
            ("ml_target_drift_detected", "target_drift", "Evidently target drift test failed."),
            ("ml_female_target_drift_detected", "female_target_drift", "Evidently female subgroup target drift test failed."),
            ("ml_fairness_valid", "fairness_valid", "Fairness metrics are assessable for both groups."),
            ("ml_fairness_di_min", "di_min", "Disparate impact minimum from the Evidently report."),
            ("ml_fairness_dpd_abs_max", "dpd_max", "Absolute DPD maximum from the Evidently report."),
            ("ml_fairness_tpr_gap_abs_max", "tpr_gap_max", "Absolute TPR gap maximum from the Evidently report."),
            ("ml_disparate_impact", "di", "Female over Male selection-rate ratio."),
            ("ml_disparate_impact_reference", "di_reference", "DI for fixed reference cohort."),
            ("ml_dpd_abs", "dpd", "Absolute demographic parity difference."),
            ("ml_tpr_gap_abs", "tpr_gap", "Absolute true positive rate gap.")):
            lines += [f"# HELP {name} {helptext}", f"# TYPE {name} gauge"]
            x = value.get(metric)
            if x is not None: lines.append(f"{name} {x}")
        for name, index, helptext in (("ml_feature_drift_score", 0, "Evidently per-feature drift score."),
                                       ("ml_feature_drift_threshold", 1, "Per-feature drift threshold."),
                                       ("ml_feature_drift_detected", 2, "Whether the per-feature Evidently drift test failed.")):
            lines += [f"# HELP {name} {helptext}", f"# TYPE {name} gauge"]
            for feature, row in value["feature"].items():
                x = row[index]
                if x is not None: lines.append(f'{name}{{feature="{feature}"}} {x}')
        for name, index, helptext in (("ml_group_selection_rate", 0, "Positive prediction rate by group."),
                                      ("ml_group_tpr", 1, "True positive rate by group.")):
            lines += [f"# HELP {name} {helptext}", f"# TYPE {name} gauge"]
            for group, row in value["groups"].items():
                x = row[index]
                if x is not None: lines.append(f'{name}{{group="{group}"}} {x}')
    return "\n".join(lines) + "\n"


class Exporter:
    def __init__(self, config: str, refresh_seconds: int = 60):
        if not 1 <= refresh_seconds <= TTL_SECONDS:
            raise ValueError(f"refresh_seconds must be between 1 and {TTL_SECONDS}")
        self.config_path = Path(config)
        self.cfg = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
        self.report_root = Path(self.cfg["paths"].get("reports", "reports")) / "evidently"
        self.manifest_path = Path(self.cfg.get("scenarios", {}).get("output_dir", "data/processed/scenarios")) / "manifest.json"
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        if set(manifest.get("scenarios", {})) != set(SCENARIOS):
            raise ValueError("scenario manifest does not match the bounded exporter scenario set")
        self.refresh_seconds = refresh_seconds
        self.state = Snapshot()
        self.stop_event = threading.Event()

    def update(self, scenario: str | None) -> None:
        if scenario is not None and scenario not in SCENARIOS: raise ValueError("unknown scenario")
        if not self.state.begin_update(): raise RuntimeError("scenario update already in progress")
        try:
            # A refresh must choose the current scenario after acquiring the lock;
            # a completed concurrent POST may have changed it while refresh waited.
            if scenario is None:
                scenario = self.state.current()[2]
            calculate_scenario(self.config_path, scenario)
            report_dir = self.report_root / scenario
            loaded = load_completed(report_dir, scenario)
            self.state.publish(metrics_from_result(loaded, scenario))
        except Exception:
            self.state.fail()
            LOGGER.exception("Evidently calculation failed for scenario %s", scenario)
            raise
        finally:
            self.state.end_update()

    def refresh_loop(self) -> None:
        while not self.stop_event.wait(self.refresh_seconds):
            try: self.update(None)
            except Exception: pass


def handler_for(app: Exporter):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_): pass
        def do_GET(self):
            path = urlparse(self.path).path
            if path == "/health":
                body, code, content = b"ok\n", 200, "text/plain; charset=utf-8"
            elif path == "/metrics":
                body, code, content = render(app.state).encode(), 200, "text/plain; version=0.0.4; charset=utf-8"
            else:
                body, code, content = b"not found\n", 404, "text/plain; charset=utf-8"
            self.send_response(code); self.send_header("Content-Type", content); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
        def do_POST(self):
            if urlparse(self.path).path != "/scenario": self.send_error(404); return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length < 2 or length > 256: raise ValueError("invalid body size")
                body = json.loads(self.rfile.read(length))
                scenario = body.get("scenario") if isinstance(body, dict) else None
                if scenario not in SCENARIOS: raise ValueError("unknown scenario")
            except (ValueError, json.JSONDecodeError): self.send_error(400, "expected JSON with a known scenario"); return
            try:
                app.update(scenario)
                out, code = json.dumps({"status": "success", "scenario": scenario}).encode(), 200
            except RuntimeError as exc:
                code, out = (409, b'{"error":"update already in progress"}') if "already in progress" in str(exc) else (500, b'{"error":"calculation failed"}')
            except Exception: code, out = 500, b'{"error":"calculation failed"}'
            self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(out))); self.end_headers(); self.wfile.write(out)
    return Handler


def serve(config: str, host: str = "127.0.0.1", port: int = 8000, refresh_seconds: int = 60):
    app = Exporter(config, refresh_seconds)
    # Keep the HTTP endpoint alive on calculation failure so Prometheus can observe
    # monitoring_available=0 and the error counter instead of only seeing a dead target.
    try:
        app.update("baseline")
    except Exception:
        pass
    refresh_thread = threading.Thread(target=app.refresh_loop, name="scenario-refresh", daemon=True)
    refresh_thread.start()
    server = ThreadingHTTPServer((host, port), handler_for(app))
    try: server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt: pass
    finally:
        app.stop_event.set(); server.shutdown(); server.server_close()
        refresh_thread.join()


def main():
    parser = argparse.ArgumentParser(prog="adult-monitoring-exporter")
    parser.add_argument("--config", default="configs/example.yaml"); parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000); parser.add_argument("--refresh-seconds", type=int, default=60)
    args = parser.parse_args(); serve(args.config, args.host, args.port, args.refresh_seconds)


if __name__ == "__main__": main()
