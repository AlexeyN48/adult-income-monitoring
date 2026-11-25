import json
import math
import threading
import time
import unittest
from unittest import mock
from datetime import datetime, timezone
from http.server import ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Thread
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from prometheus_client.parser import text_string_to_metric_families

from adult_monitoring.exporter import (FEATURES, GROUPS, SCENARIOS, Snapshot, Exporter,
    handler_for, load_completed, metrics_from_result, render)


def report(at=None):
    at = at or datetime.now(timezone.utc).isoformat()
    current = {"f1_positive": .703191, "fairness.disparate_impact": .3221,
        "fairness.abs_dpd": .1748, "fairness.abs_tpr_gap": .0657,
        "fairness.validity.all_groups": 1.0, "fairness.validity.demographic_parity_difference": 1.0,
        "fairness.validity.disparate_impact": 1.0, "fairness.validity.tpr_gap": 1.0}
    reference = {"fairness.disparate_impact": .32648}
    for group in GROUPS:
        current[f"groups.{group}.selection_rate"] = .08 if group == "Female" else .25
        current[f"groups.{group}.tpr"] = .59 if group == "Female" else .66
    return {"schema_version": 1, "status": "success", "scenario": "baseline", "calculated_at_utc": at,
        "evidently": {"evidently_metric_values": {"current": current, "reference": reference},
                      "f1_alert_threshold_from_training": .663487},
        "thresholds": {"di_min": .8, "dpd_abs": .1, "tpr_gap_abs": .1},
        "feature_drift": {f: {"score": .02, "threshold": .1, "test_status": "success"} for f in FEATURES},
        "feature_drift_aggregate": {"drifted_feature_share": 0.0},
        "target_drift": {"status": "success", "validity": True},
        "female_target_drift": {"status": "success", "validity": True}}


def accepted_values():
    return metrics_from_result(report(), "baseline")


class ExporterTests(unittest.TestCase):
    def test_refresh_selects_latest_scenario_under_lock_and_busy_keeps_last_good_data(self):
        app = Exporter.__new__(Exporter)
        app.state = Snapshot()
        app.config_path = Path("unused-test-config.yaml")
        app.report_root = Path("unused-test-reports")
        app.state.publish(accepted_values())
        latest = accepted_values(); latest["scenario"] = "fairness_defect"
        original_acquire = app.state.begin_update
        def concurrent_switch():
            app.state.publish(latest)
            return original_acquire()
        result = report(); result["scenario"] = "fairness_defect"
        with mock.patch.object(app.state, "begin_update", side_effect=concurrent_switch), \
             mock.patch("adult_monitoring.exporter.calculate_scenario") as calculation, \
             mock.patch("adult_monitoring.exporter.load_completed", return_value=result):
            app.update(None)
        calculation.assert_called_once_with(app.config_path, "fairness_defect")
        self.assertTrue(app.state.begin_update())
        try:
            with self.assertRaisesRegex(RuntimeError, "already in progress"):
                app.update("baseline")
            samples = {s.name: s.value for f in text_string_to_metric_families(render(app.state)) for s in f.samples}
            self.assertEqual(samples["ml_monitoring_available"], 1)
            self.assertEqual(samples["ml_monitoring_errors_total"], 0)
        finally:
            app.state.end_update()

    def test_completed_snapshot_pair_must_match_and_be_successful(self):
        good = report()
        with TemporaryDirectory() as td:
            root = Path(td)
            (root / "monitoring.json").write_text(json.dumps(good), encoding="utf-8")
            status = {"status": "success", "scenario": "baseline", "calculated_at_utc": good["calculated_at_utc"]}
            (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
            self.assertEqual(load_completed(root, "baseline")["scenario"], "baseline")
            status["calculated_at_utc"] = "2026-11-24T00:00:00+00:00"
            (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "timestamp mismatch"):
                load_completed(root, "baseline")
            status["calculated_at_utc"] = good["calculated_at_utc"]; status["status"] = "failed"
            (root / "status.json").write_text(json.dumps(status), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not successful"):
                load_completed(root, "baseline")

    def test_schema_errors_nan_and_future_timestamp_are_rejected(self):
        bad = report(); bad["evidently"]["evidently_metric_values"]["current"]["f1_positive"] = math.nan
        with self.assertRaisesRegex(ValueError, "F1"):
            metrics_from_result(bad, "baseline")
        bad = report(); bad["feature_drift"].pop(FEATURES[0])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            metrics_from_result(bad, "baseline")
        with TemporaryDirectory() as td:
            root = Path(td); future = report("2999-01-01T00:00:00+00:00")
            (root / "monitoring.json").write_text(json.dumps(future), encoding="utf-8")
            (root / "status.json").write_text(json.dumps({"status":"success", "scenario":"baseline", "calculated_at_utc":future["calculated_at_utc"]}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "future"):
                load_completed(root, "baseline")

    def test_undefined_fairness_is_not_exported_as_zero(self):
        source = report(); values = source["evidently"]["evidently_metric_values"]["current"]
        values["fairness.validity.disparate_impact"] = 0.0
        values["fairness.disparate_impact"] = None
        snapshot = Snapshot(); snapshot.publish(metrics_from_result(source, "baseline"))
        text = render(snapshot)
        families = {f.name: f for f in text_string_to_metric_families(text)}
        self.assertEqual(families["ml_fairness_valid"].samples[0].value, 0)
        names = {sample.name for family in text_string_to_metric_families(text) for sample in family.samples}
        self.assertNotIn("ml_disparate_impact", names)
        self.assertFalse(any(name == "ml_group_selection_rate" for name in names))

    def test_prometheus_parser_reads_contract_values_and_refresh_failure_suppresses_business_metrics(self):
        state = Snapshot(); state.publish(accepted_values())
        output = render(state)
        samples = {s.name: s.value for family in text_string_to_metric_families(output) for s in family.samples}
        self.assertEqual(samples["ml_f1_score"], .703191)
        self.assertEqual(samples['ml_feature_drift_detected'], 0)
        self.assertIn("ml_group_tpr", {family.name for family in text_string_to_metric_families(output)})
        state.fail()
        failed = render(state)
        samples2 = {s.name for family in text_string_to_metric_families(failed) for s in family.samples}
        self.assertIn("ml_monitoring_available", samples2)
        self.assertNotIn("ml_f1_score", samples2)
        self.assertIn("ml_monitoring_last_success_timestamp_seconds", samples2)

    def test_stale_report_suppresses_business_metrics_but_preserves_last_success_time(self):
        value = accepted_values()
        stamp = datetime.fromisoformat(value["calculated_at_utc"].replace("Z", "+00:00")).timestamp()
        state = Snapshot(); state.publish(value)
        output = render(state, now=stamp + 181)
        samples = {s.name: s.value for family in text_string_to_metric_families(output) for s in family.samples}
        self.assertEqual(samples["ml_monitoring_available"], 0)
        self.assertEqual(samples["ml_monitoring_last_success_timestamp_seconds"], stamp)
        self.assertNotIn("ml_f1_score", samples)

    def test_copy_on_write_never_exposes_torn_active_scenario_and_f1(self):
        state = Snapshot()
        one = accepted_values(); two = dict(one); two.update(scenario="fairness_defect", f1=.644947)
        state.publish(one)
        errors = []; reads = [0]; started = threading.Event()
        def writer():
            started.set()
            for i in range(1000):
                state.publish(two if i % 2 else one)
                time.sleep(.0001)
        thread = Thread(target=writer); thread.start()
        started.wait(timeout=1)
        while thread.is_alive() or reads[0] == 0:
            lines = render(state).splitlines()
            reads[0] += 1
            active = next(line for line in lines if line.startswith("ml_active_scenario_info"))
            f1 = float(next(line.split()[-1] for line in lines if line.startswith("ml_f1_score ")))
            if ("fairness_defect" in active) != (f1 == .644947): errors.append((active, f1))
        thread.join()
        self.assertEqual(errors, [])
        self.assertGreater(reads[0], 0)

    def test_http_rejects_unknown_scenario_and_concurrent_update(self):
        class FakeApp:
            def __init__(self): self.state = Snapshot()
            def update(self, scenario):
                if not self.state.begin_update(): raise RuntimeError("scenario update already in progress")
                try:
                    value = accepted_values(); value["scenario"] = scenario; self.state.publish(value)
                finally: self.state.end_update()
        app = FakeApp(); server = ThreadingHTTPServer(("127.0.0.1", 0), handler_for(app))
        thread = Thread(target=server.serve_forever, daemon=True); thread.start()
        url = f"http://127.0.0.1:{server.server_port}/scenario"
        try:
            req = Request(url, data=b'{"scenario":"../../evil"}', headers={"Content-Type":"application/json"}, method="POST")
            with self.assertRaises(HTTPError) as caught: urlopen(req)
            self.assertEqual(caught.exception.code, 400)
            app.state.begin_update()
            try:
                req = Request(url, data=b'{"scenario":"baseline"}', headers={"Content-Type":"application/json"}, method="POST")
                with self.assertRaises(HTTPError) as caught: urlopen(req)
                self.assertEqual(caught.exception.code, 409)
            finally: app.state.end_update()
        finally:
            server.shutdown(); server.server_close(); thread.join()


if __name__ == "__main__": unittest.main()
