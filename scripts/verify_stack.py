"""Exercise real exporter/Prometheus HTTP APIs; save evidence without browser automation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.parse import urlencode
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def request_json(url: str, payload: dict | None = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    with urlopen(request, timeout=120 if data else 15) as response:
        return json.load(response)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prometheus", default="http://127.0.0.1:9090")
    parser.add_argument("--exporter", default="http://127.0.0.1:8000")
    parser.add_argument("--alert", default="ModelF1Low")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output", type=Path, default=Path("reports/prometheus_demo/evidence.json"))
    args = parser.parse_args()
    started = time.time()
    evidence = {"started_at_utc": utc(), "source": "live Prometheus and exporter HTTP APIs", "events": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def event(phase: str, **values) -> None:
        row = {"at_utc": utc(), "phase": phase, **values}
        evidence["events"].append(row)
        print(json.dumps(row), flush=True)

    def query(expression: str) -> list:
        result = request_json(args.prometheus + "/api/v1/query?" + urlencode({"query": expression}))
        if result.get("status") != "success":
            raise RuntimeError(f"Prometheus query failed: {result}")
        return result["data"]["result"]

    def scalar(expression: str) -> float | None:
        result = query(expression)
        if not result:
            return None
        if len(result) != 1:
            raise RuntimeError(f"Expected one time series for {expression}, got {len(result)}")
        return float(result[0]["value"][1])

    def select(scenario: str) -> None:
        deadline = time.monotonic() + args.timeout
        while True:
            try:
                response = request_json(args.exporter + "/scenario", {"scenario": scenario})
                event("scenario_selected", scenario=scenario, response=response)
                return
            except HTTPError as exc:
                if exc.code != 409 or time.monotonic() >= deadline:
                    raise
                # The periodic refresh may already hold the calculation lock.
                time.sleep(1)

    def wait_for(state: str, scenario: str, *, low: bool) -> None:
        deadline = time.monotonic() + args.timeout
        previous = None
        while time.monotonic() < deadline:
            response = request_json(args.prometheus + "/api/v1/rules?type=alert")
            rules = [rule for group in response["data"]["groups"] for rule in group["rules"]
                     if rule.get("name") == args.alert]
            if len(rules) != 1:
                raise RuntimeError(f"Expected one {args.alert} rule, got {len(rules)}")
            rule = rules[0]
            actual = rule["state"]
            f1, threshold = scalar("ml_f1_score"), scalar("ml_f1_threshold")
            active = scalar(f'ml_active_scenario_info{{scenario="{scenario}"}}')
            available = scalar("ml_monitoring_available")
            observed = (actual, f1, threshold, active, available)
            if observed != previous:
                event("observation", expected_state=state, scenario=scenario, alert_state=actual,
                      f1=f1, threshold=threshold, active_scenario=active, monitoring_available=available)
                previous = observed
            matching = active == 1 and available == 1 and f1 is not None and threshold is not None
            if matching and (f1 < threshold) == low and actual == state:
                event("assertion_passed", state=state, scenario=scenario, f1=f1, threshold=threshold,
                      rule=rule)
                return
            time.sleep(2)
        raise TimeoutError(f"Did not observe {args.alert}={state} on {scenario} within {args.timeout}s")

    restored = False
    try:
        targets = request_json(args.prometheus + "/api/v1/targets")
        evidence["targets"] = targets["data"]["activeTargets"]
        if not any(t.get("health") == "up" for t in evidence["targets"]):
            raise RuntimeError("No healthy Prometheus scrape target")
        select("baseline")
        wait_for("inactive", "baseline", low=False)
        select("fairness_defect")
        wait_for("pending", "fairness_defect", low=True)
        wait_for("firing", "fairness_defect", low=True)
        firing = query(f'ALERTS{{alertname="{args.alert}",alertstate="firing"}}')
        if not firing:
            raise RuntimeError("Firing rule has no ALERTS time series")
        evidence["firing_series"] = firing
        select("recovery")
        wait_for("inactive", "recovery", low=False)
        restored = True
        # Include a recovery sample on the query_range five-second grid.
        time.sleep(6)
        evidence["result"] = "passed"
    except Exception as exc:
        evidence["result"] = "failed"
        evidence["error"] = str(exc)
        raise
    finally:
        if not restored:
            try:
                select("recovery")
            except Exception as exc:
                evidence["recovery_error"] = str(exc)
        evidence["finished_at_utc"] = utc()
        evidence["range_start"] = started
        evidence["range_end"] = time.time()
        evidence["range_series"] = {}
        for expression in ("ml_f1_score", "ml_f1_threshold",
                           f'ALERTS{{alertname="{args.alert}",alertstate="firing"}}'):
            try:
                response = request_json(args.prometheus + "/api/v1/query_range?" + urlencode({
                    "query": expression, "start": started, "end": evidence["range_end"], "step": 5}))
                evidence["range_series"][expression] = response["data"]["result"]
            except Exception as exc:
                evidence.setdefault("range_errors", {})[expression] = str(exc)
        args.output.write_text(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        print(f"Evidence saved to {args.output}", flush=True)


if __name__ == "__main__":
    main()
