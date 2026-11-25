"""Plot measured Prometheus samples and observed alert states without opening a GUI."""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("reports/prometheus_demo/evidence.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/prometheus_demo/f1.png"))
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding="utf-8"))
    if data.get("result") != "passed":
        raise ValueError("The integration run did not pass; inspect its evidence before plotting")
    local = timezone(timedelta(hours=5))
    fig, (metric_ax, alert_ax) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                            gridspec_kw={"height_ratios": [3, 1]}, layout="constrained")
    for name, label, style in (("ml_f1_score", "F1 из Prometheus", "-o"),
                               ("ml_f1_threshold", "Порог алерта", "--")):
        series = data["range_series"].get(name, [])
        if len(series) != 1 or not series[0].get("values"):
            raise ValueError(f"Expected one measured series for {name}")
        points = series[0]["values"]
        times = [datetime.fromtimestamp(float(t), local) for t, _ in points]
        values = [float(v) for _, v in points]
        metric_ax.plot(times, values, style, label=label, linewidth=2, markersize=3)
    metric_ax.set(title="Реальные данные Prometheus: baseline → дефект → восстановление", ylabel="F1 класса >50K")
    metric_ax.grid(alpha=.25)
    metric_ax.legend(loc="best")
    observed = [e for e in data["events"] if e["phase"] == "observation"]
    if not observed:
        raise ValueError("Missing observed rule states")
    state_times = [datetime.fromisoformat(e["at_utc"]).astimezone(local) for e in observed]
    state_values = [{"inactive": 0, "pending": 1, "firing": 2}[e["alert_state"]] for e in observed]
    alert_ax.step(state_times, state_values, where="post", color="#b42318", linewidth=2)
    alert_ax.set(yticks=[0, 1, 2], yticklabels=["Inactive", "Pending", "Firing"],
                 ylim=(-.2, 2.2), ylabel="ModelF1Low", xlabel="Время (UTC+5)")
    alert_ax.grid(alpha=.25)
    alert_ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M:%S", tz=local))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
