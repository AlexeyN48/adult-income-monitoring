"""CLI for data preparation, training and scenario generation."""
import argparse
import json

from .data import prepare
from .training import train
from .scenarios import generate_scenarios
from .monitoring import monitor


def main() -> None:
    parser = argparse.ArgumentParser(prog="adult_monitoring")
    parser.add_argument("command", choices=["prepare", "train", "pipeline", "scenarios", "monitor"])
    parser.add_argument("--config", default="configs/example.yaml")
    parser.add_argument("--scenario", default="all", help="Scenario alias or all (monitor command)")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config)
    elif args.command == "train":
        result = train(args.config)
    elif args.command == "scenarios":
        result = generate_scenarios(args.config)
    elif args.command == "monitor":
        result = monitor(args.config, args.scenario)
    else:
        prepare(args.config)
        result = train(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
