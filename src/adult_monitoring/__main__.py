"""CLI for data preparation, training and scenario generation."""
import argparse
import json

from .data import prepare
from .training import train
from .scenarios import generate_scenarios


def main() -> None:
    parser = argparse.ArgumentParser(prog="adult_monitoring")
    parser.add_argument("command", choices=["prepare", "train", "pipeline", "scenarios"])
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config)
    elif args.command == "train":
        result = train(args.config)
    elif args.command == "scenarios":
        result = generate_scenarios(args.config)
    else:
        prepare(args.config)
        result = train(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
