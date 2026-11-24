"""CLI: python -m adult_monitoring prepare|train."""
import argparse
import json

from .data import prepare
from .training import train


def main() -> None:
    parser = argparse.ArgumentParser(prog="adult_monitoring")
    parser.add_argument("command", choices=["prepare", "train", "pipeline"])
    parser.add_argument("--config", default="configs/example.yaml")
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.config)
    elif args.command == "train":
        result = train(args.config)
    else:
        prepare(args.config)
        result = train(args.config)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
