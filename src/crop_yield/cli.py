"""One-command acquisition, training and inference from any working directory."""

import argparse
import json
import logging
from pathlib import Path

from crop_yield.backtesting import run_backtesting
from crop_yield.data import download_data
from crop_yield.features import build_table
from crop_yield.modeling import run_training
from crop_yield.predict import predict_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["download", "train", "backtest", "residual-backtest", "predict", "identity-audit"],
    )
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path)
    parser.add_argument("--input", type=Path)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    root = args.root.resolve()
    if args.command == "identity-audit":
        from crop_yield.identity import run_identity

        print(json.dumps(run_identity(root), indent=2))
        return
    config_path = args.config or root / "configs" / "default.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.command == "predict":
        if args.input is None:
            parser.error("predict requires --input")
        print(json.dumps(predict_file(root / "models" / "model.joblib", args.input), indent=2))
        return
    raw = download_data(root / "data" / "raw")
    if args.command in {"train", "backtest", "residual-backtest"}:
        table, audit = build_table(raw, config)
        if args.command in {"backtest", "residual-backtest"}:
            result = run_backtesting(
                table, config, root, residual_only=args.command == "residual-backtest"
            )
            print(json.dumps(result["pooled"], indent=2))
            return
        processed = root / "data" / "processed"
        processed.mkdir(exist_ok=True)
        table.to_csv(processed / "features.csv", index=False)
        result = run_training(table, config, root, audit)
        print(
            json.dumps(
                {"selected_model": result["selected_model"], "test": result["test"]}, indent=2
            )
        )


if __name__ == "__main__":
    main()
