#!/usr/bin/env python3
"""Collect the 24 original zero-shot results from the two Table 9 log batches."""

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


DATASETS = ("ETTh1", "ETTh2", "ETTm1", "ETTm2", "Electricity", "Weather")
HORIZONS = (96, 192, 336, 720)
MODEL_ID = re.compile(r"Model ID:\s*VisionTS_ETTh1_(\d+)\b")
METRICS = re.compile(r"^\s*mse:\s*([^,\s]+)\s*,\s*mae:\s*([^,\s]+)\s*$", re.MULTILINE)


def read_log(path):
    contents = path.read_text(encoding="utf-8", errors="replace")
    if "Bad checkpoint file" in contents or "Traceback (most recent call last)" in contents:
        raise ValueError(f"Run error recorded in {path}")
    return contents


def metric_pair(contents, label):
    matches = METRICS.findall(contents)
    if len(matches) != 1:
        raise ValueError(f"{label}: expected one MSE/MAE line, found {len(matches)}")
    mse, mae = map(float, matches[0])
    if not all(map(math.isfinite, (mse, mae))):
        raise ValueError(f"{label}: non-finite MSE or MAE")
    return mse, mae


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save-dir", type=Path, default=Path.home() / "VisionTS/long_term_tsf/save")
    parser.add_argument("--etth1-run", default="ETTh1_zero.0dDW5i4d")
    parser.add_argument("--remaining-run", default="remaining_zero.Tz3mKGxU")
    args = parser.parse_args()
    root = args.save_dir
    results = {}

    try:
        log = root / args.etth1_run / "run.log"
        parts = MODEL_ID.split(read_log(log))
        for prediction_length, contents in zip(parts[1::2], parts[2::2]):
            horizon = int(prediction_length)
            key = ("ETTh1", horizon)
            if key in results or horizon not in HORIZONS:
                raise ValueError(f"{log}: duplicate or unexpected ETTh1 horizon {horizon}")
            results[key] = (*metric_pair(contents, f"{log} (ETTh1 {horizon})"), log)
        for horizon in HORIZONS:
            if ("ETTh1", horizon) not in results:
                raise ValueError(f"{log}: missing ETTh1 horizon {horizon}")

        for dataset in DATASETS[1:]:
            for horizon in HORIZONS:
                log = root / args.remaining_run / f"{dataset}_{horizon}" / "run.log"
                results[dataset, horizon] = (*metric_pair(read_log(log), log), log)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")

    output = root / "table9_zero_shot_summary.csv"
    with output.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("dataset", "horizon", "mse", "mae", "source_log"))
        for dataset in DATASETS:
            rows = [results[dataset, horizon] for horizon in HORIZONS]
            for horizon, (mse, mae, log) in zip(HORIZONS, rows):
                writer.writerow((dataset, horizon, mse, mae, log.relative_to(root)))
            writer.writerow((dataset, "avg", statistics.mean(row[0] for row in rows),
                             statistics.mean(row[1] for row in rows), ""))
        writer.writerow(("All", "avg", statistics.mean(row[0] for row in results.values()),
                         statistics.mean(row[1] for row in results.values()), ""))
    print(f"Collected {len(results)}/24 results: {output}")


if __name__ == "__main__":
    main()
