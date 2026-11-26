#!/usr/bin/env python3
"""Run the VisionTS column of Table 21 on independent GPUs."""

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import statistics
import subprocess
import sys
import tempfile
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace


LTSF_DIR = Path(__file__).resolve().parents[2]
CHECKPOINT = LTSF_DIR.parent / "ckpt" / "mae_visualize_vit_base.pth"
HORIZONS = (96, 192, 336, 720)
SEEDS = (2021, 2022, 2023)
BATCH_SIZE = 256
LEARNING_RATE = 1e-4


@dataclass(frozen=True)
class DatasetSpec:
    path: str
    context: int
    period: int
    norm: float = 0.4
    align: float = 0.4
    epochs: int = 1
    horizons: tuple = HORIZONS


DATASETS = {
    "ETTh1": DatasetSpec("ETT-small/ETTh1.csv", 1152, 24),
    "ETTh2": DatasetSpec("ETT-small/ETTh2.csv", 1152, 24),
    "ETTm1": DatasetSpec("ETT-small/ETTm1.csv", 2304, 96),
    "ETTm2": DatasetSpec("ETT-small/ETTm2.csv", 1152, 96),
    "Illness": DatasetSpec("illness/national_illness.csv", 104, 52, 1.0, 0.4, 100, (24, 36, 48, 60)),
    "Weather": DatasetSpec("weather/weather.csv", 576, 144, 1.0, 0.7),
    "Traffic": DatasetSpec("traffic/traffic.csv", 1152, 24),
    "Electricity": DatasetSpec("electricity/electricity.csv", 1152, 24),
}


class SeriesWindows:
    """Enumerate all (window, variable) pairs over the repository's scaled split."""

    def __init__(self, dataset, context, horizon):
        import numpy as np

        self.x = np.asarray(dataset.data_x, dtype=np.float32)
        self.y = np.asarray(dataset.data_y, dtype=np.float32)
        self.channels = self.x.shape[1]
        self.count = (len(self.x) - context - horizon + 1) * self.channels
        if self.count <= 0:
            raise ValueError("The split has no complete forecasting windows")
        self.history = np.arange(context)
        self.future = context + np.arange(horizon)

    def take(self, indices, device):
        import torch

        times = (indices // self.channels)[:, None]
        channels = (indices % self.channels)[:, None]
        history = self.x[times + self.history, channels]
        future = self.y[times + self.future, channels]
        return (torch.from_numpy(history).to(device).unsqueeze(-1),
                torch.from_numpy(future).to(device).unsqueeze(-1))


def evaluate(model, windows, device, batch_size, progress, description):
    import numpy as np
    import torch

    model.eval()
    squared = torch.zeros((), device=device, dtype=torch.float64)
    absolute = torch.zeros_like(squared)
    with torch.no_grad():
        for start in progress(range(0, windows.count, batch_size), desc=description):
            indices = np.arange(start, min(start + batch_size, windows.count))
            history, future = windows.take(indices, device)
            error = model(history) - future
            squared += error.pow(2).sum(dtype=torch.float64)
            absolute += error.abs().sum(dtype=torch.float64)
    count = windows.count * len(windows.future)
    mse, mae = squared.item() / count, absolute.item() / count
    if not np.isfinite([mse, mae]).all():
        raise ValueError("Evaluation produced non-finite errors")
    return mse, mae


def train_job(dataset, horizon, seed, gpu, output, micro_batch, progress):
    import numpy as np
    import torch
    from torch.nn import functional as F

    from data_provider.data_factory import data_dict
    from utils.tools import adjust_learning_rate
    from visionts import VisionTS

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device("cuda:0")
    spec = DATASETS[dataset]
    csv_path = LTSF_DIR / "dataset" / spec.path
    loader = data_dict[dataset if dataset.startswith("ETT") else "custom"]
    splits = {}
    for flag in ("train", "val", "test"):
        split = loader(root_path=str(csv_path.parent), data_path=csv_path.name,
                       flag=flag, size=(spec.context, 48, horizon), features="M", timeenc=1)
        splits[flag] = SeriesWindows(split, spec.context, horizon)
    print(f"{dataset} H={horizon} seed={seed} GPU={gpu} ({torch.cuda.get_device_name(0)})")
    print(f"PyTorch={torch.__version__}, CUDA={torch.version.cuda}")
    print("Series windows:", {flag: split.count for flag, split in splits.items()})

    model = VisionTS(arch="mae_base", finetune_type="ln", load_ckpt=False)
    model.vision_model.load_state_dict(torch.load(str(CHECKPOINT), map_location="cpu")["model"], strict=True)
    model.update_config(context_len=spec.context, pred_len=horizon, periodicity=spec.period,
                        norm_const=spec.norm, align_const=spec.align, interpolation="bilinear")
    model = model.to(device)
    trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
    optimizer = torch.optim.Adam(trainable.values(), lr=LEARNING_RATE)
    schedule = SimpleNamespace(learning_rate=LEARNING_RATE, lradj="type1")
    best_loss, best_epoch, best_state = float("inf"), 0, None
    train = splits["train"]
    eval_batch = max(micro_batch, 64)
    label = f"GPU {gpu} {dataset}/{horizon} seed={seed}"

    for epoch in range(1, spec.epochs + 1):
        model.train()
        order = rng.permutation(train.count)
        epoch_loss = torch.zeros((), device=device, dtype=torch.float64)
        batches = progress(range(0, train.count, BATCH_SIZE), desc=f"{label} train {epoch}/{spec.epochs}")
        for step, start in enumerate(batches, 1):
            batch = order[start:start + BATCH_SIZE]
            optimizer.zero_grad()
            for offset in range(0, len(batch), micro_batch):
                history, future = train.take(batch[offset:offset + micro_batch], device)
                loss = F.mse_loss(model(history), future, reduction="sum")
                (loss / (len(batch) * horizon)).backward()
                epoch_loss += loss.detach()
            optimizer.step()
            if step % 500 == 0:
                print(f"Epoch {epoch}, optimizer steps: {step}", flush=True)

        valid_loss, _ = evaluate(model, splits["val"], device, eval_batch, progress, f"{label} val {epoch}")
        print(f"Epoch {epoch}: train MSE={epoch_loss.item() / (train.count * horizon):.6f}, "
              f"validation MSE={valid_loss:.6f}")
        if valid_loss <= best_loss:
            best_loss, best_epoch = valid_loss, epoch
            best_state = {name: param.detach().cpu().clone() for name, param in trainable.items()}
        elif epoch - best_epoch >= 3:
            break
        adjust_learning_rate(optimizer, epoch, schedule)

    with torch.no_grad():
        for name, param in trainable.items():
            param.copy_(best_state[name])
    mse, mae = evaluate(model, splits["test"], device, eval_batch, progress, f"{label} test")
    torch.save(best_state, output / "layernorm.pt")
    print(f"Test MSE={mse:.6f}, MAE={mae:.6f}, best epoch={best_epoch}")
    return dict(dataset=dataset, horizon=horizon, seed=seed, gpu=gpu, mse=mse, mae=mae,
                best_epoch=best_epoch, validation_mse=best_loss)


def gpu_uuids(indices):
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"],
        check=True, stdout=subprocess.PIPE, text=True,
    )
    available = {index.strip(): uuid.strip() for index, uuid in csv.reader(result.stdout.splitlines())}
    missing = [index for index in indices if index not in available]
    if missing:
        raise ValueError(f"GPU indices not found in nvidia-smi: {', '.join(missing)}")
    return {index: available[index] for index in indices}


def worker(gpu, gpu_uuid, jobs, root, micro_batch, position, progress_lock):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_uuid
    os.environ["MPLBACKEND"] = "Agg"
    sys.path[:0] = [str(LTSF_DIR), str(LTSF_DIR.parent)]

    import torch
    from tqdm import tqdm

    tqdm.set_lock(progress_lock)

    def progress(batches, desc):
        return tqdm(batches, desc=desc, position=position, leave=False, dynamic_ncols=True,
                    mininterval=1, file=sys.__stderr__, disable=not sys.__stderr__.isatty())

    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible CUDA device for GPU {gpu} ({gpu_uuid})")
    print(f"Physical GPU {gpu} ({gpu_uuid}) is worker cuda:0", flush=True)
    for dataset, horizon, seed in iter(jobs.get, None):
        output = root / dataset / str(horizon) / f"seed_{seed}"
        output.mkdir(parents=True, exist_ok=True)
        tqdm.write(f"GPU {gpu}: {dataset} H={horizon} seed={seed}", file=sys.__stderr__)
        with (output / "run.log").open("a", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            try:
                print(f"Starting {dataset} H={horizon} seed={seed} on physical GPU {gpu}", flush=True)
                metrics = train_job(dataset, horizon, seed, gpu, output, micro_batch, progress)
                temporary = output / "metrics.json.tmp"
                temporary.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
                temporary.replace(output / "metrics.json")
            except Exception:
                traceback.print_exc()
                tqdm.write(f"Failed: {output / 'run.log'}", file=sys.__stderr__)
        torch.cuda.empty_cache()


def completed_result(root, task):
    dataset, horizon, seed = task
    directory = root / dataset / str(horizon) / f"seed_{seed}"
    path = directory / "metrics.json"
    weights = directory / "layernorm.pt"
    if not path.is_file() or not weights.is_file() or weights.stat().st_size == 0:
        return None
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
        if (row["dataset"], row["horizon"], row["seed"]) == task and all(
            math.isfinite(row[key]) for key in ("mse", "mae", "validation_mse")
        ):
            return row
    except (ValueError, KeyError, TypeError):
        pass
    return None


def write_summary(root, tasks, repeats):
    runs = []
    groups = {}
    for task in tasks:
        row = completed_result(root, task)
        if row is not None:
            runs.append(row)
            groups.setdefault(task[:2], []).append(row)

    fields = ("dataset", "horizon", "seed", "gpu", "mse", "mae", "best_epoch", "validation_mse")
    with (root / "runs.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(runs)
    with (root / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("dataset", "horizon", "runs", "mse_mean", "mae_mean", "mse_std", "mae_std"))
        for (dataset, horizon), rows in groups.items():
            if len(rows) != repeats:
                continue
            mse, mae = ([row[key] for row in rows] for key in ("mse", "mae"))
            mse_std, mae_std = (statistics.stdev(mse), statistics.stdev(mae)) if repeats > 1 else ("", "")
            writer.writerow((dataset, horizon, repeats, statistics.mean(mse), statistics.mean(mae),
                             mse_std, mae_std))
    return len(runs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    parser.add_argument("--repeats", type=int, choices=(1, 2, 3), default=3)
    parser.add_argument("--micro-batch", type=int, default=32)
    parser.add_argument("--gpus", nargs="+", default=["2", "3"],
                        help="Physical GPU indices shown by nvidia-smi (default: 2 3)")
    parser.add_argument("--resume", type=Path,
                        help="Existing run directory. Retry unfinished tasks and reuse completed results.")
    args = parser.parse_args()
    if args.micro_batch < 1 or args.micro_batch > BATCH_SIZE:
        parser.error(f"--micro-batch must be between 1 and {BATCH_SIZE}")
    if len(args.gpus) != len(set(args.gpus)) or not all(gpu.isdecimal() for gpu in args.gpus):
        parser.error("--gpus must contain distinct numeric GPU indices")
    datasets = list(dict.fromkeys(args.datasets))
    required = [CHECKPOINT, *(LTSF_DIR / "dataset" / DATASETS[name].path for name in datasets)]
    missing = [str(path) for path in required if not path.is_file() or path.stat().st_size == 0]
    if missing:
        parser.error("Missing or empty inputs:\n" + "\n".join(missing))
    tasks = [(dataset, horizon, seed) for dataset in datasets
             for horizon in DATASETS[dataset].horizons for seed in SEEDS[:args.repeats]]
    pending = tasks
    if args.resume is not None:
        root = args.resume.resolve()
        try:
            original = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            original_args = original["arguments"]
            if (set(original_args["datasets"]) != set(datasets)
                    or original_args["repeats"] != args.repeats
                    or original_args["micro_batch"] != args.micro_batch
                    or original["batch_size"] != BATCH_SIZE
                    or original["learning_rate"] != LEARNING_RATE
                    or original["seeds"] != list(SEEDS[:args.repeats])
                    or original["checkpoint"] != str(CHECKPOINT)):
                raise ValueError("Resume with the original datasets, repeats, micro-batch and weights")
        except (OSError, KeyError, ValueError) as error:
            parser.error(f"Cannot resume {root}: {error}")
        pending = [task for task in tasks if completed_result(root, task) is None]
        print(f"Resuming {root}: {len(tasks) - len(pending)} complete, {len(pending)} pending", flush=True)
        if not pending:
            write_summary(root, tasks, args.repeats)
            return 0

    try:
        devices = gpu_uuids(args.gpus)
    except (OSError, subprocess.CalledProcessError, ValueError) as error:
        parser.error(str(error))

    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=LTSF_DIR, check=True,
                              stdout=subprocess.PIPE, text=True).stdout.strip()
    if args.resume is None:
        base = LTSF_DIR / "save" / "table21_fullshot"
        base.mkdir(parents=True, exist_ok=True)
        root = Path(tempfile.mkdtemp(prefix="run.", dir=base))
        manifest = dict(arguments=vars(args), gpu_uuids=devices, commit=revision, checkpoint=str(CHECKPOINT),
                        batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, seeds=SEEDS[:args.repeats])
        (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    else:
        with (root / "resume.log").open("a", encoding="utf-8") as log:
            print(f"Commit {revision}: {len(pending)} pending, GPUs {devices}", file=log)
    print(f"Results: {root} ({len(tasks)} runs)", flush=True)

    context = mp.get_context("spawn")
    jobs = context.Queue()
    # Start the largest workloads early to reduce the time the other GPUs sit idle.
    priority = ("Traffic", "Electricity", "Weather", "ETTm1", "ETTm2", "ETTh1", "ETTh2", "Illness")
    for task in sorted(pending, key=lambda item: priority.index(item[0])):
        jobs.put(task)
    progress_lock = context.RLock()
    processes = [context.Process(target=worker, args=(gpu, devices[gpu], jobs, root, args.micro_batch,
                                                     position, progress_lock))
                 for position, gpu in enumerate(args.gpus[:len(pending)])]
    for process in processes:
        jobs.put(None)
        process.start()
    try:
        for process in processes:
            process.join()
    except KeyboardInterrupt:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join()
        write_summary(root, tasks, args.repeats)
        raise SystemExit(130)

    completed = write_summary(root, tasks, args.repeats)
    print(f"Completed: {completed}/{len(tasks)}. See {root / 'summary.csv'}")
    return int(completed != len(tasks) or any(process.exitcode for process in processes))


if __name__ == "__main__":
    sys.exit(main())
