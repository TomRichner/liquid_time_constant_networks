#!/usr/bin/env python3
"""
collect_results.py — Download and tabulate best-epoch results from GCS.

Usage:
    python3 cloud/collect_results.py <run_name>
    python3 cloud/collect_results.py run25ep
    python3 cloud/collect_results.py run25ep --csv results.csv
"""
import argparse
import csv
import io
import subprocess
import sys

GCS_BUCKET = "gs://liquidneuralnets-experiments/results-py"

EXPERIMENTS = ["har", "gesture", "occupancy", "smnist", "traffic", "power", "ozone", "person", "cheetah"]
CLASSIFICATION = {"har", "gesture", "occupancy", "smnist", "ozone", "person"}
REGRESSION = {"traffic", "power", "cheetah"}

MODELS = ["lstm", "ctrnn", "node", "ctgru", "ltc", "srnn"]


def gcs_cat(path):
    """Read file contents from GCS, return string or None on failure."""
    try:
        r = subprocess.run(
            ["gcloud", "storage", "cat", path],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:
        pass
    return None


def parse_csv(text):
    """Parse the 2-line CSV (header + data), return dict."""
    reader = csv.reader(io.StringIO(text))
    header = [h.strip() for h in next(reader)]
    values = [v.strip() for v in next(reader)]
    d = dict(zip(header, values))
    # Normalize column names: ozone uses 'test acc' instead of 'test accuracy'
    for short, full in [("test acc", "test accuracy"), ("valid acc", "valid accuracy"),
                        ("train acc", "train accuracy")]:
        if short in d and full not in d:
            d[full] = d[short]
    return d


def collect(run_name, max_seeds=5):
    """Collect results for all experiments/models/seeds."""
    results = {}  # (model, experiment) -> list of dicts (one per seed)

    for model in MODELS:
        for exp in EXPERIMENTS:
            seed_results = []
            for seed in range(1, max_seeds + 1):
                # Try known CSV name patterns
                for suffix in [f"{model}_{32}.csv", f"{model}_{32}_00.csv"]:
                    path = f"{GCS_BUCKET}/{run_name}/{model}/{exp}/seed{seed}/{suffix}"
                    text = gcs_cat(path)
                    if text:
                        try:
                            d = parse_csv(text)
                            d["seed"] = seed
                            seed_results.append(d)
                        except Exception:
                            pass
                        break  # found it, skip alternate suffix

            if seed_results:
                results[(model, exp)] = seed_results

    return results


def _fmt_sigfigs(val, n=3):
    """Format a float to n significant figures."""
    if val == 0:
        return "0"
    from math import log10, floor
    digits = n - 1 - floor(log10(abs(val)))
    digits = max(digits, 0)
    return f"{val:.{digits}f}"


def _get_metric(exp):
    """Return (metric_name, csv_key, format_fn) for each experiment."""
    if exp == "ozone":
        return "F1-score", "test accuracy", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
    elif exp in CLASSIFICATION:
        return "accuracy", "test accuracy", lambda m, s: f"{m:.2f}% ± {s:.2f}" if s else f"{m:.2f}%"
    elif exp in REGRESSION:
        metric = "MSE" if exp == "cheetah" else "squared error"
        return metric, "test loss", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
    return "loss", "test loss", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)


def format_table(results):
    """Format results matching the paper's Table 3 layout."""
    import statistics

    # Column width for each model
    cw = 20

    # ── Unified table ──
    print("\n" + "=" * (22 + cw * len(MODELS)))
    print("  Test performance at best validation epoch (matching Table 3 format)")
    print("=" * (22 + cw * len(MODELS)))

    header = f"{'Dataset':<14}{'Metric':<10}" + "".join(f"{m:>{cw}}" for m in MODELS)
    print(header)
    print("-" * len(header))

    for exp in EXPERIMENTS:
        metric_name, csv_key, fmt_fn = _get_metric(exp)
        row = f"{exp:<14}{metric_name:<10}"

        for model in MODELS:
            key = (model, exp)
            if key in results:
                seeds = results[key]
                vals = [float(s.get(csv_key, 0)) for s in seeds]
                mean = sum(vals) / len(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else None
                cell = fmt_fn(mean, std)
                row += f"{cell:>{cw}}"
            else:
                row += f"{'—':>{cw}}"
        print(row)

    print()
    print(f"  n = seeds per cell (paper uses n=5)")
    print()


def write_csv_output(results, filename):
    """Write all results to a single CSV file."""
    with open(filename, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "experiment", "task_type", "seed",
                         "best_epoch", "test_loss", "test_metric", "metric_name"])

        for (model, exp), seeds in sorted(results.items()):
            task_type = "classification" if exp in CLASSIFICATION else "regression"
            metric_name = "accuracy" if exp in CLASSIFICATION else "mae"
            for s in seeds:
                metric_val = s.get("test accuracy", s.get("test mae", ""))
                writer.writerow([
                    model, exp, task_type, s.get("seed", 1),
                    s.get("best epoch", ""), s.get("test loss", ""),
                    metric_val, metric_name
                ])

    print(f"  Results written to {filename}")


def main():
    parser = argparse.ArgumentParser(description="Collect experiment results from GCS")
    parser.add_argument("run_name", help="Run name (e.g., run25ep)")
    parser.add_argument("--seeds", type=int, default=5, help="Max seeds to check (default 5)")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV file")
    parser.add_argument("--models", type=str, default=None,
                        help="Space-separated list of models (default: all)")
    args = parser.parse_args()

    global MODELS
    if args.models:
        MODELS = args.models.split()

    print(f"Collecting results for run: {args.run_name}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Seeds:  1-{args.seeds}")

    results = collect(args.run_name, max_seeds=args.seeds)

    found = len(results)
    total = len(MODELS) * len(EXPERIMENTS)
    print(f"\n  Found {found}/{total} experiment results")

    format_table(results)

    if args.csv:
        write_csv_output(results, args.csv)


if __name__ == "__main__":
    main()
