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


def format_table(results):
    """Format results as a readable table, matching Table 3 from the paper."""

    # ── Classification table ──
    # Note: ozone uses F1-score, not accuracy
    print("\n" + "=" * 100)
    print("  CLASSIFICATION — Test Metric at Best Validation Epoch")
    print("  (accuracy % for most tasks; F1-score for ozone)")
    print("=" * 100)

    cls_exps = [e for e in EXPERIMENTS if e in CLASSIFICATION]
    header = f"{'Model':<8}" + "".join(f"{e:>12}" for e in cls_exps)
    print(header)
    print("-" * len(header))

    for model in MODELS:
        row = f"{model:<8}"
        for exp in cls_exps:
            key = (model, exp)
            if key in results:
                seeds = results[key]
                accs = [float(s.get("test accuracy", 0)) for s in seeds]
                mean = sum(accs) / len(accs)
                if len(accs) > 1:
                    import statistics
                    std = statistics.stdev(accs)
                    row += f"{mean:>9.2f}±{std:.1f}"
                else:
                    row += f"{mean:>12.2f}"
            else:
                row += f"{'—':>12}"
        print(row)

    # ── Regression table (MSE / MAE) ──
    print("\n" + "=" * 80)
    print("  REGRESSION — Test MSE (loss) / MAE at Best Validation Epoch")
    print("=" * 80)

    reg_exps = [e for e in EXPERIMENTS if e in REGRESSION]
    header = f"{'Model':<8}" + "".join(f"  {e+' MSE':>14}{e+' MAE':>14}" for e in reg_exps)
    print(header)
    print("-" * len(header))

    for model in MODELS:
        row = f"{model:<8}"
        for exp in reg_exps:
            key = (model, exp)
            if key in results:
                seeds = results[key]
                mses = [float(s.get("test loss", 0)) for s in seeds]
                maes = [float(s.get("test mae", 0)) for s in seeds]
                mean_mse = sum(mses) / len(mses)
                mean_mae = sum(maes) / len(maes)
                row += f"  {mean_mse:>14.6f}{mean_mae:>14.6f}"
            else:
                row += f"  {'—':>14}{'—':>14}"
        print(row)

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
