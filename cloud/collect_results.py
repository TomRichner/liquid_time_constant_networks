#!/usr/bin/env python3
"""
collect_results.py — Download and tabulate best-epoch results from GCS.

Usage:
    python3 cloud/collect_results.py <run_name>
    python3 cloud/collect_results.py run25ep --seeds 1
    python3 cloud/collect_results.py run25ep --seeds 5 --csv out.csv
"""
import argparse
import csv
import datetime
import io
import os
import statistics
import subprocess
import sys
from math import log10, floor

GCS_BUCKET = "gs://liquidneuralnets-experiments/results-py"

EXPERIMENTS = ["har", "gesture", "occupancy", "smnist", "traffic", "power", "ozone_fixed", "person", "cheetah"]
CLASSIFICATION = {"har", "gesture", "occupancy", "smnist", "ozone", "ozone_fixed", "person"}
REGRESSION = {"traffic", "power", "cheetah"}

MODELS = ["lstm", "ctrnn", "node", "ctgru", "ltc", "srnn", "srnn-per-neuron", "srnn-echo",
          "srnn-no-adapt", "srnn-sfa-only", "srnn-std-only", "srnn-E-only"]


def download_run(run_name):
    """Bulk-download all results for a run to a local temp dir. Returns local path."""
    base_dir = "/tmp/collect_results"
    local_dir = os.path.join(base_dir, run_name)
    os.makedirs(base_dir, exist_ok=True)
    gcs_path = f"{GCS_BUCKET}/{run_name}/"
    print(f"  Downloading {gcs_path} → {local_dir}/ ...")
    r = subprocess.run(
        ["gcloud", "storage", "cp", "-r", gcs_path, base_dir],
        capture_output=True, text=True, timeout=120
    )
    if r.returncode != 0:
        print(f"  WARNING: gcloud storage cp failed: {r.stderr.strip()}", file=sys.stderr)
    return local_dir


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
    """Collect results for all experiments/models/seeds from local download."""
    local_dir = download_run(run_name)
    results = {}  # (model, experiment) -> list of dicts (one per seed)

    for model in MODELS:
        for exp in EXPERIMENTS:
            seed_results = []
            for seed in range(1, max_seeds + 1):
                seed_dir = os.path.join(local_dir, model, exp, f"seed{seed}")
                if not os.path.isdir(seed_dir):
                    continue
                # Try known CSV name patterns
                for suffix in [f"{model}_32.csv", f"{model}_32_00.csv"]:
                    csv_path = os.path.join(seed_dir, suffix)
                    if os.path.isfile(csv_path):
                        try:
                            with open(csv_path) as f:
                                d = parse_csv(f.read().strip())
                            d["seed"] = seed
                            seed_results.append(d)
                        except Exception:
                            pass
                        break  # found it, skip alternate suffix

            if seed_results:
                results[(model, exp)] = seed_results

    return results


# ── Formatting helpers ────────────────────────────────────────────────

def _fmt_sigfigs(val, n=3):
    """Format a float to n significant figures."""
    if val == 0:
        return "0"
    digits = n - 1 - floor(log10(abs(val)))
    digits = max(digits, 0)
    return f"{val:.{digits}f}"


def _get_metric(exp):
    """Return (metric_name, csv_key, format_fn) for each experiment."""
    if exp in ("ozone", "ozone_fixed"):
        return "F1-score", "test accuracy", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
    elif exp in CLASSIFICATION:
        return "accuracy", "test accuracy", lambda m, s: f"{m:.2f}% ± {s:.2f}" if s else f"{m:.2f}%"
    elif exp in REGRESSION:
        metric = "MSE" if exp == "cheetah" else "squared error"
        return metric, "test loss", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
    return "loss", "test loss", lambda m, s: f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)


# Experiments where higher is better
HIGHER_IS_BETTER = CLASSIFICATION  # accuracy & F1


def _build_rows(results):
    """Build table data rows: list of (exp, metric_name, [cell_strings], [raw_means])."""
    rows = []
    for exp in EXPERIMENTS:
        metric_name, csv_key, fmt_fn = _get_metric(exp)
        cells = []
        raw_means = []
        for model in MODELS:
            key = (model, exp)
            if key in results:
                seeds = results[key]
                vals = [float(s.get(csv_key, 0)) for s in seeds]
                mean = sum(vals) / len(vals)
                std = statistics.stdev(vals) if len(vals) > 1 else None
                cells.append(fmt_fn(mean, std))
                raw_means.append(mean)
            else:
                cells.append("—")
                raw_means.append(None)
        rows.append((exp, metric_name, cells, raw_means))
    return rows


def _best_index(exp, raw_means):
    """Return index of the best model for this experiment."""
    valid = [(i, v) for i, v in enumerate(raw_means) if v is not None]
    if not valid:
        return None
    if exp in HIGHER_IS_BETTER:
        return max(valid, key=lambda x: x[1])[0]
    else:
        return min(valid, key=lambda x: x[1])[0]


# ── Plain-text table (terminal) ──────────────────────────────────────

def format_plain(results):
    """Format results as plain-text table for terminal output."""
    lines = []
    cw = 20

    lines.append("")
    lines.append("=" * (22 + cw * len(MODELS)))
    lines.append("  Test performance at best validation epoch (Table 3 format)")
    lines.append("=" * (22 + cw * len(MODELS)))

    header = f"{'Dataset':<14}{'Metric':<10}" + "".join(f"{m:>{cw}}" for m in MODELS)
    lines.append(header)
    lines.append("-" * len(header))

    for exp, metric_name, cells, raw_means in _build_rows(results):
        best = _best_index(exp, raw_means)
        parts = []
        for i, c in enumerate(cells):
            if i == best:
                parts.append(f"{'*'+c+'*':>{cw}}")
            else:
                parts.append(f"{c:>{cw}}")
        row = f"{exp:<14}{metric_name:<10}" + "".join(parts)
        lines.append(row)

    lines.append("")
    return lines


# ── Markdown table (pandoc-ready) ────────────────────────────────────

def format_markdown(results, run_name="", num_seeds=1):
    """Format results as pandoc-ready markdown with YAML frontmatter."""
    lines = []

    # YAML frontmatter for pandoc PDF
    lines.append("---")
    lines.append(f'title: "Experiment Results — {run_name}"')
    lines.append(f'date: "{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}"')
    lines.append("geometry: landscape,margin=1.5cm")
    lines.append("fontsize: 10pt")
    lines.append("---")
    lines.append("")
    lines.append(f"# Results: {run_name}")
    lines.append("")
    lines.append(f"Test performance at best validation epoch. Seeds per cell: n={num_seeds}.")
    lines.append("Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).")
    lines.append("")

    # Markdown table
    header = "| Dataset | Metric | " + " | ".join(MODELS) + " |"
    sep = "|---|---|" + "|".join(["---:" for _ in MODELS]) + "|"
    lines.append(header)
    lines.append(sep)

    for exp, metric_name, cells, raw_means in _build_rows(results):
        best = _best_index(exp, raw_means)
        styled = []
        for i, c in enumerate(cells):
            styled.append(f"**{c}**" if i == best else c)
        row = "| " + " | ".join([exp, metric_name] + styled) + " |"
        lines.append(row)

    lines.append("")
    lines.append(f"*Table: {run_name} — {num_seeds} seed(s) per cell.*")
    lines.append("")

    return lines


# ── CSV output ───────────────────────────────────────────────────────

def write_csv_output(results, filename):
    """Write all results to a single CSV file."""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
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


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect experiment results from GCS")
    parser.add_argument("run_name", help="Run name (e.g., run25ep)")
    parser.add_argument("--seeds", type=int, default=5, help="Max seeds to check (default 5)")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV file (overrides default)")
    parser.add_argument("--models", type=str, default=None,
                        help="Space-separated list of models (default: all)")
    parser.add_argument("--no-save", action="store_true", help="Don't save files, print only")
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

    # Print plain-text table to terminal
    for line in format_plain(results):
        print(line)

    # Save outputs
    if not args.no_save:
        out_dir = os.path.join("results", args.run_name)
        os.makedirs(out_dir, exist_ok=True)

        # Save markdown (pandoc-ready)
        md_lines = format_markdown(results, run_name=args.run_name, num_seeds=args.seeds)
        md_file = os.path.join(out_dir, "results.md")
        with open(md_file, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"  Markdown saved to {md_file}")
        print(f"    → PDF: pandoc {md_file} -o {os.path.join(out_dir, 'results.pdf')}")

        # Save data CSV
        csv_file = args.csv or os.path.join(out_dir, "results.csv")
        write_csv_output(results, csv_file)
        print(f"  CSV saved to {csv_file}")


if __name__ == "__main__":
    main()
