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
import json
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
          "srnn-no-adapt", "srnn-no-adapt-no-dales", "srnn-sfa-only", "srnn-std-only",
          "srnn-e-only", "srnn-e-only-echo", "srnn-e-only-per-neuron"]


def download_run(run_name, with_checkpoints=False):
    """Bulk-download all results for a run to project-local tmp dir. Returns local path."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    base_dir = os.path.join(project_dir, "tmp", "collect_results")
    local_dir = os.path.join(base_dir, run_name)
    os.makedirs(base_dir, exist_ok=True)
    gcs_path = f"{GCS_BUCKET}/{run_name}/"
    print(f"  Downloading {gcs_path} → {local_dir}/ ...")
    os.makedirs(local_dir, exist_ok=True)
    cmd = ["gcloud", "storage", "rsync", "-r", gcs_path, local_dir]
    if not with_checkpoints:
        cmd.extend(["--exclude", "checkpoint/.*"])
        print(f"  (skipping checkpoints; use --with-checkpoints to include)")
    r = subprocess.run(
        cmd, capture_output=True, text=True, timeout=600
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


def collect(run_name, max_seeds=5, with_checkpoints=False):
    """Collect results for all experiments/models/seeds from local download.

    Returns (results, timing) where:
      results: {(model, exp): [seed_dicts]}
      timing:  {(model, exp, seed): metadata_dict}  from run_metadata.json
    """
    local_dir = download_run(run_name, with_checkpoints=with_checkpoints)
    results = {}  # (model, experiment) -> list of dicts (one per seed)
    timing = {}   # (model, experiment, seed) -> metadata dict

    for model in MODELS:
        for exp in EXPERIMENTS:
            seed_results = []
            for seed in range(1, max_seeds + 1):
                seed_dir = os.path.join(local_dir, model, exp, f"seed{seed}")
                if not os.path.isdir(seed_dir):
                    continue

                # Read run_metadata.json if present
                meta_path = os.path.join(seed_dir, "run_metadata.json")
                if os.path.isfile(meta_path):
                    try:
                        with open(meta_path) as f:
                            timing[(model, exp, seed)] = json.load(f)
                    except Exception:
                        pass

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

    return results, timing


# ── Formatting helpers ────────────────────────────────────────────────

def _fmt_sigfigs(val, n=3):
    """Format a float to n significant figures."""
    if val == 0:
        return "0"
    digits = n - 1 - floor(log10(abs(val)))
    digits = max(digits, 0)
    return f"{val:.{digits}f}"


def _fmt_partial(base, n, max_n):
    """Append (n=X) annotation when not all seeds are finished."""
    if n < max_n:
        return f"{base} (n={n})"
    return base


def _get_metric(exp):
    """Return (metric_name, csv_key, format_fn) for each experiment.
    format_fn(mean, std, n, max_n) -> formatted string.
    """
    if exp in ("ozone", "ozone_fixed"):
        def fmt(m, s, n, mn):
            base = f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
            return _fmt_partial(base, n, mn)
        return "F1-score", "test accuracy", fmt
    elif exp in CLASSIFICATION:
        def fmt(m, s, n, mn):
            base = f"{m:.2f}% ± {s:.2f}" if s else f"{m:.2f}%"
            return _fmt_partial(base, n, mn)
        return "accuracy", "test accuracy", fmt
    elif exp in REGRESSION:
        metric = "MSE" if exp == "cheetah" else "squared error"
        def fmt(m, s, n, mn):
            base = f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
            return _fmt_partial(base, n, mn)
        return metric, "test loss", fmt
    def fmt(m, s, n, mn):
        base = f"{_fmt_sigfigs(m)} ± {_fmt_sigfigs(s)}" if s else _fmt_sigfigs(m)
        return _fmt_partial(base, n, mn)
    return "loss", "test loss", fmt


# Experiments where higher is better
HIGHER_IS_BETTER = CLASSIFICATION  # accuracy & F1


# ── Timing helpers ────────────────────────────────────────────────────

def _parse_utc(s):
    """Parse a UTC ISO timestamp string to datetime."""
    return datetime.datetime.fromisoformat(s.replace("Z", "+00:00"))


def compute_timing_stats(timing):
    """Compute run-level timing stats from per-cell metadata.

    Returns dict with keys: started, completed, wall_clock, cpu_hours, n_cells.
    """
    starts, ends, durations = [], [], []
    for meta in timing.values():
        completed = meta.get("completed", meta.get("failed_at"))
        dur = meta.get("duration_seconds", 0)
        if completed:
            ct = _parse_utc(completed)
            ends.append(ct)
            if dur:
                starts.append(ct - datetime.timedelta(seconds=dur))
                durations.append(dur)
    if not ends:
        return None
    return {
        "started": min(starts) if starts else None,
        "completed": max(ends),
        "wall_clock": max(ends) - min(starts) if starts else None,
        "cpu_hours": sum(durations) / 3600,
        "n_cells": len(ends),
    }


def _get_median_durations(timing):
    """Get median duration (seconds) per (exp, model). Returns {(exp, model): median_seconds}."""
    medians = {}
    for exp in EXPERIMENTS:
        for model in MODELS:
            durs = [meta["duration_seconds"]
                    for (m, e, s), meta in timing.items()
                    if m == model and e == exp and meta.get("duration_seconds")]
            if durs:
                medians[(exp, model)] = statistics.median(durs)
    return medians


def _build_duration_rows(timing):
    """Build median wall-clock duration table with human-readable format (e.g. '1.2h', '28m')."""
    medians = _get_median_durations(timing)
    rows = []
    for exp in EXPERIMENTS:
        cells = []
        for model in MODELS:
            val = medians.get((exp, model))
            if val is not None:
                median_min = val / 60
                if median_min >= 60:
                    cells.append(f"{median_min / 60:.1f}h")
                else:
                    cells.append(f"{median_min:.0f}m")
            else:
                cells.append("\u2014")
        rows.append((exp, cells))
    return rows


def _build_cpu_hours_rows(timing):
    """Build median CPU-hours table with decimal hours (e.g. '1.21')."""
    medians = _get_median_durations(timing)
    rows = []
    for exp in EXPERIMENTS:
        cells = []
        for model in MODELS:
            val = medians.get((exp, model))
            if val is not None:
                hours = val / 3600
                if hours >= 10:
                    cells.append(f"{hours:.1f}")
                else:
                    cells.append(f"{hours:.2f}")
            else:
                cells.append("\u2014")
        rows.append((exp, cells))
    return rows


def _fmt_timedelta(td):
    """Format a timedelta as e.g. '2d 11h 50m'."""
    total_sec = int(td.total_seconds())
    days = total_sec // 86400
    hours = (total_sec % 86400) // 3600
    minutes = (total_sec % 3600) // 60
    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes or not parts:
        parts.append(f"{minutes}m")
    return " ".join(parts)


def _build_rows(results, max_seeds=5):
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
                n = len(vals)
                mean = sum(vals) / n
                std = statistics.stdev(vals) if n > 1 else None
                cells.append(fmt_fn(mean, std, n, max_seeds))
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

def format_plain(results, timing=None, num_seeds=5):
    """Format results as plain-text table for terminal output."""
    lines = []
    cw = 24

    # Timing summary
    if timing:
        stats = compute_timing_stats(timing)
        if stats:
            lines.append("")
            if stats["started"]:
                lines.append(f"  Started:              {stats['started'].strftime('%Y-%m-%d %H:%M')} UTC")
            lines.append(f"  Completed:            {stats['completed'].strftime('%Y-%m-%d %H:%M')} UTC")
            if stats["wall_clock"]:
                lines.append(f"  Wall-clock elapsed:   {_fmt_timedelta(stats['wall_clock'])}")
            lines.append(f"  Total CPU-hours:      {stats['cpu_hours']:.0f}h ({stats['n_cells']} cells)")

    lines.append("")
    lines.append("=" * (22 + cw * len(MODELS)))
    lines.append("  Test performance at best validation epoch (Table 3 format)")
    lines.append("=" * (22 + cw * len(MODELS)))

    header = f"{'Dataset':<14}{'Metric':<10}" + "".join(f"{m:>{cw}}" for m in MODELS)
    lines.append(header)
    lines.append("-" * len(header))

    for exp, metric_name, cells, raw_means in _build_rows(results, max_seeds=num_seeds):
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

    # Duration tables
    if timing:
        for title, build_fn in [
            ("Median wall-clock duration per experiment/model", _build_duration_rows),
            ("Median CPU-hours per experiment/model", _build_cpu_hours_rows),
        ]:
            dur_rows = build_fn(timing)
            if any(c != "\u2014" for _, cells in dur_rows for c in cells):
                lines.append("")
                lines.append("=" * (14 + cw * len(MODELS)))
                lines.append(f"  {title}")
                lines.append("=" * (14 + cw * len(MODELS)))
                header = f"{'Dataset':<14}" + "".join(f"{m:>{cw}}" for m in MODELS)
                lines.append(header)
                lines.append("-" * len(header))
                for exp, cells in dur_rows:
                    row = f"{exp:<14}" + "".join(f"{c:>{cw}}" for c in cells)
                    lines.append(row)
                lines.append("")

    return lines


# ── Markdown table (pandoc-ready) ────────────────────────────────────

def format_markdown(results, timing=None, run_name="", num_seeds=1):
    """Format results as pandoc-ready markdown with YAML frontmatter."""
    lines = []

    # Compute timing stats for frontmatter
    stats = compute_timing_stats(timing) if timing else None

    # YAML frontmatter for pandoc PDF
    lines.append("---")
    lines.append(f'title: "Experiment Results — {run_name}"')
    if stats and stats["started"]:
        lines.append(f'date: "{stats["started"].strftime("%Y-%m-%d %H:%M")} – {stats["completed"].strftime("%Y-%m-%d %H:%M")} UTC"')
    else:
        lines.append(f'date: "{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}"')
    lines.append("geometry: landscape,margin=1.5cm")
    lines.append("fontsize: 10pt")
    lines.append("---")
    lines.append("")
    lines.append(f"# Results: {run_name}")
    lines.append("")
    lines.append(f"Test performance at best validation epoch. Seeds per cell: n={num_seeds}.")
    lines.append("Paper reference: Hasani et al. 2021, Table 3 (n=5, 200 epochs).")
    if stats:
        lines.append("")
        if stats["started"]:
            lines.append(f"- **Started:** {stats['started'].strftime('%Y-%m-%d %H:%M')} UTC")
        lines.append(f"- **Completed:** {stats['completed'].strftime('%Y-%m-%d %H:%M')} UTC")
        if stats["wall_clock"]:
            lines.append(f"- **Wall-clock elapsed:** {_fmt_timedelta(stats['wall_clock'])}")
        lines.append(f"- **Total CPU-hours:** {stats['cpu_hours']:.0f}h ({stats['n_cells']} cells)")
    lines.append("")

    # Results table
    header = "| Dataset | Metric | " + " | ".join(MODELS) + " |"
    sep = "|---|---|" + "|".join(["---:" for _ in MODELS]) + "|"
    lines.append(header)
    lines.append(sep)

    for exp, metric_name, cells, raw_means in _build_rows(results, max_seeds=num_seeds):
        best = _best_index(exp, raw_means)
        styled = []
        for i, c in enumerate(cells):
            styled.append(f"**{c}**" if i == best else c)
        row = "| " + " | ".join([exp, metric_name] + styled) + " |"
        lines.append(row)

    lines.append("")
    lines.append(f"*Table: {run_name} — {num_seeds} seed(s) per cell.*")
    lines.append("")

    # Duration tables
    if timing:
        for title, build_fn, note in [
            ("Median Wall-Clock Duration", _build_duration_rows, "Durations are median across seeds."),
            ("Median CPU-Hours", _build_cpu_hours_rows, "CPU-hours are median across seeds (1 vCPU per cell)."),
        ]:
            dur_rows = build_fn(timing)
            if any(c != "\u2014" for _, cells in dur_rows for c in cells):
                lines.append(f"## {title}")
                lines.append("")
                header = "| Dataset | " + " | ".join(MODELS) + " |"
                sep = "|---|" + "|".join(["---:" for _ in MODELS]) + "|"
                lines.append(header)
                lines.append(sep)
                for exp, cells in dur_rows:
                    row = "| " + " | ".join([exp] + cells) + " |"
                    lines.append(row)
                lines.append("")
                lines.append(f"*{note}*")
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


# ── PDF generation ───────────────────────────────────────────────────

def _run_pandoc(md_file, pdf_file, paperwidth="24in", paperheight="11in"):
    """Generate a PDF from a markdown file using pandoc."""
    try:
        subprocess.run([
            "pandoc", md_file, "-o", pdf_file,
            "-V", f"geometry:paperwidth={paperwidth}",
            "-V", f"geometry:paperheight={paperheight}",
            "-V", "geometry:margin=0.5in",
            "-V", "fontsize=8pt",
        ], check=True, capture_output=True, text=True, timeout=60)
        print(f"  PDF saved to {pdf_file}")
    except FileNotFoundError:
        print(f"  PDF skipped (pandoc not installed)")
    except subprocess.CalledProcessError as e:
        print(f"  PDF failed: {e.stderr.strip()}")


# ── SRNN parameter inspection ────────────────────────────────────────

def _run_inspect_srnn_params(run_name, out_dir):
    """Run inspect_srnn_params.py for the first experiment with SRNN checkpoints."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    inspect_script = os.path.join(script_dir, "inspect_srnn_params.py")
    if not os.path.isfile(inspect_script):
        print(f"  inspect_srnn_params.py not found, skipping")
        return

    # Auto-detect: find first experiment with SRNN checkpoint data
    local_dir = os.path.join(project_dir, "tmp", "collect_results", run_name)
    srnn_models = [m for m in MODELS if m.startswith("srnn")]
    target_exp = None
    for exp in EXPERIMENTS:
        for model in srnn_models:
            ckpt_dir = os.path.join(local_dir, model, exp, "seed1", "checkpoint")
            if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
                target_exp = exp
                break
        if target_exp:
            break

    if not target_exp:
        print(f"  No SRNN checkpoints found for any experiment, skipping param inspection")
        return

    print(f"\n  Running SRNN parameter inspection (experiment: {target_exp})...")
    try:
        subprocess.run([
            sys.executable, inspect_script,
            "--run", run_name,
            "--experiment", target_exp,
            "--seed", "1",
            "--out_dir", out_dir,
        ], check=True, capture_output=True, text=True, timeout=120)
    except subprocess.CalledProcessError as e:
        print(f"  inspect_srnn_params failed: {e.stderr.strip()[:200]}")
        return
    except FileNotFoundError:
        print(f"  Python not found for inspect_srnn_params")
        return

    # Generate PDF from srnn_params.md
    params_md = os.path.join(out_dir, "srnn_params.md")
    params_pdf = os.path.join(out_dir, "srnn_params.pdf")
    if os.path.isfile(params_md):
        _run_pandoc(params_md, params_pdf, paperwidth="11in", paperheight="8.5in")


# ── Main ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect experiment results from GCS")
    parser.add_argument("run_name", help="Run name (e.g., run25ep)")
    parser.add_argument("--seeds", type=int, default=5, help="Max seeds to check (default 5)")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV file (overrides default)")
    parser.add_argument("--models", type=str, default=None,
                        help="Space-separated list of models (default: all)")
    parser.add_argument("--no-save", action="store_true", help="Don't save files, print only")
    parser.add_argument("--with-checkpoints", action="store_true",
                        help="Also download checkpoint files (slow for large runs)")
    args = parser.parse_args()

    global MODELS
    if args.models:
        MODELS = args.models.split()

    print(f"Collecting results for run: {args.run_name}")
    print(f"  Models: {', '.join(MODELS)}")
    print(f"  Seeds:  1-{args.seeds}")

    results, timing = collect(args.run_name, max_seeds=args.seeds, with_checkpoints=args.with_checkpoints)

    found = len(results)
    total = len(MODELS) * len(EXPERIMENTS)
    print(f"\n  Found {found}/{total} experiment results")

    # Print plain-text table to terminal
    for line in format_plain(results, timing=timing, num_seeds=args.seeds):
        print(line)

    # Save outputs
    if not args.no_save:
        out_dir = os.path.join("results", args.run_name)
        os.makedirs(out_dir, exist_ok=True)

        # Save markdown (pandoc-ready)
        md_lines = format_markdown(results, timing=timing, run_name=args.run_name, num_seeds=args.seeds)
        md_file = os.path.join(out_dir, "results.md")
        with open(md_file, "w") as f:
            f.write("\n".join(md_lines) + "\n")
        print(f"  Markdown saved to {md_file}")

        # Generate PDF (extra-wide landscape for 15-model table)
        pdf_file = os.path.join(out_dir, "results.pdf")
        _run_pandoc(md_file, pdf_file)

        # Save data CSV
        csv_file = args.csv or os.path.join(out_dir, "results.csv")
        write_csv_output(results, csv_file)
        print(f"  CSV saved to {csv_file}")

        # Generate SRNN parameter tables (Init/Best/Last)
        _run_inspect_srnn_params(args.run_name, out_dir)


if __name__ == "__main__":
    main()
