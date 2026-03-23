#!/usr/bin/env python3
"""Check progress of running full300 VMs by SSHing and reading training logs.

Usage:
    python3 cloud/check_progress.py full300
    python3 cloud/check_progress.py full300 --experiment smnist
"""
import subprocess, re, sys, argparse, json
from collections import defaultdict

def get_running_vms(run_name):
    """List running VMs for a given run name."""
    r = subprocess.run(
        ["gcloud", "compute", "instances", "list",
         f"--filter=status=RUNNING AND name~{run_name}",
         "--format=json"],
        capture_output=True, text=True, timeout=30
    )
    if r.returncode != 0:
        print(f"Error listing VMs: {r.stderr}", file=sys.stderr)
        return []
    vms = json.loads(r.stdout)
    results = []
    for vm in vms:
        name = vm["name"]
        zone = vm["zone"].split("/")[-1]
        # Parse: run_name-model-experiment-seedN
        # e.g. full300-lstm-smnist-seed1
        suffix = name[len(run_name) + 1:]  # remove "full300-"
        parts = suffix.rsplit("-seed", 1)
        if len(parts) == 2:
            model_exp = parts[0]
            seed = int(parts[1])
            # The experiment is the last segment, model is everything before
            # But model names contain hyphens too... need to match known experiments
            for exp in ["har", "gesture", "occupancy", "smnist", "traffic",
                        "power", "ozone-fixed", "person", "cheetah"]:
                if model_exp.endswith(f"-{exp}"):
                    model = model_exp[:-(len(exp) + 1)]
                    results.append({"name": name, "zone": zone, "model": model,
                                    "experiment": exp, "seed": seed})
                    break
    return results


def get_latest_epoch(vm_name, zone):
    """SSH into a VM and get the latest training epoch and metrics."""
    try:
        r = subprocess.run(
            ["gcloud", "compute", "ssh", vm_name, f"--zone={zone}",
             "--tunnel-through-iap",
             "--command=tail -1 /tmp/training.log 2>/dev/null"],
            capture_output=True, text=True, timeout=30
        )
        if r.returncode != 0:
            return None
        line = r.stdout.strip()
        if not line:
            return None

        # Parse: "Epochs 238, train loss: 0.01, train accuracy: 99.85%, ..."
        m = re.search(r'Epochs (\d+)', line)
        if not m:
            return None

        result = {"epoch": int(m.group(1)), "raw": line}

        # Try to parse test accuracy
        ta = re.search(r'test accuracy: ([\d.]+)%', line)
        if ta:
            result["test_accuracy"] = float(ta.group(1))

        # Try to parse test loss
        tl = re.search(r'test loss: ([\d.]+)', line)
        if tl:
            result["test_loss"] = float(tl.group(1))

        return result
    except subprocess.TimeoutExpired:
        return None


def main():
    parser = argparse.ArgumentParser(description="Check progress of running experiment VMs")
    parser.add_argument("run_name", default="full300", nargs="?")
    parser.add_argument("--experiment", default=None, help="Filter to specific experiment")
    args = parser.parse_args()

    print(f"  Listing running VMs for {args.run_name}...")
    vms = get_running_vms(args.run_name)

    if args.experiment:
        vms = [v for v in vms if v["experiment"] == args.experiment]

    if not vms:
        print("  No running VMs found.")
        return

    # Group by experiment
    by_exp = defaultdict(list)
    for vm in vms:
        by_exp[vm["experiment"]].append(vm)

    total = len(vms)
    print(f"  Found {total} running VMs across {len(by_exp)} experiment(s)")
    print(f"  SSHing into each to check progress...\n")

    all_results = []  # (vm, progress_dict)
    for i, vm in enumerate(vms):
        sys.stdout.write(f"\r  Checking {i+1}/{total}: {vm['name']}...")
        sys.stdout.flush()
        progress = get_latest_epoch(vm["name"], vm["zone"])
        all_results.append((vm, progress))

    print(f"\r  Checked {total}/{total} VMs" + " " * 40)
    print()

    # Group results by (experiment, model) and compute median
    from statistics import median
    grouped = defaultdict(list)
    for vm, prog in all_results:
        if prog:
            grouped[(vm["experiment"], vm["model"])].append(prog)

    # Print progress table per experiment
    for exp in sorted(by_exp.keys()):
        print(f"  === {exp} ===")
        print(f"  {'Model':<28s} {'Seeds':>6s} {'Median Epoch':>14s} {'Metric':>18s}")
        print(f"  {'-'*28} {'-'*6} {'-'*14} {'-'*18}")

        models_data = []
        for (e, model), progs in sorted(grouped.items()):
            if e != exp:
                continue
            epochs = [p["epoch"] for p in progs]
            med_epoch = median(epochs)
            n = len(progs)

            # Get metric (accuracy or loss)
            if "test_accuracy" in progs[0]:
                vals = [p["test_accuracy"] for p in progs if "test_accuracy" in p]
                med_val = median(vals)
                metric_str = f"{med_val:.2f}% acc"
            elif "test_loss" in progs[0]:
                vals = [p["test_loss"] for p in progs if "test_loss" in p]
                med_val = median(vals)
                metric_str = f"{med_val:.4f} loss"
            else:
                metric_str = "—"

            print(f"  {model:<28s} {f'n={n}':>6s} {f'{int(med_epoch)}/300':>14s} {metric_str:>18s}")
            models_data.append((model, n, med_epoch, metric_str))

        print()


if __name__ == "__main__":
    main()
