"""Inspect SRNN parameters across Init/Best/Last checkpoints for all SRNN variants.

Usage:
  python3 cloud/inspect_srnn_params.py --run all50ep --experiment har --seed 1
  python3 cloud/inspect_srnn_params.py --run all50ep --experiment har --seed 1 --local /tmp/collect_results/all50ep
"""
import os, sys, argparse, subprocess, glob
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf

def softplus(x):
    return np.log(1 + np.exp(x))

# ── Model definitions ────────────────────────────────────────────────
# Shared-parameter models (scalar dynamics params)
SRNN_SHARED_MODELS = [
    "srnn",
    "srnn-echo",
    "srnn-no-adapt",
    "srnn-no-adapt-no-dales",
    "srnn-sfa-only",
    "srnn-std-only",
    "srnn-e-only",
    "srnn-e-only-echo",
]

# Per-neuron models (vector dynamics params)
SRNN_PER_NEURON_MODELS = [
    "srnn-per-neuron",
    "srnn-e-only-per-neuron",
]

# Which checkpoint variable prefix each model variant uses
def ckpt_prefix(model):
    """TF checkpoint variable prefix for SRNN models."""
    return "rnn/srnn"

# Whether model uses Dale's law (softplus weight transform)
def uses_dales(model):
    return model != "srnn-no-adapt-no-dales"

# Scalar params: (ckpt_var_suffix, display_name, transform, required_condition)
SFA_E_MODELS = ("srnn", "srnn-echo", "srnn-sfa-only", "srnn-e-only", "srnn-e-only-echo",
                 "srnn-per-neuron", "srnn-e-only-per-neuron")
SFA_I_MODELS = ("srnn", "srnn-echo", "srnn-per-neuron")
STD_E_MODELS = ("srnn", "srnn-echo", "srnn-std-only", "srnn-e-only", "srnn-e-only-echo",
                 "srnn-per-neuron", "srnn-e-only-per-neuron")
STD_I_MODELS = ("srnn", "srnn-echo", "srnn-per-neuron")

PARAMS = [
    ("log_tau_d",       "tau_d",        softplus, lambda m: True),
    ("a_0",             "a_0",          None,     lambda m: True),
    ("log_tau_a_E_lo",  "tau_a_E_lo",   softplus, lambda m: m in SFA_E_MODELS),
    ("log_tau_a_E_hi",  "tau_a_E_hi",   softplus, lambda m: m in SFA_E_MODELS),
    ("log_tau_a_I_lo",  "tau_a_I_lo",   softplus, lambda m: m in SFA_I_MODELS),
    ("log_tau_a_I_hi",  "tau_a_I_hi",   softplus, lambda m: m in SFA_I_MODELS),
    ("log_c_E",         "c_E",          softplus, lambda m: m in SFA_E_MODELS),
    ("log_c_I",         "c_I",          softplus, lambda m: m in SFA_I_MODELS),
    ("c_0_E",           "c_0_E",        None,     lambda m: m in SFA_E_MODELS),
    ("c_0_I",           "c_0_I",        None,     lambda m: m in SFA_I_MODELS),
    ("log_tau_b_E_rec", "tau_b_E_rec",  softplus, lambda m: m in STD_E_MODELS),
    ("log_tau_b_E_rel", "tau_b_E_rel",  softplus, lambda m: m in STD_E_MODELS),
    ("log_tau_b_I_rec", "tau_b_I_rec",  softplus, lambda m: m in STD_I_MODELS),
    ("log_tau_b_I_rel", "tau_b_I_rel",  softplus, lambda m: m in STD_I_MODELS),
]

CHECKPOINT_SUFFIXES = {
    "Init": "_init",
    "Best": "",       # no suffix = best validation
    "Last": "_last",
}

N, N_E = 32, 16

def _read_weights(reader, prefix, model_type):
    """Read and process weight matrices from checkpoint."""
    W_raw = reader.get_tensor(f"{prefix}/W")
    if uses_dales(model_type):
        W_sp = softplus(W_raw)
        W_E = W_sp[:, :N_E]
        W_I = -W_sp[:, N_E:]
        W_dales = np.concatenate([W_E, W_I], axis=1)
    else:
        # No Dale's law — raw weights used directly
        W_E = W_raw[:, :N_E]
        W_I = W_raw[:, N_E:]
        W_dales = W_raw
    W_in = reader.get_tensor(f"{prefix}/W_in")
    return W_E, W_I, W_dales, W_in


def read_checkpoint(ckpt_path, model_type):
    """Read all SRNN params from a checkpoint file."""
    if not os.path.exists(ckpt_path + ".index"):
        return None

    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = reader.get_variable_to_shape_map()
    prefix = ckpt_prefix(model_type)

    row = {}
    for var_suffix, display_name, transform, condition in PARAMS:
        if not condition(model_type):
            continue
        full_name = f"{prefix}/{var_suffix}"
        if full_name in var_map:
            val = reader.get_tensor(full_name).flatten()[0]
            row[display_name] = transform(val) if transform else float(val)

    W_E, W_I, W_dales, W_in = _read_weights(reader, prefix, model_type)
    row["W_E_mean"] = float(W_E.mean())
    row["W_E_std"] = float(W_E.std())
    row["W_I_mean"] = float(W_I.mean())
    row["W_I_std"] = float(W_I.std())
    row["W_all_mean"] = float(W_dales.mean())
    row["W_all_std"] = float(W_dales.std())
    row["W_in_mean"] = float(W_in.mean())
    row["W_in_std"] = float(W_in.std())

    return row

def read_per_neuron_checkpoint(ckpt_path, model_type):
    """Read per-neuron SRNN params, return dict of {display_name: (mean, std)}."""
    if not os.path.exists(ckpt_path + ".index"):
        return None

    reader = tf.train.load_checkpoint(ckpt_path)
    var_map = reader.get_variable_to_shape_map()
    prefix = "rnn/srnn"
    row = {}

    has_I = model_type in ("srnn-per-neuron",)  # full model has I neurons

    # Params that span all n neurons, split into E/I
    for var_suffix, base_name, transform in [
        ("log_tau_d", "tau_d", softplus),
        ("a_0", "a_0", None),
    ]:
        full = f"{prefix}/{var_suffix}"
        vals = reader.get_tensor(full).flatten()
        if transform:
            vals = transform(vals)
        e_vals = vals[:N_E]
        i_vals = vals[N_E:]
        row[f"{base_name} (E)"] = (float(e_vals.mean()), float(e_vals.std()))
        if has_I:
            row[f"{base_name} (I)"] = (float(i_vals.mean()), float(i_vals.std()))

    # E-neuron params
    for var_suffix, display_name, transform in [
        ("log_tau_a_E_lo", "tau_a_E_lo", softplus),
        ("log_tau_a_E_hi", "tau_a_E_hi", softplus),
        ("log_c_E", "c_E", softplus),
        ("c_0_E", "c_0_E", None),
        ("log_tau_b_E_rec", "tau_b_E_rec", softplus),
        ("log_tau_b_E_rel", "tau_b_E_rel", softplus),
    ]:
        full = f"{prefix}/{var_suffix}"
        if full in var_map:
            vals = reader.get_tensor(full).flatten()
            if transform:
                vals = transform(vals)
            row[display_name] = (float(vals.mean()), float(vals.std()))

    # I-neuron params (only for full srnn-per-neuron)
    if has_I:
        for var_suffix, display_name, transform in [
            ("log_tau_a_I_lo", "tau_a_I_lo", softplus),
            ("log_tau_a_I_hi", "tau_a_I_hi", softplus),
            ("log_c_I", "c_I", softplus),
            ("c_0_I", "c_0_I", None),
            ("log_tau_b_I_rec", "tau_b_I_rec", softplus),
            ("log_tau_b_I_rel", "tau_b_I_rel", softplus),
        ]:
            full = f"{prefix}/{var_suffix}"
            if full in var_map:
                vals = reader.get_tensor(full).flatten()
                if transform:
                    vals = transform(vals)
                row[display_name] = (float(vals.mean()), float(vals.std()))

    # Weight matrices
    W_E, W_I, W_dales, W_in = _read_weights(reader, prefix, model_type)
    row["W_E"] = (float(W_E.mean()), float(W_E.std()))
    row["W_I"] = (float(W_I.mean()), float(W_I.std()))
    row["W_all"] = (float(W_dales.mean()), float(W_dales.std()))
    row["W_in"] = (float(W_in.mean()), float(W_in.std()))

    return row

def fmt(v):
    if abs(v) < 0.001:
        return f"{v:.6f}"
    elif abs(v) > 100:
        return f"{v:.1f}"
    else:
        return f"{v:.4f}"

def get_ckpt_dir(base_dir, model, experiment, seed):
    """Get the checkpoint directory for a model from the collect_results download."""
    return os.path.join(base_dir, model, experiment, f"seed{seed}", "checkpoint")


EXPERIMENTS = ["har", "gesture", "occupancy", "smnist", "traffic", "power", "ozone_fixed", "person", "cheetah"]

def _find_experiments_with_srnn_checkpoints(base_dir, seed):
    """Find experiments that have at least one SRNN checkpoint."""
    all_models = SRNN_SHARED_MODELS + SRNN_PER_NEURON_MODELS
    found = []
    for exp in EXPERIMENTS:
        for model in all_models:
            ckpt_dir = get_ckpt_dir(base_dir, model, exp, seed)
            if os.path.isdir(ckpt_dir) and os.listdir(ckpt_dir):
                found.append(exp)
                break
    return found


def generate_tables(base_dir, out_dir, run_name, experiments, seed):
    """Generate per-model Init/Best/Last parameter tables with experiments as rows.

    Args:
        experiments: list of experiment names, or ["all"] to auto-detect.
    """
    if experiments == ["all"]:
        experiments = _find_experiments_with_srnn_checkpoints(base_dir, seed)
        if not experiments:
            print("  No experiments found with SRNN checkpoints.")
            return
        print(f"  Auto-detected experiments: {', '.join(experiments)}")

    md_lines = []
    md_lines.append(f"# SRNN Parameters: Init vs Best vs Last")
    md_lines.append(f"**Run:** {run_name} | **Seed:** {seed}\n")

    console_output = []

    # ── Shared-parameter models ──────────────────────────────────────
    for model in SRNN_SHARED_MODELS:
        # Collect data across all experiments
        model_data = {}  # {experiment: {stage: row_dict}}
        for exp in experiments:
            ckpt_dir = get_ckpt_dir(base_dir, model, exp, seed)
            exp_data = {}
            for stage, suffix in CHECKPOINT_SUFFIXES.items():
                ckpt_path = os.path.join(ckpt_dir, f"{model}{suffix}")
                row = read_checkpoint(ckpt_path, model)
                if row is not None:
                    exp_data[stage] = row
            if exp_data:
                model_data[exp] = exp_data

        if not model_data:
            print(f"  {model}: no checkpoints found in any experiment, skipping")
            continue

        print(f"  {model}: found data for {', '.join(model_data.keys())}")

        # Determine which params this model has
        param_names = [d for _, d, _, cond in PARAMS if cond(model)]
        weight_stats = ["W_E", "W_I", "W_all", "W_in"]  # mean ± std pairs

        # Write one sub-table per checkpoint stage
        md_lines.append(f"---\n## `{model}`\n")
        console_output.append(f"\n{'='*60}")
        console_output.append(f"  {model}")
        console_output.append(f"{'='*60}")

        for stage in ["Init", "Best", "Last"]:
            # Check if any experiment has this stage
            exps_with_stage = [e for e in model_data if stage in model_data[e]]
            if not exps_with_stage:
                continue

            md_lines.append(f"### {stage}\n")
            # Header: Experiment | param1 | param2 | ... | W_E | W_I | W_all | W_in
            all_cols = param_names + [f"{w} mean" for w in weight_stats] + [f"{w} std" for w in weight_stats]
            header = "| Experiment |" + " | ".join(all_cols) + " |"
            sep = "|---|" + "|".join(["---:"] * len(all_cols)) + "|"
            md_lines.append(header)
            md_lines.append(sep)

            for exp in experiments:
                if exp not in model_data or stage not in model_data[exp]:
                    continue
                d = model_data[exp][stage]
                vals = []
                for p in param_names:
                    vals.append(fmt(d[p]) if p in d else "—")
                for w in weight_stats:
                    vals.append(fmt(d.get(f"{w}_mean", 0)))
                for w in weight_stats:
                    vals.append(fmt(d.get(f"{w}_std", 0)))
                md_lines.append(f"| {exp} | " + " | ".join(vals) + " |")

            md_lines.append("")

            # Console
            console_output.append(f"\n  {stage}:")
            col_w = 10
            hdr = f"{'Experiment':<12s}" + "".join(f"{p:>{col_w}s}" for p in param_names[:6])
            console_output.append("  " + hdr)
            for exp in exps_with_stage:
                d = model_data[exp][stage]
                line = f"{exp:<12s}"
                for p in param_names[:6]:
                    line += f"{fmt(d.get(p, 0)):>{col_w}s}"
                console_output.append("  " + line)

    # ── Per-neuron models ────────────────────────────────────────────
    for model in SRNN_PER_NEURON_MODELS:
        has_I = model == "srnn-per-neuron"

        model_data = {}
        for exp in experiments:
            ckpt_dir = get_ckpt_dir(base_dir, model, exp, seed)
            exp_data = {}
            for stage, suffix in CHECKPOINT_SUFFIXES.items():
                ckpt_path = os.path.join(ckpt_dir, f"{model}{suffix}")
                row = read_per_neuron_checkpoint(ckpt_path, model)
                if row is not None:
                    exp_data[stage] = row
            if exp_data:
                model_data[exp] = exp_data

        if not model_data:
            print(f"  {model}: no checkpoints found in any experiment, skipping")
            continue

        print(f"  {model}: found data for {', '.join(model_data.keys())}")

        # Build param list
        pn_params = ["tau_d (E)"]
        if has_I:
            pn_params.append("tau_d (I)")
        pn_params.append("a_0 (E)")
        if has_I:
            pn_params.append("a_0 (I)")
        pn_params.extend(["tau_a_E_lo", "tau_a_E_hi", "c_E", "c_0_E"])
        if has_I:
            pn_params.extend(["tau_a_I_lo", "tau_a_I_hi", "c_I", "c_0_I"])
        pn_params.extend(["tau_b_E_rec", "tau_b_E_rel"])
        if has_I:
            pn_params.extend(["tau_b_I_rec", "tau_b_I_rel"])
        pn_params.extend(["W_E", "W_I", "W_all", "W_in"])

        def fmt_ms(mean, std):
            return f"{fmt(mean)}±{fmt(std)}"

        md_lines.append(f"---\n## `{model}` (mean ± std across neurons)\n")
        console_output.append(f"\n{'='*60}")
        console_output.append(f"  {model} (per-neuron)")
        console_output.append(f"{'='*60}")

        for stage in ["Init", "Best", "Last"]:
            exps_with_stage = [e for e in model_data if stage in model_data[e]]
            if not exps_with_stage:
                continue

            md_lines.append(f"### {stage}\n")
            header = "| Experiment |" + " | ".join(pn_params) + " |"
            sep = "|---|" + "|".join(["---:"] * len(pn_params)) + "|"
            md_lines.append(header)
            md_lines.append(sep)

            for exp in experiments:
                if exp not in model_data or stage not in model_data[exp]:
                    continue
                d = model_data[exp][stage]
                vals = []
                for p in pn_params:
                    if p in d:
                        m, s = d[p]
                        vals.append(fmt_ms(m, s))
                    else:
                        vals.append("—")
                md_lines.append(f"| {exp} | " + " | ".join(vals) + " |")

            md_lines.append("")

            # Console
            console_output.append(f"\n  {stage}:")
            col_w = 20
            hdr = f"{'Experiment':<12s}" + "".join(f"{p:>{col_w}s}" for p in pn_params[:4])
            console_output.append("  " + hdr)
            for exp in exps_with_stage:
                d = model_data[exp][stage]
                line = f"{exp:<12s}"
                for p in pn_params[:4]:
                    if p in d:
                        m, s = d[p]
                        line += f"{fmt_ms(m, s):>{col_w}s}"
                    else:
                        line += f"{'—':>{col_w}s}"
                console_output.append("  " + line)

    # Write outputs
    os.makedirs(out_dir, exist_ok=True)
    md_path = os.path.join(out_dir, "srnn_params.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"\n  Markdown: {md_path}")

    # Print console
    print("\n" + "=" * 60)
    print(f"  SRNN Parameters: {run_name} / seed{seed}")
    print("=" * 60)
    for line in console_output:
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Inspect SRNN parameters across checkpoints")
    parser.add_argument("--run", default="all50ep", help="Run name in GCS")
    parser.add_argument("--experiment", default="all",
                        help="Experiment name, or 'all' to auto-detect (default: all)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--local", default=None,
                        help="Path to local collect_results download dir (e.g. /tmp/collect_results/all50ep)")
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (default: results/<run>/)")
    args = parser.parse_args()

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    if args.local:
        base_dir = args.local
    else:
        # Use collect_results download location, or download if needed
        base_dir = os.path.join(base, "tmp", "collect_results", args.run)
        if not os.path.exists(base_dir):
            print(f"  No local data found. Downloading from GCS...")
            gcs_path = f"gs://liquidneuralnets-experiments/results-py/{args.run}/"
            dl_base = os.path.join(base, "tmp", "collect_results")
            os.makedirs(dl_base, exist_ok=True)
            subprocess.run(
                ["gcloud", "storage", "cp", "-r", gcs_path, dl_base],
                check=True, capture_output=True
            )

    out_dir = args.out_dir or os.path.join(base, "results", args.run)
    experiments = [args.experiment] if args.experiment != "all" else ["all"]
    generate_tables(base_dir, out_dir, args.run, experiments, args.seed)


if __name__ == "__main__":
    main()
