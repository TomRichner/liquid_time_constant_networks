#!/usr/bin/env python3
"""Standalone forward simulation of the SRNN-E-only model variant.

Uses the ACTUAL SRNNCell from srnn_model.py via TF1 session.
Produces a multi-panel figure of stimulus, x(t), r(t), sum(a_k(t)), b(t).

Usage:
    python3 simulate_srnn_eonly.py                    # fused, 300 neurons, 60s
    python3 simulate_srnn_eonly.py --solver explicit   # explicit Euler
    python3 simulate_srnn_eonly.py --n 32 --duration 3 # smaller test
"""

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import argparse
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import srnn_model
from srnn_model import SRNNCell, piecewise_sigmoid

# ══════════════════════════════════════════════════════════════════════
# Color palettes (ported from Matlab SRNNModel2)
# ══════════════════════════════════════════════════════════════════════

EXCITATORY_COLORS = np.array([
    [1.00, 0.00, 0.00],
    [1.00, 0.75, 0.00],
    [0.85, 0.20, 0.45],
    [0.90, 0.10, 0.60],
    [0.90, 0.55, 0.00],
    [0.55, 0.27, 0.27],
    [0.86, 0.08, 0.24],
    [0.60, 0.15, 0.45],
])

INHIBITORY_COLORS = np.array([
    [0.00, 0.45, 0.74],
    [0.00, 0.75, 1.00],
    [0.20, 0.47, 0.62],
    [0.00, 0.50, 0.50],
    [0.30, 0.75, 0.93],
    [0.25, 0.62, 0.75],
    [0.00, 0.80, 0.80],
    [0.15, 0.55, 0.65],
])


def get_color(neuron_idx, n_E):
    """Get color for a neuron: warm for E, cool for I."""
    if neuron_idx < n_E:
        return EXCITATORY_COLORS[neuron_idx % len(EXCITATORY_COLORS)]
    else:
        return INHIBITORY_COLORS[(neuron_idx - n_E) % len(INHIBITORY_COLORS)]


def plot_lines(ax, t, data, n_E, ylabel_text, ylim_range=None):
    """Plot lines with E/I color coding. data shape: (T, n_neurons)."""
    n_neurons = data.shape[1]
    for i in range(n_neurons):
        color = get_color(i, n_E)
        ax.plot(t, data[:, i], color=color, linewidth=0.7, alpha=0.8)
    ax.set_ylabel(ylabel_text, fontsize=10)
    if ylim_range is not None:
        ax.set_ylim(ylim_range)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="SRNN-E-only forward simulation")
    parser.add_argument('--solver', default='semi_implicit',
                        choices=['semi_implicit', 'explicit'],
                        help='ODE solver: semi_implicit (fused) or explicit Euler')
    parser.add_argument('--duration', type=float, default=60.0, help='Simulation duration (s)')
    parser.add_argument('--fs', type=float, default=400.0, help='Sample rate (Hz)')
    parser.add_argument('--n', type=int, default=300, help='Number of neurons')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--stim-amplitude', type=float, default=0.5,
                        help='Step stimulus amplitude')
    parser.add_argument('--stim-on', type=float, default=20.0,
                        help='Stimulus onset time (s)')
    parser.add_argument('--stim-off', type=float, default=40.0,
                        help='Stimulus offset time (s)')
    parser.add_argument('--stim-frac', type=float, default=0.3,
                        help='Fraction of E neurons that receive stimulus')
    parser.add_argument('--a0', type=float, default=0.35,
                        help='Activation threshold a_0 (default: 0.35)')
    parser.add_argument('--sc', type=float, default=0.0,
                        help='Piecewise sigmoid center S_c (default: 0.0)')
    parser.add_argument('--indegree', type=int, default=None,
                        help='Recurrent indegree (None = full connectivity)')
    args = parser.parse_args()

    n = args.n
    n_E = n // 2
    n_I = n - n_E
    h = 1.0 / args.fs
    T_total = args.duration
    T_steps = int(round(T_total * args.fs))
    t_stim_on = args.stim_on
    t_stim_off = args.stim_off

    solver_label = "Semi-implicit (fused)" if args.solver == "semi_implicit" else "Explicit Euler"

    # Select random subset of E neurons to stimulate
    np.random.seed(args.seed + 99)
    n_stim = max(1, int(round(args.stim_frac * n_E)))
    stim_neurons = np.sort(np.random.choice(n_E, n_stim, replace=False))

    print(f"SRNN-E-only forward simulation (using real SRNNCell from srnn_model.py)")
    print(f"  Solver:    {solver_label}")
    print(f"  Neurons:   {n} ({n_E} E, {n_I} I)")
    print(f"  Steps:     {T_steps} ({T_total:.1f}s at {args.fs} Hz)")
    print(f"  Step size: h = {h:.4f}s")
    print(f"  Stimulus:  [{t_stim_on:.0f}s, {t_stim_off:.0f}s], amplitude={args.stim_amplitude}")
    print(f"             {n_stim}/{n_E} E neurons stimulated")
    indegree_str = str(args.indegree) if args.indegree else f"{n} (full)"
    print(f"  a_0 = {args.a0},  S_c = {args.sc},  indegree = {indegree_str}")

    # Monkey-patch piecewise_sigmoid if S_c != 0.0 so the real cell uses it
    if args.sc != 0.0:
        _original_ps = srnn_model.piecewise_sigmoid
        custom_sc = args.sc
        def _patched_ps(x, S_a=0.9, S_c=None):
            return _original_ps(x, S_a=S_a, S_c=custom_sc)
        srnn_model.piecewise_sigmoid = _patched_ps
        print(f"  [patched piecewise_sigmoid with S_c={custom_sc}]")

    # ── Build TF graph ──────────────────────────────────────────────────
    tf.reset_default_graph()
    tf.set_random_seed(args.seed)

    # Input is n-dimensional so we can target specific neurons
    input_ph = tf.placeholder(tf.float32, [1, n], name='input')
    state_ph = tf.placeholder(tf.float32, [1, None], name='state')

    cell = SRNNCell(
        num_units=n, n_E=n_E,
        n_a_E=3, n_a_I=0,
        n_b_E=1, n_b_I=0,
        ode_solver_unfolds=1,
        h=h,
        solver=args.solver,
        readout="synaptic",
        per_neuron=False,
        dales=True,
        indegree=args.indegree
    )

    # Call cell once to build all variables
    with tf.variable_scope("sim"):
        output_op, new_state_op = cell(input_ph, state_ph)

    # Build ops to extract state components for recording
    # (uses the potentially-patched piecewise_sigmoid)
    with tf.variable_scope("sim", reuse=True):
        parts_dict = cell._unpack_state(state_ph)
        x_op = parts_dict['x']                         # (1, n)
        x_eff_op = cell._compute_x_eff(parts_dict)     # (1, n)
        r_op = srnn_model.piecewise_sigmoid(x_eff_op - cell.a_0)  # (1, n)
        b_op = cell._extract_b(parts_dict)             # (1, n)

        # a_E sum over k timescales: (1, n_E, 3) -> sum -> (1, n_E)
        a_E_sum_op = tf.reduce_sum(parts_dict['a_E'], axis=2) if parts_dict['a_E'] is not None else None

    # ── Session ─────────────────────────────────────────────────────────
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Override W_in = eye(n) so input maps 1-to-1 to neurons
        # (matches Matlab SRNNModel2 convention where W_in = eye(n))
        sess.run(cell.W_in.assign(np.eye(n, dtype=np.float32)))
        print("  W_in overridden to eye(n) for direct neuron targeting")

        # Override a_0 if specified
        if args.a0 != 0.35:
            a0_shape = sess.run(cell.a_0).shape
            sess.run(cell.a_0.assign(np.full(a0_shape, args.a0, dtype=np.float32)))
            print(f"  a_0 overridden to {args.a0}")

        # Print actual parameter values from the graph
        tau_d_val = sess.run(tf.nn.softplus(cell.log_tau_d))
        a_0_val = sess.run(cell.a_0)
        print(f"  tau_d   = {tau_d_val[0]:.4f}s")
        print(f"  a_0     = {a_0_val[0]:.2f}")

        # Extract the effective W matrix (with Dale's law + sparsity mask)
        W_eff_op = cell._get_W_eff()
        W_eff_val = sess.run(W_eff_op)
        n_nonzero = np.count_nonzero(W_eff_val)
        n_total = W_eff_val.size
        print(f"  W shape: {W_eff_val.shape}")
        print(f"  W range: [{W_eff_val.min():.3f}, {W_eff_val.max():.3f}]")
        print(f"  W sparsity: {n_nonzero}/{n_total} nonzero ({100*n_nonzero/n_total:.1f}%)")

        state_dim = cell.state_size
        print(f"  State dim: {state_dim}")
        print()

        # ── Initial conditions ──
        # State layout: [a_E(n_E*3), b_E(n_E), x(n)]
        np.random.seed(args.seed + 1)
        a_E_init = np.zeros(n_E * 3, dtype=np.float32)
        b_E_init = np.ones(n_E, dtype=np.float32)
        x_init = 0.1 * np.random.randn(n).astype(np.float32)
        state_val = np.concatenate([a_E_init, b_E_init, x_init]).reshape(1, -1)

        # ── Storage ──
        T = T_steps
        t_vec = np.arange(T) * h
        x_hist = np.zeros((T, n), dtype=np.float32)
        r_hist = np.zeros((T, n), dtype=np.float32)
        a_sum_hist = np.zeros((T, n_E), dtype=np.float32)
        b_hist = np.zeros((T, n_E), dtype=np.float32)
        u_hist = np.zeros((T, n), dtype=np.float32)

        # ── Simulate ──
        print_interval = max(1, T // 10)
        print("  Simulating...", end='', flush=True)
        for step in range(T):
            t_now = step * h

            # Step stimulus to random subset of E neurons
            u_val = np.zeros((1, n), dtype=np.float32)
            if t_stim_on <= t_now < t_stim_off:
                u_val[0, stim_neurons] = args.stim_amplitude

            u_hist[step] = u_val[0]

            # Record current state BEFORE stepping
            feed = {state_ph: state_val}
            fetch = [x_op, r_op, b_op]
            if a_E_sum_op is not None:
                fetch.append(a_E_sum_op)

            results = sess.run(fetch, feed_dict=feed)
            x_hist[step] = results[0][0]
            r_hist[step] = results[1][0]
            b_hist[step] = results[2][0, :n_E]
            if a_E_sum_op is not None:
                a_sum_hist[step] = results[3][0]

            # Step the cell forward
            _, state_val = sess.run(
                [output_op, new_state_op],
                feed_dict={input_ph: u_val, state_ph: state_val}
            )

            if (step + 1) % print_interval == 0:
                print(f" {100*(step+1)//T}%", end='', flush=True)

        print(" done.")

    # ── Decimate for plotting ────────────────────────────────────────────
    plot_deci = max(1, int(args.fs / 10))
    t_plot = t_vec[::plot_deci]
    x_plot = x_hist[::plot_deci]
    r_plot = r_hist[::plot_deci]
    a_sum_plot = a_sum_hist[::plot_deci]
    b_plot = b_hist[::plot_deci]
    u_plot = u_hist[::plot_deci]
    print(f"  Decimated {T} -> {len(t_plot)} points for plotting (deci={plot_deci})")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'SRNN-E-only  ({solver_label},  n={n},  h={h:.4f}s)  [real SRNNCell]',
                 fontsize=13, fontweight='bold')

    # 0) stimulus — show only the stimulated neurons
    stim_data = u_plot[:, stim_neurons]
    plot_lines(axes[0], t_plot, stim_data, n_stim,
               'u(t)\nstimulus', ylim_range=(-0.1, args.stim_amplitude * 1.3))

    # 1) x(t) — dendritic state
    plot_lines(axes[1], t_plot, x_plot, n_E, 'x(t)\ndendritic')

    # 2) r(t) — firing rate
    plot_lines(axes[2], t_plot, r_plot, n_E, 'r(t)\nfiring rate',
               ylim_range=(-0.05, 1.05))

    # 3) Σa_k(t) — summed adaptation (E neurons only)
    plot_lines(axes[3], t_plot, a_sum_plot, n_E,
               'Σa_k(t)\nadaptation (E)')

    # 4) b(t) — STD depression (E neurons only)
    plot_lines(axes[4], t_plot, b_plot, n_E,
               'b(t)\ndepression (E)', ylim_range=(-0.05, 1.05))

    axes[4].set_xlabel('Time (s)', fontsize=11)

    # Vertical lines at stimulus on/off
    for ax in axes:
        ax.axvline(t_stim_on, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
        ax.axvline(t_stim_off, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)

    plt.tight_layout()

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'srnn_eonly_simulation.png')
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n  Figure saved to {out_path}")
    plt.close()

    # ── W matrix figure (imagesc-style) ──────────────────────────────────
    fig_w, ax_w = plt.subplots(1, 1, figsize=(8, 7))
    vmax = np.max(np.abs(W_eff_val))
    im = ax_w.imshow(W_eff_val, aspect='equal', cmap='RdBu_r',
                     vmin=-vmax, vmax=vmax, interpolation='nearest')
    cbar = fig_w.colorbar(im, ax=ax_w, shrink=0.8)
    cbar.set_label('Weight', fontsize=11)

    # Mark E/I boundary
    ax_w.axvline(n_E - 0.5, color='k', linewidth=1.5, linestyle='--', alpha=0.7)
    ax_w.axhline(n_E - 0.5, color='k', linewidth=1.5, linestyle='--', alpha=0.7)

    # Quadrant labels
    fs_label = 9
    ax_w.text(n_E * 0.5, -3, 'E→', ha='center', fontsize=fs_label, fontweight='bold', color='red')
    ax_w.text(n_E + n_I * 0.5, -3, 'I→', ha='center', fontsize=fs_label, fontweight='bold', color='blue')
    ax_w.text(-3, n_E * 0.5, '→E', ha='right', va='center', fontsize=fs_label, fontweight='bold', color='red')
    ax_w.text(-3, n_E + n_I * 0.5, '→I', ha='right', va='center', fontsize=fs_label, fontweight='bold', color='blue')

    ax_w.set_xlabel('Presynaptic neuron (j)', fontsize=11)
    ax_w.set_ylabel('Postsynaptic neuron (i)', fontsize=11)
    ax_w.set_title(f'W_eff  (n={n}, Dale\'s law)\nE cols ≥ 0 (red), I cols ≤ 0 (blue)',
                   fontsize=12, fontweight='bold')

    w_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'srnn_eonly_W_matrix.png')
    fig_w.savefig(w_path, dpi=150, bbox_inches='tight')
    print(f"  W matrix figure saved to {w_path}")
    plt.close()


if __name__ == '__main__':
    main()
