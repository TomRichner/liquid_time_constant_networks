# trainable_ic.py — Burn-in and trainable initial conditions
#
# Two-phase approach for TF1 graph compatibility:
#
# Phase 1 (graph build, in __init__):
#   ic_var = create_ic_variable(state_dim, name="ic")
#   batch_ic = tile_ic_for_batch(ic_var, tf.shape(self.x)[1])
#   head, _ = tf.nn.dynamic_rnn(cell, head, initial_state=batch_ic, ...)
#
# Phase 2 (after session init, before training):
#   run_burn_in_and_assign(cell, input_size, sess, ic_var, burn_in_seconds=30)
#
# The IC variable starts as zeros. After session init, we run the cell
# forward with zero input for burn_in_seconds to find the quiescent
# attractor, then assign that state to the IC variable. Since ic_var is
# trainable, gradients during training will track the evolving attractor.

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


def create_ic_variable(state_dim, name="initial_state_0"):
    """Create a trainable IC variable initialized to zeros.

    Call during graph build (Phase 1), before dynamic_rnn.

    Args:
        state_dim: Integer — total flat state dimension of the cell.
        name: Variable name.

    Returns:
        ic_var: tf.Variable(trainable=True) of shape (state_dim,).
    """
    ic_var = tf.get_variable(
        name, shape=[state_dim],
        initializer=tf.zeros_initializer(),
        trainable=True, dtype=tf.float32)
    return ic_var


def tile_ic_for_batch(ic_var, batch_size_tensor):
    """Tile a (state_dim,) IC variable to (batch, state_dim).

    Args:
        ic_var: tf.Variable of shape (state_dim,).
        batch_size_tensor: Scalar int tensor for dynamic batch size.

    Returns:
        Tiled tensor of shape (batch, state_dim).
    """
    return tf.tile(tf.expand_dims(ic_var, 0), [batch_size_tensor, 1])


def compute_burn_in(cell, input_size, sess, burn_in_seconds=30.0):
    """Run the cell forward with zero input to find the quiescent state.

    Creates a temporary sub-graph, runs burn_in_steps of the cell with
    zero input, and returns the final state as a numpy array.

    Works with any RNNCell (SRNN, CTRNN, LSTM, NODE, etc.).

    Args:
        cell: A tf.nn.rnn_cell.RNNCell instance (already constructed).
        input_size: Dimension of the input features.
        sess: A tf.Session with the cell's variables already initialized.
        burn_in_seconds: Duration of burn-in in simulated seconds.

    Returns:
        state_np: numpy array of shape (state_dim,) — the settled state.
    """
    # Estimate steps needed
    if hasattr(cell, '_h') and hasattr(cell, '_ode_solver_unfolds'):
        # SRNN
        dt_per_step = cell._h * cell._ode_solver_unfolds
    elif hasattr(cell, '_delta_t') and hasattr(cell, '_unfolds'):
        # CTRNN, NODE
        dt_per_step = cell._delta_t * cell._unfolds
    else:
        # LSTM, CTGRU, LTC — no physical time. Treat burn_in_seconds as
        # number of steps directly.
        dt_per_step = 1.0

    n_steps = max(1, int(np.ceil(burn_in_seconds / dt_per_step)))
    print(f"  Burn-in: {n_steps} steps ({burn_in_seconds}s, dt={dt_per_step:.6f})")

    # Build burn-in input: (n_steps, 1, input_size) of zeros
    zero_input = np.zeros((n_steps, 1, input_size), dtype=np.float32)

    # Use dynamic_rnn to step through
    with tf.variable_scope("burn_in", reuse=tf.AUTO_REUSE):
        burn_in_input = tf.placeholder(
            tf.float32, [n_steps, 1, input_size], name="burn_in_input")
        _, final_state = tf.nn.dynamic_rnn(
            cell, burn_in_input, dtype=tf.float32, time_major=True)

    # Initialize any new variables created by the burn-in sub-graph
    # (dynamic_rnn may create duplicate cell variables in burn_in/ scope)
    uninit_vars = sess.run(tf.report_uninitialized_variables())
    if len(uninit_vars) > 0:
        uninit_var_names = [v.decode() for v in uninit_vars]
        vars_to_init = [v for v in tf.global_variables()
                        if v.name.split(':')[0] in uninit_var_names]
        if vars_to_init:
            sess.run(tf.variables_initializer(vars_to_init))

    # Handle LSTMStateTuple
    if isinstance(final_state, tf.nn.rnn_cell.LSTMStateTuple):
        c_val, h_val = sess.run(
            [final_state.c, final_state.h],
            feed_dict={burn_in_input: zero_input})
        state_np = np.concatenate([c_val[0], h_val[0]], axis=0)
    else:
        state_np = sess.run(
            final_state,
            feed_dict={burn_in_input: zero_input})[0]  # Remove batch dim

    return state_np


def run_burn_in_and_assign(cell, input_size, sess, ic_var,
                           burn_in_seconds=30.0):
    """Run burn-in and assign result to the IC variable.

    Call after session init (Phase 2).

    Args:
        cell: The constructed RNNCell.
        input_size: Input feature dimension.
        sess: Active tf.Session (variables must be initialized).
        ic_var: tf.Variable created by create_ic_variable().
        burn_in_seconds: Burn-in duration.

    Returns:
        ic_np: numpy array of the burn-in result.
    """
    ic_np = compute_burn_in(cell, input_size, sess, burn_in_seconds)

    # Assign to IC variable
    sess.run(ic_var.assign(ic_np))
    print(f"  IC assigned: norm={np.linalg.norm(ic_np):.4f}, "
          f"range=[{ic_np.min():.4f}, {ic_np.max():.4f}]")

    return ic_np
