# trainable_ic.py — Burn-in and trainable initial conditions
#
# Two-step approach:
# 1. Burn-in: Forward-step the cell for a specified duration with zero input
#    to find the quiescent attractor (or a point on it).
# 2. Trainable IC: Store the result as a tf.Variable(trainable=True) that
#    tracks the evolving attractor during training via gradients.
#
# Usage:
#   cell = SRNNCell(32, ...)
#   ic_var, ic_np = create_trainable_ic(
#       cell, input_size=28, sess=sess, burn_in_seconds=30.0)
#   # Then pass to dynamic_rnn:
#   batch_ic = tf.tile(tf.expand_dims(ic_var, 0), [batch_size, 1])
#   head, _ = tf.nn.dynamic_rnn(cell, head, initial_state=batch_ic, ...)

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()


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
            For SRNN: effective dt = h * ode_solver_unfolds per RNN step.
            For CTRNN: effective dt = _delta_t * _unfolds per RNN step.
            For LSTM: no physical time; use burn_in_seconds as step count.

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

    # Build burn-in input: (n_steps, 1, input_size) of zeros
    zero_input = np.zeros((n_steps, 1, input_size), dtype=np.float32)

    # Use dynamic_rnn to step through
    with tf.variable_scope("burn_in", reuse=tf.AUTO_REUSE):
        burn_in_input = tf.placeholder(
            tf.float32, [n_steps, 1, input_size], name="burn_in_input")
        _, final_state = tf.nn.dynamic_rnn(
            cell, burn_in_input, dtype=tf.float32, time_major=True)

    # Handle LSTMStateTuple
    if isinstance(final_state, tf.nn.rnn_cell.LSTMStateTuple):
        c_val, h_val = sess.run(
            [final_state.c, final_state.h],
            feed_dict={burn_in_input: zero_input})
        # Flatten LSTM state: concat [c, h]
        state_np = np.concatenate([c_val[0], h_val[0]], axis=0)
    else:
        state_np = sess.run(
            final_state,
            feed_dict={burn_in_input: zero_input})[0]  # Remove batch dim

    return state_np


def create_trainable_ic(cell, input_size, sess, burn_in_seconds=30.0,
                        name="initial_state_0"):
    """Burn-in and create a trainable initial condition variable.

    Args:
        cell: A constructed RNNCell.
        input_size: Input feature dimension.
        sess: Active tf.Session.
        burn_in_seconds: Burn-in duration (seconds or steps).
        name: Variable name.

    Returns:
        ic_var: tf.Variable(trainable=True) of shape (state_dim,).
        ic_np: numpy array of the burn-in result.
    """
    ic_np = compute_burn_in(cell, input_size, sess, burn_in_seconds)

    # Create as trainable variable
    ic_var = tf.get_variable(
        name, shape=ic_np.shape,
        initializer=tf.constant_initializer(ic_np),
        trainable=True, dtype=tf.float32)

    # Initialize this new variable
    sess.run(ic_var.initializer)

    return ic_var, ic_np


def tile_ic_for_batch(ic_var, batch_size_tensor):
    """Tile a (state_dim,) IC variable to (batch, state_dim).

    Args:
        ic_var: tf.Variable of shape (state_dim,).
        batch_size_tensor: Scalar int tensor for dynamic batch size.

    Returns:
        Tiled tensor of shape (batch, state_dim).
    """
    return tf.tile(tf.expand_dims(ic_var, 0), [batch_size_tensor, 1])
