# srnn_model.py — Stable Recurrent Neural Network (SRNN) as a TF1 RNNCell
#
# Port of srnn.jl from JuliaLang/src/models/srnn.jl.
# Implements the SRNN dynamics with optional Spike-Frequency Adaptation (SFA)
# and Short-Term Depression (STD), E/I neuron split, semi-implicit (fused)
# solver, and multiple readout modes.
#
# Drop-in replacement for LTCCell / CTRNN / etc. in the experiments.
#
# Usage:
#   cell = SRNNCell(num_units=32, n_E=32)
#   head, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major=True)

import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np

# ═══════════════════════════════════════════════════════════════════════
# ACTIVATION: Piecewise Sigmoid  (port of activations.jl)
# ═══════════════════════════════════════════════════════════════════════

def piecewise_sigmoid(x, S_a=0.9, S_c=0.0):
    """Piecewise linear/quadratic sigmoid bounded in [0, 1].

    Port of Julia's piecewise_sigmoid from activations.jl.
    - Linear region centered at S_c with slope 1
    - Quadratic rounding at corners
    - Saturates at 0 and 1

    Args:
        x: input tensor
        S_a: linear fraction parameter (0 to 1)
        S_c: center/shift parameter (default 0.0; a_0 is subtracted externally)
    """
    a = S_a / 2.0
    c = S_c
    k = 0.5 / (1.0 - 2.0 * a)

    x1 = c + a - 1.0
    x2 = c - a
    x3 = c + a
    x4 = c + 1.0 - a

    left_quad = k * tf.square(x - x1)
    linear    = (x - c) + 0.5
    right_quad = 1.0 - k * tf.square(x - x4)

    y = tf.where(x < x1, tf.zeros_like(x),
        tf.where(x < x2, left_quad,
        tf.where(x <= x3, linear,
        tf.where(x <= x4, right_quad,
        tf.ones_like(x)))))
    return y

# ═══════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════

def _inv_softplus(x):
    """Inverse of softplus: returns y such that softplus(y) = x."""
    return np.log(np.exp(x) - 1.0).astype(np.float32)


def _make_tau_range(lo, hi, n_steps):
    """Build log-spaced time constant range from trainable endpoints.

    lo, hi: tensors of shape (1,) or (n,)
    Returns: tensor of shape (n, n_steps) or (1, n_steps)
    """
    if n_steps == 1:
        t = tf.constant([[0.5]], dtype=tf.float32)  # (1, 1)
    else:
        t = tf.constant(
            [[i / (n_steps - 1) for i in range(n_steps)]],
            dtype=tf.float32
        )  # (1, n_steps)
    # lo, hi are (dim,) vectors; reshape to (dim, 1) for broadcast
    lo_2d = tf.reshape(lo, [-1, 1])
    hi_2d = tf.reshape(hi, [-1, 1])
    return lo_2d + (hi_2d - lo_2d) * t  # (dim, n_steps)


def _apply_dales(W, n_E):
    """Enforce Dale's law via column-wise softplus.

    E columns (0:n_E):    softplus(W) >= 0
    I columns (n_E:end):  -softplus(W) <= 0
    """
    W_E = tf.nn.softplus(W[:, :n_E])
    W_I = -tf.nn.softplus(W[:, n_E:])
    return tf.concat([W_E, W_I], axis=1)


def generate_rmt_matrix(n, indegree, f, level_of_chaos=1.0, E_W=0.0, seed=None):
    """Generate RMT E/I weight matrix (NumPy port of connectivity.jl).

    Args:
        n: number of neurons
        indegree: expected in-degree per neuron
        f: fraction excitatory (0 < f < 1)
        level_of_chaos: scaling factor
        E_W: mean offset
        seed: random seed

    Returns:
        W: (n, n) weight matrix
        E_indices: array of E neuron indices
        I_indices: array of I neuron indices
    """
    rng = np.random.RandomState(seed)
    n_E = int(round(f * n))
    n_I = n - n_E
    E_indices = np.arange(n_E)
    I_indices = np.arange(n_E, n)

    alpha = indegree / n
    F = 1.0 / np.sqrt(n * alpha * (2.0 - alpha))

    mu_E = 3.0 * F + E_W
    mu_I = -4.0 * F + E_W
    sigma_E = F
    sigma_I = F

    A = rng.randn(n, n)
    S = (rng.rand(n, n) < alpha).astype(np.float64)

    D = np.zeros(n)
    D[E_indices] = sigma_E
    D[I_indices] = sigma_I

    v = np.zeros(n)
    v[E_indices] = mu_E
    v[I_indices] = mu_I
    M = np.ones((n, 1)) @ v.reshape(1, -1)

    W = S * (A @ np.diag(D) + M)
    W = level_of_chaos * W

    return W.astype(np.float32), E_indices, I_indices


# ═══════════════════════════════════════════════════════════════════════
# SRNNCell
# ═══════════════════════════════════════════════════════════════════════

class SRNNCell(tf.nn.rnn_cell.RNNCell):
    """Stable Recurrent Neural Network cell for tf.nn.dynamic_rnn.

    Implements the SRNN ODE with optional SFA and STD, E/I neuron split,
    and semi-implicit (fused) or explicit Euler solver.

    Args:
        num_units: Total number of neurons (n).
        n_E: Number of excitatory neurons (n_I = num_units - n_E).
        n_a_E: Number of SFA timescales for E neurons (0 = no SFA).
        n_a_I: Number of SFA timescales for I neurons.
        n_b_E: 1 to enable STD for E neurons, 0 to disable.
        n_b_I: 1 to enable STD for I neurons, 0 to disable.
        ode_solver_unfolds: Number of ODE sub-steps per RNN step.
        h: Step size for the ODE solver.
        solver: "semi_implicit" or "explicit".
        readout: "synaptic", "rate", or "dendritic".
        per_neuron: If True, dynamics params are per-neuron vectors.
    """

    def __init__(self, num_units, n_E=None,
                 n_a_E=0, n_a_I=0, n_b_E=0, n_b_I=0,
                 ode_solver_unfolds=6, h=1.0/400.0,
                 solver="semi_implicit", readout="synaptic",
                 per_neuron=False, dales=False):
        super(SRNNCell, self).__init__()
        self._num_units = num_units
        self._n_E = n_E if n_E is not None else num_units
        self._n_I = num_units - self._n_E
        self._n_a_E = n_a_E
        self._n_a_I = n_a_I
        self._n_b_E = n_b_E
        self._n_b_I = n_b_I
        self._ode_solver_unfolds = ode_solver_unfolds
        self._h = np.float32(h)
        self._solver = solver
        self._readout = readout
        self._per_neuron = per_neuron
        self._dales = dales

        # Compute total state dimension
        self._state_dim = (
            self._n_E * n_a_E +
            self._n_I * n_a_I +
            self._n_E * n_b_E +
            self._n_I * n_b_I +
            num_units
        )

        self._is_built = False

    @property
    def state_size(self):
        return self._state_dim

    @property
    def output_size(self):
        return self._num_units

    def build(self, input_shape):
        pass

    def _get_tau_a(self, pop):
        """Get SFA time constants for population pop ('E' or 'I').

        n_a == 1: single trainable tau (no lo/hi range)
        n_a >= 2: range from lo to hi via _make_tau_range
        """
        n_a = self._n_a_E if pop == 'E' else self._n_a_I
        if n_a == 1:
            log_tau = getattr(self, 'log_tau_a_{}'.format(pop))
            return tf.nn.softplus(log_tau)  # (1,) or (n_pop,)
        else:
            lo = getattr(self, 'log_tau_a_{}_lo'.format(pop))
            hi = getattr(self, 'log_tau_a_{}_hi'.format(pop))
            return tf.nn.softplus(_make_tau_range(lo, hi, n_a))  # (dim, n_a)

    def _get_variables(self, input_size):
        """Create all trainable variables (called once on first __call__)."""
        n = self._num_units
        n_E = self._n_E
        n_I = self._n_I
        pn = self._per_neuron

        sigma_w = np.sqrt(2.0 / n).astype(np.float32)

        # Helper: scalar (1,) or per-neuron (dim,) initializer
        def _s_or_v(val, dim):
            if pn:
                return np.full(dim, val, dtype=np.float32)
            else:
                return np.array([val], dtype=np.float32)

        # ── Core weights ──
        if self._dales:
            # RMT initialization: inv_softplus(|W_rmt|), clamped
            indegree = min(n, max(1, n))  # full connectivity for small n
            f = float(n_E) / float(n)
            W_rmt, _, _ = generate_rmt_matrix(n, indegree, f)
            W_abs = np.abs(W_rmt)
            # inv_softplus: log(exp(x) - 1), clamp to avoid extremes
            W_raw = np.clip(np.log(np.exp(W_abs.astype(np.float64)) - 1.0),
                            -10.0, 10.0).astype(np.float32)
            self.W = tf.get_variable(
                'W', initializer=W_raw, dtype=tf.float32)
        else:
            self.W = tf.get_variable(
                'W', [n, n],
                initializer=tf.initializers.random_normal(stddev=sigma_w),
                dtype=tf.float32)

        self.W_in = tf.get_variable(
            'W_in', [n, input_size],
            initializer=tf.initializers.random_normal(stddev=0.1),
            dtype=tf.float32)

        self.a_0 = tf.get_variable(
            'a_0', shape=_s_or_v(0.35, n).shape,
            initializer=tf.constant_initializer(_s_or_v(0.35, n)),
            dtype=tf.float32)

        self.log_tau_d = tf.get_variable(
            'log_tau_d', shape=_s_or_v(_inv_softplus(0.1), n).shape,
            initializer=tf.constant_initializer(_s_or_v(_inv_softplus(0.1), n)),
            dtype=tf.float32)

        # ── SFA parameters for E neurons ──
        if self._n_a_E == 1:
            self.log_tau_a_E = tf.get_variable(
                'log_tau_a_E', shape=_s_or_v(_inv_softplus(1.0), n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(1.0), n_E)),
                dtype=tf.float32)
            self.log_c_E = tf.get_variable(
                'log_c_E', shape=_s_or_v(-3.0, n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(-3.0, n_E)),
                dtype=tf.float32)
            self.c_0_E = tf.get_variable(
                'c_0_E', shape=_s_or_v(0.0, n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(0.0, n_E)),
                dtype=tf.float32)
        elif self._n_a_E >= 2:
            self.log_tau_a_E_lo = tf.get_variable(
                'log_tau_a_E_lo', shape=_s_or_v(_inv_softplus(0.25), n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(0.25), n_E)),
                dtype=tf.float32)
            self.log_tau_a_E_hi = tf.get_variable(
                'log_tau_a_E_hi', shape=_s_or_v(_inv_softplus(10.0), n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(10.0), n_E)),
                dtype=tf.float32)
            self.log_c_E = tf.get_variable(
                'log_c_E', shape=_s_or_v(-3.0, n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(-3.0, n_E)),
                dtype=tf.float32)
            self.c_0_E = tf.get_variable(
                'c_0_E', shape=_s_or_v(0.0, n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(0.0, n_E)),
                dtype=tf.float32)

        # ── SFA parameters for I neurons ──
        if self._n_a_I == 1:
            self.log_tau_a_I = tf.get_variable(
                'log_tau_a_I', shape=_s_or_v(_inv_softplus(1.0), n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(1.0), n_I)),
                dtype=tf.float32)
            self.log_c_I = tf.get_variable(
                'log_c_I', shape=_s_or_v(-3.0, n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(-3.0, n_I)),
                dtype=tf.float32)
            self.c_0_I = tf.get_variable(
                'c_0_I', shape=_s_or_v(0.0, n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(0.0, n_I)),
                dtype=tf.float32)
        elif self._n_a_I >= 2:
            self.log_tau_a_I_lo = tf.get_variable(
                'log_tau_a_I_lo', shape=_s_or_v(_inv_softplus(0.25), n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(0.25), n_I)),
                dtype=tf.float32)
            self.log_tau_a_I_hi = tf.get_variable(
                'log_tau_a_I_hi', shape=_s_or_v(_inv_softplus(10.0), n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(10.0), n_I)),
                dtype=tf.float32)
            self.log_c_I = tf.get_variable(
                'log_c_I', shape=_s_or_v(-3.0, n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(-3.0, n_I)),
                dtype=tf.float32)
            self.c_0_I = tf.get_variable(
                'c_0_I', shape=_s_or_v(0.0, n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(0.0, n_I)),
                dtype=tf.float32)

        # ── STD parameters for E neurons ──
        if self._n_b_E > 0:
            self.log_tau_b_E_rec = tf.get_variable(
                'log_tau_b_E_rec', shape=_s_or_v(_inv_softplus(1.0), n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(1.0), n_E)),
                dtype=tf.float32)
            self.log_tau_b_E_rel = tf.get_variable(
                'log_tau_b_E_rel', shape=_s_or_v(_inv_softplus(0.25), n_E).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(0.25), n_E)),
                dtype=tf.float32)

        # ── STD parameters for I neurons ──
        if self._n_b_I > 0:
            self.log_tau_b_I_rec = tf.get_variable(
                'log_tau_b_I_rec', shape=_s_or_v(_inv_softplus(1.0), n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(1.0), n_I)),
                dtype=tf.float32)
            self.log_tau_b_I_rel = tf.get_variable(
                'log_tau_b_I_rel', shape=_s_or_v(_inv_softplus(0.25), n_I).shape,
                initializer=tf.constant_initializer(_s_or_v(_inv_softplus(0.25), n_I)),
                dtype=tf.float32)

    # ── State unpacking ─────────────────────────────────────────────────

    def _unpack_state(self, state):
        """Unpack flat state (batch, state_dim) into components.

        Returns dict with keys: a_E, a_I, b_E, b_I, x.
        Each value is a tensor or None if that component is disabled.
        a_E shape: (batch, n_E, n_a_E) or None
        b_E shape: (batch, n_E) or None
        x shape:   (batch, n)
        """
        n = self._num_units
        n_E, n_I = self._n_E, self._n_I
        n_a_E, n_a_I = self._n_a_E, self._n_a_I
        n_b_E, n_b_I = self._n_b_E, self._n_b_I
        batch = tf.shape(state)[0]

        idx = 0
        parts = {}

        # a_E: (batch, n_E * n_a_E) → (batch, n_E, n_a_E)
        len_a_E = n_E * n_a_E
        if len_a_E > 0:
            parts['a_E'] = tf.reshape(state[:, idx:idx+len_a_E], [-1, n_E, n_a_E])
        else:
            parts['a_E'] = None
        idx += len_a_E

        # a_I
        len_a_I = n_I * n_a_I
        if len_a_I > 0:
            parts['a_I'] = tf.reshape(state[:, idx:idx+len_a_I], [-1, n_I, n_a_I])
        else:
            parts['a_I'] = None
        idx += len_a_I

        # b_E: (batch, n_E)
        len_b_E = n_E * n_b_E
        if len_b_E > 0:
            parts['b_E'] = state[:, idx:idx+len_b_E]
        else:
            parts['b_E'] = None
        idx += len_b_E

        # b_I: (batch, n_I)
        len_b_I = n_I * n_b_I
        if len_b_I > 0:
            parts['b_I'] = state[:, idx:idx+len_b_I]
        else:
            parts['b_I'] = None
        idx += len_b_I

        # x: (batch, n)
        parts['x'] = state[:, idx:idx+n]

        return parts

    def _pack_state(self, parts):
        """Pack components back into flat state (batch, state_dim)."""
        pieces = []
        n_E, n_I = self._n_E, self._n_I
        n_a_E, n_a_I = self._n_a_E, self._n_a_I

        if parts['a_E'] is not None:
            pieces.append(tf.reshape(parts['a_E'], [-1, n_E * n_a_E]))
        if parts['a_I'] is not None:
            pieces.append(tf.reshape(parts['a_I'], [-1, n_I * n_a_I]))
        if parts['b_E'] is not None:
            pieces.append(parts['b_E'])
        if parts['b_I'] is not None:
            pieces.append(parts['b_I'])
        pieces.append(parts['x'])

        return tf.concat(pieces, axis=1)

    # ── SFA: compute x_eff ──────────────────────────────────────────────

    def _compute_x_eff(self, parts):
        """Apply SFA to dendritic state x. Returns x_eff (batch, n)."""
        x = parts['x']
        n_E = self._n_E

        if self._n_a_E > 0 and parts['a_E'] is not None:
            c_E = tf.nn.softplus(self.log_c_E)  # (dim_E,) or (1,)
            a_E_sum = tf.reduce_sum(parts['a_E'], axis=2)  # (batch, n_E)
            # x_eff_E = x[:, :n_E] - c_E * a_E_sum
            x_E = x[:, :n_E] - c_E * a_E_sum
            x_I = x[:, n_E:]
            x = tf.concat([x_E, x_I], axis=1)

        if self._n_a_I > 0 and parts['a_I'] is not None:
            c_I = tf.nn.softplus(self.log_c_I)
            a_I_sum = tf.reduce_sum(parts['a_I'], axis=2)  # (batch, n_I)
            x_E = x[:, :n_E]
            x_I = x[:, n_E:] - c_I * a_I_sum
            x = tf.concat([x_E, x_I], axis=1)

        return x

    # ── STD: extract b ───────────────────────────────────────────────────

    def _extract_b(self, parts):
        """Extract full STD vector b (batch, n). Returns ones if no STD."""
        n = self._num_units
        n_E = self._n_E
        batch = tf.shape(parts['x'])[0]

        b = tf.ones([batch, n], dtype=tf.float32)

        if self._n_b_E > 0 and parts['b_E'] is not None:
            b = tf.concat([parts['b_E'], b[:, n_E:]], axis=1)

        if self._n_b_I > 0 and parts['b_I'] is not None:
            b = tf.concat([b[:, :n_E], parts['b_I']], axis=1)

        return b

    # ── Readout ──────────────────────────────────────────────────────────

    def _compute_readout(self, parts):
        """Compute the readout from the state.

        Returns tensor of shape (batch, n).
        """
        if self._readout == "dendritic":
            return parts['x']

        x_eff = self._compute_x_eff(parts)
        r = piecewise_sigmoid(x_eff - self.a_0)

        if self._readout == "rate":
            return r

        # "synaptic" (default)
        b = self._extract_b(parts)
        return b * r

    # ── Semi-implicit (fused) solver step ────────────────────────────────

    def _fused_step(self, parts, u):
        """One semi-implicit sub-step. Returns new parts dict.

        Closed-form ratio updates:
          x_new = (x + α·(u + W·br)) / (1 + α)   where α = Δt/τ_d
          a_new = (a + (Δt/τ_a)·(c_0 + r)) / (1 + Δt/τ_a)
          b_new = (b + Δt/τ_rec) / (1 + Δt·(1/τ_rec + r/τ_rel))
        """
        n = self._num_units
        n_E, n_I = self._n_E, self._n_I
        n_a_E, n_a_I = self._n_a_E, self._n_a_I
        Δt = self._h

        x = parts['x']

        # Compute x_eff and firing rate
        x_eff = self._compute_x_eff(parts)
        r = piecewise_sigmoid(x_eff - self.a_0)  # (batch, n)

        # Apply STD
        b = self._extract_b(parts)
        br = b * r  # (batch, n)

        # ── Fused x update ──
        W_eff = _apply_dales(self.W, n_E) if self._dales else self.W
        tau_d = tf.nn.softplus(self.log_tau_d)  # (1,) or (n,)
        alpha_x = Δt / tau_d
        # W @ br: (batch, n) @ (n, n)^T → use matmul with W transposed
        Wbr = tf.matmul(br, W_eff, transpose_b=True)  # (batch, n)
        x_new = (x + alpha_x * (u + Wbr)) / (1.0 + alpha_x)

        new_parts = {'x': x_new, 'a_E': None, 'a_I': None, 'b_E': None, 'b_I': None}

        # ── Fused a_E update ──
        if n_a_E > 0 and parts['a_E'] is not None:
            tau_a_E_raw = self._get_tau_a('E')  # (dim,) if n_a==1, (dim, n_a) if n_a>=2
            tau_a_E = tf.reshape(tau_a_E_raw, [1, -1, n_a_E])
            alpha_a_E = Δt / tau_a_E

            r_E = tf.reshape(r[:, :n_E], [-1, n_E, 1])  # (batch, n_E, 1)
            c_0_E = tf.reshape(self.c_0_E, [1, -1, 1])   # (1, dim, 1)

            new_parts['a_E'] = (parts['a_E'] + alpha_a_E * (c_0_E + r_E)) / (1.0 + alpha_a_E)

        # ── Fused a_I update ──
        if n_a_I > 0 and parts['a_I'] is not None:
            tau_a_I_raw = self._get_tau_a('I')
            tau_a_I = tf.reshape(tau_a_I_raw, [1, -1, n_a_I])
            alpha_a_I = Δt / tau_a_I

            r_I = tf.reshape(r[:, n_E:], [-1, n_I, 1])
            c_0_I = tf.reshape(self.c_0_I, [1, -1, 1])

            new_parts['a_I'] = (parts['a_I'] + alpha_a_I * (c_0_I + r_I)) / (1.0 + alpha_a_I)

        # ── Fused b_E update ──
        if self._n_b_E > 0 and parts['b_E'] is not None:
            tau_b_rec = tf.nn.softplus(self.log_tau_b_E_rec)
            tau_b_rel = tf.nn.softplus(self.log_tau_b_E_rel)
            r_E = r[:, :n_E]
            new_parts['b_E'] = (parts['b_E'] + Δt / tau_b_rec) / \
                (1.0 + Δt * (1.0 / tau_b_rec + r_E / tau_b_rel))

        # ── Fused b_I update ──
        if self._n_b_I > 0 and parts['b_I'] is not None:
            tau_b_rec = tf.nn.softplus(self.log_tau_b_I_rec)
            tau_b_rel = tf.nn.softplus(self.log_tau_b_I_rel)
            r_I = r[:, n_E:]
            new_parts['b_I'] = (parts['b_I'] + Δt / tau_b_rec) / \
                (1.0 + Δt * (1.0 / tau_b_rec + r_I / tau_b_rel))

        return new_parts

    # ── Explicit Euler step ──────────────────────────────────────────────

    def _explicit_step(self, state_flat, u):
        """One explicit Euler sub-step on the flat state vector."""
        parts = self._unpack_state(state_flat)
        x = parts['x']
        n = self._num_units
        n_E, n_I = self._n_E, self._n_I
        n_a_E, n_a_I = self._n_a_E, self._n_a_I
        Δt = self._h

        # Compute x_eff and firing rate
        x_eff = self._compute_x_eff(parts)
        r = piecewise_sigmoid(x_eff - self.a_0)

        # Apply STD
        b = self._extract_b(parts)
        br = b * r

        # dx/dt
        W_eff = _apply_dales(self.W, n_E) if self._dales else self.W
        tau_d = tf.nn.softplus(self.log_tau_d)
        Wbr = tf.matmul(br, W_eff, transpose_b=True)
        dx_dt = (-x + Wbr + u) / tau_d

        # Flat dS/dt pieces
        dS_pieces = []

        # da_E/dt
        if n_a_E > 0 and parts['a_E'] is not None:
            tau_a_E_raw = self._get_tau_a('E')
            tau_a_E = tf.reshape(tau_a_E_raw, [1, -1, n_a_E])
            r_E = tf.reshape(r[:, :n_E], [-1, n_E, 1])
            c_0_E = tf.reshape(self.c_0_E, [1, -1, 1])
            da_E_dt = (c_0_E + r_E - parts['a_E']) / tau_a_E  # (batch, n_E, n_a_E)
            dS_pieces.append(tf.reshape(da_E_dt, [-1, n_E * n_a_E]))

        # da_I/dt
        if n_a_I > 0 and parts['a_I'] is not None:
            tau_a_I_raw = self._get_tau_a('I')
            tau_a_I = tf.reshape(tau_a_I_raw, [1, -1, n_a_I])
            r_I = tf.reshape(r[:, n_E:], [-1, n_I, 1])
            c_0_I = tf.reshape(self.c_0_I, [1, -1, 1])
            da_I_dt = (c_0_I + r_I - parts['a_I']) / tau_a_I
            dS_pieces.append(tf.reshape(da_I_dt, [-1, n_I * n_a_I]))

        # db_E/dt
        if self._n_b_E > 0 and parts['b_E'] is not None:
            tau_b_rec = tf.nn.softplus(self.log_tau_b_E_rec)
            tau_b_rel = tf.nn.softplus(self.log_tau_b_E_rel)
            r_E = r[:, :n_E]
            db_E_dt = (1.0 - parts['b_E']) / tau_b_rec - (parts['b_E'] * r_E) / tau_b_rel
            dS_pieces.append(db_E_dt)

        # db_I/dt
        if self._n_b_I > 0 and parts['b_I'] is not None:
            tau_b_rec = tf.nn.softplus(self.log_tau_b_I_rec)
            tau_b_rel = tf.nn.softplus(self.log_tau_b_I_rel)
            r_I = r[:, n_E:]
            db_I_dt = (1.0 - parts['b_I']) / tau_b_rec - (parts['b_I'] * r_I) / tau_b_rel
            dS_pieces.append(db_I_dt)

        dS_pieces.append(dx_dt)
        dS_dt = tf.concat(dS_pieces, axis=1)

        return state_flat + Δt * dS_dt

    # ── Main __call__ ────────────────────────────────────────────────────

    def __call__(self, inputs, state, scope=None):
        """Run one RNN step: inputs (batch, input_dim), state (batch, state_dim).

        Returns (output, new_state) where output is (batch, num_units).
        """
        with tf.variable_scope("srnn"):
            if not self._is_built:
                self._is_built = True
                self._input_size = int(inputs.shape[-1])
                self._get_variables(self._input_size)

            # Compute input drive: u = W_in @ input → (batch, n)
            # W_in is (n, n_in), inputs is (batch, n_in) → matmul(inputs, W_in^T)
            u = tf.matmul(inputs, self.W_in, transpose_b=True)  # (batch, n)

            if self._solver == "semi_implicit":
                parts = self._unpack_state(state)
                for _ in range(self._ode_solver_unfolds):
                    parts = self._fused_step(parts, u)
                new_state = self._pack_state(parts)
                output = self._compute_readout(parts)
            elif self._solver == "explicit":
                state_out = state
                for _ in range(self._ode_solver_unfolds):
                    state_out = self._explicit_step(state_out, u)
                new_state = state_out
                parts = self._unpack_state(new_state)
                output = self._compute_readout(parts)
            else:
                raise ValueError("Unknown solver: '{}'. Use 'semi_implicit' or 'explicit'".format(
                    self._solver))

        return output, new_state
