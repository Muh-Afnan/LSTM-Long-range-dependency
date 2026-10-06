# Approach — Day 38

## Derivation
The RNN overwrites h entirely each step:
    h_t = tanh(x_t @ W_xh + h_{t-1} @ W_hh + b_h)

Nothing survives. The fix is addition instead of replacement:
    c_t = c_{t-1} + something

The old value persists. But pure addition can never erase stale
information, so each term gets a learned 0-1 gate.

## The Six Equations
    f_t = σ(W_f x_t + U_f h_{t-1} + b_f)        forget gate
    i_t = σ(W_i x_t + U_i h_{t-1} + b_i)        input gate
    g_t = tanh(W_g x_t + U_g h_{t-1} + b_g)     candidate
    o_t = σ(W_o x_t + U_o h_{t-1} + b_o)        output gate

    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t
    h_t = o_t ⊙ tanh(c_t)

Three gates use sigmoid (0-1 control values). The candidate uses
tanh because it carries content, not control.

## Shapes (vocab 29, embedding 8, hidden 128, batch 32, window 6)
    W_* : (8, 128)      U_* : (128, 128)      b_* : (128,)
    x_t : (32, 8)       h, c : (32, 128)
    stacked h : (32, 6, 128) → @ W_hy (128, 29) → (32, 6, 29)

## Parameters
    per gate: (8 × 128) + (128 × 128) + 128 = 17,536
    × 4 gates                               = 70,144

Exactly 4× the RNN cell (17,536). The RNN was effectively the
candidate on its own, with no gating.

GRU drops the separate cell state and merges forget and input into
one update gate: 3 sets of weights instead of 4. Fewer parameters,
faster training and inference, comparable performance in practice.

## Initialization
Xavier on the four W_* (input weights).
Orthogonal on the four U_* (recurrent weights) — eigenvalues of
magnitude 1, so repeated application across timesteps neither
shrinks nor explodes the signal.