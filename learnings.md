# Learnings — Day 38

## Addition Is the Whole Idea
Matrix multiplication replaces the old value. Addition keeps it.
    c = c @ W          → old value gone
    c = c + something  → old value still there

## The Gradient Is Exactly 1
    ∂c_t / ∂c_{t-1} = 1     for  c_t = c_{t-1} + x

Backward through the cell state, the gradient is multiplied by 1
at every step — not by W_hh, not squashed by tanh.

    RNN:   grad × W_hh × W_hh × ... → vanishes
    LSTM:  grad × 1 × 1 × ...       → survives

The original 1997 paper calls this the constant error carousel.
In practice the factor is f_t rather than exactly 1, but f_t is a
learned value the network can hold near 1 — a gate it controls,
not a fixed matrix it is stuck with.

This is also why ResNet works: skip connections give gradients a
path multiplied by 1 instead of through every layer's weights.
Same insight, applied to depth instead of time.

## Three Gates and One Candidate
Forget gate (σ)   — how much of the old cell state to keep
Input gate (σ)    — how much of the new candidate to write
Candidate (tanh)  — NOT a gate; it is the content to write
Output gate (σ)   — which parts of c_t to expose as h_t

The candidate uses tanh because it carries information, not a
0-1 control value. Calling it the "update gate" is GRU vocabulary.

## c vs h
c is the full memory — everything the cell remembers, carried from
the beginning, never used directly for prediction.
h is the filtered view: the slice of c relevant to this timestep.

Only h feeds W_hy and only h is stacked into hidden_states. The
output gate controls what the memory exposes; closed dimensions
stay intact in c, available at a future step.

## Results vs Day 37 (same data, same hyperparameters)
    RNN:   loss 1.52 at 1000 epochs
    LSTM:  loss 1.26 at  500 epochs

    RNN:   "f the was not a little the dodo, oughs in the was n"
    LSTM:  "f the windown to she cally a thing and began the wo"

The RNN strung word fragments together. The LSTM holds structure
across several words — "a thing and began the" is grammatical.
"windown" and "cally" are plausible non-words: correct English
letter statistics, wrong words. The failure mode moved from noise
to reaching for structure and missing.

Caveat: the LSTM ran 500 epochs vs the RNN's 1000. It still wins
on both loss and output quality, but this is not a matched run.

## Still Broken
Grammar across longer spans still falls apart. Gating extends how
far information travels but recurrence is still sequential —
timestep 50 cannot be computed until timestep 49 finishes, and
information still passes through every intermediate step.

That is the wall attention goes through.

## Bugs Fixed
- torch.tanh on the output gate instead of sigmoid — a gate needs 0-1
- h = sigmoid(c * o) instead of tanh(c) * o
- init calls for W_hy and W_hh written before declaring them
- predict() accepted h but not c — cell state reset every step,
  discarding the one thing LSTM adds. Same bug class as Day 37.
- torch.tensor([idx]) gives shape (1,); forward() needs (1,1)