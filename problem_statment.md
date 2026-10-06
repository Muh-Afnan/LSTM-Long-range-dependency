# Day 38 — LSTM: Long-Range Dependency

## Problem
The Day 37 RNN learned English spelling but produced broken grammar:
"the was not a little the". Information from early timesteps decayed
because h was fully overwritten at every step, multiplied by W_hh
and squashed through tanh. Build an LSTM from scratch that preserves
long-range information.

## Core Questions
- What operation lets information survive 50 timesteps?
- What is the gradient through that operation, and why does it matter?
- What do the four gates control?
- Why keep two states instead of one?

## Requirements
- LSTMFromScratch with 14 nn.Parameter tensors (4 gates + output)
- Same hyperparameters as Day 37 for a fair comparison
- Autoregressive generation carrying both h and c
- Compare loss and output quality against the RNN baseline