# Day 38 — LSTM from Scratch

Gated recurrent cell built with raw nn.Parameter tensors, trained
on Alice in Wonderland (~49k characters).

## Core Class
LSTMFromScratch — 14 parameters: W, U, b for each of four gates,
plus W_hy and b_y for the output projection.

## Config
vocab 29 | embedding 8 | hidden 128 | window 6 | batch 32
Adam lr=0.001 | 500 epochs

## Results
| Model | Epochs | Loss | Cell params |
|-------|--------|------|-------------|
| RNN   | 1000   | 1.52 | 17,536      |
| LSTM  |  500   | 1.26 | 70,144      |

    RNN:   "f the was not a little the dodo, oughs in the was n"
    LSTM:  "f the windown to she cally a thing and began the wo"

## Key Insight
    c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t

Addition, not replacement. The gradient through that line is 1,
so error signal reaches timesteps 50 steps back intact.

c holds everything. h is what the output gate chooses to show.

## Run
python demo.py