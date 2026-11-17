#!/usr/bin/env python3
"""
itmn_e2e_ref.py - integer reference for the end-to-end ITMN datapath (Q4.11).

Recomputes the goldens in ../goldens/itmn bit-exactly from the raw 12-lead input:
encoder -> 5x ITM block (P1, Bot, B1..B4, block output; MaxPool k2s2 before blocks 2
and 4) -> GAP -> FC. The Mamba mixer output is taken from Mam_OutProj_FP (the
RTL testbenches check the mixer stage by stage); everything else is recomputed
with the RTL's integer arithmetic.

    python itmn_e2e_ref.py            # check every link against the goldens
"""
import os
import sys
import numpy as np

FB = 11
GOLDEN = os.environ.get("GOLDEN_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "goldens", "itmn"))
# (d_in, d_out, T) per ITM block
DIMS = {0: (64, 64, 1000), 1: (64, 64, 1000), 2: (64, 64, 500), 3: (64, 64, 500), 4: (64, 128, 250)}
POOL_BEFORE = {2, 4}          # MaxPool1d(2, 2) precedes these blocks
GAP_INV_T_Q15 = 131           # round(2^15 / 250)


def sat(x):
    return np.clip(x, -32768, 32767)


def load(rel):
    v = np.array([int(l, 16) for l in open(os.path.join(GOLDEN, rel)) if l.strip()], dtype=np.int64)
    v[v >= 0x8000] -= 0x10000
    return v


def p1(x, w, b):                      # k=1 conv + fused BN bias (encoder, P1)
    return sat(sat((w @ x) >> FB) + b[:, None])


def mac(x, w):                        # k=1 conv, no bias (Bot, B1)
    return sat((w @ x) >> FB)


def conv(x, w, k, pad):               # one wide accumulation over taps x C_in, zero pad
    _, t = x.shape
    acc = np.zeros((w.shape[0], t), np.int64)
    for tap in range(k):
        sh = tap - pad
        src = np.zeros_like(x)
        if sh >= 0:
            src[:, :t - sh] = x[:, sh:]
        else:
            src[:, -sh:] = x[:, :t + sh]
        acc += w[:, :, tap] @ src
    return sat(acc >> FB)


def maxpool3(x):                      # k=3 s=1 pad=1, edge-clamped (= -inf pad)
    left = np.concatenate([x[:, :1], x[:, :-1]], 1)
    right = np.concatenate([x[:, 1:], x[:, -1:]], 1)
    return np.maximum(np.maximum(left, x), right)


def maxpool2(x):                      # k=2 s=2
    return np.maximum(x[:, 0::2], x[:, 1::2])


def bn_relu(x, scale, shift):
    return np.maximum(sat(sat((x * scale[:, None]) >> FB) + shift[:, None]), 0)


def fin(inc, scale, shift, mamba_out):
    return sat(bn_relu(inc, scale, shift) + np.maximum(mamba_out, 0))


def gap(x):
    return sat((x.sum(1) * GAP_INV_T_Q15) >> 15)


def fc(g, w, b):
    return sat(sat((w @ g) >> FB) + b)


def block(i, x_in):
    """Return dict of every intermediate for ITM block i given its (pooled) input."""
    d_in, d_out, t = DIMS[i]
    dim = d_out // 4
    g = sorted(d for d in os.listdir(GOLDEN) if d.startswith("block_"))[i] + "/"
    r = {"in": x_in}
    r["p1"] = p1(x_in, load(g + "P1_Weight_Fused.txt").reshape(d_out, d_in), load(g + "P1_Bias_Fused.txt"))
    r["bot"] = mac(r["p1"], load(g + "W_Bot.txt").reshape(dim, d_out))
    r["b1"] = mac(maxpool3(r["p1"]), load(g + "W_B1.txt").reshape(dim, d_out))
    for name, k in (("b2", 9), ("b3", 19), ("b4", 39)):
        r[name] = conv(r["bot"], load(g + f"W_{name.upper()}.txt").reshape(dim, dim, k), k, k // 2)
    r["inc"] = np.concatenate([r["b1"], r["b2"], r["b3"], r["b4"]], 0)
    r["mamba"] = load(g + "Mam_OutProj_FP.txt").reshape(d_out, t)
    r["final"] = fin(r["inc"], load(g + "Inc_BN_Scale.txt"), load(g + "Inc_BN_Shift.txt"), r["mamba"])
    return r, g


def check():
    ok_all = True
    def rep(name, a, golden_rel, shape):
        nonlocal ok_all
        ok = np.array_equal(a, load(golden_rel).reshape(shape))
        ok_all &= ok
        print(("OK  " if ok else "BAD ") + name)
    wav = load("encoder/Encoder_Input_X.txt").reshape(12, -1)
    enc = p1(wav, load("encoder/Encoder_Weight_Fused.txt").reshape(64, 12), load("encoder/Encoder_Bias_Fused.txt"))
    rep("encoder", enc, "encoder/Encoder_Output_FP.txt", enc.shape)
    x = enc
    for i in range(5):
        if i in POOL_BEFORE:
            x = maxpool2(x)
        r, g = block(i, x)
        d_in, d_out, t = DIMS[i]
        rep(f"B{i} input", r["in"], g + "P1_Input_X.txt", (d_in, t))
        rep(f"B{i} P1", r["p1"], g + "P1_Output_Golden_FP.txt", (d_out, t))
        for k, f in (("bot", "Out_Bot"), ("b1", "Out_B1"), ("b2", "Out_B2"), ("b3", "Out_B3"), ("b4", "Out_B4")):
            rep(f"B{i} {k}", r[k], g + f + ".txt", (d_out // 4, t))
        rep(f"B{i} final", r["final"], g + "Final_ITM_Full_FP.txt", (d_out, t))
        x = load(g + "Final_ITM_Full_FP.txt").reshape(d_out, t)
    g_ = gap(x)
    rep("GAP", g_, "gap_fc/GAP_Output.txt", g_.shape)
    w = load("gap_fc/FC_Weight.txt")
    lg = fc(g_, w.reshape(-1, 128), load("gap_fc/FC_Bias.txt"))
    rep("FC logits", lg, "gap_fc/FC_Logits.txt", lg.shape)
    print("ALL OK" if ok_all else "MISMATCH")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if check() else 1)
