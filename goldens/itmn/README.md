# ITMN goldens (PTB-XL super-diagnostic checkpoint, Q4.11)

Hex files, one 16-bit value per line, read by the testbenches with `$readmemh` through `` `GOLDEN_ROOT``
(`common/goldens.vh`, default `../goldens/itmn` relative to a design folder). Generated from the trained model by the
extraction script of the model repository; all are integer (bit-exact) values even where the name ends in `_FP`.
Matrices are row-major [out][in]; activations are channel-major, element (c, t) at line c*T + t.

| Path | Contents |
|---|---|
| `exp_lut.txt`, `silu_lut.txt`, `softplus_lut.txt` | 256-entry activation tables (`common/*_LUT.v`) |
| `rsqrt_q97.txt` | RMSNorm reciprocal square root ROM (`common/RSqrt_ROM.v`) |
| `encoder/` | encoder input (raw 12-lead ECG, 12 x 1000), fused conv+BN weights and bias, output |
| `block_0b_layerNN/` | ITM block b: `P1_Input_X` (block input), `P1_*` (k1 conv + BN), `W_Bot`, `W_B1..W_B4` and `Out_*` (Inception branches), `Inc_BN_*` (branch BN), `Mam_W_*`, `Mam_A_signed`, `Mam_D_param`, `Mam_B_*` (Mamba weights), `Mam_*_FP` (Mamba intermediates: x_inner, z, delta, y_gated, out_proj), `Final_ITM_Full_FP` (block output) |
| `gap_fc/` | GAP output, FC weights and bias, FC logits (the network output) |

Block shapes: blocks 0-1 T = 1000, d_model 64; blocks 2-3 T = 500 (MaxPool 2 after blocks 1 and 3); block 4 T = 250,
d_model 128. d_inner = 2 d_model, d_state 16, dt_rank = d_model / 16.
