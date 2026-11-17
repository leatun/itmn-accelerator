# Results

Clock 250 MHz (4.0 ns), Kria KV260 (xck26-sfvc784-2LV-c), Vivado 2022.1, out of context, post-route.

## itmn_e2e: full ITMN, one ECG (12 leads x 1000 samples) to 5 logits

Cycles from `tb_ITM_E2E` (full length, logits bit-exact with the golden: 2978, -1926, -3695, -2539, -2052):

| Block | T | Inception phase | Mamba phase (+ block output, pooling, GAP/FC) | Total |
|---|---|---|---|---|
| 0 (with encoder) | 1000 | 444,017 | 1,916,131 | 2,360,148 |
| 1 | 1000 | 412,015 | 1,916,131 | 2,328,146 |
| 2 | 500 | 206,015 | 958,131 | 1,164,146 |
| 3 | 500 | 206,015 | 958,131 | 1,164,146 |
| 4 (with GAP, FC) | 250 | 356,015 | 1,143,410 | 1,499,425 |
| **all** | | | | **8,516,011 = 34.1 ms** |

Host DMA time between phases is not included.

Post-route (`itmn_e2e/run_synth_kv260.tcl`):

| Design | WNS (setup) | WHS (hold) | LUT | FF | BRAM tiles | URAM | DSP | Power |
|---|---|---|---|---|---|---|---|---|
| itmn_e2e | +0.036 ns | +0.055 ns | 49,287 (42 %) | 34,333 (15 %) | 114 / 144 | 32 / 64 | 76 / 1248 | 1.60 W |

## mamba_only: the Mamba mixer

Cycles from `tb_Mamba_Top_5BLOCK_VERIFY` (full length, every out_proj value bit-exact):

| Block | T | Cycles | per t |
|---|---|---|---|
| 0 | 1000 | 3,094,129 | 3,094 |
| 1 | 1000 | 3,094,129 | 3,094 |
| 2 | 500 | 1,547,129 | 3,094 |
| 3 | 500 | 1,547,129 | 3,094 |
| 4 | 250 | 1,732,257 | 6,929 |

The same testbench on `itmn_e2e` (Mamba phase only, without the block output stage): 1,890 cycles per t on blocks 0-3,
4,521 on block 4; the difference is mostly the scan (15 instead of 34 cycles per channel).

Post-route (`mamba_only/run_synth_kv260.tcl`):

| Design | WNS (setup) | WHS (hold) | LUT | FF | BRAM tiles | URAM | DSP | Power |
|---|---|---|---|---|---|---|---|---|
| mamba_only | +0.016 ns | +0.055 ns | 37,932 (32 %) | 26,432 (11 %) | 114 / 144 | 24 / 64 | 72 / 1248 | 1.42 W |

## Bench: mamba_only at other models' layer sizes

`bench.csv`, produced by `mamba_only/bench/run_bench.py`. Cycles per token and layer, measured on the RTL; no data
checked. `ms_*` are per token through all `n_layer` layers at 250 MHz; `ms_dma` adds the modelled weight streaming.
`adapters` names the compared paper(s) whose model size the row reproduces; their own techniques are not implemented.
