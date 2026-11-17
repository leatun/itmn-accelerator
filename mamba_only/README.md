# mamba_only: the Mamba mixer of ITMN

The design before the Inception part moved on chip: the same two-cluster architecture as `itmn_e2e`, running the
Mamba phase of an ITM block (RMSNorm through out_proj, over all timesteps). The host provides each block's input (the
P1 output) and reads back the mixer output. `itmn_e2e` is this design plus the Inception phase, the block output and
the head, and a faster scan (15 instead of 34 cycles per channel).

| File | Contents |
|---|---|
| `src/Mamba_Top.v`, `src/Mamba_Cluster.v`, `src/_parameter.v` | top, controller of one cluster, parameters and RAM layout |
| `tb/tb_Mamba_Top_DBG.v` | `run_dbg_sim.do`: block 0, every stage of timestep DBG_T-1 |
| `tb/tb_Mamba_Top_5BLOCK_VERIFY.v` | `run_verify_sim.do`: the 5 blocks back to back, out_proj of every t |
| `run_synth_kv260.tcl` | Vivado out-of-context synthesis + implementation |
| `bench/` | cycle benchmark at other models' sizes, below |

## bench: other Mamba models on this design

`bench/configs.csv` lists the layer sizes of the models compared in the thesis (Mamba2 130M-2.7B, Mamba1
130M-2.8B, Vim Ti/S/B, the four eMamba edge tasks) and which compared paper (`adapters`) each row stands for.
`bench/run_bench.py` compiles this design once with xsim, elaborates `bench/tb_bench.v` with each row's
parameters and runs 3 tokens of one layer:

    python bench/run_bench.py                      # all rows, about an hour
    python bench/run_bench.py --only VIM_TI,M2_130M

It writes `bench/results.csv` (compute and scan cycles per token and layer; latency per token over all layers at
250 MHz, with and without the modelled weight streaming; Vim bidirectional estimate) and checks the cycles against
the recorded ones (`ref_cyc`, `ref_m6`).

What the rows are: the same hardware, run at another layer size. No data is checked (constant input, weights not
loaded); the rival papers' own techniques (quantisation schemes, low-rank, speculative decoding, ...) are not
implemented. `dma_cyc` is a model: the layer's weights streamed once per token over the 256-bit port at one word per
cycle. Vim's bidirectional figure adds one more scan per token (upper bound). Synthesis at a bench size:

    vivado -mode batch -source run_synth_kv260.tcl -tclargs 4.0 synth_reports_m2_130m 13

(the third argument is the config's `h_addr_w`, which sizes the scan state RAM).
