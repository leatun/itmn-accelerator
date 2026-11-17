# ITMN accelerator RTL

Verilog accelerator for ITMN, an ECG classifier (encoder, 5 ITM blocks of Inception and Mamba1, global average
pooling, FC head) trained on PTB-XL (5 super-diagnostic classes), in Q4.11 fixed point. Target: Kria KV260
(xck26-sfvc784-2LV-c) at 250 MHz. Simulation is bit-exact with the integer model of the trained network.

| Folder | What it is |
|---|---|
| `itmn_e2e/` | the full network on chip: raw 12-lead ECG in, 5 class logits out |
| `mamba_only/` | the earlier design with the Mamba mixer only (the Inception part runs on the host) |
| `mamba_only/bench/` | cycle benchmark of `mamba_only` at the layer sizes of other Mamba / Mamba2 / Vim / eMamba models (the rows compared with other papers) |
| `common/` | modules shared by both designs: PE array, memories, activation tables |
| `goldens/itmn/` | weights, inputs and expected values of every stage, from the trained checkpoint |
| `ref/itmn_e2e_ref.py` | integer reference of the network outside the Mamba mixer; checks every golden link |
| `results/` | measured cycles and post-route resources |

## Architecture in one paragraph

Two identical clusters, each a controller (`src/Mamba_Cluster.v`), a 16-lane PE array with a scan state RAM
(`common/M_Cluster.v`, `common/Mamba_PE_v2.v`) and its own memories (`common/Memory_Cluster.v`: main URAM, weight
BRAM, const LUTRAM). Group-parallel stages split output groups by parity between the clusters; Inception stages split
timesteps. Writes to the shared part of the main RAM are mirrored into the other cluster's RAM; the clusters meet at a
barrier after each stage. The host loads weights and input through a 256-bit DMA port, per block and phase.
`itmn_e2e` runs per block an Inception phase (encoder on block 0, P1, maxpool3, bottleneck, branches k1/k9/k19/k39)
and a Mamba phase (RMSNorm, in_proj, conv, SiLU, x_proj, dt_proj, selective scan, gate, out_proj) with the block
output, MaxPool and GAP folded into the out_proj writeback, and the FC head after block 4.

## Tools

- ModelSim (tested: ModelSim-Altera Starter 13.0sp1, `vsim`); scripts run from the design folder.
- Vivado 2022.1 (synthesis, and xsim for the bench).
- Python 3 with numpy (`ref/`, bench script).

## Reproduce

All commands from the design folder (`itmn_e2e/` or `mamba_only/`). Each testbench prints its cycle count and `PASS`
(or a byte-exact PASS line) only when every checked value equals the golden.

| Check | Command | Expected |
|---|---|---|
| Mamba phase, block 0, every stage of timestep 1 | `vsim -c -do "set DBG_T 2; do run_dbg_sim.do"` | itmn_e2e 3,909 cycles, mamba_only 6,317 |
| same, timestep 16 | `set DBG_T 17` | 32,259 / 52,727 |
| Mamba phase of all 5 blocks, first 20 t | `vsim -c -do "set VERIFY_T 20; do run_verify_sim.do"` | per block 37,929 (block 4: 90,657) / 62,009 (138,817) |
| same, full length | `vsim -c -do "do run_verify_sim.do"` (about an hour) | see `results/` |
| Inception phase of one block (itmn_e2e) | `vsim -c -do "set ITM_BK 0; set INC_T 20; do run_inc_sim.do"` | 8,255 cycles, all 6 stages PASS |
| one full ITM block (itmn_e2e) | `vsim -c -do "set ITM_BK 0; set BLK_T 20; do run_block_sim.do"` | per t: INC 444, Mamba 1,922 cycles (block 4: 1,424 / 4,592, with GAP and FC) |
| raw ECG to logits (itmn_e2e) | `vsim -c -do "do run_e2e_sim.do"` (about an hour) | 8,516,011 cycles, logits 2978, -1926, -3695, -2539, -2052 |
| post-route timing and resources | `vivado -mode batch -source run_synth_kv260.tcl` (about 1 h) | see `results/` |
| bench (mamba_only) | `python bench/run_bench.py` (Vivado/bin on PATH) | every config equals its recorded cycles |
| integer reference | `python ref/itmn_e2e_ref.py` (from the root) | `ALL OK` |

`set LIBN <name>` gives a run its own work library, so several can run side by side. Synthesis is out of context
(the top has ~600 ports); its reports go to `<design>/synth_reports_kv260/`.

## Notes for reading the results

- Timing is out of context with the DMA port and the block configuration inputs false-pathed (they are static while
  the accelerator computes); a system integration adds the AXI interface around it.
- `mamba_only/bench` measures cycles only: the inputs are constant and the weights are not loaded, so no value is
  checked. The models above ITMN's size need far more weight storage than the chip has; their rows add a modelled
  weight-streaming time (`dma_cyc`, one 256-bit word per cycle) and are labelled size-scaled. The configurations with
  several state words per channel (`nsg` > 1: Mamba2, eMamba CIFAR-10 / WikiText-2) have never been checked against
  data, only for cycles.
- The controllers keep general-Mamba options as ports (`N_STATE_GRP`, `USE_M5`, `XP_OUT_GRP_IN`); ITMN uses 1, 1, 3.
