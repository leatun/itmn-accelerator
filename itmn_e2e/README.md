# itmn_e2e: the full ITMN network

| File | Contents |
|---|---|
| `src/Mamba_Top.v` | two clusters, their memories, mirror writes between them, host DMA |
| `src/Mamba_Cluster.v` | controller of one cluster: every stage of both phases, the MAC job table, scan, FIN / MaxPool / GAP |
| `src/_parameter.v` | number format, PE modes, main / weight / const RAM layout, stage ids |
| `tb/` | testbenches, below |
| `run_*_sim.do` | ModelSim scripts (shared part in `../sim_common.tcl`) |
| `run_synth_kv260.tcl` | Vivado out-of-context synthesis + implementation (flow in `../synth_kv260.tcl`) |

## Host sequence per block

1. DMA the Inception weights into both clusters' weight RAMs, the P1 bias into the const RAMs (block 0: also the raw
   ECG and the encoder weights); `start` with `PHASE_INC = 1` (`ENC_EN` on block 0). The block input is in
   `PT_X_IN` (from the previous block, or the encoder).
2. DMA the Mamba weights (output groups split by parity: even groups to cluster 0) and constants, the Inception BN
   scale / shift (block 4: also FC weights and bias); `start` with `PHASE_INC = 0`, `FIN_EN = 1`, `POOL_OUT` on
   blocks 1 and 3, `GAP_EN` and `GAP_INV` on block 4.
3. The block output is in `PT_INC_OUT`, which is the next block's `PT_X_IN`; after block 4 the logits are at
   `PT_LOGITS`, lanes 0..4.

`tb/tb_ITM_E2E.v` is this sequence for all 5 blocks.

## Testbenches

| Testbench | Script | Checks |
|---|---|---|
| `tb_Mamba_Top_DBG` | `run_dbg_sim.do` (`DBG_T`) | Mamba phase of block 0: x_inner, z, delta, y_gated, out_proj of timestep DBG_T-1 |
| `tb_Mamba_Top_5BLOCK_VERIFY` | `run_verify_sim.do` (`VERIFY_T`) | Mamba phase of the 5 blocks back to back, out_proj of every t |
| `tb_ITM_INC` | `run_inc_sim.do` (`ITM_BK`, `INC_T`) | Inception phase of one block: P1, bottleneck, 4 branches |
| `tb_ITM_BLOCK` | `run_block_sim.do` (`ITM_BK`, `BLK_T`) | one block, both phases, block output (+ GAP and logits on block 4) |
| `tb_ITM_E2E` | `run_e2e_sim.do` (`ITM_BK` = last block) | raw ECG to logits, every block output on the way |

The Mamba-phase testbenches start from each block's golden input, so a failure points at one block and stage.
