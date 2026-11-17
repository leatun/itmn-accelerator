`ifndef PARAMETER_V3
`define PARAMETER_V3

// Q4.11 fixed-point
`define DATA_W            16
`define ACC_W             40
`define FRAC_BITS         11

// Mamba_PE modes
`define MAMBA_PE_IDLE     3'd0
`define MAMBA_PE_MAC      3'd1
`define MAMBA_PE_MUL      3'd2
`define MAMBA_PE_ADD      3'd3
`define MAMBA_PE_MAC2     3'd5
`define MAMBA_PE_MUL2     3'd6

// H_RegFile geometry (SSM state). Synth uses these defaults; TBs override
// H_ADDR_W/H_DEPTH on Mamba_Top. Must cover every address M6 can generate.
`define H_ADDR_W          9
`define H_DEPTH           256

// ============================================================================
// Per-timestep unified memory layout (ram_main, mirrored across clusters)
//   Sizes cover B0..B4 (CH_OUT<=8, CH_M<=16).
//   Slot 0..3 = lifetime-aliased scratch (16 words each).
//   Slot CIRC = X_INNER_CIRC 4-tap circular (64 words).
//   Slot BULK = INPUT / MAMBA_OUT aliased (T x CH_OUT words).
// ============================================================================
`define PT_X_NORM         15'd0       // Slot 0 (also X_PROJ, Y_GATED - disjoint lifetimes)
`define PT_X_PROJ         15'd0
`define PT_Y_GATED        15'd0
`define PT_X_CONV         15'd16      // Slot 1 (also U, Y_SSM)
`define PT_U              15'd16
`define PT_Y_SSM          15'd16
`define PT_DELTA          15'd32      // Slot 2
`define PT_Z_GATE         15'd48      // Slot 3 (hold M1B -> M7)
`define PT_X_INNER_CIRC   15'd64      // Slot CIRC - 4x16 words, addr = (t%4)*CH_M + c_grp
`define PT_INPUT          15'd128     // Slot BULK - preload input, T x CH_OUT words
`define PT_MAMBA_OUT      15'd128     // Slot BULK - aliased with INPUT (INPUT[t] dies before MAMBA_OUT[t] is written)
// ITM regions (T x CH_OUT words each, sized for B0: T=1000, CH_OUT=4)
//   PT_X_IN / PT_INC_OUT : block input -> inception branch outputs (branch b =
//                          word b of each t) -> FIN in place -> MaxPool2 in place
//                          -> next block's input
//   PT_MP3 / PT_BOT      : maxpool3(P1) for B1, then bottleneck (1 word/t)
`define PT_X_IN           15'd4128
`define PT_INC_OUT        15'd4128
`define PT_MP3            15'd8128
`define PT_BOT            15'd8128
`define PT_MAIN_DEPTH     12128       // 128 + 3 x 4000, <= 3 URAM rows (12288)

// Main-bus address width: covers the largest RAM and is the unified DMA
// address bus width. Memory_Cluster narrows it per RAM.
`define MAIN_ADDR_W       15

// Weight RAM (BRAM, TDP for MAC2 dual read), one per cluster. Even output
// groups live in c0's RAM, odd in c1's, both compacted; SMALLS duplicated.
//   Layout sized for worst-case B4 (d_inner=256, dt_rank=8):
//     [0..1216)      SMALLS resident:
//                      W_DW      [0..64)     max = d_inner*4/16 = 64
//                      W_XPROJ   [64..832)   max = 48*d_inner/16 = 768
//                      W_DTPROJ  [832..960)  max = d_inner*dt_rank/16 = 128
//                      W_A       [960..1216) max = d_inner*d_state/16 = 256
//     [1216..3264)   W_INPROJ_X (max 2048 words)
//     [3264..5312)   W_INPROJ_Z (max 2048 words)
//     [5312..7360)   W_OUTPROJ  (max 2048 words)
//     [7360..8192)   spare
`define W_MEM_DEPTH       8192
`define W_MEM_ADDR_W      13
`define W_DW_BASE         15'd0
`define W_XPROJ_BASE      15'd64
`define W_DTPROJ_BASE     15'd832
`define W_A_BASE          15'd960
`define W_INPROJ_X_BASE   15'd1216
`define W_INPROJ_Z_BASE   15'd3264
`define W_OUTPROJ_BASE    15'd5312
`define W_FC_BASE         15'd7360    // block 4 head: FC 128 -> n_cls (1 group x 128 words), Mamba layout spare

// INC-phase weight layout. ram_weight is reloaded by the host per phase (INC
// phase, then MAMBA phase), so these overlay the Mamba bases above. Inception
// stages split work by time, so both clusters hold the full (uncompacted)
// weights: word = base + g*len + k, 16 output lanes per word.
`define W_P1_BASE         15'd0       // d_out x d_in / 16 words (B4: 512)
`define W_BOT_BASE        15'd512     // (d_out/4) x d_out / 16 (B4: 256)
`define W_B1_BASE         15'd768     // (d_out/4) x d_out / 16 (B4: 256)
`define W_B2_BASE         15'd1024    // (d_out/4) x 9  x (d_out/4) / 16 (B4: 576)
`define W_B3_BASE         15'd1600    // (d_out/4) x 19 x (d_out/4) / 16 (B4: 1216)
`define W_B4_BASE         15'd2816    // (d_out/4) x 39 x (d_out/4) / 16 (B4: 2496), ends 5312
`define W_ENC_BASE        15'd5312    // encoder 12(->16) x 64: 4 groups x 16 words

// Const RAM (LUTRAM), duplicated per cluster
`define C_MEM_DEPTH       128
`define C_MEM_ADDR_W      7
`define C_W_NORM_BASE     15'd0
`define C_B_DW_BASE       15'd8
`define C_B_DT_BASE       15'd24
`define C_D_PARAM_BASE    15'd40
`define C_P1_BIAS_BASE    15'd64      // INC phase: P1 fused-BN bias, d_out/16 words
`define C_INC_SCALE_BASE  15'd72      // FIN: inception BN scale, d_out/16 words (Mamba phase)
`define C_INC_SHIFT_BASE  15'd80      // FIN: inception BN shift, d_out/16 words
`define C_ENC_BIAS_BASE   15'd88      // encoder fused-BN bias, 4 words (INC phase, block 0)
`define C_FC_BIAS_BASE    15'd92      // FC bias, 1 word (lanes 0..n_cls-1)

// ============================================================================
// Datapath geometry
// ============================================================================
// NUM_LANES is a plain integer so it can be used in generate loop bounds and
// replication counts (Verilog requires elaboration-time ints).
`define NUM_LANES         16
`define LANE_MAX          (`NUM_LANES - 1)  // ctr_l rolls over at this value

// M2 depthwise conv: 4 taps x 16b = 1 weight word per c_grp
`define M2_WEIGHT_STRIDE  15'd4

// Fixed-point saturation output values (16-bit)
`define SAT16_MAX         16'sh7FFF
`define SAT16_MIN         16'sh8000  // = -32768 signed

// cur_stage enum (5-bit: room for the ITM stages)
`define STG_M1A           5'd0
`define STG_M1B           5'd1
`define STG_M2            5'd2
`define STG_M3            5'd3
`define STG_M4            5'd4
`define STG_M5            5'd5
`define STG_M6            5'd6
`define STG_M7            5'd7
`define STG_M8            5'd8
`define STG_RN            5'd9
`define STG_P1            5'd10       // INC phase: block P1 (k=1 conv + fused BN)
`define STG_MP3           5'd11       // INC phase: maxpool3 (k3 s1, edge-clamped) of P1 -> PT_MP3
`define STG_B1            5'd12       // INC phase: branch 1 = k1 conv of MP3 -> INC_OUT word 0
`define STG_BOT           5'd13       // INC phase: bottleneck k1 conv of P1 -> PT_BOT
`define STG_B2            5'd14       // INC phase: branch 2 = k9  conv of BOT -> INC_OUT words dim/16*1..
`define STG_B3            5'd15       // INC phase: branch 3 = k19 conv of BOT -> INC_OUT words dim/16*2..
`define STG_B4            5'd16       // INC phase: branch 4 = k39 conv of BOT -> INC_OUT words dim/16*3..
`define STG_ENC           5'd17       // INC phase, block 0: encoder k1 conv 12 -> 64 (+ bias) -> PT_X_IN
`define PT_ENC_IN         15'd8128    // raw 12-lead input, 1 word/t (lanes 12..15 = 0); MP3 region is free
`define STG_GAP           5'd18       // block 4: GAP finalize (acc * INV_T >> 15) -> PT_GAP
`define STG_FC            5'd19       // block 4: FC head GAP -> logits (+ bias) -> PT_LOGITS
`define PT_GAP            15'd8128    // GAP vector, CH_OUT words (BOT region is dead in the Mamba phase)
`define PT_LOGITS         15'd8136    // FC logits, word 0 lanes 0..n_cls-1

`endif
