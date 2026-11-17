`include "_parameter.v"

// Controller of one cluster. Top level instantiates two (CLUSTER_ID 0 and 1) running the same FSM on their own PE
// array, scan state RAM and memories; ram_main writes to the shared regions are mirrored into the peer's RAM by the
// top level (see Mamba_Top), the per-cluster scratch slots are not.
//
// Work split: in group-parallel stages cluster c computes output groups g = c, c+2, ... (ctr_g steps by 2) and
// holds only those groups' weights (compacted, index g/2). Time-split stages (the Inception phase) instead take
// timesteps t = c, c+2, ... and compute every output group of their t, with the full weights.
// A cluster that finishes a stage waits in S_BARRIER until the peer has finished it too.
//
// Mamba phase, per timestep: RN (RMSNorm), M1A/M1B (in_proj x, z), M2 (causal conv4 + bias), M3 (SiLU),
// M4 (x_proj), M5 (dt_proj + softplus), M6/M7 (scan and gate, interleaved per group), M8 (out_proj).
// With FIN_EN, M8's writeback also forms the ITM block output (FIN) and, with POOL_OUT, MaxPool1d(2,2); with GAP_EN
// the block then runs GAP and the FC head. Inception phase (PHASE_INC): [ENC], P1, MP3, B1, BOT, B2, B3, B4.
//
// Latencies the schedule is built on: main and weight RAM data are usable 3 cycles after the address is set (RAM
// output register + m_rd_data / w_rd_data_r), const RAM data 2 cycles; a PE result (cl_out_vec) is valid when the
// drain counter reaches 4, i.e. 6 cycles after the issuing state.
module Mamba_Cluster #(
    parameter CLUSTER_ID = 0,
    parameter H_ADDR_W   = `H_ADDR_W,
    parameter H_DEPTH    = `H_DEPTH
)(
    input  wire         clk,
    input  wire         rst,
    input  wire         start,

    // Stage barrier with the peer cluster
    input  wire         peer_done_stage,
    output wire         done_stage,
    output wire         done_all,

    // Block shape, sampled continuously; hold them while busy
    input  wire [9:0]   T_MAX,
    input  wire [8:0]   CH_OUT,         // d_model / 16
    input  wire [8:0]   CH_M,           // d_inner / 16 (0 = 128)
    input  wire [7:0]   DT_RANK,
    input  wire [3:0]   N_STATE_GRP,    // state words (of 16) per channel; ITMN: 1
    input  wire         USE_M5,         // dt_proj stage present; ITMN: 1
    input  wire [4:0]   XP_OUT_GRP_IN,  // x_proj output groups (0 = 3)
    input  wire         PHASE_INC,      // 1: Inception phase, 0: Mamba phase
    input  wire [8:0]   CH_IN,          // block input d_in / 16 (P1)
    input  wire         FIN_EN,         // fuse the ITM block output into M8
    input  wire         POOL_OUT,       // with FIN_EN: MaxPool1d(2,2) the block output
    input  wire         ENC_EN,         // Inception phase starts with the encoder (block 0)
    input  wire         GAP_EN,         // with FIN_EN: GAP and FC head after the t-loop (block 4)
    input  wire [15:0]  GAP_INV,        // round(2^15 / T)

    // Memories of this cluster (Memory_Cluster)
    output reg  [14:0]  m_rd_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] m_rd_data_c,

    output reg          m_we,
    output reg  [14:0]  m_wr_addr,
    output reg  [`NUM_LANES*`DATA_W-1:0] m_wr_data,

    output reg  [14:0]  w_rd_addr,
    output reg  [14:0]  w_rd_addr2,
    input  wire [`NUM_LANES*`DATA_W-1:0] w_rd_data,
    input  wire [`NUM_LANES*`DATA_W-1:0] w_rd_data2,

    output reg  [14:0]  const_rd_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0] const_rd_data
);

    // CH_M = 0 means 128 groups
    wire [9:0]  ch_m_act = (CH_M == 9'd0) ? 8'd128 : CH_M;
    wire [13:0] d_inner  = {ch_m_act, 4'b0};
    // S_H_INIT zeroes every state word M6 can address (see m6_c): d_inner words with one state word per channel,
    // d_inner*8 otherwise
    wire [17:0] h_init_limit = (N_STATE_GRP == 4'd1) ? {4'b0, d_inner}
                                                     : {d_inner, 3'b0};
    wire [12:0] d_model  = {CH_OUT, 4'b0};
    wire [9:0]  t_last   = T_MAX - 10'd1;
    // RMSNorm mean = sum >> log2(d_model); d_model must be a power of 2
    wire [3:0]  log2_dm  =
        (CH_OUT == 9'd4)   ? 4'd2  :
        (CH_OUT == 9'd8)   ? 4'd3  :
        (CH_OUT == 9'd16)  ? 4'd4  :
        (CH_OUT == 9'd32)  ? 4'd5  :
        (CH_OUT == 9'd64)  ? 4'd6  :
        (CH_OUT == 9'd128) ? 4'd7  :
                             4'd8;
    wire [5:0]  rn_total_shift = {2'b0, log2_dm} + 6'd19;
    wire [4:0]  XP_OUT_GRP = (XP_OUT_GRP_IN == 5'd0) ? 5'd3 : XP_OUT_GRP_IN;

    localparam [14:0] PT_X_NORM        = `PT_X_NORM;
    localparam [14:0] PT_X_PROJ        = `PT_X_PROJ;
    localparam [14:0] PT_Y_GATED       = `PT_Y_GATED;
    localparam [14:0] PT_X_CONV        = `PT_X_CONV;
    localparam [14:0] PT_U             = `PT_U;
    localparam [14:0] PT_Y_SSM         = `PT_Y_SSM;
    localparam [14:0] PT_DELTA         = `PT_DELTA;
    localparam [14:0] PT_Z_GATE        = `PT_Z_GATE;
    localparam [14:0] PT_X_INNER_CIRC  = `PT_X_INNER_CIRC;
    localparam [14:0] PT_INPUT         = `PT_INPUT;
    localparam [14:0] PT_MAMBA_OUT     = `PT_MAMBA_OUT;
    localparam [14:0] PT_X_IN          = `PT_X_IN;
    localparam [14:0] PT_INC_OUT       = `PT_INC_OUT;
    localparam [14:0] PT_MP3           = `PT_MP3;
    localparam [14:0] PT_BOT           = `PT_BOT;
    localparam [14:0] W_BOT_BASE       = `W_BOT_BASE;
    localparam [14:0] W_B1_BASE        = `W_B1_BASE;
    localparam [14:0] W_B2_BASE        = `W_B2_BASE;
    localparam [14:0] W_B3_BASE        = `W_B3_BASE;
    localparam [14:0] W_B4_BASE        = `W_B4_BASE;
    localparam [14:0] W_ENC_BASE       = `W_ENC_BASE;
    localparam [14:0] PT_ENC_IN        = `PT_ENC_IN;
    localparam [14:0] PT_GAP           = `PT_GAP;
    localparam [14:0] PT_LOGITS        = `PT_LOGITS;
    localparam [14:0] W_FC_BASE        = `W_FC_BASE;
    localparam [14:0] C_FC_BIAS_BASE   = `C_FC_BIAS_BASE;
    localparam [14:0] C_ENC_BIAS_BASE  = `C_ENC_BIAS_BASE;
    localparam [14:0] W_P1_BASE        = `W_P1_BASE;
    localparam [14:0] C_P1_BIAS_BASE   = `C_P1_BIAS_BASE;
    localparam [14:0] C_INC_SCALE_BASE = `C_INC_SCALE_BASE;
    localparam [14:0] C_INC_SHIFT_BASE = `C_INC_SHIFT_BASE;
    localparam [14:0] W_INPROJ_X_BASE  = `W_INPROJ_X_BASE;
    localparam [14:0] W_INPROJ_Z_BASE  = `W_INPROJ_Z_BASE;
    localparam [14:0] W_OUTPROJ_BASE   = `W_OUTPROJ_BASE;
    localparam [14:0] W_DW_BASE        = `W_DW_BASE;
    localparam [14:0] W_XPROJ_BASE     = `W_XPROJ_BASE;
    localparam [14:0] W_DTPROJ_BASE    = `W_DTPROJ_BASE;
    localparam [14:0] W_A_BASE         = `W_A_BASE;
    localparam [14:0] C_W_NORM_BASE    = `C_W_NORM_BASE;
    localparam [14:0] C_B_DW_BASE      = `C_B_DW_BASE;
    localparam [14:0] C_B_DT_BASE      = `C_B_DT_BASE;
    localparam [14:0] C_D_PARAM_BASE   = `C_D_PARAM_BASE;

    // ModelSim rejects a bit-select on the macro expression, so a sized copy
    localparam [3:0] LANE_LAST = `LANE_MAX;

    localparam [4:0] STG_M1A = `STG_M1A;
    localparam [4:0] STG_M1B = `STG_M1B;
    localparam [4:0] STG_M2  = `STG_M2;
    localparam [4:0] STG_M3  = `STG_M3;
    localparam [4:0] STG_M4  = `STG_M4;
    localparam [4:0] STG_M5  = `STG_M5;
    localparam [4:0] STG_M6  = `STG_M6;
    localparam [4:0] STG_M8  = `STG_M8;
    localparam [4:0] STG_RN  = `STG_RN;
    localparam [4:0] STG_P1  = `STG_P1;
    localparam [4:0] STG_MP3 = `STG_MP3;
    localparam [4:0] STG_B1  = `STG_B1;
    localparam [4:0] STG_BOT = `STG_BOT;
    localparam [4:0] STG_B2  = `STG_B2;
    localparam [4:0] STG_B3  = `STG_B3;
    localparam [4:0] STG_B4  = `STG_B4;
    localparam [4:0] STG_ENC = `STG_ENC;
    localparam [4:0] STG_GAP = `STG_GAP;
    localparam [4:0] STG_FC  = `STG_FC;

    localparam [6:0] S_IDLE      = 7'd0;
    localparam [6:0] S_H_INIT    = 7'd1;
    localparam [6:0] S_MAC1      = 7'd2;
    localparam [6:0] S_MAC2      = 7'd3;
    localparam [6:0] S_MAC4      = 7'd4;
    localparam [6:0] S_RN1       = 7'd6;
    localparam [6:0] S_RN2       = 7'd7;
    localparam [6:0] S_RN3       = 7'd8;
    localparam [6:0] S_RN4       = 7'd9;
    localparam [6:0] S_RN5       = 7'd10;
    localparam [6:0] S_RN6       = 7'd12;
    localparam [6:0] S_RN7       = 7'd13;
    localparam [6:0] S_RN8       = 7'd14;
    localparam [6:0] S_RN9       = 7'd15;
    localparam [6:0] S_RN10      = 7'd16;
    localparam [6:0] S_RN11      = 7'd17;
    localparam [6:0] S_RN12      = 7'd18;
    localparam [6:0] S_RN13      = 7'd19;
    localparam [6:0] S_M2_1      = 7'd21;
    localparam [6:0] S_M2_2      = 7'd22;
    localparam [6:0] S_M2_3      = 7'd23;
    localparam [6:0] S_M2_4      = 7'd24;
    localparam [6:0] S_M2_5      = 7'd27;
    localparam [6:0] S_M2_6      = 7'd28;
    localparam [6:0] S_M2_7      = 7'd29;
    localparam [6:0] S_M3_1      = 7'd31;
    localparam [6:0] S_M5_1      = 7'd36;
    localparam [6:0] S_M5_2      = 7'd37;
    localparam [6:0] S_M5_3      = 7'd38;
    localparam [6:0] S_M5_4      = 7'd39;
    localparam [6:0] S_M5_5      = 7'd40;
    localparam [6:0] S_M5_6      = 7'd41;
    localparam [6:0] S_M5_DT_R1  = 7'd42;
    localparam [6:0] S_M5_DT_R2  = 7'd43;
    localparam [6:0] S_M5_7      = 7'd44;
    localparam [6:0] S_M5_8      = 7'd45;
    localparam [6:0] S_M5_9      = 7'd46;
    localparam [6:0] S_M6_1      = 7'd58;
    localparam [6:0] S_M6_2      = 7'd59;
    localparam [6:0] S_M6_3      = 7'd60;
    localparam [6:0] S_M6_4      = 7'd74;
    localparam [6:0] S_M6_5      = 7'd75;
    localparam [6:0] S_M6_6      = 7'd76;
    localparam [6:0] S_M6_8      = 7'd99;
    localparam [6:0] S_M6_9      = 7'd82;
    localparam [6:0] S_M6_12     = 7'd87;
    localparam [6:0] S_M6_13     = 7'd88;
    localparam [6:0] S_M6_14     = 7'd89;
    localparam [6:0] S_M6_15     = 7'd101;
    localparam [6:0] S_M6_18     = 7'd92;
    localparam [6:0] S_M7_3      = 7'd50;
    localparam [6:0] S_M7_5      = 7'd53;
    localparam [6:0] S_M7_6      = 7'd54;
    localparam [6:0] S_M7_7      = 7'd55;
    localparam [6:0] S_MAC5      = 7'd98;
    localparam [6:0] S_MAC3      = 7'd102;
    localparam [6:0] S_MP_RUN    = 7'd110;
    localparam [6:0] S_FIN       = 7'd111;
    localparam [6:0] S_GAP_RUN   = 7'd112;
    localparam [6:0] S_BARRIER   = 7'd120;
    localparam [6:0] S_DONE      = 7'd127;

    (* MAX_FANOUT = 50 *) reg [6:0]   state;
    (* MAX_FANOUT = 50 *) reg [4:0]   cur_stage;
    reg [9:0]   t_cnt;
    reg [2:0]   wait_cnt;
    (* MAX_FANOUT = 30 *) reg [8:0]   ctr_g;
    (* MAX_FANOUT = 30 *) reg [12:0]  ctr_k;
    (* MAX_FANOUT = 30 *) reg [1:0]   tap_cnt;
    (* MAX_FANOUT = 30 *) reg [7:0]   ctr_r;
    (* MAX_FANOUT = 30 *) reg [8:0]   m3_wr_ptr;
    reg [2:0]   ctr_load;
    (* MAX_FANOUT = 30 *) reg [2:0]   ctr_s;
    (* MAX_FANOUT = 30 *) reg [3:0]   ctr_l;
    reg [15:0]  S_reg;
    reg [15:0]  h_init_cnt;

    reg [4:0]   next_stage_q;
    reg [6:0]   next_state_q;

    reg [`NUM_LANES*`DATA_W-1:0] m5_dt_word_reg;
    reg [`NUM_LANES*`DATA_W-1:0] silu_z_reg;
    reg [`NUM_LANES*`DATA_W-1:0] m6_ssm_grp_acc;
    reg [`NUM_LANES*`DATA_W-1:0] m6_delta_word, m6_u_word, m6_D_word;
    reg [`NUM_LANES*`DATA_W-1:0] m6_w0_reg, m6_w1_reg, m6_w2_reg;
    reg [`NUM_LANES*`DATA_W-1:0] m6_w3_reg;
    // Packed B/C of this timestep are loaded (first group only): M7 writes PT_Y_GATED, which aliases PT_X_PROJ, so later
    // groups cannot re-read them
    reg m6_bc_cached;
    reg signed [23:0]            m6_y_ch_reg;
    // D*u from the scan MUL2, captured in the one cycle it is on out_vec2
    reg signed [15:0]            m6_du_reg;
    reg                          m6_pref_c_valid;

    reg  [2:0]                        cl_op_mode;
    reg                               cl_clear_acc;
    reg  [`NUM_LANES*`DATA_W-1:0]     cl_in_W1_vec, cl_in_H_ext, cl_in_W2_vec, cl_in_X_vec;
    wire [`NUM_LANES*`DATA_W-1:0]     cl_out_vec, cl_out_vec2;
    wire [`NUM_LANES*`ACC_W-1:0]      cl_acc_raw_vec;

    reg                               m6_h_from_rf_r;
    reg  [H_ADDR_W-1:0]               m6_h_rd_addr_r;
    reg                               m6_h_wr_en_r;
    reg  [H_ADDR_W-1:0]               m6_h_wr_addr_r;
    reg                               m6_h_wr_from_pe_r;
    reg  [`NUM_LANES*`DATA_W-1:0]     m6_h_wr_data_ext_r;

    reg  [`NUM_LANES*`DATA_W-1:0]     m_rd_data;
    reg  [`NUM_LANES*`DATA_W-1:0]     w_rd_data_r;
    reg  [`NUM_LANES*`DATA_W-1:0]     w_rd_data2_r;
    reg  [14:0]                       w_grp_base_r;
    reg  [14:0]                       w_grp_base_next_r;
    wire [14:0]                       w_grp_base_next_comb;
    wire [14:0]                       barrier_mac_w_base;
    reg  [14:0]                       rn_t_offset_x_r;
    wire [14:0]                       rn_t_offset_x_comb;
    wire [14:0]                       rn_t_offset_x_comb_next;
    // defined with the MAC addresses below
    wire                              mac_last_iter_fwd;

    // MAC2 operands: the two consecutive input elements ctr_k, ctr_k+1 of the current input word, each broadcast to all
    // lanes. keep stops Vivado from sharing the select LUTs between the two.
    wire [3:0]   y_lane_sel  = ctr_k[3:0];
    wire [15:0]  y_scalar    = m_rd_data[y_lane_sel*`DATA_W +: `DATA_W];
    (* keep = "true" *) wire [`NUM_LANES*`DATA_W-1:0] y_broadcast = {`NUM_LANES{y_scalar}};
    wire [3:0]   y_lane_sel2 = ctr_k[3:0] + 4'd1;
    wire [15:0]  y_scalar2   = m_rd_data[y_lane_sel2*`DATA_W +: `DATA_W];
    (* keep = "true" *) wire [`NUM_LANES*`DATA_W-1:0] y_broadcast2 = {`NUM_LANES{y_scalar2}};

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            m_rd_data         <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_data_r       <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_data2_r      <= {`NUM_LANES*`DATA_W{1'b0}};
            w_grp_base_r      <= 15'd0;
            w_grp_base_next_r <= 15'd0;
            rn_t_offset_x_r   <= 15'd0;
        end else begin
            m_rd_data         <= m_rd_data_c;
            w_rd_data_r       <= w_rd_data;
            w_rd_data2_r      <= w_rd_data2;
            w_grp_base_next_r <= w_grp_base_next_comb;
            rn_t_offset_x_r   <= rn_t_offset_x_comb;
            if (state == S_BARRIER && peer_done_stage) begin
                w_grp_base_r <= barrier_mac_w_base;
                // M8 -> RN advances t: preload (t+1)*CH_OUT
                if (cur_stage == STG_M8 && next_stage_q == STG_RN) begin
                    rn_t_offset_x_r <= rn_t_offset_x_comb_next;
                end
            end
            // group loopback: the next group's base must be in place for the S_MAC2 read issued next cycle
            if (state == S_MAC5 && wait_cnt == 3'd4 && !mac_last_iter_fwd) begin
                w_grp_base_r <= w_grp_base_next_r;
            end
        end
    end

    reg  [12:0]  rsqrt_idx_r;
    wire [15:0]  rsqrt_data;
    wire [`NUM_LANES*`DATA_W-1:0] silu_out_w, sp_out_w, exp_out_w;
    wire [`NUM_LANES*`DATA_W-1:0] silu_in_drv, sp_in_drv, exp_in_drv;

    assign silu_in_drv = (state == S_M3_1) ? m_rd_data :
                         (state == S_M7_3) ? m_rd_data :
                                              {`NUM_LANES*`DATA_W{1'b0}};
    assign sp_in_drv   = (state == S_M5_9) ? cl_out_vec : {`NUM_LANES*`DATA_W{1'b0}};
    assign exp_in_drv  = (state == S_M6_6) ? cl_out_vec : {`NUM_LANES*`DATA_W{1'b0}};

    LUT_Bank u_lut (
        .silu_in_flat  (silu_in_drv),
        .sp_in_flat    (sp_in_drv),
        .exp_in_flat   (exp_in_drv),
        .silu_out_flat (silu_out_w),
        .sp_out_flat   (sp_out_w),
        .exp_out_flat  (exp_out_w),
        .rsqrt_idx     (rsqrt_idx_r),
        .rsqrt_data    (rsqrt_data)
    );

    // The state RAM is driven only by S_H_INIT and the scan
    wire h_ctrl_active = (cur_stage == STG_M6) || (state == S_H_INIT);
    wire                              cl_h_from_rf_w     = h_ctrl_active ? m6_h_from_rf_r     : 1'b0;
    wire [H_ADDR_W-1:0]               cl_h_rd_addr_w     = h_ctrl_active ? m6_h_rd_addr_r     : {H_ADDR_W{1'b0}};
    wire                              cl_h_wr_en_w       = h_ctrl_active ? m6_h_wr_en_r       : 1'b0;
    wire [H_ADDR_W-1:0]               cl_h_wr_addr_w     = h_ctrl_active ? m6_h_wr_addr_r     : {H_ADDR_W{1'b0}};
    wire                              cl_h_wr_from_pe_w  = h_ctrl_active ? m6_h_wr_from_pe_r  : 1'b0;
    wire [`NUM_LANES*`DATA_W-1:0]     cl_h_wr_data_ext_w = h_ctrl_active ? m6_h_wr_data_ext_r : {`NUM_LANES*`DATA_W{1'b0}};

    M_Cluster #(.H_ADDR_W(H_ADDR_W), .H_DEPTH(H_DEPTH)) u_mc (
        .clk          (clk),
        .rst          (rst),
        .op_mode      (cl_op_mode),
        .clear_acc    (cl_clear_acc),
        .in_W1_vec    (cl_in_W1_vec),
        .in_H_ext     (cl_in_H_ext),
        .in_W2_vec    (cl_in_W2_vec),
        .in_X_vec     (cl_in_X_vec),
        .h_from_rf    (cl_h_from_rf_w),
        .h_rd_addr    (cl_h_rd_addr_w),
        .h_wr_en      (cl_h_wr_en_w),
        .h_wr_addr    (cl_h_wr_addr_w),
        .h_wr_from_pe (cl_h_wr_from_pe_w),
        .h_wr_data_ext(cl_h_wr_data_ext_w),
        .out_vec      (cl_out_vec),
        .acc_raw_vec  (cl_acc_raw_vec),
        .out_vec2     (cl_out_vec2)
    );

    // Lane sum of the PE accumulators, 2 cycles after acc_raw_vec
    wire signed [`ACC_W+4:0] sum_d_wide_c;
    reg  signed [`ACC_W+4:0] sum_d_wide;
    Reduce16Wide u_rw (
        .clk    (clk),
        .rst    (rst),
        .in_vec (cl_acc_raw_vec),
        .out_sum(sum_d_wide_c)
    );
    always @(posedge clk or posedge rst) begin
        if (rst) sum_d_wide <= {(`ACC_W+5){1'b0}};
        else     sum_d_wide <= sum_d_wide_c;
    end

    wire signed [`ACC_W+4:0] mean_i_signed = sum_d_wide >>> rn_total_shift;
    wire [12:0] mean_i_clip =
        (mean_i_signed < 0)         ? 13'd0    :
        (mean_i_signed > 45'sd8191) ? 13'd8191 :
                                       mean_i_signed[12:0];
    wire [`NUM_LANES*`DATA_W-1:0] S_broadcast = {`NUM_LANES{S_reg}};

    // MAC job table. Every MAC-shaped stage runs the shared S_MAC1..5 chain: 16 outputs per group, each the dot product
    // of a weight row with len input elements. A stage is one row:
    //   len, grp_count   input length, output groups
    //   in_base          ram_main word holding input elements 0..15
    //   wr_base          ram_main word of output group 0 (+ ctr_g)
    //   next stage/state after the last group (through S_BARRIER)
    //   bias             optional per-group bias word in ram_const, added with saturation at writeback
    // The weight base is kept separately (mac_w_base_of) so S_BARRIER can preload the next stage's.
    //
    // Time-split (tsplit) stages run over the whole sequence: each cluster takes t = CLUSTER_ID, +2, ... and computes
    // every output group of its t. Input and output advance per t by in_wpt / out_wpt words; tin_r / tout_r hold the
    // current t's bases (set at barrier release, stepped with t), so there is no t multiply on the address path.
    function [14:0] mac_w_base_of;
        input [4:0] stg;
        case (stg)
            STG_M1A: mac_w_base_of = W_INPROJ_X_BASE;
            STG_M1B: mac_w_base_of = W_INPROJ_Z_BASE;
            STG_M4:  mac_w_base_of = W_XPROJ_BASE;
            STG_P1:  mac_w_base_of = W_P1_BASE;
            STG_B1:  mac_w_base_of = W_B1_BASE;
            STG_BOT: mac_w_base_of = W_BOT_BASE;
            STG_B2:  mac_w_base_of = W_B2_BASE;
            STG_B3:  mac_w_base_of = W_B3_BASE;
            STG_B4:  mac_w_base_of = W_B4_BASE;
            STG_ENC: mac_w_base_of = W_ENC_BASE;
            STG_FC:  mac_w_base_of = W_FC_BASE;
            default: mac_w_base_of = W_OUTPROJ_BASE;
        endcase
    endfunction

    function mac_tsplit_of;
        input [4:0] stg;
        mac_tsplit_of = (stg == STG_P1) || (stg == STG_MP3) || (stg == STG_B1) || (stg == STG_BOT) ||
                        (stg == STG_B2) || (stg == STG_B3) || (stg == STG_B4) || (stg == STG_ENC);
    endfunction

    function [14:0] tsplit_in_base_of;
        input [4:0] stg;
        case (stg)
            STG_MP3: tsplit_in_base_of = PT_INPUT;
            STG_B1:  tsplit_in_base_of = PT_MP3;
            STG_BOT: tsplit_in_base_of = PT_INPUT;
            STG_B2, STG_B3, STG_B4: tsplit_in_base_of = PT_BOT;
            STG_ENC: tsplit_in_base_of = PT_ENC_IN;
            default: tsplit_in_base_of = PT_X_IN;
        endcase
    endfunction
    function [14:0] tsplit_out_base_of;
        input [4:0] stg; input [8:0] ch_out;
        case (stg)
            STG_MP3: tsplit_out_base_of = PT_MP3;
            STG_B1:  tsplit_out_base_of = PT_INC_OUT;
            STG_BOT: tsplit_out_base_of = PT_BOT;
            STG_ENC: tsplit_out_base_of = PT_X_IN;
            STG_B2:  tsplit_out_base_of = PT_INC_OUT + {8'b0, ch_out[8:2]};
            STG_B3:  tsplit_out_base_of = PT_INC_OUT + {7'b0, ch_out[8:2], 1'b0};
            STG_B4:  tsplit_out_base_of = PT_INC_OUT + {7'b0, ch_out[8:2], 1'b0} + {8'b0, ch_out[8:2]};
            default: tsplit_out_base_of = PT_INPUT;
        endcase
    endfunction
    function [8:0] tsplit_in_wpt_of;
        input [4:0] stg; input [8:0] ch_in; input [8:0] ch_out;
        tsplit_in_wpt_of = (stg == STG_ENC) ? 9'd1 :
                           (stg == STG_P1) ? ch_in :
                           (stg == STG_B2 || stg == STG_B3 || stg == STG_B4) ? {2'b0, ch_out[8:2]} : ch_out;
    endfunction
    function [8:0] tsplit_out_wpt_of;
        input [4:0] stg; input [8:0] ch_in; input [8:0] ch_out;
        tsplit_out_wpt_of = (stg == STG_BOT) ? {2'b0, ch_out[8:2]} :
                            (stg == STG_ENC) ? ch_in : ch_out;
    endfunction

    // job_r follows job_of(cur_stage) one cycle late and is preloaded with job_of(next_stage_q) at the barrier release, so
    // the first state of a stage already sees its row; this keeps the stage decode (and the conv K*dim multiply) off the
    // FSM paths
    reg  [14:0] mac_w_base;
    always @(posedge clk or posedge rst)
        if (rst) mac_w_base <= 15'd0;
        else     mac_w_base <= (state == S_BARRIER && peer_done_stage) ? mac_w_base_of(next_stage_q)
                                                                   : mac_w_base_of(cur_stage);
    localparam JOB_W = 117;
    function [JOB_W-1:0] job_of;
        input [4:0] stg;
        reg [13:0] len; reg [9:0] grp; reg [14:0] inb, wrb; reg [4:0] nstg; reg [6:0] nst;
        reg ben; reg [14:0] bbase; reg conv; reg [14:0] pad; reg tsp; reg [8:0] iwpt, owpt;
        begin
            ben = 1'b0; bbase = 15'd0; conv = 1'b0; pad = 15'd0;
            tsp  = mac_tsplit_of(stg);
            iwpt = tsplit_in_wpt_of (stg, CH_IN, CH_OUT);
            owpt = tsplit_out_wpt_of(stg, CH_IN, CH_OUT);
            case (stg)
                STG_ENC: begin
                    len   = 14'd16;
                    grp   = {1'b0, CH_IN};
                    inb   = PT_ENC_IN;
                    wrb   = PT_X_IN;
                    ben   = 1'b1;
                    bbase = C_ENC_BIAS_BASE;
                    nstg  = STG_P1;
                    nst   = S_MAC1;
                end
                STG_P1: begin
                    len   = {1'b0, CH_IN, 4'b0};
                    grp   = {1'b0, CH_OUT};
                    inb   = PT_X_IN;
                    wrb   = PT_INPUT;
                    ben   = 1'b1;
                    bbase = C_P1_BIAS_BASE;
                    nstg  = STG_MP3;
                    nst   = S_MP_RUN;
                end
                STG_MP3: begin
                    len   = {1'b0, CH_OUT, 4'b0};
                    grp   = {1'b0, CH_OUT};
                    inb   = PT_INPUT;
                    wrb   = PT_MP3;
                    nstg  = STG_B1;
                    nst   = S_MAC1;
                end
                STG_B1: begin
                    len   = {1'b0, CH_OUT, 4'b0};
                    grp   = {3'b0, CH_OUT[8:2]};
                    inb   = PT_MP3;
                    wrb   = PT_INC_OUT;
                    nstg  = STG_BOT;
                    nst   = S_MAC1;
                end
                STG_BOT: begin
                    len   = {1'b0, CH_OUT, 4'b0};
                    grp   = {3'b0, CH_OUT[8:2]};
                    inb   = PT_INPUT;
                    wrb   = PT_BOT;
                    nstg  = STG_B2;
                    nst   = S_MAC1;
                end
                STG_B2: begin
                    len   = {CH_OUT, 2'b0} * 14'd9;
                    grp   = {3'b0, CH_OUT[8:2]};
                    inb   = PT_BOT;
                    wrb   = PT_INC_OUT;
                    pad   = {CH_OUT, 2'b0} * 15'd4  / 15'd16;
                    conv  = 1'b1;
                    nstg  = STG_B3;
                    nst   = S_MAC1;
                end
                STG_B3: begin
                    len   = {CH_OUT, 2'b0} * 14'd19;
                    grp   = {3'b0, CH_OUT[8:2]};
                    inb   = PT_BOT;
                    wrb   = PT_INC_OUT;
                    pad   = {CH_OUT, 2'b0} * 15'd9  / 15'd16;
                    conv  = 1'b1;
                    nstg  = STG_B4;
                    nst   = S_MAC1;
                end
                STG_B4: begin
                    len   = {CH_OUT, 2'b0} * 14'd39;
                    grp   = {3'b0, CH_OUT[8:2]};
                    inb   = PT_BOT;
                    wrb   = PT_INC_OUT;
                    pad   = {CH_OUT, 2'b0} * 15'd19 / 15'd16;
                    conv  = 1'b1;
                    nstg  = stg;
                    nst   = S_DONE;
                end
                STG_M1A: begin
                    len   = {1'b0, d_model};
                    grp   = ch_m_act;
                    inb   = PT_X_NORM;
                    wrb   = PT_X_INNER_CIRC + ({13'b0, t_cnt[1:0]} * {5'b0, ch_m_act});
                    nstg  = STG_M1B;
                    nst   = S_MAC1;
                end
                STG_M1B: begin
                    len   = {1'b0, d_model};
                    grp   = ch_m_act;
                    inb   = PT_X_NORM;
                    wrb   = PT_Z_GATE;
                    nstg  = STG_M2;
                    nst   = S_M2_1;
                end
                STG_M4: begin
                    len   = d_inner;
                    grp   = {5'b0, XP_OUT_GRP};
                    inb   = PT_U;
                    wrb   = PT_X_PROJ;
                    nstg  = USE_M5 ? STG_M5 : STG_M6;
                    nst   = USE_M5 ? S_M5_1 : S_M6_1;
                end
                STG_M8: begin
                    len   = d_inner;
                    grp   = {1'b0, CH_OUT};
                    inb   = PT_Y_GATED;
                    wrb   = PT_MAMBA_OUT + rn_t_offset_x_r;
                    nstg  = (t_cnt != t_last) ? STG_RN : GAP_EN ? STG_GAP   : stg;
                    nst   = (t_cnt != t_last) ? S_RN1  : GAP_EN ? S_GAP_RUN : S_DONE;
                end
                STG_FC: begin
                    len   = {1'b0, CH_OUT, 4'b0};
                    grp   = 10'd1;
                    inb   = PT_GAP;
                    wrb   = PT_LOGITS;
                    ben   = 1'b1;
                    bbase = C_FC_BIAS_BASE;
                    nstg  = stg;
                    nst   = S_DONE;
                end
                default: begin
                    len   = {1'b0, d_model};
                    grp   = ch_m_act;
                    inb   = PT_X_NORM;
                    wrb   = PT_MAMBA_OUT + rn_t_offset_x_r;
                    nstg  = stg;
                    nst   = S_IDLE;
                end
            endcase
            job_of = {len, grp, inb, wrb, nstg, nst, ben, bbase, conv, pad, tsp, iwpt, owpt};
        end
    endfunction
    reg  [JOB_W-1:0] job_r;
    always @(posedge clk or posedge rst)
        if (rst) job_r <= {JOB_W{1'b0}};
        else     job_r <= (state == S_BARRIER && peer_done_stage) ? job_of(next_stage_q)
                                                               : job_of(cur_stage);
    wire [13:0] mac_len        = job_r[116:103];
    wire [9:0]  mac_grp_count  = job_r[102:93];
    wire [14:0] mac_in_base    = job_r[92:78];
    wire [14:0] mac_wr_base    = job_r[77:63];
    wire [4:0]  mac_next_stage = job_r[62:58];
    wire [6:0]  mac_next_state = job_r[57:51];
    wire        mac_bias_en    = job_r[50];
    wire [14:0] mac_bias_base  = job_r[49:35];
    wire        mac_conv       = job_r[34];
    wire [14:0] mac_pad_words  = job_r[33:19];
    wire        mac_tsplit     = job_r[18];
    wire [8:0]  mac_in_wpt     = job_r[17:9];
    wire [8:0]  mac_out_wpt    = job_r[8:0];
    reg  [14:0] tin_r, tout_r;
    // maxpool3: max(x[t-1], x[t], x[t+1]), edges clamped (= -inf padding)
    reg  [2:0]  mp_ph;
    reg  [`NUM_LANES*`DATA_W-1:0] mp_acc;
    wire [14:0] mp_cur_addr  = tin_r + {6'b0, ctr_g};
    wire [14:0] mp_prev_addr = (t_cnt == 10'd0)            ? mp_cur_addr : mp_cur_addr - {6'b0, CH_OUT};
    wire [14:0] mp_next_addr = ({1'b0, t_cnt} + 11'd1 >= {1'b0, T_MAX}) ? mp_cur_addr : mp_cur_addr + {6'b0, CH_OUT};
    function [`NUM_LANES*`DATA_W-1:0] max_vec;
        input [`NUM_LANES*`DATA_W-1:0] a, b;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            max_vec[i*`DATA_W +: `DATA_W] =
                ($signed(a[i*`DATA_W +: `DATA_W]) > $signed(b[i*`DATA_W +: `DATA_W]))
                ? a[i*`DATA_W +: `DATA_W] : b[i*`DATA_W +: `DATA_W];
    endfunction
    function [`NUM_LANES*`DATA_W-1:0] relu_vec;
        input [`NUM_LANES*`DATA_W-1:0] a;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            relu_vec[i*`DATA_W +: `DATA_W] = a[(i+1)*`DATA_W-1] ? {`DATA_W{1'b0}} : a[i*`DATA_W +: `DATA_W];
    endfunction

    // FIN (ITM block output), fused into M8's writeback:
    //   final = sat_add(relu(sat_add(sat16(inc*scale >> 11), shift)), relu(m8)), written over INC_OUT[t][g]
    wire        fin_mode = FIN_EN && (cur_stage == STG_M8);
    wire [14:0] fin_addr = PT_INC_OUT + rn_t_offset_x_r + {6'b0, ctr_g};
    // MaxPool2: even t stores final[t] in slot t; odd t reads final[t-1] back and stores the max in slot t/2. Slot t/2
    // <= t-1 is already consumed and later inc[t'] slots are never touched, so the compaction is safe in place.
    wire        fin_pool_odd = POOL_OUT && t_cnt[0];
    wire [14:0] fin_prev_t   = rn_t_offset_x_r - {6'b0, CH_OUT};
    wire [14:0] fin_prev_addr = PT_INC_OUT + fin_prev_t + {6'b0, ctr_g};
    wire [14:0] fin_pool_addr = PT_INC_OUT + {1'b0, fin_prev_t[14:1]} + {6'b0, ctr_g};
    reg  [3:0]  fin_ph;
    reg         fin_ready;
    reg  [`NUM_LANES*`DATA_W-1:0] fin_m8, fin_inc, fin_scale, fin_shift, fin_res, fin_prev, fin_a;

    // GAP: each cluster sums its own groups' FIN outputs over T in 25-bit lanes (T*32767 < 2^24 for T <= 512), indexed
    // by ctr_g/2. S_GAP_RUN then emits sat16((sum * GAP_INV) >>> 15) one lane per cycle; lane select and product are
    // registered to keep the 25x17 multiply off the clock period.
    localparam GAP_W = 25;
    reg  [`NUM_LANES*GAP_W-1:0] gap_acc [0:3];
    reg  [4:0]  gap_lane;
    reg  signed [GAP_W+17-1:0] gap_prod;
    reg  [`NUM_LANES*`DATA_W-1:0] gap_word;
    wire [GAP_W-1:0] gap_lane_sum = gap_acc[ctr_g[2:1]][gap_lane[3:0]*GAP_W +: GAP_W];
    reg  [GAP_W-1:0] gap_sum_r;
    wire signed [GAP_W+17-1:0] gap_shifted = gap_prod >>> 15;
    wire [`DATA_W-1:0] gap_sat16 =
        (gap_shifted > 32767)  ? 16'h7FFF :
        (gap_shifted < -32768) ? 16'h8000 : gap_shifted[`DATA_W-1:0];
    function [`NUM_LANES*GAP_W-1:0] gap_add;
        input [`NUM_LANES*GAP_W-1:0] acc;
        input [`NUM_LANES*`DATA_W-1:0] x;
        integer i;
        for (i = 0; i < `NUM_LANES; i = i + 1)
            gap_add[i*GAP_W +: GAP_W] = acc[i*GAP_W +: GAP_W] +
                {{(GAP_W-`DATA_W){x[(i+1)*`DATA_W-1]}}, x[i*`DATA_W +: `DATA_W]};
    endfunction
    // FIN epilogue in two cycles: fin_a = relu(sat(inc*scale>>11 + shift)), then fin_final = fin_a + relu(m8) (and
    // the pooling max)
    wire [`NUM_LANES*`DATA_W-1:0] fin_a_next = relu_vec(sat_add_vec(cl_out_vec, fin_shift));
    wire [`NUM_LANES*`DATA_W-1:0] fin_final  = sat_add_vec(fin_a, relu_vec(fin_m8));
    wire [14:0] tin_next  = tin_r  + {5'b0, mac_in_wpt,  1'b0};
    wire [14:0] tout_next = tout_r + {5'b0, mac_out_wpt, 1'b0};
    // conv: element k reads word tin_r - pad + k/16; words outside [in_base, in_base + T*wpt) are the zero padding
    wire [14:0] mac_in_base_eff = mac_tsplit ? tin_r - mac_pad_words : mac_in_base;
    wire [14:0] conv_region_end = mac_in_base + ((mac_in_wpt == 9'd2) ? {4'b0, T_MAX, 1'b0} : {5'b0, T_MAX});
    wire [14:0] mac_wr_addr = (mac_tsplit ? tout_r : mac_wr_base) + {6'b0, ctr_g};

    // tsplit position: last group of this t, last t of this cluster
    wire        mac_g_last_in_t = ({1'b0, ctr_g} + 10'd1 >= mac_grp_count);
    wire        mac_t_last      = ({1'b0, t_cnt} + 11'd2 >= {1'b0, T_MAX});

    function [`NUM_LANES*`DATA_W-1:0] sat_add_vec;
        input [`NUM_LANES*`DATA_W-1:0] a, b;
        integer i;
        reg signed [`DATA_W:0] s;
        begin
            for (i = 0; i < `NUM_LANES; i = i + 1) begin
                s = $signed(a[i*`DATA_W +: `DATA_W]) + $signed(b[i*`DATA_W +: `DATA_W]);
                sat_add_vec[i*`DATA_W +: `DATA_W] =
                    (s > 17'sd32767)  ? 16'h7FFF :
                    (s < -17'sd32768) ? 16'h8000 : s[`DATA_W-1:0];
            end
        end
    endfunction
    wire [14:0] mac_len_ext  = {1'b0, mac_len};

    // Weight RAMs hold only this cluster's groups, compacted: group g is at index g/2 (tsplit stages hold all groups and index by
    // ctr_g)
    wire [8:0]  g_bank_idx    = {1'b0, ctr_g[8:1]};

    // w_grp_base_r is event-driven: loaded at the barrier release and at the S_MAC5 group loopback, held otherwise
    // (ctr_g is stable inside the MAC loop). w_grp_base_next_r is first used >= 16 cycles after any ctr_g or stage
    // change, so its 2-cycle lag is safe.
    wire [12:0] ctr_k_p2 = ctr_k + 13'd2;
    wire [12:0] ctr_k_p4 = ctr_k + 13'd4;
    wire [12:0] ctr_k_p6 = ctr_k + 13'd6;

    wire [14:0] mac_rd_addr_k0 = mac_in_base_eff + {6'b0, ctr_k[12:4]};
    // Conv zero padding: the word in use (index kw = ctr_k >> 4) is padding when kw < in_base - base_eff or
    // kw >= region_end - base_eff. Both limits move only with tin_r or the stage, and the first MAC fires >= 2 cycles
    // later, so they are registered and the per-cycle test is one 9-bit compare.
    wire signed [16:0] conv_base_s = mac_tsplit
        ? $signed({2'b0, tin_r}) - $signed({2'b0, mac_pad_words})
        : $signed({2'b0, mac_in_base});
    reg  signed [16:0] conv_lo_r, conv_hi_r;
    always @(posedge clk or posedge rst)
        if (rst) begin
            conv_lo_r <= 17'sd0;
            conv_hi_r <= 17'sd0;
        end else begin
            conv_lo_r <= $signed({2'b0, mac_in_base})     - conv_base_s;
            conv_hi_r <= $signed({2'b0, conv_region_end}) - conv_base_s;
        end
    wire signed [16:0] conv_kw  = $signed({8'b0, ctr_k[12:4]});
    wire        conv_zero      = mac_conv && ((conv_kw < conv_lo_r) ||
                                              (conv_kw >= conv_hi_r));
    wire [`NUM_LANES*`DATA_W-1:0] mac_y1 = conv_zero ? {`NUM_LANES*`DATA_W{1'b0}} : y_broadcast;
    wire [`NUM_LANES*`DATA_W-1:0] mac_y2 = conv_zero ? {`NUM_LANES*`DATA_W{1'b0}} : y_broadcast2;
    wire [14:0] mac_rd_addr_k2 = mac_in_base_eff + {6'b0, ctr_k_p2[12:4]};
    // The loop consumes two input elements per cycle and RAM data arrives 3 cycles after the address, so the input
    // address runs 6 elements ahead
    wire [14:0] mac_rd_addr_k6 = mac_in_base_eff + {6'b0, ctr_k_p6[12:4]};
    wire [14:0] mac_w_addr_k0  = w_grp_base_r + {2'b0, ctr_k};
    wire [14:0] mac_w_addr_k2  = w_grp_base_r + {2'b0, ctr_k_p2};
    wire [14:0] mac_w_addr_k4  = w_grp_base_r + {2'b0, ctr_k_p4};
    wire [14:0] mac_w_addr_k6  = w_grp_base_r + {2'b0, ctr_k_p6};

    wire [8:0]  ctr_g_p2       = ctr_g + 9'd2;
    wire [8:0]  g_bank_idx_next = {1'b0, ctr_g_p2[8:1]};
    wire [8:0]  w_grp_idx_next  = mac_tsplit ? (mac_g_last_in_t ? 9'd0 : ctr_g + 9'd1)
                                             : g_bank_idx_next;
    reg  [8:0]  w_grp_idx_next_r;
    always @(posedge clk or posedge rst)
        if (rst) w_grp_idx_next_r <= 9'd0;
        else     w_grp_idx_next_r <= w_grp_idx_next;
    assign      w_grp_base_next_comb = mac_w_base + ({6'b0, w_grp_idx_next_r} * mac_len_ext);
    wire [14:0] mac_w_addr_gnext = w_grp_base_next_r;

    // group index at a barrier release is 0 for every stage
    assign      barrier_mac_w_base = mac_w_base_of(next_stage_q);
    wire [14:0] mac_rd_addr_g0   = (mac_tsplit && mac_g_last_in_t) ? tin_next - mac_pad_words : mac_in_base_eff;

    // last iteration: the next group of this cluster would be past the last group
    wire mac_last_iter = mac_tsplit ? (mac_g_last_in_t && mac_t_last)
                                    : (ctr_g_p2 >= {1'b0, mac_grp_count[8:0]});
    assign mac_last_iter_fwd = mac_last_iter;

    // RN: both clusters sum the squares of the full input word (it is mirrored, so they get the same mean); the
    // scaling pass splits groups by cluster
    wire [12:0] ctr_k_p1 = ctr_k + 13'd1;
    wire [9:0]  rn_x_per_t     = CH_OUT;
    wire [9:0]  rn_x_grp_last  = rn_x_per_t - 10'd1;
    wire [14:0] rn_x_per_t_ext = {5'b0, rn_x_per_t};
    // t_cnt * CH_OUT, registered (rn_t_offset_x_r); the barrier that advances t loads (t+1) * CH_OUT
    assign      rn_t_offset_x_comb      = t_cnt           * rn_x_per_t_ext;
    assign      rn_t_offset_x_comb_next = (t_cnt + 10'd1) * rn_x_per_t_ext;
    wire [14:0] rn_t_offset_x = rn_t_offset_x_r;
    wire [14:0] rn_sq_rd_addr_k0 = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k[6:0]};
    wire [14:0] rn_sq_rd_addr_k1 = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k_p1[6:0]};
    wire [14:0] rn_sq_rd_addr_k2 = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k_p2[6:0]};
    wire [12:0] ctr_k_p3         = ctr_k + 13'd3;
    wire [14:0] rn_sq_rd_addr_k3 = PT_INPUT + rn_t_offset_x + {8'b0, ctr_k_p3[6:0]};
    wire [10:0] rn_sq_last       = {1'b0, rn_x_grp_last};
    wire [14:0] rn_ap_rd_addr_g0 = PT_INPUT + rn_t_offset_x + {6'b0, ctr_g};
    wire [14:0] rn_ap_rd_addr_g2 = PT_INPUT + rn_t_offset_x + {6'b0, ctr_g_p2};
    wire [14:0] rn_ap_c_addr_g0  = C_W_NORM_BASE + {6'b0, ctr_g};
    wire [14:0] rn_ap_c_addr_g2  = C_W_NORM_BASE + {6'b0, ctr_g_p2};
    wire [14:0] rn_wr_addr       = PT_X_NORM + {6'b0, ctr_g};
    wire        rn_last_iter     = (ctr_g_p2 >= {1'b0, rn_x_per_t[8:0]});
    // M2: causal conv over the last 4 x_inner, kept in a 4-slot circular buffer (slot = t mod 4); taps before t = 0
    // are zero
    wire signed [10:0] eff_t_for_tap     = {1'b0, t_cnt} - 11'd3 + {9'b0, tap_cnt};
    wire               m2_pad            = (eff_t_for_tap < 0);
    wire [1:0]         eff_t_slot        = eff_t_for_tap[1:0];
    wire [2:0]         tap_p1            = {1'b0, tap_cnt} + 3'd1;
    wire [2:0]         tap_p2            = {1'b0, tap_cnt} + 3'd2;
    wire [2:0]         tap_p3            = {1'b0, tap_cnt} + 3'd3;
    wire signed [11:0] eff_t_p1          = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p1});
    wire signed [11:0] eff_t_p2          = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p2});
    wire signed [11:0] eff_t_p3          = $signed({2'b0, t_cnt}) - 12'sd3 + $signed({9'b0, tap_p3});
    wire [1:0]         eff_t_slot_p1     = eff_t_p1[1:0];
    wire [1:0]         eff_t_slot_p2     = eff_t_p2[1:0];
    wire [1:0]         eff_t_slot_p3     = eff_t_p3[1:0];
    wire [14:0] m2_rd_addr_tap0 = PT_X_INNER_CIRC
                                  + ({13'b0, eff_t_slot}    * {5'b0, ch_m_act})
                                  + {6'b0, ctr_g};
    wire [14:0] m2_rd_addr_tap1 = PT_X_INNER_CIRC
                                  + ({13'b0, eff_t_slot_p1} * {5'b0, ch_m_act})
                                  + {6'b0, ctr_g};
    wire [14:0] m2_rd_addr_tap2 = PT_X_INNER_CIRC
                                  + ({13'b0, eff_t_slot_p2} * {5'b0, ch_m_act})
                                  + {6'b0, ctr_g};
    wire [14:0] m2_rd_addr_tap3 = PT_X_INNER_CIRC
                                  + ({13'b0, eff_t_slot_p3} * {5'b0, ch_m_act})
                                  + {6'b0, ctr_g};
    wire [14:0] m2_w_addr_tap0 = W_DW_BASE + ({5'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {13'b0, tap_cnt};
    wire [14:0] m2_w_addr_tap1 = W_DW_BASE + ({5'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {12'b0, tap_p1};
    wire [14:0] m2_w_addr_tap2 = W_DW_BASE + ({5'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {12'b0, tap_p2};
    wire [14:0] m2_w_addr_tap3 = W_DW_BASE + ({5'b0, ctr_g} * `M2_WEIGHT_STRIDE) + {12'b0, tap_p3};
    wire [14:0] m2_c_addr      = C_B_DW_BASE + {6'b0, ctr_g};
    wire [14:0] m2_wr_addr     = PT_X_CONV   + {6'b0, ctr_g};
    wire signed [10:0] eff_t_tap0      = {1'b0, t_cnt} - 11'd3;
    wire [1:0]         eff_t_slot_tap0 = eff_t_tap0[1:0];
    wire [14:0] m2_rd_addr_g2 = PT_X_INNER_CIRC
                                + ({13'b0, eff_t_slot_tap0} * {5'b0, ch_m_act})
                                + {6'b0, ctr_g_p2};
    wire [14:0] m2_w_addr_g2  = W_DW_BASE   + ({5'b0, ctr_g_p2} * `M2_WEIGHT_STRIDE);
    wire [14:0] m2_c_addr_g2  = C_B_DW_BASE + {6'b0, ctr_g_p2};
    wire        m2_last_iter  = (ctr_g_p2 >= {1'b0, ch_m_act[8:0]});

    // M3: streaming SiLU, write pointer 3 cycles behind the read pointer
    wire [14:0] m3_rd_addr    = PT_X_CONV + {6'b0, ctr_g};
    wire [14:0] m3_wr_addr    = PT_U      + {6'b0, m3_wr_ptr};
    wire        m3_last_write = ((m3_wr_ptr + 9'd2) >= {1'b0, ch_m_act[8:0]});

    // M5: delta[g] = softplus(W_dt[g] . dt + b_dt), one rank per cycle; dt is broadcast lane by lane from the word in
    // m5_dt_word_reg, reloaded every 16 ranks
    wire [7:0]  dt_rank_last = DT_RANK - 8'd1;
    wire [7:0]  ctr_r_p1     = ctr_r + 8'd1;
    wire [7:0]  ctr_r_p2     = ctr_r + 8'd2;
    wire [7:0]  ctr_r_p3     = ctr_r + 8'd3;
    wire [14:0] m5_dt_addr   = PT_X_PROJ + 15'd0;
    wire [15:0] m5_dt_scalar = m5_dt_word_reg[ctr_l*`DATA_W +: `DATA_W];
    wire [`NUM_LANES*`DATA_W-1:0] m5_dt_broadcast = {`NUM_LANES{m5_dt_scalar}};
    wire        m5_last_iter = (ctr_g_p2 >= {1'b0, ch_m_act[8:0]});
    // Weight bases of group ctr_g and ctr_g+2, registered (the ctr_g * DT_RANK product limited the clock); at the
    // S_M5_9 group step m5_base_r takes m5_base_g2_r in the same edge as ctr_g advances
    wire [14:0] m5_base_c    = W_DTPROJ_BASE + ({6'b0, ctr_g}    * {7'b0, DT_RANK});
    wire [14:0] m5_base_g2_c = W_DTPROJ_BASE + ({6'b0, ctr_g_p2} * {7'b0, DT_RANK});
    reg  [14:0] m5_base_r, m5_base_g2_r;
    always @(posedge clk or posedge rst)
        if (rst) begin
            m5_base_r    <= 15'd0;
            m5_base_g2_r <= 15'd0;
        end else begin
            m5_base_g2_r <= m5_base_g2_c;
            m5_base_r    <= (state == S_M5_9 && !m5_last_iter) ? m5_base_g2_r : m5_base_c;
        end
    wire [14:0] m5_w_addr_r0 = m5_base_r + {7'b0, ctr_r};
    wire [14:0] m5_w_addr_r1 = m5_base_r + {7'b0, ctr_r_p1};
    wire [14:0] m5_w_addr_r2 = m5_base_r + {7'b0, ctr_r_p2};
    wire [14:0] m5_w_addr_r3 = m5_base_r + {7'b0, ctr_r_p3};
    wire [14:0] m5_c_addr    = C_B_DT_BASE + {6'b0, ctr_g};
    wire [14:0] m5_wr_addr   = PT_DELTA    + {6'b0, ctr_g};
    wire [14:0] m5_w_addr_g2_r0 = m5_base_g2_r;
    wire [14:0] m5_w_addr_g2_r1 = m5_w_addr_g2_r0 + 15'd1;
    wire [14:0] m5_c_addr_g2    = C_B_DT_BASE   + {6'b0, ctr_g_p2};

    // M6: one channel per ctr_l (16 per group), N_STATE_GRP state words per channel (ctr_s). ITMN has one state word
    // per channel; the scan then overlaps the next channel's dt*A / dt*B with the current one (fast path).
    wire [2:0]   n_state_grp_max_w = N_STATE_GRP[2:0] - 3'd1;
    wire [10:0]  m6_channel        = {ctr_g, ctr_l};
    // State RAM address: this cluster holds every other group, so it is indexed by g/2, and the state-word field is
    // dropped with one word per channel
    wire         m6_ns1            = (N_STATE_GRP == 4'd1);
    wire [2:0]   ctr_s_p1_pref     = ctr_s + 3'd1;
    wire [15:0]  m6_c              = m6_ns1 ? {4'b0, g_bank_idx[7:0], ctr_l}
                                            : {1'b0, g_bank_idx[7:0], ctr_l, ctr_s};
    wire [15:0]  m6_c_next         = m6_ns1 ? {4'b0, g_bank_idx[7:0], ctr_l}
                                            : {1'b0, g_bank_idx[7:0], ctr_l, ctr_s_p1_pref};
    wire [15:0]  m6_c_next_lane    = {4'b0, g_bank_idx[7:0], (ctr_l + 4'd1)};
    wire [14:0]  m6_B_addr_s       = PT_X_PROJ + {12'b0, ctr_s};
    wire [14:0]  m6_C_addr_s       = PT_X_PROJ + {11'b0, N_STATE_GRP} + {12'b0, ctr_s};
    wire [14:0]  m6_A_stride_offset = (N_STATE_GRP == 4'd1)
                                      ? {4'b0,  m6_channel}
                                      : {1'b0,  m6_channel, 3'b0};
    wire [14:0]  m6_A_addr_s       = W_A_BASE  + m6_A_stride_offset + {12'b0, ctr_s};
    wire [10:0]  m6_channel_next_l    = {ctr_g, (ctr_l + 4'd1)};
    wire [14:0]  m6_A_addr_next_l_s0  = W_A_BASE + {4'b0, m6_channel_next_l};
    wire [14:0]  m6_B_addr_s_next  = PT_X_PROJ + {12'b0, ctr_s_p1_pref};
    wire [14:0]  m6_C_addr_s_next  = PT_X_PROJ + {11'b0, N_STATE_GRP} + {12'b0, ctr_s_p1_pref};
    wire [14:0]  m6_A_addr_s_next  = W_A_BASE  + m6_A_stride_offset + {12'b0, ctr_s_p1_pref};
    wire [14:0]  m6_delta_addr     = PT_DELTA        + {6'b0, ctr_g};
    wire [14:0]  m6_u_addr         = PT_U            + {6'b0, ctr_g};
    wire [14:0]  m6_D_addr_const   = C_D_PARAM_BASE  + {6'b0, ctr_g};

    // Mamba1 x_proj output is packed [dt (DT_RANK), B (16), C (16), pad] over words 0..2, so B and C straddle word
    // boundaries: B = words {1,0} >> DT_RANK*16, C = words {2,1} >> DT_RANK*16. With several state words per channel
    // each (B, C) word is read directly.
    wire [8:0]   m6_dt_shift     = {1'b0, DT_RANK} << 4;
    wire [8:0]   m6_dt_comp      = 9'd256 - m6_dt_shift;
    wire [`NUM_LANES*`DATA_W-1:0] m6_B_shifted =
        (m6_w1_reg << m6_dt_comp) | (m6_w0_reg >> m6_dt_shift);
    wire [`NUM_LANES*`DATA_W-1:0] m6_C_shifted =
        (m6_w3_reg << m6_dt_comp) | (m6_w1_reg >> m6_dt_shift);
    wire m6_pack_mode = (N_STATE_GRP == 4'd1) && (DT_RANK != 8'd0);
    wire [`NUM_LANES*`DATA_W-1:0] m6_B_word = m6_pack_mode ? m6_B_shifted : m6_w0_reg;
    wire [`NUM_LANES*`DATA_W-1:0] m6_C_word = m6_pack_mode ? m6_C_shifted : m6_w1_reg;

    wire [15:0]  m6_dt_scalar    = m6_delta_word[ctr_l*`DATA_W +: `DATA_W];
    wire [`NUM_LANES*`DATA_W-1:0] m6_dt_broadcast = {`NUM_LANES{m6_dt_scalar}};
    wire [3:0]   ctr_l_p1        = ctr_l + 4'd1;
    wire [15:0]  m6_dt_next_scalar = m6_delta_word[ctr_l_p1*`DATA_W +: `DATA_W];
    wire [`NUM_LANES*`DATA_W-1:0] m6_dt_next_broadcast = {`NUM_LANES{m6_dt_next_scalar}};
    wire [15:0]  m6_u_scalar     = m6_u_word[ctr_l*`DATA_W +: `DATA_W];
    wire [`NUM_LANES*`DATA_W-1:0] m6_u_broadcast  = {`NUM_LANES{m6_u_scalar}};
    wire [15:0]  m6_D_scalar     = m6_D_word[ctr_l*`DATA_W +: `DATA_W];
    wire [`NUM_LANES*`DATA_W-1:0] m6_D_broadcast  = {`NUM_LANES{m6_D_scalar}};

    wire signed [`ACC_W+4:0] m6_y_ch_full = sum_d_wide >>> `FRAC_BITS;
    wire signed [15:0] m6_y_ch_sat =
        (m6_y_ch_full >  45'sd32767)  ? 16'sh7FFF :
        (m6_y_ch_full < -45'sd32768)  ? 16'sh8000 :
                                         m6_y_ch_full[15:0];
    wire signed [15:0] m6_y_ch_reg_sat =
        (m6_y_ch_reg >  24'sd32767)  ? 16'sh7FFF :
        (m6_y_ch_reg < -24'sd32768)  ? 16'sh8000 :
                                        m6_y_ch_reg[15:0];

    // h_new = sat16(dA*h + dB*u) and y_ssm = sat16(y + D*u) are added in fabric, bit-identical to a PE ADD (17-bit sum,
    // saturate, no shift)
    function automatic [`DATA_W-1:0] sat_add16;
        input [`DATA_W-1:0] a, b;
        reg   [`DATA_W:0]   s;
        begin
            s = {a[`DATA_W-1], a} + {b[`DATA_W-1], b};
            sat_add16 = (s[`DATA_W] != s[`DATA_W-1])
                      ? (s[`DATA_W] ? `SAT16_MIN : `SAT16_MAX)
                      : s[`DATA_W-1:0];
        end
    endfunction

    wire [`NUM_LANES*`DATA_W-1:0] m6_h_new_w;
    genvar m6_hl;
    generate
        for (m6_hl = 0; m6_hl < `NUM_LANES; m6_hl = m6_hl + 1) begin : M6_HNEW
            assign m6_h_new_w[m6_hl*`DATA_W +: `DATA_W] =
                sat_add16(cl_out_vec [m6_hl*`DATA_W +: `DATA_W],
                          cl_out_vec2[m6_hl*`DATA_W +: `DATA_W]);
        end
    endgenerate

    // D*u: the fast path latched it in S_M6_12, otherwise it is the S_M6_14 MUL output
    wire [15:0] m6_du_sel  = (N_STATE_GRP == 4'd1) ? m6_du_reg : cl_out_vec[15:0];
    wire [15:0] m6_y_ssm_w = sat_add16(m6_y_ch_reg_sat, m6_du_sel);

    // M7: y_gated[g] = y_ssm[g] * silu(z[g])
    wire [14:0] m7_z_addr    = PT_Z_GATE  + {6'b0, ctr_g};
    wire [14:0] m7_wr_addr   = PT_Y_GATED + {6'b0, ctr_g};
    wire        m7_last_iter = (ctr_g_p2 >= {1'b0, ch_m_act[8:0]});

    assign done_stage = (state == S_BARRIER);
    assign done_all   = (state == S_DONE);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            state              <= S_IDLE;
            cur_stage          <= 5'd0;
            t_cnt              <= 10'd0;
            wait_cnt           <= 3'd0;
            ctr_g              <= 9'd0;
            ctr_k              <= 13'd0;
            tap_cnt            <= 2'd0;
            ctr_r              <= 8'd0;
            m3_wr_ptr          <= 9'd0;
            ctr_l              <= 4'd0;
            ctr_load           <= 3'd0;
            ctr_s              <= 3'd0;
            m5_dt_word_reg     <= {`NUM_LANES*`DATA_W{1'b0}};
            silu_z_reg         <= {`NUM_LANES*`DATA_W{1'b0}};
            m_rd_addr          <= 15'd0;
            m_we               <= 1'b0;
            m_wr_addr          <= 15'd0;
            m_wr_data          <= {`NUM_LANES*`DATA_W{1'b0}};
            w_rd_addr          <= 15'd0;
            w_rd_addr2         <= 15'd0;
            const_rd_addr      <= 15'd0;
            cl_op_mode         <= `MAMBA_PE_IDLE;
            cl_clear_acc       <= 1'b0;
            cl_in_W1_vec       <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_H_ext        <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_W2_vec       <= {`NUM_LANES*`DATA_W{1'b0}};
            cl_in_X_vec        <= {`NUM_LANES*`DATA_W{1'b0}};
            rsqrt_idx_r        <= 13'd0;
            S_reg              <= 16'd0;
            h_init_cnt         <= 16'd0;
            m6_h_from_rf_r     <= 1'b0;
            m6_h_rd_addr_r     <= {H_ADDR_W{1'b0}};
            m6_h_wr_en_r       <= 1'b0;
            m6_h_wr_addr_r     <= {H_ADDR_W{1'b0}};
            m6_h_wr_from_pe_r  <= 1'b0;
            m6_h_wr_data_ext_r <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_ssm_grp_acc     <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_delta_word      <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_u_word          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_D_word          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_w0_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_w1_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_w3_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_bc_cached       <= 1'b0;
            m6_w2_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_y_ch_reg        <= 24'sd0;
            m6_du_reg          <= 16'sd0;
            m6_pref_c_valid    <= 1'b0;
            next_stage_q       <= 5'd0;
            next_state_q       <= S_IDLE;
            tin_r              <= 15'd0;
            tout_r             <= 15'd0;
            mp_ph              <= 3'd0;
            fin_ph             <= 4'd0;
            fin_ready          <= 1'b0;
            fin_m8             <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_inc            <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_scale          <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_shift          <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_res            <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_prev           <= {`NUM_LANES*`DATA_W{1'b0}};
            fin_a              <= {`NUM_LANES*`DATA_W{1'b0}};
            gap_acc[0]         <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[1]         <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[2]         <= {`NUM_LANES*GAP_W{1'b0}};
            gap_acc[3]         <= {`NUM_LANES*GAP_W{1'b0}};
            gap_lane           <= 5'd0;
            gap_prod           <= {(GAP_W+17){1'b0}};
            gap_sum_r          <= {GAP_W{1'b0}};
            gap_word           <= {`NUM_LANES*`DATA_W{1'b0}};
            mp_acc             <= {`NUM_LANES*`DATA_W{1'b0}};
        end else begin
            cl_op_mode   <= `MAMBA_PE_IDLE;
            cl_clear_acc <= 1'b0;
            m_we         <= 1'b0;
            m6_h_wr_en_r <= 1'b0;

            case (state)
                S_IDLE: begin
                    if (start) begin
                        t_cnt      <= 10'd0;
                        ctr_g      <= {8'd0, CLUSTER_ID[0]};
                        ctr_k      <= 13'd0;
                        tap_cnt    <= 2'd0;
                        ctr_r      <= 8'd0;
                        ctr_l      <= 4'd0;
                        ctr_s      <= 3'd0;
                        h_init_cnt <= 16'd0;
                        m6_bc_cached <= 1'b0;
                        gap_acc[0] <= {`NUM_LANES*GAP_W{1'b0}}; gap_acc[1] <= {`NUM_LANES*GAP_W{1'b0}};
                        gap_acc[2] <= {`NUM_LANES*GAP_W{1'b0}}; gap_acc[3] <= {`NUM_LANES*GAP_W{1'b0}};
                        // The Inception phase enters through the barrier so the release sets its tsplit counters and weight base
                        if (PHASE_INC) begin
                            next_stage_q <= ENC_EN ? STG_ENC : STG_P1;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else begin
                            cur_stage  <= STG_M6;
                            state      <= S_H_INIT;
                        end
                    end
                end

                // Zero the state RAM, then start the first timestep
                S_H_INIT: begin
                    if ({2'b0, h_init_cnt} < h_init_limit) begin
                        m6_h_wr_en_r       <= 1'b1;
                        m6_h_wr_addr_r     <= h_init_cnt[H_ADDR_W-1:0];
                        m6_h_wr_from_pe_r  <= 1'b0;
                        m6_h_wr_data_ext_r <= {`NUM_LANES*`DATA_W{1'b0}};
                        h_init_cnt         <= h_init_cnt + 16'd1;
                    end else begin
                        h_init_cnt   <= 16'd0;
                        next_stage_q <= STG_RN;
                        next_state_q <= S_RN1;
                        state        <= S_BARRIER;
                    end
                end

                // MAC chain (all MAC-shaped stages, see the job table)
                S_MAC1: begin
                    m_rd_addr  <= mac_rd_addr_k0;
                    w_rd_addr  <= mac_w_addr_k0;
                    w_rd_addr2 <= mac_w_addr_k0 + 15'd1;
                    if (mac_bias_en)
                        const_rd_addr <= mac_bias_base + {6'b0, ctr_g};
                    state      <= S_MAC2;
                end

                // Three addresses in flight before the first MAC2 fire in S_MAC3
                S_MAC2: begin
                    if (wait_cnt == 3'd0) begin
                        wait_cnt   <= 3'd1;
                        w_rd_addr  <= mac_w_addr_k2;
                        w_rd_addr2 <= mac_w_addr_k2 + 15'd1;
                        m_rd_addr  <= mac_rd_addr_k2;
                        ctr_k      <= 13'd0;
                    end else begin
                        wait_cnt   <= 3'd0;
                        w_rd_addr  <= mac_w_addr_k4;
                        w_rd_addr2 <= mac_w_addr_k4 + 15'd1;
                        state      <= S_MAC3;
                    end
                end

                S_MAC3: begin
                    cl_in_W1_vec <= w_rd_data_r;
                    cl_in_W2_vec <= w_rd_data2_r;
                    cl_in_H_ext  <= mac_y1;
                    cl_in_X_vec  <= mac_y2;
                    cl_op_mode   <= `MAMBA_PE_MAC2;
                    cl_clear_acc <= 1'b1;
                    m_rd_addr    <= mac_rd_addr_k6;
                    w_rd_addr    <= mac_w_addr_k6;
                    w_rd_addr2   <= mac_w_addr_k6 + 15'd1;
                    ctr_k        <= ctr_k_p2;
                    state        <= S_MAC4;
                end

                // The exit iteration does not fire: the lookahead reads already reach into the next group's weights
                S_MAC4: begin
                    if (ctr_k != mac_len) begin
                        cl_op_mode   <= `MAMBA_PE_MAC2;
                        cl_in_W1_vec <= w_rd_data_r;
                        cl_in_W2_vec <= w_rd_data2_r;
                        cl_in_H_ext  <= mac_y1;
                        cl_in_X_vec  <= mac_y2;
                    end
                    m_rd_addr    <= mac_rd_addr_k6;
                    w_rd_addr    <= mac_w_addr_k6;
                    w_rd_addr2   <= mac_w_addr_k6 + 15'd1;
                    ctr_k        <= ctr_k_p2;
                    if (ctr_k == mac_len) begin
                        state <= S_MAC5;
                    end
                end

                // Drain, write the group, then loop to the next group or go to the barrier
                S_MAC5: begin
                    if (wait_cnt != 3'd4) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    // M8 result ready: compute FIN first; S_FIN returns here with wait_cnt = 4 to commit it
                    end else if (fin_mode && !fin_ready) begin
                        fin_m8   <= cl_out_vec;
                        fin_ph   <= 4'd0;
                        wait_cnt <= 3'd0;
                        state    <= S_FIN;
                    end else begin
                        wait_cnt  <= 3'd0;
                        fin_ready <= 1'b0;
                        m_we      <= 1'b1;
                        m_wr_addr <= !fin_mode   ? mac_wr_addr  :
                                     fin_pool_odd ? fin_pool_addr : fin_addr;
                        if (fin_mode && GAP_EN)
                            gap_acc[g_bank_idx[1:0]] <= gap_add(gap_acc[g_bank_idx[1:0]], fin_res);
                        m_wr_data <= fin_mode    ? fin_res :
                                     mac_bias_en ? sat_add_vec(cl_out_vec, const_rd_data)
                                                 : cl_out_vec;
                        if (mac_last_iter) begin
                            ctr_g <= {8'd0, CLUSTER_ID[0]};
                            ctr_k <= 13'd0;
                            next_stage_q <= mac_next_stage;
                            next_state_q <= mac_next_state;
                            state        <= S_BARRIER;
                        end else begin
                            if (mac_tsplit) begin
                                ctr_g <= mac_g_last_in_t ? 9'd0 : ctr_g + 9'd1;
                                if (mac_g_last_in_t) begin
                                    t_cnt  <= t_cnt + 10'd2;
                                    tin_r  <= tin_next;
                                    tout_r <= tout_next;
                                end
                                if (mac_bias_en)
                                    const_rd_addr <= mac_bias_base +
                                        (mac_g_last_in_t ? 15'd0 : {6'b0, ctr_g} + 15'd1);
                            end else begin
                                ctr_g      <= ctr_g_p2;
                            end
                            ctr_k      <= 13'd0;
                            m_rd_addr  <= mac_rd_addr_g0;
                            w_rd_addr  <= mac_w_addr_gnext;
                            w_rd_addr2 <= mac_w_addr_gnext + 15'd1;
                            state      <= S_MAC2;
                        end
                    end
                end

                // GAP (block 4): per own group, lanes 0..15 through select -> product -> sat16 into gap_word, then one write to
                // PT_GAP + ctr_g; after the last group, FC
                S_GAP_RUN: begin
                    gap_lane <= gap_lane + 5'd1;
                    gap_sum_r <= gap_lane_sum;
                    if (gap_lane >= 5'd1 && gap_lane <= 5'd16)
                        gap_prod <= $signed(gap_sum_r) * $signed({1'b0, GAP_INV});
                    if (gap_lane >= 5'd2)
                        gap_word[(gap_lane - 5'd2)*`DATA_W +: `DATA_W] <= gap_sat16;
                    if (gap_lane == 5'd18) begin
                        gap_lane  <= 5'd0;
                        m_we      <= 1'b1;
                        m_wr_addr <= PT_GAP + {6'b0, ctr_g};
                        m_wr_data <= gap_word;
                        if (ctr_g_p2 >= {1'b0, CH_OUT[8:0]}) begin
                            next_stage_q <= STG_FC;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else begin
                            ctr_g <= ctr_g_p2;
                        end
                    end
                end

                // FIN. ph0 read INC_OUT[t][g] and scale[g]; ph1 read shift[g]; ph2 latch scale (pooled odd t: read final[t-1]);
                // ph3 latch shift and inc; ph4 PE MUL inc*scale; ph10/11 fabric epilogue, then back to S_MAC5
                S_FIN: begin
                    fin_ph <= fin_ph + 4'd1;
                    case (fin_ph)
                        4'd0: begin
                            m_rd_addr     <= fin_addr;
                            const_rd_addr <= C_INC_SCALE_BASE + {6'b0, ctr_g};
                        end
                        4'd1: const_rd_addr <= C_INC_SHIFT_BASE + {6'b0, ctr_g};
                        4'd2: begin
                            fin_scale <= const_rd_data;
                            if (fin_pool_odd) m_rd_addr <= fin_prev_addr;
                        end
                        4'd3: begin
                            fin_shift <= const_rd_data;
                            fin_inc   <= m_rd_data;
                        end
                        4'd5: fin_prev <= m_rd_data;
                        4'd4: begin
                            cl_op_mode   <= `MAMBA_PE_MUL;
                            cl_in_W1_vec <= fin_inc;
                            cl_in_H_ext  <= fin_scale;
                        end
                        4'd10: fin_a <= fin_a_next;
                        4'd11: begin
                            fin_res   <= fin_pool_odd ? max_vec(fin_prev, fin_final) : fin_final;
                            fin_ready <= 1'b1;
                            fin_ph    <= 4'd0;
                            wait_cnt  <= 3'd4;
                            state     <= S_MAC5;
                        end
                        default: ;
                    endcase
                end

                // MP3 (Inception): phases 0-2 read t-1, t, t+1, phases 3-5 fold them; phase 5 writes and steps (g, t) like a tsplit
                // MAC loopback
                S_MP_RUN: begin
                    mp_ph <= mp_ph + 3'd1;
                    case (mp_ph)
                        3'd0: m_rd_addr <= mp_prev_addr;
                        3'd1: m_rd_addr <= mp_cur_addr;
                        3'd2: m_rd_addr <= mp_next_addr;
                        3'd3: mp_acc    <= m_rd_data;
                        3'd4: mp_acc    <= max_vec(mp_acc, m_rd_data);
                        default: begin
                            mp_ph     <= 3'd0;
                            m_we      <= 1'b1;
                            m_wr_addr <= tout_r + {6'b0, ctr_g};
                            m_wr_data <= max_vec(mp_acc, m_rd_data);
                            if (mac_g_last_in_t) begin
                                ctr_g <= 9'd0;
                                if (mac_t_last) begin
                                    next_stage_q <= mac_next_stage;
                                    next_state_q <= mac_next_state;
                                    state        <= S_BARRIER;
                                end else begin
                                    t_cnt  <= t_cnt + 10'd2;
                                    tin_r  <= tin_next;
                                    tout_r <= tout_next;
                                end
                            end else begin
                                ctr_g <= ctr_g + 9'd1;
                            end
                        end
                    endcase
                end

                // Wait for the peer, then enter next_state_q and reset the per-stage counters
                S_BARRIER: begin
                    if (peer_done_stage) begin
                        cur_stage <= next_stage_q;
                        state     <= next_state_q;
                        // M8 -> RN is the next timestep: B/C must be reloaded
                        if (next_stage_q == STG_RN && cur_stage == STG_M8) begin
                            t_cnt        <= t_cnt + 10'd1;
                            m6_bc_cached <= 1'b0;
                        end
                        // S_M5_1 consumes the dt word, so issue its read now
                        if (next_state_q == S_M5_1) begin
                            m_rd_addr <= m5_dt_addr;
                        end
                        ctr_g    <= {8'd0, CLUSTER_ID[0]};
                        ctr_k    <= 13'd0;
                        ctr_l    <= 4'd0;
                        ctr_s    <= 3'd0;
                        ctr_r    <= 8'd0;
                        tap_cnt  <= 2'd0;
                        ctr_load <= 3'd0;
                        wait_cnt <= 3'd0;
                        // tsplit stage: all groups from 0, t from CLUSTER_ID
                        if (mac_tsplit_of(next_stage_q)) begin
                            ctr_g  <= 9'd0;
                            t_cnt  <= {9'd0, CLUSTER_ID[0]};
                            tin_r  <= tsplit_in_base_of(next_stage_q) +
                                      (CLUSTER_ID[0] ? {6'b0, tsplit_in_wpt_of(next_stage_q, CH_IN, CH_OUT)} : 15'd0);
                            tout_r <= tsplit_out_base_of(next_stage_q, CH_OUT) +
                                      (CLUSTER_ID[0] ? {6'b0, tsplit_out_wpt_of(next_stage_q, CH_IN, CH_OUT)} : 15'd0);
                        end
                    end
                end

                // A start pulse restarts directly (next block)
                S_DONE: begin
                    if (start) begin
                        t_cnt      <= 10'd0;
                        ctr_g      <= {8'd0, CLUSTER_ID[0]};
                        ctr_k      <= 13'd0;
                        tap_cnt    <= 2'd0;
                        ctr_r      <= 8'd0;
                        ctr_l      <= 4'd0;
                        ctr_s      <= 3'd0;
                        h_init_cnt <= 16'd0;
                        // the previous block's last timestep left it set
                        m6_bc_cached <= 1'b0;
                        gap_acc[0] <= {`NUM_LANES*GAP_W{1'b0}}; gap_acc[1] <= {`NUM_LANES*GAP_W{1'b0}};
                        gap_acc[2] <= {`NUM_LANES*GAP_W{1'b0}}; gap_acc[3] <= {`NUM_LANES*GAP_W{1'b0}};
                        if (PHASE_INC) begin
                            next_stage_q <= ENC_EN ? STG_ENC : STG_P1;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else begin
                            cur_stage  <= STG_M6;
                            state      <= S_H_INIT;
                        end
                    end
                end

                // RN: sum of squares -> rsqrt (ROM) -> x * gamma * rsqrt per group
                S_RN1: begin
                    ctr_k     <= 13'd0;
                    m_rd_addr <= rn_sq_rd_addr_k0;
                    state     <= S_RN2;
                end

                // Words 0, 1, 2 are issued in RN1, RN2, RN2 and RN3 stays 3 words ahead of the word it consumes; the hold cycle
                // must advance the address, a repeated word only shows from t = 1
                S_RN2: begin
                    if (wait_cnt != 3'd1) begin
                        wait_cnt  <= wait_cnt + 3'd1;
                        m_rd_addr <= rn_sq_rd_addr_k1;
                    end else begin
                        wait_cnt  <= 3'd0;
                        m_rd_addr <= rn_sq_rd_addr_k2;
                        state     <= S_RN3;
                    end
                end

                S_RN3: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (ctr_k == 13'd0);
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= m_rd_data;
                    m_rd_addr    <= rn_sq_rd_addr_k3;
                    ctr_k        <= ctr_k_p1;
                    if (ctr_k == rn_sq_last) state <= S_RN4;
                end

                // PE drain + lane-sum register
                S_RN4: begin
                    if (wait_cnt == 3'd5) begin
                        wait_cnt <= 3'd0;
                        state    <= S_RN5;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_RN5: begin
                    rsqrt_idx_r <= mean_i_clip;
                    state       <= S_RN6;
                end

                S_RN6: begin
                    S_reg <= rsqrt_data;
                    ctr_g <= {8'd0, CLUSTER_ID[0]};
                    state <= S_RN7;
                end

                S_RN7: begin
                    m_rd_addr     <= rn_ap_rd_addr_g0;
                    const_rd_addr <= rn_ap_c_addr_g0;
                    state         <= S_RN8;
                end

                S_RN8: begin
                    if (wait_cnt != 3'd1) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        wait_cnt <= 3'd0;
                        state    <= S_RN9;
                    end
                end

                S_RN9: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m_rd_data;
                    cl_in_H_ext  <= const_rd_data;
                    state        <= S_RN10;
                end

                S_RN10: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_RN11;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // x*gamma*S, and prefetch the next group's x and gamma
                S_RN11: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= S_broadcast;
                    if (!rn_last_iter) begin
                        m_rd_addr     <= rn_ap_rd_addr_g2;
                        const_rd_addr <= rn_ap_c_addr_g2;
                    end
                    state <= S_RN12;
                end

                S_RN12: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_RN13;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_RN13: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= rn_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (rn_last_iter) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        next_stage_q <= STG_M1A;
                        next_state_q <= S_MAC1;
                        state        <= S_BARRIER;
                    end else begin
                        ctr_g <= ctr_g_p2;
                        state <= S_RN9;
                    end
                end
                // M2: 4-tap MAC, bias add (PE ADD), write
                S_M2_1: begin
                    tap_cnt       <= 2'd0;
                    m_rd_addr     <= m2_rd_addr_tap0;
                    w_rd_addr     <= m2_w_addr_tap0;
                    const_rd_addr <= m2_c_addr;
                    state         <= S_M2_2;
                end

                S_M2_2: begin
                    if (wait_cnt == 3'd0) begin
                        wait_cnt  <= 3'd1;
                        m_rd_addr <= m2_rd_addr_tap1;
                        w_rd_addr <= m2_w_addr_tap1;
                    end else begin
                        wait_cnt  <= 3'd0;
                        m_rd_addr <= m2_rd_addr_tap2;
                        w_rd_addr <= m2_w_addr_tap2;
                        state     <= S_M2_3;
                    end
                end

                // One tap per cycle, reads 3 taps ahead; the reads past tap 3 are never consumed
                S_M2_3: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (tap_cnt == 2'd0);
                    cl_in_W1_vec <= w_rd_data_r;
                    cl_in_H_ext  <= m2_pad ? {`NUM_LANES*`DATA_W{1'b0}} : m_rd_data;
                    m_rd_addr    <= m2_rd_addr_tap3;
                    w_rd_addr    <= m2_w_addr_tap3;
                    tap_cnt      <= tap_p1[1:0];
                    if (tap_cnt == 2'd3) state <= S_M2_4;
                end

                S_M2_4: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M2_5;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M2_5: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= const_rd_data;
                    state        <= S_M2_6;
                end

                S_M2_6: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M2_7;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M2_7: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m2_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (m2_last_iter) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        m3_wr_ptr    <= {8'd0, CLUSTER_ID[0]};
                        next_stage_q <= STG_M3;
                        next_state_q <= S_M3_1;
                        state        <= S_BARRIER;
                    end else begin
                        ctr_g         <= ctr_g_p2;
                        tap_cnt       <= 2'd0;
                        m_rd_addr     <= m2_rd_addr_g2;
                        w_rd_addr     <= m2_w_addr_g2;
                        const_rd_addr <= m2_c_addr_g2;
                        state         <= S_M2_2;
                    end
                end

                // wait_cnt counts the 3 warm-up reads before the first write
                S_M3_1: begin
                    if (wait_cnt < 3'd3) begin
                        if (ctr_g < {1'b0, ch_m_act[8:0]}) begin
                            m_rd_addr <= m3_rd_addr;
                            ctr_g     <= ctr_g_p2;
                        end
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        if (ctr_g < {1'b0, ch_m_act[8:0]}) begin
                            m_rd_addr <= m3_rd_addr;
                            ctr_g     <= ctr_g_p2;
                        end
                        m_we      <= 1'b1;
                        m_wr_addr <= m3_wr_addr;
                        m_wr_data <= silu_out_w;
                        if (m3_last_write) begin
                            ctr_g        <= {8'd0, CLUSTER_ID[0]};
                            ctr_k        <= 13'd0;
                            m3_wr_ptr    <= 9'd0;
                            wait_cnt     <= 3'd0;
                            next_stage_q <= STG_M4;
                            next_state_q <= S_MAC1;
                            state        <= S_BARRIER;
                        end else begin
                            m3_wr_ptr <= m3_wr_ptr + 9'd2;
                        end
                    end
                end
                S_M5_1: begin
                    if (wait_cnt == 3'd1) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M5_2;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M5_2: begin
                    m5_dt_word_reg <= m_rd_data;
                    ctr_l          <= 4'd0;
                    ctr_r          <= 8'd0;
                    state          <= S_M5_3;
                end

                S_M5_3: begin
                    w_rd_addr     <= m5_w_addr_r0;
                    const_rd_addr <= m5_c_addr;
                    state         <= S_M5_4;
                end

                // Three weight addresses (r0 in S_M5_3, r1 and r2 here) before the first MAC; the loop then reads 3 ranks
                // ahead. The next group re-enters at wait_cnt = 1 to issue its r2.
                S_M5_4: begin
                    if (wait_cnt != 3'd1) begin
                        wait_cnt  <= wait_cnt + 3'd1;
                        w_rd_addr <= m5_w_addr_r1;
                    end else begin
                        wait_cnt  <= 3'd0;
                        w_rd_addr <= m5_w_addr_r2;
                        state     <= S_M5_5;
                    end
                end

                S_M5_5: begin
                    cl_op_mode   <= `MAMBA_PE_MAC;
                    cl_clear_acc <= (ctr_r == 8'd0);
                    cl_in_W1_vec <= w_rd_data_r;
                    cl_in_H_ext  <= m5_dt_broadcast;
                    w_rd_addr    <= m5_w_addr_r3;
                    ctr_l        <= ctr_l + 4'd1;
                    ctr_r        <= ctr_r_p1;
                    if (ctr_r == dt_rank_last)
                        state <= S_M5_6;
                    else if (ctr_l == 4'd15) begin
                        m_rd_addr <= PT_X_PROJ + {11'b0, ctr_r_p1[7:4]};
                        w_rd_addr <= m5_w_addr_r1;
                        state     <= S_M5_DT_R1;
                    end
                end

                S_M5_DT_R1: begin
                    if (wait_cnt != 3'd1) begin
                        wait_cnt  <= wait_cnt + 3'd1;
                        w_rd_addr <= m5_w_addr_r1;
                    end else begin
                        wait_cnt <= 3'd0;
                        state    <= S_M5_DT_R2;
                    end
                end

                S_M5_DT_R2: begin
                    m5_dt_word_reg <= m_rd_data;
                    state          <= S_M5_5;
                end

                S_M5_6: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M5_7;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M5_7: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= cl_out_vec;
                    cl_in_H_ext  <= const_rd_data;
                    state        <= S_M5_8;
                end

                S_M5_8: begin
                    if (wait_cnt != 3'd4) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        wait_cnt <= 3'd0;
                        if (!m5_last_iter) begin
                            w_rd_addr <= m5_w_addr_g2_r0;
                        end
                        state <= S_M5_9;
                    end
                end

                S_M5_9: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m5_wr_addr;
                    m_wr_data <= sp_out_w;
                    ctr_l     <= 4'd0;
                    ctr_r     <= 8'd0;
                    if (m5_last_iter) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_s        <= 3'd0;
                        ctr_load     <= 3'd0;
                        next_stage_q <= STG_M6;
                        next_state_q <= S_M6_1;
                        state        <= S_BARRIER;
                    end else begin
                        ctr_g         <= ctr_g_p2;
                        w_rd_addr     <= m5_w_addr_g2_r1;
                        const_rd_addr <= m5_c_addr_g2;
                        wait_cnt      <= 3'd1;
                        state         <= S_M5_4;
                    end
                end

                // Load slots: 0 B (+ A[c] on the weight port, h on the state RAM), 1 C, 2 packed C high word (Mamba1 layout,
                // first group of a timestep), 3 delta + D, 4 u. A full load runs at (l, s) = (0, 0); later (l, s) reload B and C
                S_M6_1: begin
                    case (ctr_load)
                        3'd0: begin
                            m_rd_addr      <= m6_B_addr_s;
                            m6_h_rd_addr_r <= m6_c;
                            w_rd_addr      <= m6_A_addr_s;
                        end
                        3'd1: m_rd_addr <= m6_C_addr_s;
                        3'd2: m_rd_addr <= PT_X_PROJ + 15'd2;
                        3'd3: begin
                            m_rd_addr     <= m6_delta_addr;
                            const_rd_addr <= m6_D_addr_const;
                        end
                        3'd4: m_rd_addr <= m6_u_addr;
                        default: ;
                    endcase
                    state <= S_M6_2;
                end

                S_M6_2: begin
                    if (wait_cnt == 3'd1) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_3;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M6_3: begin
                    case (ctr_load)
                        3'd0: begin
                            // packed B/C are cached, and PT_X_PROJ word 0 now holds y_gated
                            if (!(m6_pack_mode && m6_bc_cached))
                                m6_w0_reg <= m_rd_data;
                            m6_w2_reg <= w_rd_data_r;
                        end
                        3'd1: m6_w1_reg <= m_rd_data;
                        3'd2: begin
                            m6_w3_reg    <= m_rd_data;
                            m6_bc_cached <= 1'b1;
                        end
                        3'd3: begin
                            m6_delta_word <= m_rd_data;
                            m6_D_word     <= const_rd_data;
                        end
                        3'd4: m6_u_word <= m_rd_data;
                        default: ;
                    endcase
                    if (ctr_load == 3'd4) begin
                        m6_pref_c_valid <= 1'b0;
                        state           <= S_M6_4;
                    end else if (ctr_load == 3'd0 && m6_pack_mode && m6_bc_cached) begin
                        ctr_load <= 3'd3;
                        state    <= S_M6_1;
                    end else if (ctr_load == 3'd1 &&
                                 !(ctr_l == 4'd0 && ctr_s == 3'd0)) begin
                        m6_pref_c_valid <= 1'b1;
                        state           <= S_M6_4;
                    end else if (ctr_load == 3'd1 && m6_pack_mode) begin
                        ctr_load <= 3'd2;
                        state    <= S_M6_1;
                    end else if (ctr_load == 3'd1) begin
                        ctr_load <= 3'd3;
                        state    <= S_M6_1;
                    end else if (ctr_load == 3'd2) begin
                        ctr_load <= 3'd3;
                        state    <= S_M6_1;
                    end else begin
                        ctr_load <= ctr_load + 3'd1;
                        state    <= S_M6_1;
                    end
                end

                // dt*A and dt*B (MUL2)
                S_M6_4: begin
                    cl_op_mode   <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec <= m6_dt_broadcast;
                    cl_in_H_ext  <= m6_w2_reg;
                    cl_in_W2_vec <= m6_dt_broadcast;
                    cl_in_X_vec  <= m6_B_word;
                    if (m6_pref_c_valid)
                        m6_w1_reg <= m_rd_data;
                    state <= S_M6_5;
                end

                S_M6_5: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_6;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // dA = exp(dt*A), dB = dt*B arrive this cycle: fire the state update MUL2(dA, h) + (dB, u). h_from_rf rises here,
                // so PE stage A sees the state word next cycle.
                S_M6_6: begin
                    cl_op_mode     <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec   <= exp_out_w;
                    cl_in_W2_vec   <= cl_out_vec2;
                    cl_in_X_vec    <= m6_u_broadcast;
                    m6_h_from_rf_r <= 1'b1;
                    // fast path: A[c+1] for the dt*A overlapped into S_M6_12
                    if (N_STATE_GRP == 4'd1 && ctr_l != LANE_LAST)
                        w_rd_addr <= m6_A_addr_next_l_s0;
                    state          <= S_M6_8;
                end

                S_M6_8: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_9;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // h_new in fabric: write it to the state RAM and fire the output MAC in the same cycle. Fast path: one MUL2 gives
                // h*C (m1) and D*u (m2).
                S_M6_9: begin
                    m6_h_from_rf_r     <= 1'b0;
                    m6_h_wr_en_r       <= 1'b1;
                    m6_h_wr_addr_r     <= m6_c;
                    m6_h_wr_from_pe_r  <= 1'b0;
                    m6_h_wr_data_ext_r <= m6_h_new_w;
                    if (N_STATE_GRP == 4'd1) begin
                        cl_op_mode   <= `MAMBA_PE_MUL2;
                        cl_in_W1_vec <= m6_h_new_w;
                        cl_in_H_ext  <= m6_C_word;
                        cl_in_W2_vec <= m6_D_broadcast;
                        cl_in_X_vec  <= m6_u_broadcast;
                    end else begin
                        cl_op_mode   <= `MAMBA_PE_MAC;
                        cl_clear_acc <= 1'b1;
                        cl_in_W1_vec <= m6_h_new_w;
                        cl_in_H_ext  <= m6_C_word;
                    end
                    if (ctr_s != n_state_grp_max_w) begin
                        m_rd_addr      <= m6_B_addr_s_next;
                        w_rd_addr      <= m6_A_addr_s_next;
                        m6_h_rd_addr_r <= m6_c_next;
                    end else if (ctr_l == LANE_LAST) begin
                        // last channel of the group: the z read for M7 is in m_rd_data by S_M6_18
                        m_rd_addr <= m7_z_addr;
                    end
                    state <= S_M6_12;
                end

                // Fast path, not the last channel: dt*A / dt*B of channel c+1 go into the idle PE slot at wait_cnt = 2 and come
                // out exactly when S_M6_18 hands over to S_M6_6 for c+1; channel c's outputs are consumed before that. h[c+1] is
                // read here too.
                S_M6_12: begin
                    if (wait_cnt == 3'd2 && N_STATE_GRP == 4'd1 &&
                        ctr_l != LANE_LAST) begin
                        cl_op_mode     <= `MAMBA_PE_MUL2;
                        cl_in_W1_vec   <= m6_dt_next_broadcast;
                        cl_in_H_ext    <= w_rd_data_r;
                        cl_in_W2_vec   <= m6_dt_next_broadcast;
                        cl_in_X_vec    <= m6_B_word;
                        m6_h_rd_addr_r <= m6_c_next_lane;
                    end
                    if (wait_cnt != 3'd5) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        wait_cnt <= 3'd0;
                        if (ctr_s != n_state_grp_max_w)
                            m_rd_addr <= m6_C_addr_s_next;
                        // out_vec2 holds D*u only in this cycle
                        if (N_STATE_GRP == 4'd1 && ctr_s == n_state_grp_max_w)
                            m6_du_reg <= cl_out_vec2[15:0];
                        state <= S_M6_13;
                    end
                end

                // y accumulates over the state words
                S_M6_13: begin
                    if (ctr_s == 3'd0)
                        m6_y_ch_reg <= {{8{m6_y_ch_sat[15]}}, m6_y_ch_sat};
                    else
                        m6_y_ch_reg <= m6_y_ch_reg + {{8{m6_y_ch_sat[15]}}, m6_y_ch_sat};

                    if (ctr_s == n_state_grp_max_w) begin
                        if (N_STATE_GRP == 4'd1) begin
                            state     <= S_M6_18;
                        end else begin
                            state <= S_M6_14;
                        end
                    end else begin
                        m6_w0_reg       <= m_rd_data;
                        m6_w2_reg       <= w_rd_data_r;
                        m6_pref_c_valid <= 1'b1;
                        ctr_s           <= ctr_s + 3'd1;
                        state           <= S_M6_4;
                    end
                end

                S_M6_14: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_D_broadcast;
                    cl_in_H_ext  <= m6_u_broadcast;
                    state        <= S_M6_15;
                end

                S_M6_15: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_18;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // y_ssm into lane ctr_l of the group word; the last channel goes to M7
                S_M6_18: begin
                    m6_ssm_grp_acc[ctr_l*`DATA_W +: `DATA_W] <= m6_y_ssm_w;
                    if (ctr_l == LANE_LAST) begin
                        state     <= S_M7_3;
                    // dt*A of c+1 is already in flight. h_from_rf was cleared in S_M6_9, so that MUL2 saw H = A[c+1], not h (dA would
                    // otherwise be exp(dt*h): invisible at t = 0, wrong from t = 1)
                    end else if (N_STATE_GRP == 4'd1) begin
                        ctr_l          <= ctr_l + 4'd1;
                        ctr_s          <= 3'd0;
                        ctr_load       <= 3'd0;
                        state          <= S_M6_6;
                    end else begin
                        ctr_l    <= ctr_l + 4'd1;
                        ctr_s    <= 3'd0;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end
                // M7, entered from the last channel of a group; loops back to M6 for the next group
                S_M7_3: begin
                    silu_z_reg <= silu_out_w;
                    state      <= S_M7_5;
                end

                S_M7_5: begin
                    cl_op_mode   <= `MAMBA_PE_MUL;
                    cl_in_W1_vec <= m6_ssm_grp_acc;
                    cl_in_H_ext  <= silu_z_reg;
                    state        <= S_M7_6;
                end

                S_M7_6: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M7_7;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M7_7: begin
                    m_we      <= 1'b1;
                    m_wr_addr <= m7_wr_addr;
                    m_wr_data <= cl_out_vec;
                    if (m7_last_iter) begin
                        ctr_g        <= {8'd0, CLUSTER_ID[0]};
                        ctr_k        <= 13'd0;
                        next_stage_q <= STG_M8;
                        next_state_q <= S_MAC1;
                        state        <= S_BARRIER;
                    end else begin
                        ctr_g    <= ctr_g_p2;
                        ctr_l    <= 4'd0;
                        ctr_s    <= 3'd0;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end

                default: begin
                    state <= S_IDLE;
                end
            endcase
        end
    end

endmodule
