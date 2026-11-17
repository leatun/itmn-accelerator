`include "_parameter.v"

// Controller of one cluster. Top level instantiates two (CLUSTER_ID 0 and 1) running the same FSM on their own PE
// array, scan state RAM and memories; ram_main writes to the shared regions are mirrored into the peer's RAM by the
// top level (see Mamba_Top), the per-cluster scratch slots are not.
//
// Work split: in group-parallel stages cluster c computes output groups g = c, c+2, ... (ctr_g steps by 2) and
// holds only those groups' weights (compacted, index g/2).
// A cluster that finishes a stage waits in S_BARRIER until the peer has finished it too.
//
// Mamba phase, per timestep: RN (RMSNorm), M1A/M1B (in_proj x, z), M2 (causal conv4 + bias), M3 (SiLU),
// M4 (x_proj), M5 (dt_proj + softplus), M6/M7 (scan and gate, interleaved per group), M8 (out_proj).
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
    localparam [6:0] S_M6_7      = 7'd80;
    localparam [6:0] S_M6_8      = 7'd99;
    localparam [6:0] S_M6_9      = 7'd82;
    localparam [6:0] S_M6_10     = 7'd81;
    localparam [6:0] S_M6_11     = 7'd86;
    localparam [6:0] S_M6_12     = 7'd87;
    localparam [6:0] S_M6_13     = 7'd88;
    localparam [6:0] S_M6_14     = 7'd89;
    localparam [6:0] S_M6_15     = 7'd101;
    localparam [6:0] S_M6_16     = 7'd90;
    localparam [6:0] S_M6_17     = 7'd91;
    localparam [6:0] S_M6_18     = 7'd92;
    localparam [6:0] S_M7_2      = 7'd49;
    localparam [6:0] S_M7_3      = 7'd50;
    localparam [6:0] S_M7_4      = 7'd52;
    localparam [6:0] S_M7_5      = 7'd53;
    localparam [6:0] S_M7_6      = 7'd54;
    localparam [6:0] S_M7_7      = 7'd55;
    localparam [6:0] S_MAC5      = 7'd98;
    localparam [6:0] S_MAC3      = 7'd102;
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
    reg [`NUM_LANES*`DATA_W-1:0] m6_dA_reg, m6_dB_reg;
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
    wire [14:0]                       w_grp_base_comb;
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
            w_grp_base_r      <= w_grp_base_comb;
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
    // The weight base is kept separately (mac_w_base_of) so S_BARRIER can preload the next stage's.
    function [14:0] mac_w_base_of;
        input [4:0] stg;
        case (stg)
            STG_M1A: mac_w_base_of = W_INPROJ_X_BASE;
            STG_M1B: mac_w_base_of = W_INPROJ_Z_BASE;
            STG_M4:  mac_w_base_of = W_XPROJ_BASE;
            default: mac_w_base_of = W_OUTPROJ_BASE;
        endcase
    endfunction

    reg  [13:0] mac_len;
    reg  [9:0]  mac_grp_count;
    reg  [14:0] mac_in_base;
    reg  [14:0] mac_wr_base;
    reg  [4:0]  mac_next_stage;
    reg  [6:0]  mac_next_state;
    wire [14:0] mac_w_base = mac_w_base_of(cur_stage);
    always @(*) begin
        case (cur_stage)
            STG_M1A: begin
                mac_len        = {1'b0, d_model};
                mac_grp_count  = ch_m_act;
                mac_in_base    = PT_X_NORM;
                mac_wr_base    = PT_X_INNER_CIRC + ({13'b0, t_cnt[1:0]} * {5'b0, ch_m_act});
                mac_next_stage = STG_M1B;
                mac_next_state = S_MAC1;
            end
            STG_M1B: begin
                mac_len        = {1'b0, d_model};
                mac_grp_count  = ch_m_act;
                mac_in_base    = PT_X_NORM;
                mac_wr_base    = PT_Z_GATE;
                mac_next_stage = STG_M2;
                mac_next_state = S_M2_1;
            end
            STG_M4: begin
                mac_len        = d_inner;
                mac_grp_count  = {5'b0, XP_OUT_GRP};
                mac_in_base    = PT_U;
                mac_wr_base    = PT_X_PROJ;
                mac_next_stage = USE_M5 ? STG_M5 : STG_M6;
                mac_next_state = USE_M5 ? S_M5_1 : S_M6_1;
            end
            STG_M8: begin
                mac_len        = d_inner;
                mac_grp_count  = {1'b0, CH_OUT};
                mac_in_base    = PT_Y_GATED;
                mac_wr_base    = PT_MAMBA_OUT + rn_t_offset_x_r;
                mac_next_stage = (t_cnt == t_last) ? cur_stage : STG_RN;
                mac_next_state = (t_cnt == t_last) ? S_DONE    : S_RN1;
            end
            default: begin
                mac_len        = {1'b0, d_model};
                mac_grp_count  = ch_m_act;
                mac_in_base    = PT_X_NORM;
                mac_wr_base    = PT_MAMBA_OUT + rn_t_offset_x_r;
                mac_next_stage = cur_stage;
                mac_next_state = S_IDLE;
            end
        endcase
    end
    wire [14:0] mac_wr_addr = mac_wr_base + {6'b0, ctr_g};
    wire [14:0] mac_len_ext  = {1'b0, mac_len};

    // Weight RAMs hold only this cluster's groups, compacted: group g is at index g/2
    wire [8:0]  g_bank_idx    = {1'b0, ctr_g[8:1]};

    assign      w_grp_base_comb      = mac_w_base + ({8'b0, g_bank_idx} * mac_len_ext);

    // w_grp_base_r is event-driven: loaded at the barrier release and at the S_MAC5 group loopback, held otherwise
    // (ctr_g is stable inside the MAC loop). w_grp_base_next_r is first used >= 16 cycles after any ctr_g or stage
    // change, so its 2-cycle lag is safe.
    wire [12:0] ctr_k_p2 = ctr_k + 13'd2;
    wire [12:0] ctr_k_p4 = ctr_k + 13'd4;
    wire [12:0] ctr_k_p6 = ctr_k + 13'd6;

    wire [14:0] mac_rd_addr_k0 = mac_in_base + {6'b0, ctr_k[12:4]};
    wire [14:0] mac_rd_addr_k2 = mac_in_base + {6'b0, ctr_k_p2[12:4]};
    // The loop consumes two input elements per cycle and RAM data arrives 3 cycles after the address, so the input
    // address runs 6 elements ahead
    wire [14:0] mac_rd_addr_k6 = mac_in_base + {6'b0, ctr_k_p6[12:4]};
    wire [14:0] mac_w_addr_k0  = w_grp_base_r + {2'b0, ctr_k};
    wire [14:0] mac_w_addr_k2  = w_grp_base_r + {2'b0, ctr_k_p2};
    wire [14:0] mac_w_addr_k4  = w_grp_base_r + {2'b0, ctr_k_p4};
    wire [14:0] mac_w_addr_k6  = w_grp_base_r + {2'b0, ctr_k_p6};

    wire [8:0]  ctr_g_p2       = ctr_g + 9'd2;
    wire [8:0]  g_bank_idx_next = {1'b0, ctr_g_p2[8:1]};
    assign      w_grp_base_next_comb = mac_w_base + ({8'b0, g_bank_idx_next} * mac_len_ext);
    wire [14:0] mac_w_addr_gnext = w_grp_base_next_r;

    // group index at a barrier release is 0 for every stage
    assign      barrier_mac_w_base = mac_w_base_of(next_stage_q);
    wire [14:0] mac_rd_addr_g0   = mac_in_base;

    // last iteration: the next group of this cluster would be past the last group
    wire mac_last_iter = (ctr_g_p2 >= {1'b0, mac_grp_count[8:0]});
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
    wire [14:0] m5_w_addr_r0 = W_DTPROJ_BASE + ({6'b0, ctr_g} * {7'b0, DT_RANK}) + {7'b0, ctr_r};
    wire [14:0] m5_w_addr_r1 = W_DTPROJ_BASE + ({6'b0, ctr_g} * {7'b0, DT_RANK}) + {7'b0, ctr_r_p1};
    wire [14:0] m5_w_addr_r2 = W_DTPROJ_BASE + ({6'b0, ctr_g} * {7'b0, DT_RANK}) + {7'b0, ctr_r_p2};
    wire [14:0] m5_w_addr_r3 = W_DTPROJ_BASE + ({6'b0, ctr_g} * {7'b0, DT_RANK}) + {7'b0, ctr_r_p3};
    wire [14:0] m5_c_addr    = C_B_DT_BASE + {6'b0, ctr_g};
    wire [14:0] m5_wr_addr   = PT_DELTA    + {6'b0, ctr_g};
    wire [14:0] m5_w_addr_g2_r0 = W_DTPROJ_BASE + ({6'b0, ctr_g_p2} * {7'b0, DT_RANK});
    wire [14:0] m5_w_addr_g2_r1 = m5_w_addr_g2_r0 + 15'd1;
    wire [14:0] m5_c_addr_g2    = C_B_DT_BASE   + {6'b0, ctr_g_p2};
    wire        m5_last_iter    = (ctr_g_p2 >= {1'b0, ch_m_act[8:0]});

    // M6: one channel per ctr_l (16 per group), N_STATE_GRP state words per channel (ctr_s). ITMN has one state word
    // per channel; the scan then starts the next channel at dt*A without a load (fast path).
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
            m6_dA_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_dB_reg          <= {`NUM_LANES*`DATA_W{1'b0}};
            m6_y_ch_reg        <= 24'sd0;
            m6_du_reg          <= 16'sd0;
            m6_pref_c_valid    <= 1'b0;
            next_stage_q       <= 5'd0;
            next_state_q       <= S_IDLE;
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
                        cur_stage  <= STG_M6;
                        state      <= S_H_INIT;
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
                    cl_in_H_ext  <= y_broadcast;
                    cl_in_X_vec  <= y_broadcast2;
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
                        cl_in_H_ext  <= y_broadcast;
                        cl_in_X_vec  <= y_broadcast2;
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
                    end else begin
                        wait_cnt  <= 3'd0;
                        m_we      <= 1'b1;
                        m_wr_addr <= mac_wr_addr;
                        m_wr_data <= cl_out_vec;
                        if (mac_last_iter) begin
                            ctr_g <= {8'd0, CLUSTER_ID[0]};
                            ctr_k <= 13'd0;
                            next_stage_q <= mac_next_stage;
                            next_state_q <= mac_next_state;
                            state        <= S_BARRIER;
                        end else begin
                            ctr_g      <= ctr_g_p2;
                            ctr_k      <= 13'd0;
                            m_rd_addr  <= mac_rd_addr_g0;
                            w_rd_addr  <= mac_w_addr_gnext;
                            w_rd_addr2 <= mac_w_addr_gnext + 15'd1;
                            state      <= S_MAC2;
                        end
                    end
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
                        m6_bc_cached <= 1'b0;
                        cur_stage  <= STG_M6;
                        state      <= S_H_INIT;
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

                // dA = exp(dt*A), dB = dt*B; h_from_rf rises so the next MUL2 sees the state word
                S_M6_6: begin
                    m6_dA_reg      <= exp_out_w;
                    m6_dB_reg      <= cl_out_vec2;
                    m6_h_from_rf_r <= 1'b1;
                    state          <= S_M6_7;
                end

                // state update: dA*h and dB*u (MUL2), then h_new = their sum (PE ADD) written back
                S_M6_7: begin
                    cl_op_mode   <= `MAMBA_PE_MUL2;
                    cl_in_W1_vec <= m6_dA_reg;
                    cl_in_W2_vec <= m6_dB_reg;
                    cl_in_X_vec  <= m6_u_broadcast;
                    state        <= S_M6_8;
                end

                S_M6_8: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_9;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M6_9: begin
                    cl_op_mode     <= `MAMBA_PE_ADD;
                    cl_in_W1_vec   <= cl_out_vec;
                    cl_in_H_ext    <= cl_out_vec2;
                    m6_h_from_rf_r <= 1'b0;
                    state          <= S_M6_10;
                end

                S_M6_10: begin
                    if (wait_cnt != 3'd4) begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end else begin
                        wait_cnt          <= 3'd0;
                        m6_h_wr_en_r      <= 1'b1;
                        m6_h_wr_addr_r    <= m6_c;
                        m6_h_wr_from_pe_r <= 1'b1;
                        state             <= S_M6_11;
                    end
                end

                // output: fast path one MUL2 gives h*C (m1) and D*u (m2); otherwise MAC over the state words
                S_M6_11: begin
                    if (N_STATE_GRP == 4'd1) begin
                        cl_op_mode   <= `MAMBA_PE_MUL2;
                        cl_in_W1_vec <= cl_out_vec;
                        cl_in_H_ext  <= m6_C_word;
                        cl_in_W2_vec <= m6_D_broadcast;
                        cl_in_X_vec  <= m6_u_broadcast;
                    end else begin
                        cl_op_mode   <= `MAMBA_PE_MAC;
                        cl_clear_acc <= 1'b1;
                        cl_in_W1_vec <= cl_out_vec;
                        cl_in_H_ext  <= m6_C_word;
                    end
                    if (ctr_s != n_state_grp_max_w) begin
                        m_rd_addr      <= m6_B_addr_s_next;
                        w_rd_addr      <= m6_A_addr_s_next;
                        m6_h_rd_addr_r <= m6_c_next;
                    // fast path: A[c+1] now, it is stable long before S_M6_18 latches it
                    end else if (N_STATE_GRP == 4'd1 && ctr_l != LANE_LAST) begin
                        w_rd_addr <= m6_A_addr_next_l_s0;
                    end
                    state <= S_M6_12;
                end

                S_M6_12: begin
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
                            state     <= S_M6_16;
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
                        state    <= S_M6_16;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // y_ssm = y + D*u (PE ADD)
                S_M6_16: begin
                    cl_op_mode   <= `MAMBA_PE_ADD;
                    cl_in_W1_vec <= {`NUM_LANES{m6_y_ch_reg_sat}};
                    cl_in_H_ext  <= (N_STATE_GRP == 4'd1) ? {`NUM_LANES{m6_du_reg}}
                                                          : {`NUM_LANES{cl_out_vec[15:0]}};
                    state        <= S_M6_17;
                end

                S_M6_17: begin
                    if (wait_cnt == 3'd4) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M6_18;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                // y_ssm into lane ctr_l of the group word; the last channel issues the z read and goes to M7
                S_M6_18: begin
                    m6_ssm_grp_acc[ctr_l*`DATA_W +: `DATA_W] <= cl_out_vec[15:0];
                    if (ctr_l == LANE_LAST) begin
                        m_rd_addr <= m7_z_addr;
                        state     <= S_M7_2;
                    // Fast path: latch A[c+1] and go straight to dt*A. h_from_rf stays 0 until S_M6_6, so that MUL2 sees H = A[c+1],
                    // not h (dA would otherwise be exp(dt*h): invisible at t = 0, wrong from t = 1).
                    end else if (N_STATE_GRP == 4'd1) begin
                        ctr_l          <= ctr_l + 4'd1;
                        ctr_s          <= 3'd0;
                        ctr_load       <= 3'd0;
                        m6_w2_reg      <= w_rd_data_r;
                        m6_h_rd_addr_r <= m6_c_next_lane;
                        state          <= S_M6_4;
                    end else begin
                        ctr_l    <= ctr_l + 4'd1;
                        ctr_s    <= 3'd0;
                        ctr_load <= 3'd0;
                        state    <= S_M6_1;
                    end
                end

                // M7, entered from the last channel of a group; loops back to M6 for the next group
                S_M7_2: begin
                    if (wait_cnt == 3'd1) begin
                        wait_cnt <= 3'd0;
                        state    <= S_M7_3;
                    end else begin
                        wait_cnt <= wait_cnt + 3'd1;
                    end
                end

                S_M7_3: begin
                    silu_z_reg <= silu_out_w;
                    state      <= S_M7_4;
                end

                S_M7_4: state <= S_M7_5;

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
