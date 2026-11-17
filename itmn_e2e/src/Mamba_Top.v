`include "_parameter.v"

// Two clusters (Mamba_Cluster), each with its own memories (Memory_Cluster), a stage barrier between them and the
// host DMA port.
//
// Mirror writes: a ram_main write of one cluster is also written into the peer's ram_main when it falls in a shared
// region, addresses [0, 32) (X_NORM / X_PROJ / Y_GATED, X_CONV / U / Y_SSM) and [128, ...) (block data). [32, 128)
// (DELTA, Z_GATE, the x_inner circular buffer) is per-cluster scratch: each cluster reads only its own groups there.
// U is shared because x_proj (M4) reads every channel.
//
// DMA write targets: 0 ram_main (both clusters), 2 cluster 0 weights, 3 ram_const (both), 4 cluster 1 weights.
// DMA read targets: 0 cluster 0 ram_main, 2 cluster 0 weights, 3 cluster 0 constants, 4 cluster 1 weights,
// 5 cluster 1 ram_main.
module Mamba_Top #(
    parameter H_ADDR_W = `H_ADDR_W,
    parameter H_DEPTH  = `H_DEPTH
)(
    input  wire                              clk,
    input  wire                              rst,
    input  wire                              start,
    output wire                              done_stage,
    output wire                              done_all,

    input  wire [9:0]                        T_MAX,
    input  wire [8:0]                        CH_OUT,   // d_model / 16
    input  wire [8:0]                        CH_M,   // d_inner / 16 (0 = 128)
    input  wire [7:0]                        DT_RANK,
    input  wire [3:0]                        N_STATE_GRP,   // state words per channel; ITMN: 1
    input  wire                              USE_M5,   // dt_proj present; ITMN: 1
    input  wire [4:0]                        XP_OUT_GRP_IN,   // x_proj output groups (0 = 3)
    input  wire                              PHASE_INC,   // 1: Inception phase, 0: Mamba phase
    input  wire [8:0]                        CH_IN,   // block input d_in / 16
    input  wire                              FIN_EN,   // fuse the ITM block output into M8
    input  wire                              POOL_OUT,   // with FIN_EN: MaxPool1d(2,2) the block output
    input  wire                              ENC_EN,   // Inception phase starts with the encoder (block 0)
    input  wire                              GAP_EN,   // with FIN_EN: GAP and FC head (block 4)
    input  wire [15:0]                       GAP_INV,   // round(2^15 / T)

    input  wire                              dma_write_en,
    input  wire [2:0]                        dma_target,
    input  wire [14:0]                       dma_addr,
    input  wire [`NUM_LANES*`DATA_W-1:0]     dma_wdata,
    input  wire                              dma_read_en,
    input  wire [2:0]                        dma_rtarget,
    input  wire [14:0]                       dma_raddr,
    output wire [`NUM_LANES*`DATA_W-1:0]     dma_rdata
);

    wire [14:0]                      c0_m_rd_addr,   c1_m_rd_addr;
    wire [`NUM_LANES*`DATA_W-1:0]    c0_m_rd_data,   c1_m_rd_data;
    wire                             c0_m_we,        c1_m_we;
    wire [14:0]                      c0_m_wr_addr,   c1_m_wr_addr;
    wire [`NUM_LANES*`DATA_W-1:0]    c0_m_wr_data,   c1_m_wr_data;
    wire [14:0]                      c0_w_rd_addr,   c1_w_rd_addr;
    wire [14:0]                      c0_w_rd_addr2,  c1_w_rd_addr2;
    wire [`NUM_LANES*`DATA_W-1:0]    c0_w_rd_data,   c1_w_rd_data;
    wire [`NUM_LANES*`DATA_W-1:0]    c0_w_rd_data2,  c1_w_rd_data2;
    wire [14:0]                      c0_const_rd_addr, c1_const_rd_addr;
    wire [`NUM_LANES*`DATA_W-1:0]    c0_const_rd_data, c1_const_rd_data;

    // Stage barrier: each cluster's peer_done_stage is the other's done_stage
    wire c0_done_stage, c1_done_stage;
    wire c0_done_all,   c1_done_all;

    function mirror_region;
        input [14:0] addr;
        begin
            mirror_region = (addr < 15'd32) || (addr >= 15'd128);
        end
    endfunction

    wire mirror_c0_wr = c0_m_we && mirror_region(c0_m_wr_addr);
    wire mirror_c1_wr = c1_m_we && mirror_region(c1_m_wr_addr);

    wire dma_main_wr   = dma_write_en && (dma_target == 3'd0);
    wire dma_const_wr  = dma_write_en && (dma_target == 3'd3);
    wire dma_wA_wr     = dma_write_en && (dma_target == 3'd2);
    wire dma_wB_wr     = dma_write_en && (dma_target == 3'd4);

    wire       c0_dma_we     = dma_main_wr || dma_const_wr || dma_wA_wr;
    wire [2:0] c0_dma_target = dma_wA_wr    ? 3'd2 :
                               dma_const_wr ? 3'd3 :
                                              3'd0;

    wire       c1_dma_we     = dma_main_wr || dma_const_wr || dma_wB_wr;
    wire [2:0] c1_dma_target = dma_wB_wr    ? 3'd2 :
                               dma_const_wr ? 3'd3 :
                                              3'd0;

    wire       dma_rd_wB     = dma_read_en && (dma_rtarget == 3'd4);
    wire       dma_rd_c1main = dma_read_en && (dma_rtarget == 3'd5);
    wire       dma_rd_toc1   = dma_rd_wB || dma_rd_c1main;
    wire       c0_dma_rd_en  = dma_read_en && !dma_rd_toc1;
    wire       c1_dma_rd_en  = dma_rd_toc1;
    wire [2:0] c1_dma_rtarget = dma_rd_wB    ? 3'd2 :
                                dma_rd_c1main ? 3'd0 :
                                                dma_rtarget;

    wire [`NUM_LANES*`DATA_W-1:0] c0_dma_rdata, c1_dma_rdata;
    assign dma_rdata = dma_rd_toc1 ? c1_dma_rdata : c0_dma_rdata;

    Mamba_Cluster #(
        .CLUSTER_ID(0),
        .H_ADDR_W  (H_ADDR_W),
        .H_DEPTH   (H_DEPTH)
    ) u_c0 (
        .clk              (clk),
        .rst              (rst),
        .start            (start),
        .peer_done_stage  (c1_done_stage),
        .done_stage       (c0_done_stage),
        .done_all         (c0_done_all),

        .T_MAX            (T_MAX),
        .CH_OUT           (CH_OUT),
        .CH_M             (CH_M),
        .DT_RANK          (DT_RANK),
        .N_STATE_GRP      (N_STATE_GRP),
        .USE_M5           (USE_M5),
        .XP_OUT_GRP_IN    (XP_OUT_GRP_IN),
        .PHASE_INC        (PHASE_INC),
        .CH_IN            (CH_IN),
        .FIN_EN           (FIN_EN),
        .POOL_OUT         (POOL_OUT),
        .ENC_EN           (ENC_EN),
        .GAP_EN           (GAP_EN),
        .GAP_INV          (GAP_INV),

        .m_rd_addr        (c0_m_rd_addr),
        .m_rd_data_c      (c0_m_rd_data),
        .m_we             (c0_m_we),
        .m_wr_addr        (c0_m_wr_addr),
        .m_wr_data        (c0_m_wr_data),
        .w_rd_addr        (c0_w_rd_addr),
        .w_rd_addr2       (c0_w_rd_addr2),
        .w_rd_data        (c0_w_rd_data),
        .w_rd_data2       (c0_w_rd_data2),
        .const_rd_addr    (c0_const_rd_addr),
        .const_rd_data    (c0_const_rd_data)
    );

    Mamba_Cluster #(
        .CLUSTER_ID(1),
        .H_ADDR_W  (H_ADDR_W),
        .H_DEPTH   (H_DEPTH)
    ) u_c1 (
        .clk              (clk),
        .rst              (rst),
        .start            (start),
        .peer_done_stage  (c0_done_stage),
        .done_stage       (c1_done_stage),
        .done_all         (c1_done_all),

        .T_MAX            (T_MAX),
        .CH_OUT           (CH_OUT),
        .CH_M             (CH_M),
        .DT_RANK          (DT_RANK),
        .N_STATE_GRP      (N_STATE_GRP),
        .USE_M5           (USE_M5),
        .XP_OUT_GRP_IN    (XP_OUT_GRP_IN),
        .PHASE_INC        (PHASE_INC),
        .CH_IN            (CH_IN),
        .FIN_EN           (FIN_EN),
        .POOL_OUT         (POOL_OUT),
        .ENC_EN           (ENC_EN),
        .GAP_EN           (GAP_EN),
        .GAP_INV          (GAP_INV),

        .m_rd_addr        (c1_m_rd_addr),
        .m_rd_data_c      (c1_m_rd_data),
        .m_we             (c1_m_we),
        .m_wr_addr        (c1_m_wr_addr),
        .m_wr_data        (c1_m_wr_data),
        .w_rd_addr        (c1_w_rd_addr),
        .w_rd_addr2       (c1_w_rd_addr2),
        .w_rd_data        (c1_w_rd_data),
        .w_rd_data2       (c1_w_rd_data2),
        .const_rd_addr    (c1_const_rd_addr),
        .const_rd_data    (c1_const_rd_data)
    );

    Memory_Cluster u_mem0 (
        .clk             (clk),
        .reset           (rst),
        .core_rd_addr    (c0_m_rd_addr),
        .core_rd_data    (c0_m_rd_data),
        .core_we         (c0_m_we),
        .core_wr_addr    (c0_m_wr_addr),
        .core_wr_data    (c0_m_wr_data),
        .xc_we           (mirror_c1_wr),
        .xc_wr_addr      (c1_m_wr_addr),
        .xc_wr_data      (c1_m_wr_data),
        .w_rd_addr       (c0_w_rd_addr),
        .w_rd_data       (c0_w_rd_data),
        .w_rd_addr2      (c0_w_rd_addr2),
        .w_rd_data2      (c0_w_rd_data2),
        .const_rd_addr   (c0_const_rd_addr),
        .const_rd_data   (c0_const_rd_data),
        .dma_write_en    (c0_dma_we),
        .dma_target      (c0_dma_target),
        .dma_addr        (dma_addr),
        .dma_wdata       (dma_wdata),
        .dma_read_en     (c0_dma_rd_en),
        .dma_rtarget     (dma_rtarget),
        .dma_raddr       (dma_raddr),
        .dma_rdata       (c0_dma_rdata)
    );

    Memory_Cluster u_mem1 (
        .clk             (clk),
        .reset           (rst),
        .core_rd_addr    (c1_m_rd_addr),
        .core_rd_data    (c1_m_rd_data),
        .core_we         (c1_m_we),
        .core_wr_addr    (c1_m_wr_addr),
        .core_wr_data    (c1_m_wr_data),
        .xc_we           (mirror_c0_wr),
        .xc_wr_addr      (c0_m_wr_addr),
        .xc_wr_data      (c0_m_wr_data),
        .w_rd_addr       (c1_w_rd_addr),
        .w_rd_data       (c1_w_rd_data),
        .w_rd_addr2      (c1_w_rd_addr2),
        .w_rd_data2      (c1_w_rd_data2),
        .const_rd_addr   (c1_const_rd_addr),
        .const_rd_data   (c1_const_rd_data),
        .dma_write_en    (c1_dma_we),
        .dma_target      (c1_dma_target),
        .dma_addr        (dma_addr),
        .dma_wdata       (dma_wdata),
        .dma_read_en     (c1_dma_rd_en),
        .dma_rtarget     (c1_dma_rtarget),
        .dma_raddr       (dma_raddr),
        .dma_rdata       (c1_dma_rdata)
    );

    assign done_all   = c0_done_all & c1_done_all;
    assign done_stage = done_all;   // kept for the host interface

endmodule
