`timescale 1ns/1ps
`include "_parameter.v"

// Cycle benchmark: the mamba-only design at another model's layer size (parameters below, set per config by
// run_bench.py through xelab -generic_top). Runs T_TEST tokens of one layer with a constant input and whatever the
// weight RAMs hold, so it measures cycles only; the data is not checked. Reports per-stage cycles of cluster 0
// and the per-token compute (total minus the state RAM clear) and scan (M6) cycles.
module tb_bench;

    parameter integer CH_OUT   = 4;      // d_model / 16
    parameter integer CH_M     = 8;      // d_inner / 16
    parameter integer DT_RANK  = 4;
    parameter integer NSG      = 1;      // state words (of 16) per channel
    parameter integer USE_M5   = 1;
    parameter integer XP_GRP   = 3;      // x_proj output groups
    parameter integer H_ADDR_W = 9;
    parameter integer T_TEST   = 3;

    localparam integer H_DEPTH = 1 << H_ADDR_W;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
    reg          dma_write_en = 0;
    reg  [2:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top #(.H_ADDR_W(H_ADDR_W), .H_DEPTH(H_DEPTH)) dut (
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .T_MAX(T_TEST[9:0]), .CH_OUT(CH_OUT[8:0]), .CH_M(CH_M[8:0]), .DT_RANK(DT_RANK[7:0]),
        .N_STATE_GRP(NSG[3:0]), .USE_M5(USE_M5[0]), .XP_OUT_GRP_IN(XP_GRP[4:0]),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(1'b0), .dma_rtarget(3'd0), .dma_raddr(15'd0), .dma_rdata(dma_rdata)
    );

    // Cluster 0 cycles per stage (ids 0 M1A 1 M1B 2 M2 3 M3 4 M4 5 M5 6 M6/M7 8 M8 9 RN) and the S_H_INIT time
    integer stage_cyc [0:9];
    integer h_init_cyc = 0;
    integer i, cyc_start, compute;
    initial for (i = 0; i < 10; i = i + 1) stage_cyc[i] = 0;
    always @(posedge clk)
        if (!rst && start === 1'b0 && dut.u_c0.state != 7'd0 && dut.u_c0.state != 7'd127) begin
            if (dut.u_c0.state == 7'd1)
                h_init_cyc = h_init_cyc + 1;
            else if (dut.u_c0.cur_stage < 10)
                stage_cyc[dut.u_c0.cur_stage] = stage_cyc[dut.u_c0.cur_stage] + 1;
        end

    initial begin
        repeat (5) @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        @(negedge clk);
        dma_write_en = 1; dma_target = 3'd0;
        for (i = 0; i < CH_OUT * T_TEST; i = i + 1) begin
            dma_addr  = `PT_INPUT + i;
            dma_wdata = {16{16'sh0100}};
            @(negedge clk);
        end
        dma_write_en = 0;

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start = $time / 10;
        wait (done_all == 1'b1);
        compute = ($time / 10 - cyc_start - h_init_cyc) / T_TEST;

        $display("[BENCH] CH_OUT=%0d CH_M=%0d DT_RANK=%0d NSG=%0d USE_M5=%0d XP_GRP=%0d H_DEPTH=%0d T=%0d",
                 CH_OUT, CH_M, DT_RANK, NSG, USE_M5, XP_GRP, H_DEPTH, T_TEST);
        $display("[BENCH] stage cycles (all tokens): RN %0d  M1A %0d  M1B %0d  M2 %0d  M3 %0d  M4 %0d  M5 %0d  M6/M7 %0d  M8 %0d",
                 stage_cyc[9], stage_cyc[0], stage_cyc[1], stage_cyc[2], stage_cyc[3], stage_cyc[4], stage_cyc[5],
                 stage_cyc[6], stage_cyc[8]);
        $display("[BENCH] h_init %0d", h_init_cyc);
        $display("[BENCH] compute_per_token %0d", compute);
        $display("[BENCH] m6_per_token %0d", stage_cyc[6] / T_TEST);
        $finish;
    end

    initial begin
        #2000000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
