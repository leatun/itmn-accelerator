`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// Mamba phase of block 0 for DBG_T timesteps (default 1). After the run, the intermediates of timestep DBG_T-1
// still sit in their RAM slots; each is read back over DMA and compared with the golden:
//   M1A x_inner (circular slot), M1B z_gate, M5 delta, M7 y_gated, M8 mamba_out.
// x_inner, z_gate and delta are written only by the cluster that owns the group (odd groups in cluster 1's RAM,
// read with DMA target 5); y_gated and mamba_out are mirrored.
module tb_Mamba_Top_DBG;

    localparam D_MODEL   = 64;
    localparam D_INNER   = 128;
    localparam D_STATE   = 16;
    localparam DT_RANK_V = 4;
    localparam N_PAD     = 48;
    localparam T_TOT     = 1000;
`ifdef DBG_T
    localparam T_TEST    = `DBG_T;
`else
    localparam T_TEST    = 1;
`endif
    localparam TC        = T_TEST - 1;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
    reg  [9:0]   T_MAX = T_TEST;

    reg          dma_write_en = 0;
    reg  [2:0]   dma_target   = 0;
    reg  [14:0]  dma_addr     = 0;
    reg  [255:0] dma_wdata    = 0;
    reg          dma_read_en  = 0;
    reg  [2:0]   dma_rtarget  = 0;
    reg  [14:0]  dma_raddr    = 0;
    wire [255:0] dma_rdata;

    Mamba_Top dut (
        .clk(clk), .rst(rst), .start(start),
        .done_stage(done_stage), .done_all(done_all),
        .T_MAX(T_MAX),
        .CH_OUT(9'd4), .CH_M(9'd8), .DT_RANK(8'd4),
        .N_STATE_GRP(4'd1), .USE_M5(1'b1), .XP_OUT_GRP_IN(5'd3),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    // Golden files are channel-major: element (ch, t) at ch*T_TOT + t
    reg [15:0] wx_mem   [0:D_INNER*D_MODEL-1];
    reg [15:0] wz_mem   [0:D_INNER*D_MODEL-1];
    reg [15:0] wo_mem   [0:D_MODEL*D_INNER-1];
    reg [15:0] wdw_mem  [0:D_INNER*4-1];
    reg [15:0] wxp_mem  [0:N_PAD*D_INNER-1];
    reg [15:0] wdt_mem  [0:D_INNER*DT_RANK_V-1];
    reg [15:0] wA_mem   [0:D_INNER*D_STATE-1];
    reg [15:0] gam_mem  [0:D_MODEL-1];
    reg [15:0] bdw_mem  [0:D_INNER-1];
    reg [15:0] bdt_mem  [0:D_INNER-1];
    reg [15:0] Dp_mem   [0:D_INNER-1];
    reg [15:0] xn_mem   [0:D_MODEL*T_TOT-1];
    reg [15:0] xi_exp   [0:D_INNER*T_TOT-1];
    reg [15:0] zg_exp   [0:D_INNER*T_TOT-1];
    reg [15:0] dlt_exp  [0:D_INNER*T_TOT-1];
    reg [15:0] ygat_exp [0:D_INNER*T_TOT-1];
    reg [15:0] mout_exp [0:D_MODEL*T_TOT-1];

    integer errors_xi = 0, errors_zg = 0, errors_dlt = 0, errors_ygat = 0, errors_mout = 0;
    integer g, c_in, k, t_cur, lane, sub_err, cyc_start_t, total;
    reg [255:0] word_tmp, readback;

    task dma_wr;
        input [2:0]   target;
        input [14:0]  addr;
        input [255:0] data;
        begin
            @(negedge clk);
            dma_write_en = 1;
            dma_target   = target;
            dma_addr     = addr;
            dma_wdata    = data;
        end
    endtask

    task dma_rd;
        input [2:0]  rt;
        input [14:0] addr;
        begin
            @(negedge clk);
            dma_read_en = 1'b1;
            dma_rtarget = rt;
            dma_raddr   = addr;
            @(posedge clk);
            @(negedge clk);
            readback    = dma_rdata;
            dma_read_en = 1'b0;
        end
    endtask

    // Group-parity-split weight word: even output groups to cluster 0's RAM (target 2), odd to cluster 1's (4)
    task dma_wr_split;
        input [14:0]  base;
        input integer grp, len, idx;
        input [255:0] data;
        dma_wr(grp[0] ? 3'd4 : 3'd2, base + (grp >> 1) * len + idx, data);
    endtask

    task dma_wr_both;
        input [14:0]  addr;
        input [255:0] data;
        begin
            dma_wr(3'd2, addr, data);
            dma_wr(3'd4, addr, data);
        end
    endtask

    // which: 0 x_inner, 1 z_gate, 2 delta, 4 y_gated, 5 mamba_out
    task compare_word;
        input [15*8:0] name;
        input [14:0]   addr;
        input integer  c_grp;
        input [2:0]    which;
        output integer errs;
        integer        lane_i, ch, idx;
        reg signed [15:0] gv, ev;
        begin
            errs = 0;
            dma_rd((which <= 3'd2 && c_grp[0]) ? 3'd5 : 3'd0, addr);
            for (lane_i = 0; lane_i < 16; lane_i = lane_i + 1) begin
                ch  = c_grp * 16 + lane_i;
                idx = ch * T_TOT + TC;
                gv  = readback[lane_i*16 +: 16];
                case (which)
                    3'd0:    ev = xi_exp[idx];
                    3'd1:    ev = zg_exp[idx];
                    3'd2:    ev = dlt_exp[idx];
                    3'd4:    ev = ygat_exp[idx];
                    default: ev = mout_exp[idx];
                endcase
                if (gv !== ev) begin
                    if (errs < 4)
                        $display("  %0s FAIL c=%0d  got=%6d (%04h)  exp=%6d (%04h)", name, ch, gv, gv, ev, ev);
                    errs = errs + 1;
                end
            end
        end
    endtask

    // Cycles of cluster 0 per stage, and how many of them wait in S_BARRIER (state 120) for the peer.
    // Stage ids: 0 M1A 1 M1B 2 M2 3 M3 4 M4 5 M5 6 M6/M7 8 M8 9 RN.
    integer prof_cyc [0:31];
    integer prof_bar [0:31];
    integer prof_i;
    initial
        for (prof_i = 0; prof_i < 32; prof_i = prof_i + 1) begin
            prof_cyc[prof_i] = 0;
            prof_bar[prof_i] = 0;
        end
    always @(posedge clk)
        if (!rst && dut.u_c0.state != 7'd0 && dut.u_c0.state != 7'd127) begin
            prof_cyc[dut.u_c0.cur_stage] = prof_cyc[dut.u_c0.cur_stage] + 1;
            if (dut.u_c0.state == 7'd120)
                prof_bar[dut.u_c0.cur_stage] = prof_bar[dut.u_c0.cur_stage] + 1;
        end

    initial begin
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_InProj_X.txt"},      wx_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_InProj_Z.txt"},      wz_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_OutProj.txt"},       wo_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_Conv.txt"},          wdw_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_XProj.txt"},         wxp_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_DtProj.txt"},        wdt_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_A_signed.txt"},        wA_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_W_Norm.txt"},          gam_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_B_Conv.txt"},          bdw_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_B_DtProj.txt"},        bdt_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_D_param.txt"},         Dp_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/P1_Output_Golden_FP.txt"}, xn_mem);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_X_Inner_FP.txt"},      xi_exp);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_Z_Gate_FP.txt"},       zg_exp);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_Delta_FP.txt"},        dlt_exp);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_Y_Gated_FP.txt"},      ygat_exp);
        $readmemh({`GOLDEN_ROOT, "/block_00_layer00/Mam_OutProj_FP.txt"},      mout_exp);

        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        // Weights: output-group-parallel matrices are split by group parity, the rest duplicated
        for (g = 0; g < D_INNER/16; g = g + 1)
            for (c_in = 0; c_in < D_MODEL; c_in = c_in + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wx_mem[(g*16+lane)*D_MODEL + c_in];
                dma_wr_split(`W_INPROJ_X_BASE, g, D_MODEL, c_in, word_tmp);
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wz_mem[(g*16+lane)*D_MODEL + c_in];
                dma_wr_split(`W_INPROJ_Z_BASE, g, D_MODEL, c_in, word_tmp);
            end
        for (g = 0; g < D_MODEL/16; g = g + 1)
            for (c_in = 0; c_in < D_INNER; c_in = c_in + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wo_mem[(g*16+lane)*D_INNER + c_in];
                dma_wr_split(`W_OUTPROJ_BASE, g, D_INNER, c_in, word_tmp);
            end
        for (g = 0; g < N_PAD/16; g = g + 1)
            for (c_in = 0; c_in < D_INNER; c_in = c_in + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wxp_mem[(g*16+lane)*D_INNER + c_in];
                dma_wr_split(`W_XPROJ_BASE, g, D_INNER, c_in, word_tmp);
            end
        for (g = 0; g < D_INNER/16; g = g + 1) begin
            for (k = 0; k < 4; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wdw_mem[(g*16+lane)*4 + k];
                dma_wr_both(`W_DW_BASE + g*4 + k, word_tmp);
            end
            for (k = 0; k < DT_RANK_V; k = k + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wdt_mem[(g*16+lane)*DT_RANK_V + k];
                dma_wr_both(`W_DTPROJ_BASE + g*DT_RANK_V + k, word_tmp);
            end
        end
        for (c_in = 0; c_in < D_INNER; c_in = c_in + 1) begin
            for (lane = 0; lane < D_STATE; lane = lane + 1)
                word_tmp[lane*16 +: 16] = wA_mem[c_in*D_STATE + lane];
            dma_wr_both(`W_A_BASE + c_in, word_tmp);
        end

        // Constants (target 3 writes both clusters' const RAMs)
        for (g = 0; g < D_MODEL/16; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1)
                word_tmp[lane*16 +: 16] = gam_mem[g*16 + lane];
            dma_wr(3'd3, `C_W_NORM_BASE + g, word_tmp);
        end
        for (g = 0; g < D_INNER/16; g = g + 1) begin
            for (lane = 0; lane < 16; lane = lane + 1)
                word_tmp[lane*16 +: 16] = bdw_mem[g*16 + lane];
            dma_wr(3'd3, `C_B_DW_BASE + g, word_tmp);
            for (lane = 0; lane < 16; lane = lane + 1)
                word_tmp[lane*16 +: 16] = bdt_mem[g*16 + lane];
            dma_wr(3'd3, `C_B_DT_BASE + g, word_tmp);
            for (lane = 0; lane < 16; lane = lane + 1)
                word_tmp[lane*16 +: 16] = Dp_mem[g*16 + lane];
            dma_wr(3'd3, `C_D_PARAM_BASE + g, word_tmp);
        end

        // Block input (the Mamba phase input = P1 output), T_TEST x D_MODEL/16 words
        for (t_cur = 0; t_cur < T_TEST; t_cur = t_cur + 1)
            for (g = 0; g < D_MODEL/16; g = g + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = xn_mem[(g*16+lane)*T_TOT + t_cur];
                dma_wr(3'd0, `PT_INPUT + t_cur*(D_MODEL/16) + g, word_tmp);
            end
        @(negedge clk); dma_write_en = 0;

        @(negedge clk); start = 1;
        @(negedge clk); start = 0;
        cyc_start_t = $time;
        wait (done_all == 1'b1);
        $display("[FSM] done at %0t  (compute = %0d cycles)", $time, ($time - cyc_start_t) / 10);
        for (prof_i = 0; prof_i < 10; prof_i = prof_i + 1)
            if (prof_cyc[prof_i] > 0)
                $display("[PROF]   stage %0d  total %0d  busy %0d  barrier %0d",
                         prof_i, prof_cyc[prof_i], prof_cyc[prof_i] - prof_bar[prof_i], prof_bar[prof_i]);
        @(negedge clk); @(negedge clk); @(negedge clk);

        for (g = 0; g < D_INNER/16; g = g + 1) begin
            compare_word("X_INNER", `PT_X_INNER_CIRC + (TC % 4)*(D_INNER/16) + g, g, 3'd0, sub_err);
            errors_xi = errors_xi + sub_err;
            compare_word("Z_GATE ", `PT_Z_GATE + g, g, 3'd1, sub_err);
            errors_zg = errors_zg + sub_err;
            compare_word("DELTA  ", `PT_DELTA + g, g, 3'd2, sub_err);
            errors_dlt = errors_dlt + sub_err;
            compare_word("Y_GATED", `PT_Y_GATED + g, g, 3'd4, sub_err);
            errors_ygat = errors_ygat + sub_err;
        end
        for (g = 0; g < D_MODEL/16; g = g + 1) begin
            compare_word("M_OUT  ", `PT_MAMBA_OUT + TC*(D_MODEL/16) + g, g, 3'd5, sub_err);
            errors_mout = errors_mout + sub_err;
        end

        total = errors_xi + errors_zg + errors_dlt + errors_ygat + errors_mout;
        $display("---- tb_Mamba_Top_DBG (t=%0d of T=%0d) ----", TC, T_TEST);
        $display("  M1A X_INNER  : %0d / %0d errors", errors_xi, D_INNER);
        $display("  M1B Z_GATE   : %0d / %0d errors", errors_zg, D_INNER);
        $display("  M5  DELTA    : %0d / %0d errors", errors_dlt, D_INNER);
        $display("  M7  Y_GATED  : %0d / %0d errors", errors_ygat, D_INNER);
        $display("  M8  MAMBA_OUT: %0d / %0d errors", errors_mout, D_MODEL);
        $display("%s", (total == 0) ? "PASS" : "FAIL");
        $finish;
    end

    initial begin
        #200000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
