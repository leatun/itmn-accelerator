`timescale 1ns/1ps
`include "_parameter.v"
`include "goldens.vh"

// Mamba phase of the 5 ITMN blocks back to back (the hardware is not reset between blocks). Each block's input is
// its golden P1 output; after done_all the whole mamba_out buffer (T x D_MODEL) is read back over DMA and compared
// with Mam_OutProj_FP. VERIFY_T limits every block to its first N timesteps.
module tb_Mamba_Top_5BLOCK_VERIFY;

    // Sized for block 4
    reg [15:0] wx_mem   [0:32767];
    reg [15:0] wz_mem   [0:32767];
    reg [15:0] wo_mem   [0:32767];
    reg [15:0] wdw_mem  [0:1023];
    reg [15:0] wxp_mem  [0:12287];
    reg [15:0] wdt_mem  [0:2047];
    reg [15:0] wA_mem   [0:4095];
    reg [15:0] gam_mem  [0:127];
    reg [15:0] bdw_mem  [0:255];
    reg [15:0] bdt_mem  [0:255];
    reg [15:0] Dp_mem   [0:255];
    reg [15:0] xn_mem   [0:63999];
    reg [15:0] mo_mem   [0:63999];

    localparam D_STATE = 16;
    localparam N_PAD   = 48;
    integer d_model, d_inner, dt_rank, n_t, t_tot;

    reg [8:0]  CH_OUT, CH_M;
    reg [7:0]  DT_RANK;
    reg [9:0]  T_MAX;

    reg clk = 0; reg rst = 1;
    always #5 clk = ~clk;

    reg          start = 0;
    wire         done_stage, done_all;
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
        .T_MAX(T_MAX), .PHASE_INC(1'b0), .CH_IN(9'd4), .FIN_EN(1'b0), .POOL_OUT(1'b0), .ENC_EN(1'b0), .GAP_EN(1'b0), .GAP_INV(16'd0),
        .CH_OUT(CH_OUT), .CH_M(CH_M), .DT_RANK(DT_RANK),
        .N_STATE_GRP(4'd1), .USE_M5(1'b1), .XP_OUT_GRP_IN(5'd3),
        .dma_write_en(dma_write_en), .dma_target(dma_target),
        .dma_addr(dma_addr), .dma_wdata(dma_wdata),
        .dma_read_en(dma_read_en), .dma_rtarget(dma_rtarget),
        .dma_raddr(dma_raddr), .dma_rdata(dma_rdata)
    );

    integer cycles [0:4];
    integer steps  [0:4];
    integer mism   [0:4];
    integer bk, g, c_in, k, t_cur, lane, shown, total_mm, cyc0;
    reg [15:0]  exp_val, got_val;
    reg [255:0] word_tmp;
    reg [8*18:1] bdir;

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

    task load_block;
        input integer b;
        begin
            case (b)
                0: bdir = "/block_00_layer00/";
                1: bdir = "/block_01_layer01/";
                2: bdir = "/block_02_layer03/";
                3: bdir = "/block_03_layer04/";
                default: bdir = "/block_04_layer06/";
            endcase
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_InProj_X.txt"},      wx_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_InProj_Z.txt"},      wz_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_OutProj.txt"},       wo_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Conv.txt"},          wdw_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_XProj.txt"},         wxp_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_DtProj.txt"},        wdt_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_A_signed.txt"},        wA_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_W_Norm.txt"},          gam_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_B_Conv.txt"},          bdw_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_B_DtProj.txt"},        bdt_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_D_param.txt"},         Dp_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "P1_Output_Golden_FP.txt"}, xn_mem);
            $readmemh({`GOLDEN_ROOT, bdir, "Mam_OutProj_FP.txt"},      mo_mem);
            d_model = (b == 4) ? 128 : 64;
            d_inner = 2 * d_model;
            dt_rank = d_model / 16;
            t_tot   = (b < 2) ? 1000 : (b < 4) ? 500 : 250;
            n_t     = t_tot;
`ifdef VERIFY_T
            if (n_t > `VERIFY_T) n_t = `VERIFY_T;
`endif
            CH_OUT  = d_model / 16;
            CH_M    = d_inner / 16;
            DT_RANK = dt_rank;
            T_MAX   = n_t;
        end
    endtask

    task preload_block;
        begin
            for (g = 0; g < d_inner/16; g = g + 1)
                for (c_in = 0; c_in < d_model; c_in = c_in + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wx_mem[(g*16+lane)*d_model + c_in];
                    dma_wr_split(`W_INPROJ_X_BASE, g, d_model, c_in, word_tmp);
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wz_mem[(g*16+lane)*d_model + c_in];
                    dma_wr_split(`W_INPROJ_Z_BASE, g, d_model, c_in, word_tmp);
                end
            for (g = 0; g < d_model/16; g = g + 1)
                for (c_in = 0; c_in < d_inner; c_in = c_in + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wo_mem[(g*16+lane)*d_inner + c_in];
                    dma_wr_split(`W_OUTPROJ_BASE, g, d_inner, c_in, word_tmp);
                end
            for (g = 0; g < N_PAD/16; g = g + 1)
                for (c_in = 0; c_in < d_inner; c_in = c_in + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wxp_mem[(g*16+lane)*d_inner + c_in];
                    dma_wr_split(`W_XPROJ_BASE, g, d_inner, c_in, word_tmp);
                end
            for (g = 0; g < d_inner/16; g = g + 1) begin
                for (k = 0; k < 4; k = k + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wdw_mem[(g*16+lane)*4 + k];
                    dma_wr_both(`W_DW_BASE + g*4 + k, word_tmp);
                end
                for (k = 0; k < dt_rank; k = k + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = wdt_mem[(g*16+lane)*dt_rank + k];
                    dma_wr_both(`W_DTPROJ_BASE + g*dt_rank + k, word_tmp);
                end
            end
            for (c_in = 0; c_in < d_inner; c_in = c_in + 1) begin
                for (lane = 0; lane < D_STATE; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = wA_mem[c_in*D_STATE + lane];
                dma_wr_both(`W_A_BASE + c_in, word_tmp);
            end

            for (g = 0; g < d_model/16; g = g + 1) begin
                for (lane = 0; lane < 16; lane = lane + 1)
                    word_tmp[lane*16 +: 16] = gam_mem[g*16 + lane];
                dma_wr(3'd3, `C_W_NORM_BASE + g, word_tmp);
            end
            for (g = 0; g < d_inner/16; g = g + 1) begin
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

            for (t_cur = 0; t_cur < n_t; t_cur = t_cur + 1)
                for (g = 0; g < d_model/16; g = g + 1) begin
                    for (lane = 0; lane < 16; lane = lane + 1)
                        word_tmp[lane*16 +: 16] = xn_mem[(g*16+lane)*t_tot + t_cur];
                    dma_wr(3'd0, `PT_INPUT + t_cur*(d_model/16) + g, word_tmp);
                end
            @(negedge clk); dma_write_en = 0;
        end
    endtask

    // mamba_out is mirrored, so cluster 0's RAM (target 0) holds all of it; read data is valid 2 cycles after the address
    task verify_block;
        begin
            mism[bk] = 0;
            shown    = 0;
            @(negedge clk); @(negedge clk);
            for (t_cur = 0; t_cur < n_t; t_cur = t_cur + 1)
                for (g = 0; g < d_model/16; g = g + 1) begin
                    @(negedge clk);
                    dma_read_en = 1;
                    dma_rtarget = 3'd0;
                    dma_raddr   = `PT_MAMBA_OUT + t_cur * (d_model/16) + g;
                    @(negedge clk);
                    @(negedge clk);
                    for (lane = 0; lane < 16; lane = lane + 1) begin
                        exp_val = mo_mem[(g*16 + lane) * t_tot + t_cur];
                        got_val = dma_rdata[lane*16 +: 16];
                        if (exp_val !== got_val) begin
                            mism[bk] = mism[bk] + 1;
                            if (shown < 10) begin
                                $display("  MISMATCH bk=%0d t=%0d g=%0d lane=%0d  exp=%04h  got=%04h",
                                         bk, t_cur, g, lane, exp_val, got_val);
                                shown = shown + 1;
                            end
                        end
                    end
                end
            @(negedge clk);
            dma_read_en = 0;
            $display("  VERIFY bk=%0d:  %s  (%0d mismatches out of %0d values)",
                     bk, (mism[bk] == 0) ? "PASS" : "FAIL", mism[bk], n_t * d_model);
        end
    endtask

    initial begin
        rst = 1;
        @(posedge clk); @(posedge clk); @(posedge clk);
        @(negedge clk); rst = 0;
        @(posedge clk);

        for (bk = 0; bk < 5; bk = bk + 1) begin
            $display("======== BLOCK %0d ========", bk);
            load_block(bk);
            preload_block;
            @(negedge clk); start = 1;
            @(negedge clk); start = 0;
            cyc0 = $time / 10;
            wait (done_all == 1'b1);
            cycles[bk] = $time / 10 - cyc0;
            steps[bk]  = n_t;
            $display("  compute = %0d cycles", cycles[bk]);
            verify_block;
            @(negedge clk); @(negedge clk); @(negedge clk);
        end

        $display("========== 5-BLOCK VERIFY SUMMARY ==========");
        $display("| Blk |  T   | Compute  |  cyc/t | Mismatches |");
        $display("|-----|------|----------|--------|------------|");
        total_mm = 0;
        for (bk = 0; bk < 5; bk = bk + 1) begin
            $display("| B%0d  | %4d | %8d | %6d | %10d |", bk, steps[bk], cycles[bk], cycles[bk] / steps[bk], mism[bk]);
            total_mm = total_mm + mism[bk];
        end
        $display("%s", (total_mm == 0) ? "PASS" : "FAIL");
        $finish;
    end

    initial begin
        #500000000; #500000000; #500000000; #500000000; #500000000;
        #500000000; #500000000; #500000000; #500000000; #500000000;
        $display("ERROR: timeout");
        $finish;
    end

endmodule
