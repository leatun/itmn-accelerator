`include "_parameter.v"

// Memories of one cluster:
//   ram_main   URAM, port A write (core > queued mirror write > new mirror write > DMA), port B read (core or DMA)
//   ram_weight BRAM, two reads per cycle for MAC2 (port B, and port A when the DMA is not writing)
//   ram_const  LUTRAM
// Mirror writes from the peer that collide with a local write wait in a queue and drain in the next cycles
// without a local write. DMA targets: 0 ram_main, 2 ram_weight, 3 ram_const.
module Memory_Cluster (
    input                              clk,
    input                              reset,

    // Core
    input  [`MAIN_ADDR_W-1:0]          core_rd_addr,
    output [`NUM_LANES*`DATA_W-1:0]    core_rd_data,
    input                              core_we,
    input  [`MAIN_ADDR_W-1:0]          core_wr_addr,
    input  [`NUM_LANES*`DATA_W-1:0]    core_wr_data,

    // Mirror write from the peer
    input                              xc_we,
    input  [`MAIN_ADDR_W-1:0]          xc_wr_addr,
    input  [`NUM_LANES*`DATA_W-1:0]    xc_wr_data,

    // Weights
    input  [`MAIN_ADDR_W-1:0]          w_rd_addr,
    output [`NUM_LANES*`DATA_W-1:0]    w_rd_data,
    input  [`MAIN_ADDR_W-1:0]          w_rd_addr2,
    output [`NUM_LANES*`DATA_W-1:0]    w_rd_data2,

    // Constants
    input  [`MAIN_ADDR_W-1:0]          const_rd_addr,
    output [`NUM_LANES*`DATA_W-1:0]    const_rd_data,

    // Host DMA
    input                              dma_write_en,
    input  [2:0]                       dma_target,
    input  [`MAIN_ADDR_W-1:0]          dma_addr,
    input  [`NUM_LANES*`DATA_W-1:0]    dma_wdata,
    input                              dma_read_en,
    input  [2:0]                       dma_rtarget,
    input  [`MAIN_ADDR_W-1:0]          dma_raddr,
    output [`NUM_LANES*`DATA_W-1:0]    dma_rdata
);

    // Queue depth: block 4 streams 8 mirrored writes per cluster back to back
    localparam XC_FIFO_DEPTH   = 8;
    localparam XC_FIFO_LOG     = 3;

    reg [`MAIN_ADDR_W-1:0]        xc_fifo_addr [0:XC_FIFO_DEPTH-1];
    reg [`NUM_LANES*`DATA_W-1:0]  xc_fifo_data [0:XC_FIFO_DEPTH-1];
    reg [XC_FIFO_LOG:0]           xc_fifo_count;
    reg [XC_FIFO_LOG-1:0]         xc_fifo_head;
    reg [XC_FIFO_LOG-1:0]         xc_fifo_tail;

    wire xc_fifo_empty = (xc_fifo_count == {(XC_FIFO_LOG+1){1'b0}});

    wire dma_main_we    = dma_write_en && (dma_target == 3'd0);
    wire replay_pending = !xc_fifo_empty && !core_we;
    wire accept_new_xc  = xc_we && !core_we && xc_fifo_empty;

    wire we_main    = core_we || replay_pending || accept_new_xc ||
                      (dma_main_we && !core_we && xc_fifo_empty && !xc_we);

    wire [`MAIN_ADDR_W-1:0] addr_main_wr =
        core_we         ? core_wr_addr                     :
        replay_pending  ? xc_fifo_addr[xc_fifo_head]       :
        accept_new_xc   ? xc_wr_addr                       :
                          dma_addr;

    wire [`NUM_LANES*`DATA_W-1:0] din_main =
        core_we         ? core_wr_data                     :
        replay_pending  ? xc_fifo_data[xc_fifo_head]       :
        accept_new_xc   ? xc_wr_data                       :
                          dma_wdata;

    wire do_enqueue = xc_we && !accept_new_xc;
    wire do_dequeue = replay_pending;

    integer xc_i;
    always @(posedge clk or posedge reset) begin
        if (reset) begin
            xc_fifo_count <= {(XC_FIFO_LOG+1){1'b0}};
            xc_fifo_head  <= {XC_FIFO_LOG{1'b0}};
            xc_fifo_tail  <= {XC_FIFO_LOG{1'b0}};
            for (xc_i = 0; xc_i < XC_FIFO_DEPTH; xc_i = xc_i + 1) begin
                xc_fifo_addr[xc_i] <= {`MAIN_ADDR_W{1'b0}};
                xc_fifo_data[xc_i] <= {`NUM_LANES*`DATA_W{1'b0}};
            end
        end else begin
            if (do_enqueue) begin
                xc_fifo_addr[xc_fifo_tail] <= xc_wr_addr;
                xc_fifo_data[xc_fifo_tail] <= xc_wr_data;
                xc_fifo_tail               <= xc_fifo_tail + {{(XC_FIFO_LOG-1){1'b0}}, 1'b1};
            end
            if (do_dequeue)
                xc_fifo_head <= xc_fifo_head + {{(XC_FIFO_LOG-1){1'b0}}, 1'b1};
            case ({do_enqueue, do_dequeue})
                2'b10:   xc_fifo_count <= xc_fifo_count + {{XC_FIFO_LOG{1'b0}}, 1'b1};
                2'b01:   xc_fifo_count <= xc_fifo_count - {{XC_FIFO_LOG{1'b0}}, 1'b1};
                default: ;
            endcase
        end
    end

    wire [`MAIN_ADDR_W-1:0] addr_main_rd =
        (dma_read_en && dma_rtarget == 3'd0) ? dma_raddr : core_rd_addr;

    wire [`NUM_LANES*`DATA_W-1:0] out_ram_main;

    BRAM_256b #(
        .ADDR_WIDTH(`MAIN_ADDR_W),
        .RAM_STYLE ("ultra"),
        .DEPTH     (`PT_MAIN_DEPTH)
    ) u_ram_main (
        .clk    (clk),
        .we_a   (we_main),
        .addr_a (addr_main_wr),
        .din_a  (din_main),
        .dout_a (),
        .addr_b (addr_main_rd),
        .dout_b (out_ram_main)
    );

    wire we_weight = dma_write_en && (dma_target == 3'd2);

    wire [`W_MEM_ADDR_W-1:0] addr_weight_a =
        we_weight ? dma_addr[`W_MEM_ADDR_W-1:0] : w_rd_addr2[`W_MEM_ADDR_W-1:0];

    wire [`W_MEM_ADDR_W-1:0] addr_weight_b =
        (dma_read_en && dma_rtarget == 3'd2) ? dma_raddr[`W_MEM_ADDR_W-1:0]
                                             : w_rd_addr [`W_MEM_ADDR_W-1:0];

    wire [`NUM_LANES*`DATA_W-1:0] out_weight_a, out_weight_b;

    // Cascades of at most 4 BRAMs: the natural 8-deep cascade was a critical path
    BRAM_256b #(
        .ADDR_WIDTH     (`W_MEM_ADDR_W),
        .RAM_STYLE      ("block"),
        .DEPTH          (`W_MEM_DEPTH),
        .CASCADE_HEIGHT (4)
    ) u_ram_weight (
        .clk    (clk),
        .we_a   (we_weight),
        .addr_a (addr_weight_a),
        .din_a  (dma_wdata),
        .dout_a (out_weight_a),
        .addr_b (addr_weight_b),
        .dout_b (out_weight_b)
    );

    localparam CONST_ADDR_W = `C_MEM_ADDR_W;
    wire we_const = dma_write_en && (dma_target == 3'd3);

    wire [CONST_ADDR_W-1:0] addr_const_a =
        we_const ? dma_addr[CONST_ADDR_W-1:0] : {CONST_ADDR_W{1'b0}};

    wire [CONST_ADDR_W-1:0] addr_const_b =
        (dma_read_en && dma_rtarget == 3'd3) ? dma_raddr[CONST_ADDR_W-1:0]
                                             : const_rd_addr[CONST_ADDR_W-1:0];

    wire [`NUM_LANES*`DATA_W-1:0] out_const;

    BRAM_256b #(
        .ADDR_WIDTH(CONST_ADDR_W),
        .RAM_STYLE ("distributed"),
        .DEPTH     (`C_MEM_DEPTH)
    ) u_ram_const (
        .clk    (clk),
        .we_a   (we_const),
        .addr_a (addr_const_a),
        .din_a  (dma_wdata),
        .dout_a (),
        .addr_b (addr_const_b),
        .dout_b (out_const)
    );

    assign core_rd_data  = out_ram_main;
    assign w_rd_data     = out_weight_b;
    assign w_rd_data2    = out_weight_a;
    assign const_rd_data = out_const;

    assign dma_rdata = (dma_rtarget == 3'd2) ? out_weight_b :
                       (dma_rtarget == 3'd3) ? out_const    :
                                                out_ram_main;

endmodule
