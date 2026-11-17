# Out-of-context synthesis and implementation of one design for the Kria KV260 (xck26-sfvc784-2LV-c). Sourced by
# <design>/run_synth_kv260.tcl, which sets DESIGN_DIR. Out of context because Mamba_Top has ~600 ports (256-bit
# DMA buses), more than the package has pins.
#   -tclargs [period_ns = 4.0] [report_dir = synth_reports_kv260] [h_addr_w]
# h_addr_w sizes the scan state RAM (2^h_addr_w words per cluster) for the bench configs; default `H_ADDR_W (9).
# SYNTH_ONLY set in the environment stops after synthesis.

set TOP  Mamba_Top
set PART xck26-sfvc784-2LV-c
set CLK_PERIOD_NS [expr {$argc > 0 ? [lindex $argv 0] : 4.0}]
set REPORT_DIR    [file join $DESIGN_DIR [expr {$argc > 1 ? [lindex $argv 1] : "synth_reports_kv260"}]]
set COMMON_DIR    [file normalize [file join $DESIGN_DIR .. common]]
set GOLDEN_ROOT   [file normalize [file join $DESIGN_DIR .. goldens itmn]]
set GENERICS      [expr {$argc > 2 ? [list -generic H_ADDR_W=[lindex $argv 2] -generic H_DEPTH=[expr {1 << [lindex $argv 2]}]] : {}}]
set FREQ_MHZ      [format "%.1f" [expr {1000.0 / $CLK_PERIOD_NS}]]
file mkdir $REPORT_DIR

set SRC_V {}
foreach f {BRAM_256b Exp_LUT Silu_LUT Softplus_LUT RSqrt_ROM LUT_Bank Memory_Cluster H_RegFile Reduce16Wide
           Mamba_PE_v2 M_Cluster} {
    lappend SRC_V [file join $COMMON_DIR $f.v]
}
lappend SRC_V [file join $DESIGN_DIR src Mamba_Cluster.v] [file join $DESIGN_DIR src Mamba_Top.v]

puts "SYNTH [file tail $DESIGN_DIR] on $PART at $FREQ_MHZ MHz ($CLK_PERIOD_NS ns), tables from $GOLDEN_ROOT"
read_verilog $SRC_V

file delete -force [file join $DESIGN_DIR .Xil]
# Serial start-up: the parallel preparation step intermittently fails to find its files on Windows
set_param general.maxThreads 1

synth_design -top $TOP -part $PART -mode out_of_context -flatten_hierarchy rebuilt {*}$GENERICS \
    -include_dirs [list [file join $DESIGN_DIR src] $COMMON_DIR] \
    -verilog_define "GOLDEN_ROOT=\"$GOLDEN_ROOT\""

create_clock -name clk -period $CLK_PERIOD_NS [get_ports clk]
set_input_delay  -clock clk 0.0 [all_inputs]
set_output_delay -clock clk 0.0 [all_outputs]
# Reset, start, the DMA port and the block configuration are static while the accelerator computes; at integration
# the host interface (AXI) owns their timing
foreach p {rst start dma_write_en dma_target dma_addr dma_wdata dma_read_en dma_rtarget dma_raddr
           T_MAX CH_OUT CH_M DT_RANK N_STATE_GRP USE_M5 XP_OUT_GRP_IN
           PHASE_INC CH_IN FIN_EN POOL_OUT ENC_EN GAP_EN GAP_INV} {
    if {[llength [get_ports -quiet $p]] > 0} { set_false_path -from [get_ports $p] }
}
set_false_path -to [get_ports dma_rdata]

set ram_main [get_cells -quiet -hier -filter {NAME =~ *u_mem*/u_ram_main/ram_reg*}]
if {[llength $ram_main] > 0} { set_property RAM_STYLE ULTRA $ram_main }

write_checkpoint -force [file join $REPORT_DIR post_synth.dcp]
report_utilization    -file [file join $REPORT_DIR util_synth.rpt]
report_timing_summary -max_paths 10 -file [file join $REPORT_DIR timing_synth.rpt]
report_timing -max_paths 50 -nworst 1 -unique_pins -file [file join $REPORT_DIR timing_synth_top50.rpt]
if {[info exists ::env(SYNTH_ONLY)]} { puts "SYNTH_ONLY set: skipping implementation"; exit }

opt_design
place_design    -directive Explore
phys_opt_design -directive Explore
route_design    -directive Explore
phys_opt_design -directive AggressiveExplore
phys_opt_design -hold_fix

write_checkpoint -force [file join $REPORT_DIR post_route.dcp]
report_timing_summary -max_paths 20 -file [file join $REPORT_DIR timing_route.rpt]
report_timing -nworst 20 -delay_type max -slack_less_than 0.0 -file [file join $REPORT_DIR timing_violations.rpt]
report_utilization -file [file join $REPORT_DIR util_route.rpt]
report_power       -file [file join $REPORT_DIR power_route.rpt]

set util_str  [report_utilization    -return_string]
set time_str  [report_timing_summary -return_string]
set power_str [report_power          -return_string]
puts "RESULT [file tail $DESIGN_DIR] @ $FREQ_MHZ MHz ($CLK_PERIOD_NS ns)"
if {[regexp {WNS\(ns\)\s+[^\n]*\n[^\n]*\n\s*([-\d.]+)} $time_str -> wns]} {
    puts [format "  WNS   : %+.3f ns (%s), fmax %.1f MHz" $wns [expr {$wns >= 0 ? "MET" : "VIOLATED"}] \
          [expr {1000.0 / ($CLK_PERIOD_NS - $wns)}]]
}
foreach {label pat} {LUT {CLB LUTs\s*\|\s*(\d+)} FF {CLB Registers\s*\|\s*(\d+)} BRAM {Block RAM Tile\s*\|\s*([\d.]+)}
                     URAM {URAM\s*\|\s*(\d+)} DSP {DSPs\s*\|\s*(\d+)}} {
    if {[regexp $pat $util_str -> val]} { puts [format "  %-6s: %s" $label $val] }
}
if {[regexp {Total On-Chip Power \(W\)\s*\|\s*([\d.]+)} $power_str -> pwr]} { puts "  Power : $pwr W" }
puts "  Reports: $REPORT_DIR"
