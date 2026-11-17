# Shared by the run_*_sim.do scripts; run them from a design folder (itmn_e2e/ or mamba_only/).
# Optional Tcl variables: GOLD_DIR = golden root (default ../goldens/itmn), LIBN = work library.
if {![info exists defs]}    { set defs "" }
if {[info exists GOLD_DIR]} { append defs " +define+GOLDEN_ROOT=\\\"$GOLD_DIR\\\"" }
if {![info exists LIBN]}    { set LIBN $DEFAULT_LIB }

set RTL "../common/BRAM_256b.v ../common/Exp_LUT.v ../common/Silu_LUT.v ../common/Softplus_LUT.v \
         ../common/RSqrt_ROM.v ../common/LUT_Bank.v ../common/Memory_Cluster.v ../common/H_RegFile.v \
         ../common/Reduce16Wide.v ../common/Mamba_PE_v2.v ../common/M_Cluster.v \
         src/Mamba_Cluster.v src/Mamba_Top.v"

proc run_tb {tb} {
    global defs RTL LIBN
    vlib $LIBN
    eval vlog -work $LIBN +incdir+src +incdir+../common -O0 $defs $RTL tb/$tb.v
    vsim -c -lib $LIBN -voptargs=+acc $tb
    run -all
    quit -f
}
