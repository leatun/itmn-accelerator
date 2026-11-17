# Mamba phase of block 0, every intermediate of timestep DBG_T-1 (default 1) checked.
# vsim -c -do "set DBG_T 2; do run_dbg_sim.do"
set defs ""
if {[info exists DBG_T]} { append defs " +define+DBG_T=$DBG_T" }
set DEFAULT_LIB work_dbg
source ../sim_common.tcl
run_tb tb_Mamba_Top_DBG
