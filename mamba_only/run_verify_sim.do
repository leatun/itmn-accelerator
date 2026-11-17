# Mamba phase of all 5 blocks, block output checked; VERIFY_T = first N timesteps per block (default all).
# vsim -c -do "set VERIFY_T 20; do run_verify_sim.do"
set defs ""
if {[info exists VERIFY_T]} { append defs " +define+VERIFY_T=$VERIFY_T" }
set DEFAULT_LIB work_5bv
source ../sim_common.tcl
run_tb tb_Mamba_Top_5BLOCK_VERIFY
