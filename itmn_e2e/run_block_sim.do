# One ITM block, both phases (+ GAP and FC on block 4); BLK_T = first N timesteps.
# vsim -c -do "set ITM_BK 4; set BLK_T 20; do run_block_sim.do"
set defs ""
if {[info exists ITM_BK]} { append defs " +define+ITM_BK=$ITM_BK" }
if {[info exists BLK_T]} { append defs " +define+BLK_T=$BLK_T" }
set DEFAULT_LIB work_blk
source ../sim_common.tcl
run_tb tb_ITM_BLOCK
