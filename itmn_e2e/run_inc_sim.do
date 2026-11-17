# Inception phase of block ITM_BK (default 0), every branch checked; INC_T = first N timesteps.
# vsim -c -do "set ITM_BK 4; set INC_T 20; do run_inc_sim.do"
set defs ""
if {[info exists ITM_BK]} { append defs " +define+ITM_BK=$ITM_BK" }
if {[info exists INC_T]} { append defs " +define+INC_T=$INC_T" }
set DEFAULT_LIB work_inc
source ../sim_common.tcl
run_tb tb_ITM_INC
