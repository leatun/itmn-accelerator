# Raw 12-lead ECG to logits; ITM_BK = last block to run (default 4).
# vsim -c -do "do run_e2e_sim.do"
set defs ""
if {[info exists ITM_BK]} { append defs " +define+E2E_LAST=$ITM_BK" }
set DEFAULT_LIB work_e2e
source ../sim_common.tcl
run_tb tb_ITM_E2E
