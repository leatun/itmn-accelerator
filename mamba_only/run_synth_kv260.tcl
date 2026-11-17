# vivado -mode batch -source run_synth_kv260.tcl [-tclargs period_ns report_dir]
set DESIGN_DIR [file dirname [file normalize [info script]]]
source [file join $DESIGN_DIR .. synth_kv260.tcl]
