// Root of the files read with $readmemh (activation tables, testbench data), relative to the simulator's working
// directory (a design folder). Override with +define+GOLDEN_ROOT=\"<path>\"; the synth scripts pass an absolute one.
`ifndef GOLDEN_ROOT
  `define GOLDEN_ROOT "../goldens/itmn"
`endif
