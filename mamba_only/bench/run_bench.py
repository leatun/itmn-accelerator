"""Cycle benchmark of the mamba-only design at the layer sizes in configs.csv (xsim, Vivado 2022.1).

    python bench/run_bench.py [--only NAME,...] [--jobs N] [--clock-mhz 250]

Run from mamba_only/. Compiles once, elaborates tb_bench per config with that config's parameters, runs it and
writes bench/results.csv. A config passes when its compute and scan cycles per token equal the recorded ones
(ref_cyc, ref_m6). Derived columns, all for one token through n_layer layers at the given clock:
  ms_compute  compute only
  ms_dma      compute plus dma_cyc per layer, a model of streaming the layer's weights over a 256-bit port at one
              word per cycle (the weights do not fit on chip at these sizes)
  bidir_cyc   Vim configs: compute + one more scan per token (backward direction), an upper bound
"""
import argparse, csv, os, re, shutil, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN = HERE.parent
COMMON = DESIGN.parent / 'common'
RTL = [COMMON / f'{m}.v' for m in ('BRAM_256b', 'Exp_LUT', 'Silu_LUT', 'Softplus_LUT', 'RSqrt_ROM', 'LUT_Bank',
                                   'Memory_Cluster', 'H_RegFile', 'Reduce16Wide', 'Mamba_PE_v2', 'M_Cluster')]
RTL += [DESIGN / 'src' / 'Mamba_Cluster.v', DESIGN / 'src' / 'Mamba_Top.v', HERE / 'tb_bench.v']


def tool(name):
    path = shutil.which(name) or shutil.which(name + '.bat')
    if not path and os.environ.get('XILINX_VIVADO'):
        path = str(Path(os.environ['XILINX_VIVADO']) / 'bin' / (name + ('.bat' if os.name == 'nt' else '')))
    if not path:
        sys.exit(f'{name} not found: put Vivado/bin on PATH or set XILINX_VIVADO')
    return path


def run(cmd, log):
    line = ' '.join(f'"{c}"' if '=' in str(c) or ' ' in str(c) else str(c) for c in cmd)
    with open(log, 'w') as f:   # on Windows the tools are .bat files and cmd splits an unquoted argument at '='
        rc = subprocess.call(line if os.name == 'nt' else cmd, cwd=DESIGN, stdout=f, stderr=subprocess.STDOUT)
    if rc:
        sys.exit(f'failed ({rc}): {line}, see {log}')


def bench(cfg, logdir):
    name = cfg['name']
    generics = {'CH_OUT': cfg['ch_out'], 'CH_M': cfg['ch_m'], 'DT_RANK': cfg['dt_rank'], 'NSG': cfg['nsg'],
                'USE_M5': cfg['use_m5'], 'XP_GRP': cfg['xp_grp'], 'H_ADDR_W': cfg['h_addr_w']}
    cmd = [tool('xelab'), 'tb_bench', '-s', f'bench_{name}', '-timescale', '1ns/1ps']
    for k, v in generics.items():
        cmd += ['-generic_top', f'{k}={v}']
    run(cmd, logdir / f'{name}_xelab.log')
    log = logdir / f'{name}.log'
    run([tool('xsim'), f'bench_{name}', '-R'], log)
    text = log.read_text()
    got = {k: int(v) for k, v in re.findall(r'\[BENCH\] (compute_per_token|m6_per_token) (\d+)', text)}
    if len(got) != 2:
        sys.exit(f'{name}: no result in {log}')
    return got['compute_per_token'], got['m6_per_token']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', help='comma-separated config names')
    ap.add_argument('--jobs', type=int, default=2)
    ap.add_argument('--clock-mhz', type=float, default=250.0)
    a = ap.parse_args()

    cfgs = list(csv.DictReader(open(HERE / 'configs.csv')))
    if a.only:
        keep = set(a.only.split(','))
        cfgs = [c for c in cfgs if c['name'] in keep]
    logdir = HERE / 'logs'
    logdir.mkdir(exist_ok=True)

    # cwd is the design folder, so the activation tables resolve through goldens.vh's default ../goldens/itmn
    run([tool('xvlog'), '-i', 'src', '-i', '../common'] + [str(p) for p in RTL],
        logdir / 'xvlog.log')
    with ThreadPoolExecutor(a.jobs) as ex:
        results = list(ex.map(lambda c: bench(c, logdir), cfgs))

    rows, fails = [], 0
    for c, (cyc, m6) in zip(cfgs, results):
        ok = cyc == int(c['ref_cyc']) and m6 == int(c['ref_m6'])
        fails += not ok
        n, dma = int(c['n_layer']), int(c['dma_cyc'])
        ms = lambda x: f'{x * n / (a.clock_mhz * 1e3):.3f}'
        rows.append({'name': c['name'], 'model': c['model'], 'adapters': c['adapters'], 'compute_cyc': cyc,
                     'm6_cyc': m6, 'ref_cyc': c['ref_cyc'], 'match': 'yes' if ok else 'NO',
                     'ms_compute': ms(cyc), 'ms_dma': ms(cyc + dma),
                     'bidir_cyc': cyc + m6 if c['name'].startswith('VIM') else ''})
        print(f"{c['name']:13s} compute {cyc:8d} (ref {c['ref_cyc']:>8s})  m6 {m6:7d}  {'ok' if ok else 'MISMATCH'}")
    out = HERE / 'results.csv'
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='
')
        w.writeheader()
        w.writerows(rows)
    print(f'{len(rows) - fails}/{len(rows)} match the recorded cycles; {out}')
    sys.exit(1 if fails else 0)


if __name__ == '__main__':
    main()
