# Learning when to act: self-triggered RL for TCP congestion control

Code, trained policies and raw results for the paper

> S. Bhagi and S. Narang, *Learning When to Act: Self-Triggered Reinforcement Learning for Supervising TCP Congestion Control*, 2026. Submitted to *Information Sciences*. Preprint: arXiv:XXXX.XXXXX

A learned supervisor sets the parameters of a lightweight congestion-control executor that runs in the transport stack. Most designs call the supervisor every few tens of milliseconds. This repository contains everything needed to compare that with alternatives under identical conditions:

- fixed decision periods: **FP-100**, **FP-400** and **FP-800**;
- a learned watcher in the style of SPINE: **WT-100**;
- our **self-triggered** supervisor (**ST**), which chooses its own holding time (0.1–1.6 s) and can be interrupted by a Page–Hinkley change detector inside the executor.

The comparison runs in two places:

1. **OMNeT++/INET simulation**, used to train all supervisors and for the main study.
2. **The Linux kernel.** The executor is ported to an eBPF `struct_ops` congestion-control module and tested on a Mahimahi-emulated bottleneck against the kernel's CUBIC, BBR and Reno. The supervisors trained in simulation are used without retraining.

## Layout

The tree mirrors the two working directories used during the study, `~/sim` and `~/emu`. Unpack it into your home directory and the scripts run without path changes.

| Path | Contents |
|---|---|
| `sim/stcc/src/StTcp.{h,cc}` | INET congestion-control module: the delay-tolerance executor, Karn-safe RTT sampling, the Page–Hinkley tripwire and the socket link to the supervisor |
| `sim/stcc/simulations/` | Dumbbell network (`Dumbbell.ned`) and base `omnetpp.ini` |
| `sim/stcc/py/scenario.py` | Seeded scenario generator: capacity process, RTT, buffer, UDP and TCP cross-traffic |
| `sim/stcc/py/runtime.py` | Runs one simulated episode; observation features and reward |
| `sim/stcc/py/agent.py` | PPO with hybrid actions and time-based (semi-Markov) discounting |
| `sim/stcc/py/train.py` | Training driver (`--mode stl\|st\|fp\|wt`) |
| `sim/stcc/py/evaluate.py` | Simulation suites: `main`, `lambda`, `steps`, `ood`, `frontier`, `fair`, `friendly` |
| `sim/stcc/py/static_grid.py`, `static_front.py` | Static-parameter grid search and the oracle static front |
| `sim/stcc/py/analysis.py` | Simulation figures, tables and the LaTeX number macros |
| `sim/runs/<run>/` | Trained policy (`final.pt`), arguments (`args.json`) and training log (`train.csv`) for every run |
| `sim/eval*/` | Raw per-episode simulation records (CSV) and time series (JSON) |
| `emu/bpf/stcc.bpf.c` | **Linux executor**: eBPF `struct_ops` TCP congestion control (`bpf_stcc`) with the window rule, delay-based slow-start exit, fixed-point Page–Hinkley tripwire and competition mode |
| `emu/bpf/stcc_link.c` | User-space agent: loads the module and relays reports and parameters between kernel flows and the supervisor |
| `emu/bpf/build.sh` | Builds both |
| `emu/py/emu.py` | Runs one scenario on Mahimahi with real Linux TCP and computes all metrics |
| `emu/py/emu_sender.py` | Traffic generator that runs inside the emulator |
| `emu/py/emu_eval.py` | Emulation suites: `main`, `ood`, `fair`, `friendly_cubic`, `friendly_bbr` (resumable) |
| `emu/py/emu_analysis.py` | Statistics used in the paper: per-scenario seed means, paired Wilcoxon tests, bootstrap CIs |
| `emu/paper/make_linux_tables.py` | Generates the Linux tables and number macros of the paper from `emu/results` |
| `emu/results/` | Raw per-episode emulation records. `superseded/` holds earlier runs that the paper does not use: before the overlap-window metric, and the first version of competition mode. |

Naming: in code the proposed supervisor is mode `stl` (holds 0.1–1.6 s) and the short-hold ablation is mode `st` (0.05–0.8 s). The paper calls them ST and ST-short. In the emulation CSVs, `STL-C` and `FP-800-C` denote runs with competition mode enabled.

## Requirements

**Simulation**
- OMNeT++ 6.0.3 and INET 4.5, release build. Only the TcpInet, TcpCommon, Ipv4, Ppp, Udp, Queueing and ProtocolElement features are needed.
- Python 3.11+ with PyTorch, NumPy, pandas, SciPy and matplotlib. Tested with Python 3.13, PyTorch 2.15, NumPy 2.5, pandas 3.0 and SciPy 1.18.

**Linux emulation**, in addition
- Linux 6.x with BTF and BPF `struct_ops` congestion control (tested on 6.18), and root access.
- clang 15 or later (tested with 18), libelf, zlib, and bpftool/libbpf (https://github.com/libbpf/bpftool).
- Mahimahi (https://github.com/ravinet/mahimahi), built with `-std=c++20` on recent compilers.
- An unprivileged user `mm`; Mahimahi refuses to run as root.
- `net.ipv4.ip_forward=1`.

## Build

```bash
# simulation module
source ~/sim/omnetpp-6.0.3/setenv
cd ~/sim/stcc/src
opp_makemake -f --deep -s -o stcc -O out -KINET_PROJ=$HOME/sim/inet4.5 -DINET_IMPORT \
  -I'$(INET_PROJ)/src' -L'$(INET_PROJ)/src' -lINET'$(D)'
make MODE=release

# Linux executor
BPFTOOL_SRC=~/emu/bpftool/src ~/emu/bpf/build.sh
```

Set `INET_DIR` and `OPP_BIN` if INET and OMNeT++ are not under `~/sim/`.

## Reproduce

**Training** (each schedule with seeds 1–3; the trained policies are already in `sim/runs`):
```bash
cd ~/sim/stcc/py
python3 train.py --name stl_c0.1_s1 --mode stl --cost 0.1 --seed 1 --episodes 1500 --port 5301   # ST
python3 train.py --name st_c0.1_s1  --mode st  --cost 0.1 --seed 1 --episodes 1500 --port 5304   # ST-short
python3 train.py --name fp800_s1    --mode fp --hold 0.8 --seed 1 --episodes 1500 --port 5302   # FP-800
python3 train.py --name wt_c0.1_s1  --mode wt --hold 0.1 --cost 0.1 --seed 1 --episodes 1500 --port 5303   # WT-100
```

**Simulation results:**
```bash
python3 evaluate.py --suite main --n 40
python3 evaluate.py --suite ood --n 20
python3 analysis.py          # figures, tables and macros
```

**Linux results** (as root):
```bash
sysctl -w net.ipv4.ip_forward=1
cd ~/emu/py
python3 smoke.py                                       # one ST and one CUBIC episode
python3 emu_eval.py --suite main --n 40                # all schemes
python3 emu_eval.py --suite main --n 40 --methods comp # competition-mode variants
python3 emu_eval.py --suite ood --n 20
python3 emu_eval.py --suite friendly_cubic --n 12 --methods all+comp
python3 emu_eval.py --suite friendly_bbr   --n 12 --methods all+comp
python3 emu_eval.py --suite fair --n 9     --methods all+comp
python3 ../paper/make_linux_tables.py <output dir>     # Linux tables and macros
```
Runs are resumable: every episode is appended to `~/emu/results/<suite>.csv` and skipped on restart. An episode takes about 20–40 s of wall-clock time. If a run is killed hard, a leftover `stcc_link` keeps the module registered ("attach failed: File exists"); kill it before restarting.

**Determinism.** Every episode is set by its seed. Training seeds are `1e6*s + i` (run seed `s`, episode `i`); evaluation seeds start at `9e6`. In simulation the seed fixes the path and OMNeT++'s random streams, so episodes replay exactly. In emulation the path is identical but kernel timing is not, so results agree to within run-to-run noise.

## Notes
- Several training runs (`fp800_s1`, `fp400_s1..3`, `stl_c0.1_s1..3`, `st_c0.3_s1`) were resumed from their last 200-episode checkpoint after hitting a compute time limit. The optimiser state was re-initialised and episodes after the checkpoint were re-run.
- INET's Reno and NewReno models are excluded from the simulated out-of-distribution suite because individual episodes took several minutes of wall-clock time.
- The competition-mode constants (`COMP_K`, `COMP_PROBE`, `COMP_DROP_DIV`, `COMP_EXIT_DIV`, `COMP_BETA_MILLI` in `stcc.bpf.c`) were set by hand on four scenarios and not tuned further. The mode is off by default; enable it with `stcc_link --comp 1`.

## Licence
- Code: MIT (`LICENSE`).
- `emu/bpf/stcc.bpf.c`: GPL-2.0, as required for kernel BPF programs using GPL-only helpers (`LICENSES/GPL-2.0.txt`).
- Data (`sim/runs`, `sim/eval*`, `emu/results`): CC BY 4.0 (`LICENSES/CC-BY-4.0.txt`).
- OMNeT++, INET, Mahimahi and libbpf are not included and keep their own licences.

## Citation
```bibtex
@misc{bhagi2026learning,
  title  = {Learning When to Act: Self-Triggered Reinforcement Learning for Supervising {TCP} Congestion Control},
  author = {Bhagi, Samaksh and Narang, Sudha},
  year   = {2026},
  eprint = {XXXX.XXXXX},
  archivePrefix = {arXiv},
  primaryClass  = {cs.NI}
}
```

Contact: Samaksh Bhagi, samakshb.research@gmail.com
