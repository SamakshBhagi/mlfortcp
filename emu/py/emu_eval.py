"""Emulation suites on real Linux TCP (Mahimahi). Resumable: every episode is
appended to <out>/<suite>.csv and skipped on restart.

    python3 emu_eval.py --suite main --n 40 [--budget 6900]
Scenario-major order, so every prefix of the run is a balanced, paired set.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emu import run_episode, RUNS  # noqa: E402
from scenario import Scenario, sample_train, sample_ood, sample_change_rate  # noqa: E402

EVAL_SEED0 = 9_000_000
OUT = os.path.expanduser("~/emu/results")


def rl(name, run, mode, hold=0.1, tripwire=None):
    return dict(name=name, kind="rl", run=run, mode=mode, hold=hold, tripwire=tripwire, cost=0.1)


def methods(which="all"):
    m = [dict(name="CUBIC", kind="tcp", cc="cubic"),
         dict(name="BBR", kind="tcp", cc="bbr"),
         dict(name="Reno", kind="tcp", cc="reno")]
    for s in (1, 2, 3):
        m += [rl(f"STL#s{s}", f"stl_c0.1_s{s}", "stl", 0.1, True),
              rl(f"FP-100#s{s}", f"fp100_s{s}", "fp", 0.1, None),
              rl(f"FP-800#s{s}", f"fp800_s{s}", "fp", 0.8, None)]
        if which == "all":
            m += [rl(f"FP-400#s{s}", f"fp400_s{s}", "fp", 0.4, None),
                  rl(f"ST#s{s}", f"st_c0.1_s{s}", "st", 0.1, True),
                  rl(f"WT-100#s{s}", f"wt_c0.1_s{s}", "wt", 0.1, None),
                  rl(f"STL-noTW#s{s}", f"stl_c0.1_s{s}", "stl", 0.1, False)]
    if which in ("comp", "all+comp"):
        m = [x for x in m if which == "all+comp" or x["kind"] != "rl"] if which == "comp" else m
        for s in (1, 2, 3):
            m += [dict(rl(f"STL-C#s{s}", f"stl_c0.1_s{s}", "stl", 0.1, True), comp=True),
                  dict(rl(f"FP-800-C#s{s}", f"fp800_s{s}", "fp", 0.8, None), comp=True)]
    return [x for x in m if x["kind"] != "rl" or os.path.exists(os.path.join(RUNS, x["run"], "final.pt"))]


def scenarios(suite, n, start=0):
    if suite == "main":
        return [sample_train(EVAL_SEED0 + i, label="test") for i in range(start, start + n)]
    if suite == "ood":
        return [sample_ood(EVAL_SEED0 + 10_000 + i) for i in range(start, start + n)]
    if suite == "lambda":
        scs = []
        for k, lam in enumerate([0.1, 0.3, 1.0, 3.0]):
            scs += [sample_change_rate(EVAL_SEED0 + 20_000 + 100 * k + i, lam) for i in range(start, start + n)]
        return scs
    if suite in ("fair", "friendly_cubic", "friendly_bbr"):
        scs = []
        for i in range(start, start + n):
            if suite == "fair":
                scs.append(Scenario(seed=EVAL_SEED0 + 40_000 + i, duration=40.0,
                                    bottleneck_delay=0.02 + 0.01 * (i % 3), capacity=[(0.0, 30e6)],
                                    buffer_pkts=200, rl_flows=3, rl_starts=(0.1, 8.0, 16.0), label="fair"))
            else:
                scs.append(Scenario(seed=EVAL_SEED0 + 50_000 + i, duration=30.0,
                                    bottleneck_delay=0.02 + 0.01 * (i % 3), capacity=[(0.0, 30e6)],
                                    buffer_pkts=150 + 50 * (i % 3), competitor=(5.0, 30.0), label=suite))
        return scs
    raise ValueError(suite)


FIELDS = ["method", "seed", "label", "thr_mbps", "util_fair", "qdelay_mean_ms", "qdelay_p95_ms",
          "rtt_mean_ms", "infl_mean", "drops", "link_util", "gen_rate", "watch_rate", "early_rate",
          "mean_hold", "calls", "comp_thr_mbps", "thr1_mbps", "thr2_mbps", "wall", "owd_ms", "C0",
          "n_changes", "base_rtt_ms", "udp", "comp", "ov_from", "ov_share0", "ov_jain", "ov_rl_jain", "comp_enters", "comp_epochs", "error"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--methods", default="all")
    ap.add_argument("--only", default="")
    ap.add_argument("--budget", type=float, default=1e9, help="stop cleanly after this many seconds")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, f"{args.suite}.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                if not r.get("error"):
                    done.add((r["method"], int(r["seed"])))
    ms = methods(args.methods)
    if args.only:
        keep = args.only.split(",")
        ms = [m for m in ms if any(m["name"].startswith(k) for k in keep)]
    if args.suite == "friendly_bbr":
        for m in ms:
            m["comp_cc"] = "bbr"
    elif args.suite == "friendly_cubic":
        for m in ms:
            m["comp_cc"] = "cubic"
    scs = scenarios(args.suite, args.n, args.start)
    jobs = [(sc, m) for sc in scs for m in ms if (m["name"], sc.seed) not in done]
    print(f"[{args.suite}] {len(done)} done, {len(jobs)} to go", flush=True)

    if os.path.exists(path):
        with open(path) as f:
            rows = list(csv.reader(f))
        if rows and rows[0] != FIELDS:  # schema grew: rewrite under the new header
            h = rows[0]
            with open(path, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
                w.writeheader()
                for r in rows[1:]:
                    w.writerow(dict(zip(FIELDS if len(r) == len(FIELDS) else h, r)))
    new = not os.path.exists(path)
    f = open(path, "a", newline="")
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    if new:
        w.writeheader()
    t0 = time.time()
    for k, (sc, m) in enumerate(jobs):
        if time.time() - t0 > args.budget:
            print(f"[{args.suite}] budget reached after {k} episodes", flush=True)
            break
        try:
            row = run_episode(m, sc, os.path.join(os.path.expanduser("~/emu/work"), args.suite))
        except Exception as e:
            traceback.print_exc()
            row = dict(method=m["name"], seed=sc.seed, label=sc.label, error=str(e)[:300])
        w.writerow(row)
        f.flush()
        if k % 10 == 0:
            print(f"[{args.suite}] {k + 1}/{len(jobs)} {time.time() - t0:.0f}s  {m['name']} seed={sc.seed} "
                  f"util={row.get('util_fair', float('nan')):.3f} infl={row.get('infl_mean', float('nan')):.3f}",
                  flush=True)
    f.close()
    print(f"[{args.suite}] finished pass", flush=True)


if __name__ == "__main__":
    main()
