"""Evaluation suites. Every method sees exactly the same scenarios (same seeds).

    python evaluate.py --suite main --out ~/sim/eval
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import json
import os
import time
import traceback
from multiprocessing import Pool

import numpy as np
import pandas as pd

from agent import PPO
from metrics import read_vec, summarize, flow_series
from runtime import Episode
from scenario import (Scenario, sample_train, sample_change_rate, sample_ood, step_scenario, sample_steps)
from train import Supervisor

RUNS = os.path.expanduser("~/sim/runs")
EVAL_SEED0 = 9_000_000          # never used in training (training seeds are s*1e6 + i, i < 1e6)


# ---------------------------------------------------------------------------
# method registry
# ---------------------------------------------------------------------------

def rl(name, run, mode, hold=0.1, tripwire=None, cost=0.1):
    return dict(name=name, kind="rl", run=run, mode=mode, hold=hold, tripwire=tripwire, cost=cost)


def methods_default():
    m = [dict(name="Reno-SACK", kind="tcp", cls="TcpReno"),
         dict(name="NewReno", kind="tcp", cls="TcpNewReno"),
         dict(name="Westwood", kind="tcp", cls="TcpWestwood"),
         dict(name="Vegas", kind="tcp", cls="TcpVegas"),
         dict(name="Static-best", kind="static", theta=None)]
    for s in (1, 2, 3):
        for (lab, run, mode, hold, tw) in [("FP-100", f"fp100_s{s}", "fp", 0.1, None),
                                          ("WT-100", f"wt_c0.1_s{s}", "wt", 0.1, None),
                                          ("ST", f"st_c0.1_s{s}", "st", 0.1, True),
                                          ("ST-noTW", f"st_c0.1_s{s}", "st", 0.1, False)]:
            if os.path.exists(os.path.join(RUNS, run, "final.pt")):
                m.append(rl(f"{lab}#s{s}", run, mode, hold, tw))
    for s in (1, 2, 3):
        run = f"stl_c0.1_s{s}"
        if os.path.exists(os.path.join(RUNS, run, "final.pt")):
            m.append(rl(f"STL#s{s}", run, "stl", 0.1, True))
            m.append(rl(f"STL-noTW#s{s}", run, "stl", 0.1, False))
    for T in (400, 800):
        for s in (1, 2, 3):
            run = f"fp{T}_s{s}"
            if os.path.exists(os.path.join(RUNS, run, "final.pt")):
                m.append(rl(f"FP-{T}#s{s}", run, "fp", T / 1000, None))
    return m


def load_static_best():
    p = os.path.join(RUNS, "static_grid", "best.json")
    return json.load(open(p))["theta"] if os.path.exists(p) else [0, 0, 0, 0]


# ---------------------------------------------------------------------------

def run_one(args):
    method, sc_json, workdir, port, num_rl, keep_series = args
    sc = Scenario.from_json(sc_json)
    try:
        if method["kind"] == "tcp":
            ep = Episode(sc, workdir, port, tripwire=False, record=True, rl_class=method["cls"],
                         other_class="TcpNewReno", num_rl=num_rl)
            flows = ep.run(lambda *a: None)
            sup = None
        elif method["kind"] == "static":
            theta = method["theta"] if method["theta"] is not None else load_static_best()
            ep = Episode(sc, workdir, port, tripwire=False, record=True, static_theta=theta, num_rl=num_rl)
            flows = ep.run(lambda *a: None)
            sup = None
        else:
            agent = PPO.load(os.path.join(RUNS, method["run"], "final.pt"))
            sup = Supervisor(agent, method["mode"], method["cost"], method["hold"],
                             deterministic=True, train=False)
            tw = method["tripwire"] if method["tripwire"] is not None else False
            ep = Episode(sc, workdir, port, tripwire=tw, record=True, num_rl=num_rl,
                         ph_hq=method.get("ph_hq"), ph_hr=method.get("ph_hr"))
            flows = ep.run(sup.act, sup.on_reward)
        vec = f"{workdir}/ep_{sc.seed}.vec"
        sca = f"{workdir}/ep_{sc.seed}.sca"
        row = dict(method=method["name"], seed=sc.seed, label=sc.label, wall=ep.wall,
                   C0=sc.capacity[0][1] / 1e6, n_changes=len(sc.capacity) - 1,
                   base_rtt_ms=sc.base_rtt * 1e3, udp=sc.udp_rate_bps > 0, comp=sc.competitor is not None)
        row.update(summarize(vec, sca, sc))
        if flows:
            ctx = flows[min(flows)]
            T = sc.duration - 0.1
            row["gen_rate"] = ctx.decisions / T
            row["watch_rate"] = ctx.watcher_calls / T
            row["early_rate"] = sum(1 for x in ctx.log if x.get("decision_reason") in (1, 2)) / T
            holds = [x["hold"] for x in ctx.log if "hold" in x]
            row["mean_hold"] = float(np.mean(holds)) if holds else np.nan
        else:
            row["gen_rate"] = row["watch_rate"] = row["early_rate"] = 0.0
            row["mean_hold"] = np.nan
        series = None
        if keep_series:
            v = read_vec(vec)
            series = {}
            for f in range(num_rl + (1 if sc.competitor else 0)):
                t, x = flow_series(v, f, 0.1, sc.duration)
                series[f"thr{f}"] = x.tolist()
            series["t"] = t.tolist()
            q = v.get("Dumbbell.r1.ppp[0].queue.queueingTime:vector", np.zeros((0, 2)))
            if len(q):
                edges = np.arange(0, sc.duration + 0.1, 0.1)
                idx = np.digitize(q[:, 0], edges) - 1
                qm = np.array([q[idx == i, 1].mean() if np.any(idx == i) else np.nan for i in range(len(edges) - 1)])
                series["qdelay_ms"] = (qm * 1e3).tolist()
            if flows:
                ctx = flows[min(flows)]
                series["decisions"] = [(x["t"], x.get("decision_reason", 0), x.get("hold", 0),
                                        x.get("theta", [0] * 4)) for x in ctx.log]
            if num_rl > 1:
                series["per_flow"] = {}
                for f in range(num_rl):
                    t, x = flow_series(v, f, 1.0, sc.duration)
                    series["per_flow"][f] = x.tolist()
        for p in glob.glob(f"{workdir}/ep_{sc.seed}.*") + glob.glob(f"{workdir}/scen_{sc.seed}.xml"):
            os.remove(p)
        return row, series
    except Exception as e:
        traceback.print_exc()
        return dict(method=method["name"], seed=sc.seed, label=sc.label, error=str(e)), None


def run_suite(methods, scenarios, out, tag, workers=2, num_rl=1, keep_series=False):
    os.makedirs(out, exist_ok=True)
    jobs = []
    for i, m in enumerate(methods):
        for j, sc in enumerate(scenarios):
            k = len(jobs)
            jobs.append((m, sc.to_json(), os.path.join(out, "work", f"w{k % workers}_{tag}"),
                         6000 + (k % workers) * 7 + (hash(tag) % 5), num_rl, keep_series))
    # one port per worker slot: run in waves so slots never collide
    rows, series = [], {}
    t0 = time.time()
    part = os.path.join(out, f"{tag}.partial.csv")
    if os.path.exists(part) and not keep_series:
        prev = pd.read_csv(part)
        if "error" in prev:
            prev = prev[prev["error"].isna()]
        rows = prev.to_dict("records")
        done = {(r["method"], int(r["seed"])) for r in rows}
        jobs = [j for j in jobs if (j[0]["name"], json.loads(j[1])["seed"]) not in done]
        print(f"[{tag}] resuming: {len(rows)} done, {len(jobs)} to go", flush=True)
    with Pool(workers) as pool:
        for i in range(0, len(jobs), workers):
            wave = jobs[i:i + workers]
            wave = [(w[0], w[1], os.path.join(out, "work", f"slot{s}_{tag}"), 6000 + s + 10 * (abs(hash(tag)) % 50),
                     w[4], w[5]) for s, w in enumerate(wave)]
            for (row, ser) in pool.map(run_one, wave):
                rows.append(row)
                if ser is not None:
                    series[f"{row['method']}|{row['seed']}"] = ser
            if (i // workers) % 10 == 0:
                print(f"[{tag}] {len(rows)}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)
                pd.DataFrame(rows).to_csv(os.path.join(out, f"{tag}.partial.csv"), index=False)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, f"{tag}.csv"), index=False)
    if series:
        json.dump(series, open(os.path.join(out, f"{tag}_series.json"), "w"))
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", required=True)
    ap.add_argument("--out", default=os.path.expanduser("~/sim/eval"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--only", default="")
    ap.add_argument("--start", type=int, default=0, help="first scenario index (to extend a suite)")
    args = ap.parse_args()
    methods = methods_default()
    if args.only:
        keep = args.only.split(",")
        methods = [m for m in methods if any(m["name"].startswith(k) for k in keep)]

    if args.suite == "main":
        scs = [sample_train(EVAL_SEED0 + i, label="test") for i in range(args.n)]
        run_suite(methods, scs, args.out, "main", args.workers)
    elif args.suite == "frontier":
        # FP-100 policies evaluated at longer/shorter periods, ST at several invocation costs
        fm = []
        for sd in (1, 2, 3):
            if os.path.exists(os.path.join(RUNS, f"fp100_s{sd}", "final.pt")):
                for T in (0.05, 0.1, 0.2, 0.4, 0.8):
                    fm.append(rl(f"FP@{int(T*1000)}ms#s{sd}", f"fp100_s{sd}", "fp", T, None))
        for run, lab in [("st_c0.02_s1", "ST c=0.02"), ("st_c0.1_s1", "ST c=0.1"), ("st_c0.3_s1", "ST c=0.3"),
                         ("st_c0.1_s2", "ST c=0.1"), ("st_c0.1_s3", "ST c=0.1")]:
            if os.path.exists(os.path.join(RUNS, run, "final.pt")):
                fm.append(rl(f"{lab}#{run}", run, "st", 0.1, True))
                fm.append(rl(f"{lab} noTW#{run}", run, "st", 0.1, False))
        for sd in (1, 2, 3):
            if os.path.exists(os.path.join(RUNS, f"wt_c0.1_s{sd}", "final.pt")):
                fm.append(rl(f"WT-100#s{sd}", f"wt_c0.1_s{sd}", "wt", 0.1, None))
        for T in (400, 800):
            if os.path.exists(os.path.join(RUNS, f"fp{T}_s1", "final.pt")):
                fm.append(rl(f"FP-{T} trained#s1", f"fp{T}_s1", "fp", T / 1000, None))
        scs = [sample_train(EVAL_SEED0 + i, label="test") for i in range(args.n)]
        run_suite(fm, scs, args.out, "frontier", args.workers)
    elif args.suite == "ood":
        scs = [sample_ood(EVAL_SEED0 + 10_000 + i) for i in range(args.start, args.start + args.n)]
        run_suite(methods, scs, args.out, "ood", args.workers)
    elif args.suite == "lambda":
        scs = []
        for k, lam in enumerate([0.1, 0.3, 1.0, 3.0]):
            scs += [sample_change_rate(EVAL_SEED0 + 20_000 + 100 * k + i, lam) for i in range(args.n)]
        run_suite(methods, scs, args.out, "lambda", args.workers)
    elif args.suite == "steps":
        scs = [sample_steps(EVAL_SEED0 + 60_000 + i) for i in range(args.n)]
        run_suite(methods, scs, args.out, "steps", args.workers, keep_series=True)
    elif args.suite == "step":
        run_suite(methods, [step_scenario(EVAL_SEED0 + 30_000)], args.out, "step", args.workers, keep_series=True)
    elif args.suite == "fair":
        scs = []
        for i in range(args.n):
            sc = Scenario(seed=EVAL_SEED0 + 40_000 + i, duration=40.0, bottleneck_delay=0.02 + 0.01 * (i % 3),
                          capacity=[(0.0, 30e6)], buffer_pkts=200, rl_flows=3, rl_starts=(0.1, 8.0, 16.0),
                          label="fair")
            scs.append(sc)
        run_suite(methods, scs, args.out, "fair", args.workers, num_rl=3, keep_series=True)
    elif args.suite == "friendly":
        scs = []
        for i in range(args.n):
            sc = Scenario(seed=EVAL_SEED0 + 50_000 + i, duration=30.0, bottleneck_delay=0.02 + 0.01 * (i % 3),
                          capacity=[(0.0, 30e6)], buffer_pkts=150 + 50 * (i % 3), competitor=(5.0, 30.0),
                          label="friendly")
            scs.append(sc)
        run_suite(methods, scs, args.out, "friendly", args.workers, keep_series=True)


if __name__ == "__main__":
    main()
