"""Train a supervisor variant.

    python train.py --name st_c0.1_s1 --mode st --cost 0.1 --seed 1
    python train.py --name fp100_s1  --mode fp --hold 0.1 --seed 1
    python train.py --name wt_s1     --mode wt --cost 0.1 --seed 1
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import time
import traceback

import numpy as np

from agent import PPO, PPOConfig
from runtime import Episode, FlowCtx, HOLDS, HOLDS_LONG, features
from scenario import sample_train

RUNS = os.path.expanduser("~/sim/runs")


class Supervisor:
    """Glue between the PPO agent and an Episode, for one RL flow or many."""

    def __init__(self, agent: PPO, mode: str, cost: float, hold: float = 0.1,
                 deterministic: bool = False, train: bool = True):
        self.agent, self.mode, self.cost, self.fixed_hold = agent, mode, cost, hold
        self.det, self.train = deterministic, train
        self.trajs = {}

    def act(self, obs, ctx: FlowCtx, m):
        a = self.agent.act(obs, deterministic=self.det)
        theta_raw = a["theta"]
        theta = np.clip(theta_raw, -1, 1)
        if self.mode in ("st", "stl"):
            hold = float((HOLDS if self.mode == "st" else HOLDS_LONG)[a["hold"]])
            cost = self.cost
            ctx.decisions += 1
            ctx.prev_hold_idx = a["hold"]
        elif self.mode == "fp":
            hold, cost = self.fixed_hold, self.cost
            ctx.decisions += 1
        else:  # wt
            hold = self.fixed_hold
            ctx.watcher_calls += 1
            if a["trig"] > 0.5 or ctx.decisions == 0 and int(m[14]) == 3:
                cost = self.cost
                ctx.decisions += 1
            else:
                theta = ctx.prev_theta.copy()
                cost = 0.0
        ctx.prev_theta = theta.astype(np.float32)
        ctx.pending = dict(obs_raw=obs, obs_n=self.agent.norm(obs), theta=theta_raw, hold=a["hold"],
                           trig=a["trig"], logp=a["logp"], v=a["v"], cost=cost, applied_hold=hold,
                           reason=int(m[14]), theta_applied=theta.copy())
        return [*theta.tolist(), hold, 0.0]

    def on_reward(self, ctx: FlowCtx, rw, final, m):
        p = ctx.pending
        if p is None:
            return
        p["r"] = rw["r"] - p["cost"]
        p["dur"] = rw["dur"]
        p["done"] = bool(final)
        if final and self.train:
            p["boot_v"] = self.agent.value(features(m, copy.deepcopy(ctx)))
        ctx.log[-1].update(theta=p["theta_applied"].tolist(), hold=p["applied_hold"],
                           cost=p["cost"], decision_reason=p["reason"])
        self.trajs.setdefault(ctx.flow_id, []).append(p)
        ctx.pending = None


def episode_summary(flows, sc):
    out = {}
    for fid, ctx in flows.items():
        L = ctx.log
        if not L:
            continue
        dur = np.array([x["dur"] for x in L])
        T = dur.sum()
        out = dict(
            ret=float(sum(x["r"] for x in L) - sum(x.get("cost", 0) for x in L)),
            util=float(np.sum([x["util"] * x["dur"] for x in L]) / T),
            infl=float(np.sum([x["infl"] * x["dur"] for x in L]) / T),
            loss=float(sum(x["loss"] for x in L)), rto=float(sum(x["rto"] for x in L)),
            gen_rate=ctx.decisions / T, watch_rate=ctx.watcher_calls / T,
            early=float(sum(1 for x in L if x.get("decision_reason") in (1, 2)) / T),
        )
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--mode", choices=["st", "stl", "fp", "wt"], required=True)
    ap.add_argument("--cost", type=float, default=0.1)
    ap.add_argument("--hold", type=float, default=0.1)
    ap.add_argument("--tripwire", type=int, default=1)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--episodes", type=int, default=800)
    ap.add_argument("--per_iter", type=int, default=8)
    ap.add_argument("--port", type=int, default=0)
    ap.add_argument("--resume", type=int, default=0, help="resume from ckpt_<it>.pt")
    args = ap.parse_args()

    if args.mode not in ("st", "stl"):
        args.tripwire = 0
    out = os.path.join(RUNS, args.name)
    os.makedirs(out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(out, "args.json"), "w"), indent=1)
    port = args.port or (5200 + (abs(hash(args.name)) % 700))

    agent = PPO(args.mode, PPOConfig(), seed=args.seed)
    ep_i, it = 0, 0
    if args.resume:
        old = PPO.load(os.path.join(out, f"ckpt_{args.resume}.pt"))
        agent.pi.load_state_dict(old.pi.state_dict())
        agent.norm = old.norm
        it = args.resume
        ep_i = it * args.per_iter
    sup = Supervisor(agent, args.mode, args.cost, args.hold)
    log = open(os.path.join(out, "train.csv"), "a" if args.resume else "w", newline="")
    wr = None
    if args.resume:
        wr = csv.DictWriter(log, fieldnames=["ret", "util", "infl", "loss", "rto", "gen_rate", "watch_rate",
                                             "early", "episode", "seed", "wall", "iter", "elapsed"])
    t0 = time.time()
    while ep_i < args.episodes:
        batch, obs_raw = [], []
        for _ in range(args.per_iter):
            seed = args.seed * 1_000_000 + ep_i
            sc = sample_train(seed)
            sup.trajs = {}
            try:
                ep = Episode(sc, os.path.join(out, "tmp"), port, tripwire=bool(args.tripwire))
                flows = ep.run(sup.act, sup.on_reward)
            except Exception as e:
                print("episode failed", seed, e, flush=True)
                traceback.print_exc()
                ep_i += 1
                continue
            for traj in sup.trajs.values():
                if traj:
                    batch.append(traj)
                    obs_raw += [s["obs_raw"] for s in traj]
            s = episode_summary(flows, sc)
            s.update(episode=ep_i, seed=seed, wall=round(ep.wall, 2), iter=it,
                     elapsed=round(time.time() - t0))
            if wr is None:
                wr = csv.DictWriter(log, fieldnames=list(s.keys()))
                wr.writeheader()
            wr.writerow(s)
            log.flush()
            ep_i += 1
        if batch:
            st = agent.update(batch)
            agent.norm.update(np.array(obs_raw))
            it += 1
            print(f"[{args.name}] it={it} ep={ep_i} t={time.time()-t0:.0f}s "
                  + " ".join(f"{k}={v:.3f}" for k, v in st.items()), flush=True)
            if it % 25 == 0:
                agent.save(os.path.join(out, f"ckpt_{it}.pt"))
    agent.save(os.path.join(out, "final.pt"))
    print("done", args.name, time.time() - t0)


if __name__ == "__main__":
    main()
