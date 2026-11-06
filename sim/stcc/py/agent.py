"""PPO with a hybrid action space and semi-Markov (variable-duration) discounting.

Supervisor variants (all drive the same executor):
  st : self-triggered   -> action = (theta, hold index); optional tripwire
  fp : fixed period     -> action = theta; hold fixed
  wt : learned watcher  -> action = (trigger, theta) every monitor interval;
                           theta is only applied (and charged) when trigger=1
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from runtime import HOLDS, OBS_DIM, TIME_UNIT

torch.set_num_threads(1)


class RunningNorm:
    def __init__(self, dim):
        self.n = 1e-4
        self.mean = np.zeros(dim)
        self.var = np.ones(dim)

    def update(self, x: np.ndarray):
        bm, bv, bn = x.mean(0), x.var(0), x.shape[0]
        d = bm - self.mean
        tot = self.n + bn
        self.mean = self.mean + d * bn / tot
        self.var = (self.var * self.n + bv * bn + d ** 2 * self.n * bn / tot) / tot
        self.n = tot

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -8, 8).astype(np.float32)

    def state_dict(self):
        return dict(n=self.n, mean=self.mean.tolist(), var=self.var.tolist())

    def load_state_dict(self, d):
        self.n, self.mean, self.var = d["n"], np.array(d["mean"]), np.array(d["var"])


def mlp(i, o, h=64):
    return nn.Sequential(nn.Linear(i, h), nn.Tanh(), nn.Linear(h, h), nn.Tanh(), nn.Linear(h, o))


class Policy(nn.Module):
    def __init__(self, mode: str, obs_dim: int = OBS_DIM):
        super().__init__()
        self.mode = mode
        self.trunk = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, 64), nn.Tanh())
        self.mu = nn.Linear(64, 4)
        self.log_std = nn.Parameter(torch.full((4,), -0.7))
        self.hold = nn.Linear(64, len(HOLDS)) if mode in ("st", "stl") else None
        self.trig = nn.Linear(64, 1) if mode == "wt" else None
        self.v = mlp(obs_dim, 1)
        for m in [self.mu] + ([self.hold] if self.hold else []) + ([self.trig] if self.trig else []):
            nn.init.orthogonal_(m.weight, 0.01)
            nn.init.zeros_(m.bias)

    def dists(self, obs: torch.Tensor):
        h = self.trunk(obs)
        mu = torch.tanh(self.mu(h))
        d = {"theta": torch.distributions.Normal(mu, self.log_std.exp().expand_as(mu))}
        if self.hold is not None:
            d["hold"] = torch.distributions.Categorical(logits=self.hold(h))
        if self.trig is not None:
            d["trig"] = torch.distributions.Bernoulli(logits=self.trig(h).squeeze(-1))
        return d

    def value(self, obs):
        return self.v(obs).squeeze(-1)

    def logp_entropy(self, obs, theta, hold, trig):
        d = self.dists(obs)
        lp_theta = d["theta"].log_prob(theta).sum(-1)
        ent = d["theta"].entropy().sum(-1)
        if self.mode == "wt":
            lp = d["trig"].log_prob(trig) + trig * lp_theta
            ent = ent * 0.0 + d["trig"].entropy() + 0.1 * d["theta"].entropy().sum(-1)
        else:
            lp = lp_theta
        if self.mode in ("st", "stl"):
            lp = lp + d["hold"].log_prob(hold)
            ent = ent + d["hold"].entropy()
        return lp, ent


@dataclass
class PPOConfig:
    gamma: float = 0.97          # per TIME_UNIT (100 ms)
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 1e-3
    epochs: int = 10
    minibatches: int = 4
    ent_coef: float = 0.005
    vf_coef: float = 0.5
    max_grad: float = 0.5
    reward_scale: float = 0.1


class PPO:
    def __init__(self, mode: str, cfg: PPOConfig = PPOConfig(), seed: int = 0):
        torch.manual_seed(seed)
        self.mode, self.cfg = mode, cfg
        self.pi = Policy(mode)
        self.opt = torch.optim.Adam(self.pi.parameters(), lr=cfg.lr, eps=1e-5)
        self.norm = RunningNorm(OBS_DIM)
        self.rng = np.random.default_rng(seed)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def act(self, obs_raw: np.ndarray, deterministic=False):
        o = torch.as_tensor(self.norm(obs_raw)[None])
        d = self.pi.dists(o)
        if deterministic:
            theta = d["theta"].mean
            hold = d["hold"].probs.argmax(-1) if "hold" in d else torch.zeros(1, dtype=torch.long)
            trig = (d["trig"].probs > 0.5).float() if "trig" in d else torch.zeros(1)
        else:
            theta = d["theta"].sample()
            hold = d["hold"].sample() if "hold" in d else torch.zeros(1, dtype=torch.long)
            trig = d["trig"].sample() if "trig" in d else torch.zeros(1)
        lp, _ = self.pi.logp_entropy(o, theta, hold, trig)
        v = self.pi.value(o)
        return dict(theta=theta[0].numpy(), hold=int(hold[0]), trig=float(trig[0]),
                    logp=float(lp[0]), v=float(v[0]))

    @torch.no_grad()
    def value(self, obs_raw):
        return float(self.pi.value(torch.as_tensor(self.norm(obs_raw)[None]))[0])

    # ---------------------------------------------------------------
    def update(self, trajs: List[List[dict]]) -> Dict[str, float]:
        c = self.cfg
        obs, th, ho, tr, lp_old, adv, ret = [], [], [], [], [], [], []
        for traj in trajs:
            n = len(traj)
            if n == 0:
                continue
            a = np.zeros(n)
            last = 0.0
            for t in reversed(range(n)):
                s = traj[t]
                g = c.gamma ** (s["dur"] / TIME_UNIT)
                if s["done"]:
                    nv, nonterm = s.get("boot_v", 0.0), 0.0
                    delta = c.reward_scale * s["r"] + g * nv - s["v"]
                else:
                    nv = traj[t + 1]["v"]
                    delta = c.reward_scale * s["r"] + g * nv - s["v"]
                    nonterm = 1.0
                last = delta + g * c.lam * nonterm * last
                a[t] = last
            for t, s in enumerate(traj):
                obs.append(s["obs_n"]); th.append(s["theta"]); ho.append(s["hold"])
                tr.append(s["trig"]); lp_old.append(s["logp"]); adv.append(a[t]); ret.append(a[t] + s["v"])
        obs = torch.as_tensor(np.array(obs), dtype=torch.float32)
        th = torch.as_tensor(np.array(th), dtype=torch.float32)
        ho = torch.as_tensor(np.array(ho), dtype=torch.long)
        tr = torch.as_tensor(np.array(tr), dtype=torch.float32)
        lp_old = torch.as_tensor(np.array(lp_old), dtype=torch.float32)
        adv = torch.as_tensor(np.array(adv), dtype=torch.float32)
        ret = torch.as_tensor(np.array(ret), dtype=torch.float32)
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
        N = obs.shape[0]
        mb = max(N // c.minibatches, 32)
        stats = dict(pl=0.0, vl=0.0, ent=0.0, kl=0.0, n=0)
        for _ in range(c.epochs):
            perm = torch.as_tensor(self.rng.permutation(N))
            for i in range(0, N, mb):
                idx = perm[i:i + mb]
                lp, ent = self.pi.logp_entropy(obs[idx], th[idx], ho[idx], tr[idx])
                ratio = (lp - lp_old[idx]).exp()
                pg = -torch.min(ratio * adv_n[idx],
                                ratio.clamp(1 - c.clip, 1 + c.clip) * adv_n[idx]).mean()
                vl = ((self.pi.value(obs[idx]) - ret[idx]) ** 2).mean()
                loss = pg + c.vf_coef * vl - c.ent_coef * ent.mean()
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.pi.parameters(), c.max_grad)
                self.opt.step()
                with torch.no_grad():
                    stats["kl"] += float((lp_old[idx] - lp).mean())
                stats["pl"] += float(pg); stats["vl"] += float(vl); stats["ent"] += float(ent.mean()); stats["n"] += 1
        n = max(stats.pop("n"), 1)
        out = {k: v / n for k, v in stats.items()}
        out["samples"] = N
        out["std"] = float(self.pi.log_std.exp().mean())
        return out

    def save(self, path):
        torch.save(dict(pi=self.pi.state_dict(), norm=self.norm.state_dict(), mode=self.mode), path)

    @staticmethod
    def load(path) -> "PPO":
        d = torch.load(path, weights_only=False)
        p = PPO(d["mode"])
        p.pi.load_state_dict(d["pi"])
        p.norm.load_state_dict(d["norm"])
        return p
