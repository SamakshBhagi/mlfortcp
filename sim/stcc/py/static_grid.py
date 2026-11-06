"""Grid search for the best fixed executor parameters (no supervisor).

Score per episode = min(util, 1) - 0.5 * RTT inflation (same weights as the
RL reward); averaged over 16 training-distribution scenarios.
"""
import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from evaluate import run_suite
from scenario import sample_train

OUT = os.path.expanduser("~/sim/runs/static_grid")
vals = [-0.7, 0.0, 0.7]
methods = [dict(name="th=" + ",".join(map(str, th)), kind="static", theta=list(th))
           for th in itertools.product(vals, repeat=4)]
scs = [sample_train(8_000_000 + i, label="grid") for i in range(16)]
workers = int(sys.argv[1]) if len(sys.argv) > 1 else 1
df = run_suite(methods, scs, OUT, "grid", workers=workers)
df["score"] = df["util_fair"].clip(upper=1.0) - 0.5 * df["infl_mean"]
g = df.groupby("method")["score"].mean().sort_values(ascending=False)
print(g.head(10))
best = g.index[0]
theta = [float(x) for x in best.split("=")[1].split(",")]
json.dump(dict(theta=theta, score=float(g.iloc[0])), open(os.path.join(OUT, "best.json"), "w"))
print("best", theta)
