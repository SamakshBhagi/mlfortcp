"""Evaluate the Pareto-optimal static thetas (from the grid) on a test suite."""
import os
import sys

from evaluate import run_suite, EVAL_SEED0
from scenario import sample_train, sample_ood, sample_change_rate

suite = sys.argv[1] if len(sys.argv) > 1 else "main"
fn = sys.argv[2] if len(sys.argv) > 2 else "pareto.txt"
names = open(os.path.expanduser("~/sim/runs/static_grid/" + fn)).read().split()
methods = [dict(name="Static" + n.replace("th=", "["), kind="static",
                theta=[float(x) for x in n.split("=")[1].split(",")]) for n in names]
if suite == "main":
    scs = [sample_train(EVAL_SEED0 + i, label="test") for i in range(40)]
elif suite == "ood":
    scs = [sample_ood(EVAL_SEED0 + 10_000 + i) for i in range(20)]
else:
    scs = []
    for k, lam in enumerate([0.1, 0.3, 1.0, 3.0]):
        scs += [sample_change_rate(EVAL_SEED0 + 20_000 + 100 * k + i, lam) for i in range(15)]
run_suite(methods, scs, os.path.expanduser("~/sim/eval"), f"staticfront_{suite}" + ("_dense" if "dense" in fn else ""), workers=1)
