"""Summaries of the emulation suites, with the paper's statistics
(utility = min(util,1) - 0.5 infl; per-scenario means over the three
training seeds; paired Wilcoxon tests; bootstrap CIs)."""
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = os.path.expanduser("~/emu/results")
RENAME = {"STL": "ST", "STL-noTW": "ST-noTW", "ST": "ST-short", "STL-C": "ST+C", "FP-800-C": "FP-800+C"}
ORDER = ["ST", "ST-noTW", "ST-short", "WT-100", "FP-100", "FP-400", "FP-800", "ST+C", "FP-800+C", "CUBIC", "BBR", "Reno"]


def family(m):
    f = m.split("#")[0]
    return RENAME.get(f, f)


def load(tag):
    d = pd.read_csv(os.path.join(R, f"{tag}.csv"))
    if "error" in d:
        d = d[d["error"].isna()]
    d = d.drop_duplicates(["method", "seed"], keep="last")
    d["family"] = d["method"].map(family)
    d["util_c"] = d["util_fair"].clip(upper=1.0)
    d["score"] = d["util_c"] - 0.5 * d["infl_mean"]
    return d


def boot(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def paired(d, a, b, col="score"):
    x = d[d.family == a].groupby("seed")[col].mean()
    y = d[d.family == b].groupby("seed")[col].mean()
    j = x.index.intersection(y.index)
    if len(j) < 5:
        return np.nan, np.nan, np.nan, len(j)
    diff = (x[j] - y[j]).values
    try:
        p = stats.wilcoxon(x[j], y[j]).pvalue
    except ValueError:
        p = 1.0
    m, lo, hi = boot(diff)
    return m, p, (lo, hi), len(j)


def table(d):
    # complete scenarios only (every family present), so the comparison is paired
    fams = [f for f in ORDER if f in set(d.family)]
    per = d.groupby(["family", "seed"])[["score", "util_c", "infl_mean", "qdelay_p95_ms", "calls",
                                         "gen_rate", "watch_rate", "mean_hold", "early_rate"]].mean().reset_index()
    cnt = per.groupby("seed").family.nunique()
    full = cnt[cnt == len(fams)].index
    per = per[per.seed.isin(full)]
    rows = []
    for f in fams:
        x = per[per.family == f]
        s, lo, hi = boot(x.score)
        rows.append(dict(family=f, n=len(x), score=s, ci=f"[{lo:.3f},{hi:.3f}]", util=x.util_c.mean(),
                         infl=x.infl_mean.mean(), p95q=x.qdelay_p95_ms.mean(), calls=x.calls.mean(),
                         hold_ms=1000 * x.mean_hold.mean(), early=x.early_rate.mean()))
    return pd.DataFrame(rows), per, full


def main(tags):
    for tag in tags:
        if not os.path.exists(os.path.join(R, f"{tag}.csv")):
            continue
        d = load(tag)
        t, per, full = table(d)
        print(f"\n=== {tag}: {len(full)} complete scenarios ===")
        print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
        dd = d[d.seed.isin(full)]
        print("paired utility differences (mean, Wilcoxon p, 95% CI, n):")
        for a, b in [("ST", "FP-100"), ("ST", "FP-800"), ("ST", "FP-400"), ("FP-800", "FP-100"),
                     ("ST", "ST-short"), ("ST", "ST-noTW"), ("ST", "WT-100"), ("ST", "CUBIC"), ("ST", "BBR")]:
            if a in set(dd.family) and b in set(dd.family):
                m, p, ci, n = paired(dd, a, b)
                print(f"  {a:>8} - {b:<8} {m:+.3f}  p={p:.2g}  CI=[{ci[0]:+.3f},{ci[1]:+.3f}]  n={n}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["main", "ood", "lambda", "friendly_cubic", "friendly_bbr", "fair"])
