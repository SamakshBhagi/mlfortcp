"""Figures, tables and LaTeX number macros from evaluation outputs.

All numbers quoted in the paper come from paper/results.tex, which this
script writes; nothing is typed by hand.
"""
from __future__ import annotations

import glob
import json
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EVAL = os.path.expanduser("~/sim/eval")
RUNS = os.path.expanduser("~/sim/runs")
PAPER = os.path.expanduser("~/sim/paper")
FIG = os.path.join(PAPER, "figs")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.titlesize": 8.5, "axes.labelsize": 8.5,
    "legend.fontsize": 7.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "lines.linewidth": 1.4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})
COL = {"ST-short": "#7fb3d8", "ST-short-noTW": "#c6dbef", "FP-800": "#e67e22", "FP-400": "#f1948a", "ST": "#0b3d91", "ST-noTW": "#6a8fd1", "FP-100": "#c0392b", "WT-100": "#2e8b57",
       "Static-best": "#8e44ad", "Reno-SACK": "#7f7f7f", "NewReno": "#b0b0b0",
       "Westwood": "#d68910", "Vegas": "#5d4037"}
ORDER = ["ST", "ST-noTW", "ST-short", "ST-short-noTW", "WT-100", "FP-100", "FP-400", "FP-800", "Static-best", "Westwood", "Reno-SACK", "NewReno", "Vegas"]
MACROS = {}


from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator


def plain_log(ax, axis="x", ticks=(0.5, 1, 2, 5, 10, 20)):
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xscale if axis == "x" else ax.set_yscale)("log")
    a.set_major_locator(FixedLocator(ticks))
    a.set_minor_locator(NullLocator())
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))


def paired_delta(d, fam, ref="FP-100", col="score"):
    """Per-scenario difference to the reference family (mean over its seeds), with bootstrap CI."""
    r = d[d.family == ref].groupby("seed")[col].mean()
    x = d[d.family == fam].groupby("seed")[col].mean()
    j = x.index.intersection(r.index)
    return boot_ci((x[j] - r[j]).values)


FAMKEY = {"ST-short": "STshort", "ST-short-noTW": "STshortnoTW", "FP-100": "FP", "FP-400": "FPfour", "FP-800": "FPeight", "WT-100": "WT"}


def famkey(f):
    if f in FAMKEY:
        return FAMKEY[f]
    return re.sub("[^A-Za-z]", "", f)


def fmtp(p):
    if p != p:
        return "n/a"
    if p >= 1e-3:
        return f"{p:.2g}"
    e = int(np.floor(np.log10(p)))
    m = p / 10 ** e
    return f"{m:.0f}\\times10^{{{e}}}"


def macro(name, value, fmt="{:.2f}"):
    MACROS[name] = fmt.format(value) if not isinstance(value, str) else value


RENAME = {"STL": "ST", "STL-noTW": "ST-noTW", "ST": "ST-short", "ST-noTW": "ST-short-noTW"}


def family(m):
    f = m.split("#")[0]
    return RENAME.get(f, f)


def boot_ci(x, n=10000, seed=0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return x.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)


def load(tag):
    p = os.path.join(EVAL, f"{tag}.csv")
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p)
    for ex in ("~/sim/eval_extra", "~/sim/eval_extra2", "~/sim/eval_extra3", "~/sim/eval_extra4", "~/sim/eval_ood_ext", "~/sim/eval_ood_ext2"):
        extra = os.path.join(os.path.expanduser(ex), f"{tag}.csv")
        if os.path.exists(extra):
            d = pd.concat([d, pd.read_csv(extra)], ignore_index=True)
    d = d.drop_duplicates(subset=["method", "seed"], keep="last")
    if "error" in d:
        bad = d["error"].notna()
        if bad.any():
            print(f"[{tag}] {bad.sum()} failed episodes dropped")
        d = d[~bad]
    d["family"] = d["method"].map(family)
    d["calls"] = d["gen_rate"].fillna(0) + d["watch_rate"].fillna(0)
    d["util_c"] = d["util_fair"].clip(upper=1.0)
    d["score"] = d["util_c"] - 0.5 * d["infl_mean"]
    return d


# ---------------------------------------------------------------------------
def fig_learning():
    fams = {"ST": "stl_c0.1_s*", "ST-short": "st_c0.1_s*", "FP-100": "fp100_s*", "FP-800": "fp800_s*", "WT-100": "wt_c0.1_s*"}
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.0))
    for lab, pat in fams.items():
        curves = []
        for run in sorted(glob.glob(os.path.join(RUNS, pat))):
            p = os.path.join(run, "train.csv")
            if not os.path.exists(p):
                continue
            d = pd.read_csv(p)
            d["score"] = d["util"].clip(upper=1) - 0.5 * d["infl"]
            curves.append(d.set_index("episode")[["score", "gen_rate", "watch_rate"]])
        if not curves:
            continue
        n = min(len(c) for c in curves)
        for k, (col, ylabel) in enumerate([("score", "utility (util $-$ 0.5 infl.)"),
                                           ("gen_rate", "generator calls / s"),
                                           ("watch_rate", "watcher calls / s")]):
            arr = np.stack([c[col].values[:n] for c in curves])
            sm = pd.DataFrame(arr.T).rolling(60, min_periods=10).mean().values.T
            mu = sm.mean(0)
            x = np.arange(n)
            ax[k].plot(x, mu, color=COL[lab], label=lab)
            if len(curves) > 1:
                ax[k].fill_between(x, sm.min(0), sm.max(0), color=COL[lab], alpha=0.15, lw=0)
            ax[k].set_xlabel("training episode")
            ax[k].set_ylabel(ylabel)
    ax[0].legend(frameon=False)
    plain_log(ax[1], "y", (1, 2, 3, 5, 10))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "learning.pdf"))
    plt.close(fig)


# ---------------------------------------------------------------------------
def table_main(d, tag="main", caption="", label="tab:main"):
    rows = []
    for f in ORDER:
        x = d[d.family == f]
        if x.empty:
            continue
        r = {"Scheme": f}
        for col, name in [("util_fair", "Util."), ("infl_mean", "Infl."), ("qdelay_p95_ms", "p95 qdelay (ms)"),
                          ("drops", "Drops"), ("gen_rate", "Gen./s"), ("watch_rate", "Watch./s")]:
            m, lo, hi = boot_ci(x[col].values)
            r[name] = (m, lo, hi)
        rows.append(r)
    lines = [r"\begin{tabular}{@{}lrrrrrr@{}}", r"\toprule",
             r"Scheme & Utilisation & RTT infl. & p95 queue (ms) & Drops & Gen.\ calls/s & Watch.\ calls/s \\",
             r"\midrule"]
    for r in rows:
        def f2(v, k=2):
            m, lo, hi = v
            return f"{m:.{k}f} {{\\scriptsize [{lo:.{k}f}, {hi:.{k}f}]}}"
        def f0(v):
            return f"{v[0]:.0f}"
        def fr(v):
            return "--" if v[0] == 0 or np.isnan(v[0]) else f"{v[0]:.2f}"
        lines.append(f"{r['Scheme']} & {f2(r['Util.'])} & {f2(r['Infl.'])} & {r['p95 qdelay (ms)'][0]:.1f} & "
                     f"{f0(r['Drops'])} & {fr(r['Gen./s'])} & {fr(r['Watch./s'])} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    open(os.path.join(PAPER, f"table_{tag}.tex"), "w").write("\n".join(lines) + "\n")
    return rows


def paired(d, a, b, col):
    x = d[d.family == a].groupby("seed")[col].mean()
    y = d[d.family == b].groupby("seed")[col].mean()
    j = x.index.intersection(y.index)
    if len(j) < 5:
        return np.nan, np.nan
    diff = (x[j] - y[j]).values
    try:
        p = stats.wilcoxon(x[j], y[j]).pvalue
    except ValueError:
        p = 1.0
    return diff.mean(), p


# ---------------------------------------------------------------------------
def fig_scatter(d, name="tradeoff"):
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5), gridspec_kw=dict(width_ratios=[1.15, 1]))
    learned = ("ST", "ST-noTW", "ST-short", "WT-100", "FP-100", "FP-400", "FP-800")
    for f in ORDER:
        x = d[d.family == f]
        if x.empty:
            continue
        mu, ul, uh = boot_ci(x["util_fair"])
        mi, il, ih = boot_ci(x["infl_mean"])
        ax[0].errorbar(mi, mu, xerr=[[mi - il], [ih - mi]], yerr=[[mu - ul], [uh - mu]], fmt="o",
                       color=COL.get(f, "k"), ms=3.5, capsize=1.5, lw=0.8)
        if f not in learned:
            ax[0].annotate(f, (mi, mu), xytext=(4, 3), textcoords="offset points", fontsize=6.5, color=COL.get(f, "k"))
    lx = d[d.family.isin(learned)]
    ax[0].annotate("learned supervisors\n(ST, ST-short, FP-$T$, WT-100)", (lx["infl_mean"].mean(), lx["util_fair"].mean()),
                   xytext=(0.52, 0.42), textcoords="data", fontsize=6.5, color=COL["ST"],
                   arrowprops=dict(arrowstyle="-", color=COL["ST"], lw=0.6))
    ax[0].set_xlabel("RTT inflation (queueing delay / propagation RTT)")
    ax[0].set_ylabel("utilisation of fair share")
    ax[0].invert_xaxis()
    for f in learned:
        x = d[d.family == f]
        c = x["calls"].mean()
        m, lo, hi = paired_delta(d, f) if f != "FP-100" else (0.0, 0.0, 0.0)
        ax[1].errorbar(c, m, yerr=[[m - lo], [hi - m]], fmt="o", color=COL[f], ms=4, capsize=2,
                       mfc=COL[f] if f != "ST-noTW" else "white")
        ax[1].annotate(f, (c, m), xytext=(5, 2), textcoords="offset points", fontsize=7, color=COL[f])
    ax[1].axhline(0, color="#999", lw=0.6, ls="--")
    plain_log(ax[1], "x", (1, 2, 5, 10, 20))
    ax[1].set_xlim(1.0, 25)
    ax[1].set_xlabel("learned-model calls per second")
    ax[1].set_ylabel("utility difference to FP-100 (paired)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    plt.close(fig)


def fig_frontier(d):
    """Paired utility difference to FP-100 vs learned-model calls."""
    d = d[~d["family"].str.contains("trained")].copy()
    m = load("main")
    if m is not None:
        add = m[m.family.isin(["FP-400", "FP-800"])].copy()
        add["family"] = add["family"] + " trained"
        add2 = m[m.family.isin(["ST", "ST-noTW"])].copy()
        add2["family"] = add2["family"].replace({"ST": "STLONG", "ST-noTW": "STLONG-noTW"})
        d = pd.concat([d, add, add2], ignore_index=True)
    d["family"] = d["family"].replace({"FP@100ms": "FP-100"})
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    pts = []
    fams = sorted(d.family.unique())
    for f in fams:
        m, lo, hi = paired_delta(d, f) if f != "FP-100" else (0.0, 0.0, 0.0)
        pts.append((f, d[d.family == f]["calls"].mean(), m, lo, hi))
    P = {p[0]: p for p in pts}
    fp = sorted([p for p in pts if p[0].startswith("FP@") or p[0] == "FP-100"], key=lambda p: p[1])
    ax.plot([p[1] for p in fp], [p[2] for p in fp], "-", color=COL["FP-100"], lw=1, zorder=1)
    for p in fp:
        ax.errorbar(p[1], p[2], yerr=[[p[2] - p[3]], [p[4] - p[2]]], fmt="o", color=COL["FP-100"], ms=4, capsize=2)
        lab = "100 ms" if p[0] == "FP-100" else p[0].split("@")[1].replace("ms", " ms")
        ax.annotate(lab, (p[1], p[2]), xytext=(3, 5), textcoords="offset points", fontsize=6.5, color=COL["FP-100"])
    for p in pts:
        if "trained" in p[0]:
            ax.errorbar(p[1], p[2], yerr=[[p[2] - p[3]], [p[4] - p[2]]], fmt="D", color=COL["FP-100"], ms=4.5,
                        capsize=2, mfc="white")
            off = (-6, 9) if "800" in p[0] else (5, 4)
            ax.annotate(p[0].split()[0] + " (trained)", (p[1], p[2]), xytext=off, textcoords="offset points",
                        fontsize=6.5, color=COL["FP-100"], ha="center" if "800" in p[0] else "left")
    for c, mk in [("0.02", "s"), ("0.1", "s"), ("0.3", "s")]:
        for tw in (True, False):
            k = f"ST c={c}" + ("" if tw else " noTW")
            if k not in P:
                continue
            p = P[k]
            col = COL["ST-short"]
            ax.errorbar(p[1], p[2], yerr=[[p[2] - p[3]], [p[4] - p[2]]], fmt=mk, color=col, ms=4.5, capsize=2,
                        mfc=col if tw else "white")
            off = {("0.02", True): (2, 7), ("0.1", True): (-10, 9), ("0.3", True): (-26, 6),
                   ("0.02", False): (5, -9), ("0.1", False): (-6, -14), ("0.3", False): (5, -3)}[(c, tw)]
            ax.annotate(f"c={c}", (p[1], p[2]), xytext=off, textcoords="offset points", fontsize=6.3, color=col)
    for k, mfc in (("STLONG", COL["ST"]), ("STLONG-noTW", "white")):
        if k in P:
            p = P[k]
            ax.errorbar(p[1], p[2], yerr=[[p[2] - p[3]], [p[4] - p[2]]], fmt="*", color=COL["ST"], ms=9,
                        capsize=2, mfc=mfc, zorder=5)
            ax.annotate("ST" if k == "STLONG" else "ST-noTW", (p[1], p[2]), xytext=(-14, 8) if k == "STLONG" else (5, -10),
                        textcoords="offset points", fontsize=7, color=COL["ST"], fontweight="bold")
    if "WT-100" in P:
        p = P["WT-100"]
        ax.errorbar(p[1], p[2], yerr=[[p[2] - p[3]], [p[4] - p[2]]], fmt="^", color=COL["WT-100"], ms=5, capsize=2)
        ax.annotate("WT-100", (p[1], p[2]), xytext=(4, -10), textcoords="offset points", fontsize=6.5,
                    color=COL["WT-100"])
    ax.axhline(0, color="#999", lw=0.6, ls="--")
    plain_log(ax, "x", (1, 1.5, 2, 3, 5, 10, 20))
    ax.set_xlabel("learned-model calls per second (generator + watcher)")
    ax.set_ylabel("utility difference to FP-100\n(paired, same scenarios)")
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=COL["FP-100"], marker="o", ls="-", ms=4, label="FP, 100 ms-trained policy at period $T$"),
         Line2D([], [], color=COL["FP-100"], marker="D", ls="", mfc="white", ms=4, label="FP, policy trained at $T$"),
         Line2D([], [], color=COL["ST"], marker="*", ls="", ms=8, label="ST (holds 0.1-1.6 s), with / without tripwire"),
         Line2D([], [], color=COL["ST-short"], marker="s", ls="", ms=4, label="ST-short (holds 0.05-0.8 s), cost $c$, with tripwire"),
         Line2D([], [], color=COL["ST-short"], marker="s", ls="", mfc="white", ms=4, label="ST-short without tripwire"),
         Line2D([], [], color=COL["WT-100"], marker="^", ls="", ms=4, label="learned watcher")]
    ax.legend(handles=h, frameon=False, loc="lower right", fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "frontier.pdf"))
    plt.close(fig)
    return pts


def fig_lambda(d):
    d = d.copy()
    d["lam"] = d["label"].str.extract(r"lambda=([0-9.]+)").astype(float)
    fams = [f for f in ["ST", "ST-noTW", "ST-short", "WT-100", "FP-100", "FP-400", "FP-800", "Static-best"] if f in d.family.unique()]
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.2))
    fits = {}
    for f in fams:
        x = d[d.family == f].groupby("lam")
        lam = np.array(sorted(d.lam.unique()))
        sc = x["score"].mean().reindex(lam)
        ca = x["calls"].mean().reindex(lam)
        ge = x["gen_rate"].mean().reindex(lam)
        st = x["qdelay_mean_ms"].mean().reindex(lam)
        ax[0].plot(lam, sc.values, "-o", ms=3, color=COL[f], label=f)
        if f != "Static-best":
            ax[1].plot(lam, ca.values, "-o", ms=3, color=COL[f], label=f)
        ax[2].plot(lam, x["util_fair"].mean().reindex(lam).values, "-o", ms=3, color=COL[f], label=f)
        if f in ("ST",):
            # affine fit of the call rate in lambda (prediction P1)
            per = d[d.family == f][["lam", "gen_rate"]].dropna()
            res = stats.linregress(per["lam"], per["gen_rate"])
            fits[f] = res
            xx = np.linspace(lam.min(), lam.max(), 50)
            ax[1].plot(xx, res.intercept + res.slope * xx, ":", color=COL[f], lw=1)
    for a in ax:
        plain_log(a, "x", (0.1, 0.3, 1, 3))
        a.set_xlabel(r"capacity change rate $\lambda$ (1/s)")
    ax[0].set_ylabel("utility")
    ax[1].set_ylabel("learned-model calls / s")
    plain_log(ax[1], "y", (1, 2, 3, 5, 10, 15))
    ax[2].set_ylabel("utilisation")
    ax[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "lambda.pdf"))
    plt.close(fig)
    return fits


def fig_step(series_path):
    S = json.load(open(series_path))
    keys = {family(k.split("|")[0]): k for k in S}
    fams = [f for f in ["ST", "FP-100", "WT-100", "Westwood"] if f in keys]
    fig, ax = plt.subplots(len(fams), 1, figsize=(7.0, 1.25 * len(fams)), sharex=True)
    ax = np.atleast_1d(ax)
    cap_t = [0, 10, 20, 30]
    cap_v = [40, 10, 60]
    for i, f in enumerate(fams):
        s = S[keys[f]]
        t = np.array(s["t"])
        a = ax[i]
        a.step(cap_t[:-1] + [30], cap_v + [cap_v[-1]], where="post", color="#999", lw=1, ls="--", label="capacity")
        a.plot(t, s["thr0"], color=COL[f], lw=1, label="throughput")
        a.set_ylabel("Mb/s")
        a.set_ylim(0, 70)
        a2 = a.twinx()
        a2.plot(t, s.get("qdelay_ms", [np.nan] * len(t)), color="#444", lw=0.7, alpha=0.7)
        a2.set_ylabel("queue (ms)", color="#444")
        a2.set_ylim(0, 120)
        a2.grid(False)
        a2.spines["right"].set_visible(True)
        if "decisions" in s and s["decisions"]:
            dec = s["decisions"]
            for (tt, reason, hold, th) in dec:
                if reason in (1, 2):
                    a.axvline(tt, color="#e67e22", lw=0.5, alpha=0.8)
            a.plot([x[0] for x in dec], [66] * len(dec), "|", color="k", ms=4, mew=0.5)
            n = len(dec)
            a.set_title(f"{f}: {n} supervisor decisions"
                        + (f", {sum(1 for x in dec if x[1] in (1, 2))} tripwire-initiated" if f == "ST" else ""),
                        loc="left", fontsize=7.5)
        else:
            a.set_title(f, loc="left", fontsize=7.5)
    ax[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "step.pdf"))
    plt.close(fig)


def jain(x):
    x = np.asarray(x, float)
    return x.sum() ** 2 / (len(x) * (x ** 2).sum()) if (x ** 2).sum() > 0 else np.nan


def _merged_series(series_path):
    S = json.load(open(series_path))
    name = os.path.basename(series_path)
    for ex in ("~/sim/eval_extra2", "~/sim/eval_extra3", "~/sim/eval_extra4"):
        pe = os.path.join(os.path.expanduser(ex), name)
        if os.path.exists(pe):
            S.update(json.load(open(pe)))
    return S


def fairness(series_path):
    S = _merged_series(series_path)
    out = {}
    for k, s in S.items():
        f = family(k.split("|")[0])
        pf = s.get("per_flow")
        if not pf:
            continue
        arr = np.array([pf[str(i)] if str(i) in pf else pf[i] for i in range(3)])
        last = arr[:, 24:40].mean(1)      # all three flows active (16-40 s, skip 8 s of convergence)
        out.setdefault(f, []).append((jain(last), last.sum()))
    return {f: (np.mean([x[0] for x in v]), np.mean([x[1] for x in v])) for f, v in out.items()}


def recovery_time(series_path, t_step=20.0, target=60.0, frac=0.9):
    """Seconds after the 10->60 Mb/s step until 1-s mean throughput >= frac*target."""
    S = json.load(open(series_path))
    out = {}
    for k, s in S.items():
        f = family(k.split("|")[0])
        t = np.array(s["t"]); x = np.array(s["thr0"])
        sm = pd.Series(x).rolling(10, min_periods=10).mean().values  # 1 s window (0.1 s bins)
        idx = np.where((t >= t_step) & (sm >= frac * target))[0]
        out[f] = float(t[idx[0]] - t_step) if len(idx) else float("inf")
        dec = s.get("decisions") or []
        out[f + "_decisions"] = len(dec)
    return out


def training_macros():
    for name, pat in (("ST", "stl_c0.1_s*"), ("STshort", "st_c0.1_s*")):
        ini, fin = [], []
        for run in glob.glob(os.path.join(RUNS, pat)):
            p = os.path.join(run, "train.csv")
            if os.path.exists(p):
                d = pd.read_csv(p).sort_values("episode")
                ini.append(d.head(50)["gen_rate"].mean())
                fin.append(d.tail(300)["gen_rate"].mean())
        if ini:
            macro(f"{name}initcalls", float(np.mean(ini)), "{:.1f}")
            macro(f"{name}finalcalls", float(np.mean(fin)), "{:.1f}")


def outcomes():
    """Headline quantitative outcomes, written as LaTeX macros."""
    training_macros()
    d = load("main")
    if d is not None:
        fam = d.groupby("family")
        st, fp, wt = fam.get_group("ST"), fam.get_group("FP-100"), fam.get_group("WT-100")
        macro("redSTvsFP", fp["calls"].mean() / st["calls"].mean(), "{:.1f}")
        macro("redSTvsWT", wt["calls"].mean() / st["calls"].mean(), "{:.1f}")
        macro("callsSTmain", st["calls"].mean())
        macro("callsWTmain", wt["calls"].mean(), "{:.1f}")
        macro("genWTmain", wt["gen_rate"].mean())
        macro("holdSTmain", st["mean_hold"].mean() * 1000, "{:.0f}")
        if "ST-short" in fam.groups:
            ss = fam.get_group("ST-short")
            macro("redSTshortvsFP", fp["calls"].mean() / ss["calls"].mean(), "{:.1f}")
            macro("holdSTshortmain", ss["mean_hold"].mean() * 1000, "{:.0f}")
        if "FP-800" in fam.groups:
            macro("redSTvsEight", fam.get_group("FP-800")["calls"].mean() / st["calls"].mean(), "{:.2f}")
        macro("earlySTmain", st["early_rate"].mean())
        for b in ["Reno-SACK", "Westwood", "NewReno", "Vegas"]:
            if b in fam.groups:
                x = fam.get_group(b)
                k = famkey(b)
                macro(f"thrgain{k}", 100 * (st["util_fair"].mean() / x["util_fair"].mean() - 1), "{:.0f}")
                macro(f"qred{k}", 100 * (1 - st["qdelay_p95_ms"].mean() / x["qdelay_p95_ms"].mean()), "{:.0f}")
        # learned vs oracle static front at the ST delay level
        sf = [os.path.join(EVAL, f) for f in ("staticfront_main.csv", "staticfront_main_dense.csv")]
        sf = [f for f in sf if os.path.exists(f)]
        if sf:
            s = pd.concat([pd.read_csv(f) for f in sf]).groupby("method")[["util_fair", "infl_mean"]].mean()
            ok = s[s.infl_mean <= st["infl_mean"].mean() + 1e-9]
            macro("staticAtSTdelay", ok.util_fair.max())
            if "FP-800" in fam.groups:
                e8 = fam.get_group("FP-800")
                ok8 = s[s.infl_mean <= e8["infl_mean"].mean() + 1e-9]
                macro("staticAtEightdelay", ok8.util_fair.max())
                macro("redEightvsFP", fp["calls"].mean() / e8["calls"].mean(), "{:.0f}")
                macro("redEightvsWT", wt["calls"].mean() / e8["calls"].mean(), "{:.0f}")
            macro("nStaticFront", len(s), "{:d}")
    dl = load("lambda")
    if dl is not None:
        dl["lam"] = dl["label"].str.extract(r"lambda=([0-9.]+)").astype(float)
        for f in ["ST", "FP-100", "WT-100", "ST-noTW", "FP-800", "FP-400", "ST-short"]:
            if f not in set(dl.family):
                continue
            x = dl[dl.family == f].groupby("lam")
            k = famkey(f)
            for lam, v in x["calls"].mean().items():
                macro(f"calls{k}L{str(lam).replace('.', 'p')}", v)
            for lam, v in x["score"].mean().items():
                macro(f"score{k}L{str(lam).replace('.', 'p')}", v, "{:.3f}")
            if f in ("ST", "ST-short"):
                kk = famkey(f)
                for lam, v in x["mean_hold"].mean().items():
                    macro(f"hold{kk}L{str(lam).replace('.', 'p')}", v * 1000, "{:.0f}")
                for lam, v in x["early_rate"].mean().items():
                    macro(f"early{kk}L{str(lam).replace('.', 'p')}", v)
                r = stats.spearmanr(dl[dl.family == f]["lam"], dl[dl.family == f]["mean_hold"])
                macro(f"holdRho{kk}", r.correlation)
                macro(f"holdP{kk}", fmtp(r.pvalue))
        for a, b in [("ST", "FP-100"), ("ST", "FP-800"), ("ST", "FP-400"), ("ST", "ST-short"), ("ST", "ST-noTW"),
                     ("ST", "WT-100")]:
            diff, pp = paired(dl, a, b, "score")
            macro(f"lamd{famkey(a)}{famkey(b)}", diff, "{:+.3f}")
            macro(f"lamp{famkey(a)}{famkey(b)}", fmtp(pp))
        st_ = dl[dl.family == "ST"].groupby("lam")["calls"].mean()
        fp_ = dl[dl.family == "FP-100"].groupby("lam")["calls"].mean()
        macro("redMinL", (fp_ / st_).min(), "{:.1f}")
        macro("redMaxL", (fp_ / st_).max(), "{:.1f}")
    df = load("frontier")
    if df is not None:
        g = df.groupby("family")
        for f in g.groups:
            k = re.sub("[^A-Za-z0-9]", "", f).replace("0", "z").replace("1", "o").replace("2", "t").replace(
                "3", "th").replace("4", "f").replace("5", "fi").replace("8", "e")
            macro(f"fr{k}score", g.get_group(f)["score"].mean(), "{:.3f}")
            macro(f"fr{k}calls", g.get_group(f)["calls"].mean())
        if "ST c=0.3" in g.groups and "ST c=0.3 noTW" in g.groups:
            macro("twGain", g.get_group("ST c=0.3")["score"].mean() - g.get_group("ST c=0.3 noTW")["score"].mean(), "{:.3f}")
            dd = df.copy()
            a_ = dd[dd.family == "ST c=0.3"].set_index("seed")["score"]
            b_ = dd[dd.family == "ST c=0.3 noTW"].set_index("seed")["score"]
            jj = a_.index.intersection(b_.index)
            macro("twGainP", fmtp(stats.wilcoxon(a_[jj], b_[jj]).pvalue))
        note = ""
        if "FP-800 trained" in g.groups:
            note = (f"; a policy trained specifically for 800\\,ms reaches "
                    f"{g.get_group('FP-800 trained')['score'].mean():.3f}")
        if "FP-400 trained" in g.groups:
            macro("frFPfourTrained", g.get_group("FP-400 trained")["score"].mean(), "{:.3f}")
        macro("FPeightTrainedNote", note)
        print("frontier macros:", [k for k in MACROS if k.startswith("fr")])
    p = os.path.join(EVAL, "step_series.json")
    if os.path.exists(p):
        rt = recovery_time(p)
        print("recovery", rt)
        for f, v in rt.items():
            k = famkey(f)
            macro(f"rec{k}", ("$>10$" if not np.isfinite(v) else (f"{int(v)}" if f.endswith("decisions") else f"{v:.1f}")))


DIGITS = str.maketrans("0123456789", "ZOTHFVSNEG")


def texname(k):
    return k.translate(DIGITS)


def friendliness(series_path, t0=10.0, t1=30.0):
    """Share of capacity taken by the flow under test while a NewReno flow competes."""
    S = _merged_series(series_path)
    out = {}
    for k, s in S.items():
        f = family(k.split("|")[0])
        t = np.array(s["t"]); a = np.array(s["thr0"]); b = np.array(s.get("thr1", np.zeros_like(a)))
        m = (t >= t0) & (t < t1)
        tot = a[m].sum() + b[m].sum()
        out.setdefault(f, []).append(a[m].sum() / tot if tot > 0 else np.nan)
    return {f: float(np.nanmean(v)) for f, v in out.items()}


def ood_fair_outputs():
    d = load("ood")
    if d is not None:
        table_main(d, tag="ood")
        g = d.groupby("family")
        for f in g.groups:
            k = famkey(f)
            macro(f"ood{k}util", g.get_group(f)["util_fair"].mean())
            macro(f"ood{k}infl", g.get_group(f)["infl_mean"].mean())
            macro(f"ood{k}calls", g.get_group(f)["calls"].mean())
            macro(f"ood{k}score", g.get_group(f)["score"].mean(), "{:.3f}")
        for a, b in [("ST", "FP-100"), ("ST", "WT-100"), ("ST", "Static-best"), ("ST", "FP-800"),
                     ("FP-800", "FP-100"), ("ST", "ST-short"), ("ST", "ST-noTW"), ("ST", "FP-400")]:
            diff, p = paired(d, a, b, "score")
            macro(f"oodd{famkey(a)}{famkey(b)}", diff, "{:+.3f}")
            macro(f"oodp{famkey(a)}{famkey(b)}", fmtp(p), "{}")
        if "ST" in g.groups:
            macro("nOOD", int(g.get_group("ST")["seed"].nunique()), "{:d}")
        if "ST" in g.groups and "FP-100" in g.groups:
            macro("oodredSTvsFP", g.get_group("FP-100")["calls"].mean() / g.get_group("ST")["calls"].mean(), "{:.0f}")
        print("OOD", d.groupby("family")[["util_fair", "infl_mean", "calls", "score"]].mean().round(3))
    p = os.path.join(EVAL, "fair_series.json")
    if os.path.exists(p):
        fj = fairness(p)
        for f, (j, tot) in fj.items():
            macro(f"jain{famkey(f)}", j, "{:.2f}")
        df = load("fair")
        for f, x in df.groupby("family"):
            k = famkey(f)
            macro(f"fairq{k}", x["qdelay_p95_ms"].mean(), "{:.0f}")
            macro(f"fairdrops{k}", x["drops"].mean(), "{:.0f}")
        rows = [(f, fj[f][0], df[df.family == f]["qdelay_p95_ms"].mean(), df[df.family == f]["drops"].mean(),
                 df[df.family == f]["calls"].mean()) for f in ORDER if f in fj]
    else:
        rows = []
    p = os.path.join(EVAL, "friendly_series.json")
    fr = friendliness(p) if os.path.exists(p) else {}
    for f, v in fr.items():
        macro(f"share{famkey(f)}", 100 * v, "{:.0f}")
    print("fairness rows", rows, "friendly", fr)
    if rows:
        L = [r"\begin{tabular}{@{}lrrrr@{}}", r"\toprule",
             r"Scheme & Jain index & p95 queue (ms) & Drops & Share vs NewReno (\%) \\", r"\midrule"]
        for (f, j, q, dr, c) in rows:
            sh = f"{100 * fr[f]:.0f}" if f in fr else "--"
            L.append(f"{f} & {j:.2f} & {q:.0f} & {dr:.0f} & {sh} \\\\")
        L += [r"\bottomrule", r"\end{tabular}"]
        open(os.path.join(PAPER, "table_fair.tex"), "w").write("\n".join(L) + "\n")


def steps_analysis(path=os.path.join(EVAL, "steps_series.json"), horizon=8.0):
    """Per-event reaction metrics on the randomised step suite."""
    from scenario import sample_steps
    if not os.path.exists(path):
        return None
    S = json.load(open(path))
    for ex in ("~/sim/eval_extra2", "~/sim/eval_extra3", "~/sim/eval_extra4"):
        pe = os.path.join(os.path.expanduser(ex), "steps_series.json")
        if os.path.exists(pe):
            S.update(json.load(open(pe)))
    rec = []
    for k, s in S.items():
        meth, seed = k.split("|")
        f = family(meth)
        sc = sample_steps(int(seed))
        t = np.array(s["t"]); x = np.array(s["thr0"])
        q = np.array(s.get("qdelay_ms", [np.nan] * len(t)), float)
        thr = pd.Series(x).rolling(10, min_periods=10).mean().values
        qs = pd.Series(q).rolling(10, min_periods=1).mean().values
        base = sc.base_rtt * 1e3
        caps = sc.capacity
        for i in range(1, len(caps)):
            te, c_new, c_old = caps[i][0], caps[i][1] / 1e6, caps[i - 1][1] / 1e6
            w = (t >= te) & (t < te + horizon)
            if c_new > c_old:
                idx = np.where(w & (thr >= 0.8 * c_new))[0]
                rt = float(t[idx[0]] - te) if len(idx) else horizon
                rec.append(dict(family=f, method=meth, seed=int(seed), kind="up", t_react=rt,
                                reached=len(idx) > 0, util=float(np.mean(x[w]) / c_new)))
            else:
                # time until the 1-s mean queueing delay is back below 25% of the propagation RTT
                after = w & (t >= te + 1.0)
                idx = np.where(after & (qs <= 0.25 * base))[0]
                rt = float(t[idx[0]] - te) if len(idx) else horizon
                rec.append(dict(family=f, method=meth, seed=int(seed), kind="down", t_react=rt,
                                reached=len(idx) > 0, qexcess=float(np.nanmean(q[w]) / base)))
    R = pd.DataFrame(rec)
    R.to_csv(os.path.join(EVAL, "steps_events.csv"), index=False)
    up = R[R.kind == "up"].groupby("family").agg(t_up=("t_react", "median"), up_mean=("t_react", "mean"),
                                                   reached=("reached", "mean"), util=("util", "mean"))
    dn = R[R.kind == "down"].groupby("family").agg(t_down=("t_react", "median"), dn_mean=("t_react", "mean"),
                                                     drained=("reached", "mean"), qexcess=("qexcess", "mean"))
    T = up.join(dn)
    print(T.round(2))
    for f, r in T.iterrows():
        k = famkey(f)
        macro(f"stUp{k}", r.up_mean, "{:.1f}")
        macro(f"stUpMed{k}", r.t_up, "{:.1f}")
        macro(f"stReach{k}", 100 * r.reached, "{:.0f}")
        macro(f"stDn{k}", r.dn_mean, "{:.1f}")
        macro(f"stUtil{k}", r.util, "{:.2f}")
    # paired test on up-step reaction: ST vs others (mean over seeds per scenario-event)
    key = ["seed", "kind"]
    def ev(f):
        x = R[(R.family == f) & (R.kind == "up")].copy()
        x["ev"] = x.groupby(["method", "seed"]).cumcount()
        return x.groupby(["seed", "ev"])["t_react"].mean()
    if "ST" in T.index:
        a = ev("ST")
        for b in T.index:
            if b == "ST":
                continue
            bb = ev(b)
            j = a.index.intersection(bb.index)
            if len(j) > 5:
                p = stats.wilcoxon(a[j], bb[j]).pvalue
                k = famkey(b)
                macro(f"stP{k}", fmtp(p), "{}")
                print("ST vs", b, "up-react p=", p, "n=", len(j))
    # figure: distribution of up-step reaction times
    fams = [f for f in ["ST", "ST-noTW", "ST-short", "FP-800", "FP-400", "FP-100", "WT-100", "Static-best", "Westwood"] if f in T.index]
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.3))
    for i, kind in enumerate(["up", "down"]):
        data = [R[(R.family == f) & (R.kind == kind)]["t_react"].values for f in fams]
        bp = ax[i].boxplot(data, vert=True, patch_artist=True, widths=0.6, showfliers=False, whis=(5, 95),
                           medianprops=dict(color="k", lw=1))
        for patch, f in zip(bp["boxes"], fams):
            patch.set_facecolor(COL.get(f, "#bbb")); patch.set_alpha(0.75)
        ax[i].set_xticks(range(1, len(fams) + 1))
        ax[i].set_xticklabels(fams, rotation=35, ha="right")
        ax[i].set_ylabel("seconds (censored at 8 s)")
        ax[i].set_title("capacity rise: time to 80% of new capacity" if kind == "up" else
                        "capacity drop: time until queue $<$ 0.25 RTT", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "steps.pdf"))
    plt.close(fig)
    return T


def write_macros():
    with open(os.path.join(PAPER, "results.tex"), "w") as f:
        for k, v in sorted(MACROS.items()):
            f.write(f"\\newcommand{{\\{texname(k)}}}{{{v}}}\n")


if __name__ == "__main__":
    fig_learning()
    d = load("main")
    if d is not None:
        table_main(d)
        fig_scatter(d)
        for f in d.family.unique():
            x = d[d.family == f]
            key = famkey(f)
            macro(f"util{key}", x["util_fair"].mean())
            macro(f"infl{key}", x["infl_mean"].mean())
            macro(f"calls{key}", x["calls"].mean())
            macro(f"gen{key}", x["gen_rate"].mean())
            macro(f"score{key}", x["score"].mean(), "{:.3f}")
            macro(f"qpn{key}", x["qdelay_p95_ms"].mean(), "{:.1f}")
            macro(f"drops{key}", x["drops"].mean(), "{:.0f}")
        print(d.groupby("family")[["util_fair", "infl_mean", "qdelay_p95_ms", "drops", "gen_rate",
                                   "watch_rate", "early_rate", "mean_hold", "score"]].mean().round(3))
        for a, b in [("ST", "FP-100"), ("ST", "WT-100"), ("ST", "ST-noTW"), ("ST", "Static-best"),
                     ("FP-100", "Static-best"), ("WT-100", "FP-100"), ("FP-800", "FP-100"), ("FP-800", "ST"),
                     ("FP-800", "WT-100"), ("FP-800", "Static-best"), ("FP-400", "FP-100"),
                     ("ST", "ST-short"), ("ST", "FP-800"), ("ST", "FP-400"), ("ST-short", "FP-100")]:
            for col in ["score", "util_fair", "infl_mean"]:
                diff, p = paired(d, a, b, col)
                print(f"{a} - {b} {col}: {diff:+.4f} p={p:.3g}")
                macro(f"d{famkey(a)}{famkey(b)}{col.split('_')[0]}", diff, "{:+.3f}")
                macro(f"p{famkey(a)}{famkey(b)}{col.split('_')[0]}",
                      fmtp(p), "{}")
    for tag in ("ood",):
        d2 = load(tag)
        if d2 is not None:
            table_main(d2, tag=tag)
            print(tag, d2.groupby("family")[["util_fair", "infl_mean", "calls", "score"]].mean().round(3))
    dl = load("lambda")
    if dl is not None:
        fits = fig_lambda(dl)
        for f, r in fits.items():
            macro(f"fitint{f.replace('-', '')}", r.intercept)
            macro(f"fitslope{f.replace('-', '')}", r.slope)
            macro(f"fitr{f.replace('-', '')}", r.rvalue ** 2)
            print("fit", f, r.intercept, r.slope, r.rvalue ** 2)
    df = load("frontier")
    if df is not None:
        pts = fig_frontier(df)
        print(pts)
    p = os.path.join(EVAL, "step_series.json")
    if os.path.exists(p):
        fig_step(p)
    p = os.path.join(EVAL, "fair_series.json")
    if os.path.exists(p):
        fj = fairness(p)
        print("fairness", fj)
        for f, (j, tot) in fj.items():
            macro(f"jain{famkey(f)}", j, "{:.3f}")
    outcomes()
    ood_fair_outputs()
    steps_analysis()
    write_macros()
