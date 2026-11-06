"""Generates table_linux.tex and linux_macros.tex from the emulation CSVs."""
import sys, os
sys.path.insert(0, os.path.expanduser("~/emu/py"))
import emu_analysis as A
OUT = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "~/sim/paper_rev")
ROWS = ["ST", "ST-noTW", "ST-short", "WT-100", "FP-100", "FP-400", "FP-800", "ST+C", "FP-800+C", "CUBIC", "BBR", "Reno"]
NAME = {"ST-noTW": "ST w/o tripwire", "ST+C": "ST + comp.\\ mode", "FP-800+C": "FP-800 + comp.\\ mode"}
lines = [r"\begin{tabular}{lccccc}", r"\toprule",
         r"Scheme & Utility & Util. & Infl. & p95 queue (ms) & Calls/s \\"]
for tag, title in [("main", "Test scenarios (40)"), ("ood", "Out-of-distribution scenarios (20)")]:
    t, per, full = A.table(A.load(tag))
    t = t.set_index("family")
    lines += [r"\midrule", rf"\multicolumn{{6}}{{l}}{{\emph{{{title}}}}} \\"]
    best = t.score.max()
    for f in ROWS:
        if f not in t.index:
            continue
        r = t.loc[f]
        s = f"{r.score:.3f}"
        s = rf"\textbf{{{s}}}" if abs(r.score - best) < 1e-9 else s
        calls = "--" if f in ("CUBIC", "BBR", "Reno") else f"{r.calls:.1f}"
        lines.append(f"{NAME.get(f, f)} & {s} {r.ci} & {r.util:.2f} & {r.infl:.2f} & {r.p95q:.0f} & {calls} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "table_linux.tex"), "w").write("\n".join(lines) + "\n")

d = A.load("main"); full = A.table(d)[2]; d = d[d.seed.isin(full)]
mac = {}
for key, b in [("FPh", "FP-100"), ("FPf", "FP-400"), ("WT", "WT-100")]:
    m, p, ci, n = A.paired(d, "ST", b)
    mac[f"linSTv{key}"] = f"{m:.3f}"
    mac[f"linSTv{key}CI"] = f"$[{ci[0]:.3f}, {ci[1]:.3f}]$"
with open(os.path.join(OUT, "linux_macros.tex"), "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print(open(os.path.join(OUT, "table_linux.tex")).read()); print(mac)

# ---- coexistence with kernel competitors ----
F = ["ST", "FP-100", "FP-800", "ST+C", "FP-800+C", "Reno", "CUBIC", "BBR"]
TAG = {"FP-100": "FP", "FP-800": "FPeight", "ST+C": "STC", "FP-800+C": "FPeightC"}
L = [r"\begin{tabular}{lcccccc}", r"\toprule",
     r" & \multicolumn{3}{c}{vs.\ CUBIC} & \multicolumn{3}{c}{vs.\ BBR} \\",
     r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
     r"Scheme & Share & Jain & p95 q. (ms) & Share & Jain & p95 q. (ms) \\", r"\midrule"]
res = {}
for t, suf in [("friendly_cubic", "c"), ("friendly_bbr", "b")]:
    d = A.load(t)
    per = d.groupby(["family", "seed"])[["ov_share0", "ov_jain", "qdelay_p95_ms"]].mean()
    for f in F:
        x = per.loc[f]
        m, lo, hi = A.boot(x.ov_share0)
        res[(f, suf)] = (m, lo, hi, x.ov_jain.mean(), x.qdelay_p95_ms.mean())
        mac[f"linShare{TAG.get(f, f)}{suf}"] = f"{100 * m:.0f}"
for f in F:
    cells = []
    for suf in "cb":
        m, lo, hi, j, q = res[(f, suf)]
        cells += [f"{m:.2f} [{lo:.2f},{hi:.2f}]", f"{j:.2f}", f"{q:.0f}"]
    L.append(f"{f} & " + " & ".join(cells) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]
open(os.path.join(OUT, "table_linux_friendly.tex"), "w").write("\n".join(L) + "\n")
for t, suf in [("friendly_cubic", "c"), ("friendly_bbr", "b")]:
    d = A.load(t); m, p, ci, n = A.paired(d, "ST+C", "ST", col="ov_share0")
    mac[f"linSTCvST{suf}"] = f"{m:.2f} [{ci[0]:.2f}, {ci[1]:.2f}]"
# cost of competition mode on single-flow paths (main suite)
d = A.load("main"); per = d.groupby(["family", "seed"]).score.mean().unstack(0)
both = per[["ST", "ST+C"]].dropna()
m, lo, hi = A.boot(both["ST+C"] - both["ST"])
mac["linSTCScore"] = f"{both['ST+C'].mean():.3f}"; mac["linSTScore"] = f"{both['ST'].mean():.3f}"
mac["linSTCvST"] = f"{m:+.3f}"; mac["linSTCvSTCI"] = f"$[{lo:.3f}, {hi:.3f}]$"
print("comp cost on main: n=", len(both), mac["linSTCvST"], mac["linSTCvSTCI"])
with open(os.path.join(OUT, "linux_macros.tex"), "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print("\n".join(L))

# ---- three-flow fairness ----
d = A.load("fair")
per = d.groupby(["family", "seed"])[["ov_rl_jain", "link_util", "qdelay_p95_ms"]].mean()
for f in ["ST", "FP-100", "FP-800", "ST+C", "FP-800+C", "Reno", "CUBIC", "BBR"]:
    x = per.loc[f]; k = TAG.get(f, f)
    m, lo, hi = A.boot(x.ov_rl_jain)
    mac[f"linJain{k}"] = f"{m:.2f}"
    mac[f"linJain{k}CI"] = f"$[{lo:.2f}, {hi:.2f}]$"
    mac[f"linUtil{k}"] = f"{x.link_util.mean():.2f}"
    mac[f"linQ{k}"] = f"{x.qdelay_p95_ms.mean():.0f}"
with open(os.path.join(OUT, "linux_macros.tex"), "w") as f:
    for k, v in mac.items():
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")
print({k: v for k, v in mac.items() if k.startswith(("linJain", "linUtil", "linQ"))})
