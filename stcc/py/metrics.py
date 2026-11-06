"""Algorithm-agnostic metrics from OMNeT++ result files.

Throughput is measured at each receiver; delay is the per-packet queueing
delay at the bottleneck plus the scenario's propagation RTT; loss is the
number of bottleneck queue overflows.
"""
from __future__ import annotations

import re
from collections import defaultdict
from typing import Dict

import numpy as np

_VEC_HDR = re.compile(r"^vector\s+(\d+)\s+(\S+)\s+(\S+)")


def read_vec(path: str) -> Dict[str, np.ndarray]:
    names, data = {}, defaultdict(list)
    with open(path) as f:
        for line in f:
            if not line or not line[0].isdigit():
                m = _VEC_HDR.match(line)
                if m:
                    names[int(m.group(1))] = f"{m.group(2)}.{m.group(3)}"
                continue
            p = line.split("\t")
            if len(p) >= 4:
                data[int(p[0])].append((float(p[2]), float(p[3])))
    return {names[k]: np.asarray(v) for k, v in data.items() if k in names}


def read_sca(path: str) -> Dict[str, float]:
    out = {}
    with open(path) as f:
        for line in f:
            if line.startswith("scalar "):
                parts = line.split()
                out[f"{parts[1]}.{parts[2]}"] = float(parts[3])
    return out


def flow_series(vec: Dict[str, np.ndarray], flow: int, bin_s: float = 0.1, t_end: float = None):
    key = f"Dumbbell.receiver[{flow}].app[0].packetReceived:vector(packetBytes)"
    if key not in vec:
        return np.array([]), np.array([])
    tv = vec[key]
    t_end = t_end or (tv[-1, 0] if len(tv) else 0)
    edges = np.arange(0, t_end + bin_s, bin_s)
    b, _ = np.histogram(tv[:, 0], bins=edges, weights=tv[:, 1])
    return edges[:-1], b * 8 / bin_s / 1e6   # Mb/s


def summarize(vec_path: str, sca_path: str, sc, t_from: float = 1.0) -> Dict[str, float]:
    vec = read_vec(vec_path)
    sca = read_sca(sca_path)
    out = {}
    key = "Dumbbell.receiver[0].app[0].packetReceived:vector(packetBytes)"
    tv = vec.get(key, np.zeros((0, 2)))
    sel = tv[:, 0] >= t_from if len(tv) else np.array([], bool)
    delivered = tv[sel, 1].sum() if len(tv) else 0.0
    span = sc.duration - t_from
    out["thr_mbps"] = delivered * 8 / span / 1e6
    fair = sc.fair_share_bytes(t_from, sc.duration)
    out["util_fair"] = delivered / max(fair, 1.0)

    q = vec.get("Dumbbell.r1.ppp[0].queue.queueingTime:vector", np.zeros((0, 2)))
    qs = q[q[:, 0] >= t_from, 1] if len(q) else np.array([0.0])
    if len(qs) == 0:
        qs = np.array([0.0])
    out["qdelay_mean_ms"] = float(qs.mean() * 1e3)
    out["qdelay_p95_ms"] = float(np.percentile(qs, 95) * 1e3)
    out["rtt_mean_ms"] = sc.base_rtt * 1e3 + out["qdelay_mean_ms"]
    out["rtt_p95_ms"] = sc.base_rtt * 1e3 + out["qdelay_p95_ms"]
    out["infl_mean"] = out["qdelay_mean_ms"] / (sc.base_rtt * 1e3)
    out["drops"] = sca.get("Dumbbell.r1.ppp[0].queue.droppedPacketsQueueOverflow:count", 0.0)

    # all-flow link utilization (vs capacity minus UDP)
    tot = 0.0
    for k, v in vec.items():
        if "packetReceived" in k and len(v):
            tot += v[v[:, 0] >= t_from, 1].sum()
    avail = 0.0
    for a, b, c in sc._segments(t_from, sc.duration):
        avail += c * (b - a) / 8
    out["link_util"] = tot / max(avail, 1.0)
    return out
