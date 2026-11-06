"""Per-episode scenario sampling and OMNeT++ configuration.

Every episode is fully determined by an integer seed. The seed controls
(i) the path parameters drawn here and (ii) OMNeT++'s own RNG streams
(Poisson UDP cross-traffic), via `seed-set`.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
from typing import List, Optional, Tuple

import numpy as np

ACCESS_DELAY_S = 0.002     # each access link, one way
MSS = 1448
PKT_BYTES = 1500
SIM_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "simulations"))
BASE_INI = os.path.join(SIM_DIR, "omnetpp.ini")


@dataclasses.dataclass
class Scenario:
    seed: int
    duration: float
    bottleneck_delay: float                 # one-way, seconds
    capacity: List[Tuple[float, float]]     # (time, bps), piecewise constant
    buffer_pkts: int
    udp_rate_bps: float = 0.0               # 0 = off
    udp_on: Tuple[float, float] = (0.0, 0.0)
    competitor: Optional[Tuple[float, float]] = None  # (start, stop) of NewReno flow
    rl_flows: int = 1                       # senders running StTcp
    rl_starts: Tuple[float, ...] = (0.1,)
    label: str = "train"

    # ---------------------------------------------------------------
    @property
    def base_rtt(self) -> float:
        # propagation only; serialization of a 1500 B packet is < 0.6 ms
        return 2.0 * (self.bottleneck_delay + 2 * ACCESS_DELAY_S)

    def capacity_at(self, t: float) -> float:
        c = self.capacity[0][1]
        for (ti, ci) in self.capacity:
            if ti <= t:
                c = ci
            else:
                break
        return c

    def _segments(self, t0: float, t1: float):
        """Yield (a, b, capacity) pieces covering [t0, t1]."""
        times = [t for (t, _) in self.capacity] + [math.inf]
        for i, (ti, ci) in enumerate(self.capacity):
            a, b = max(t0, ti), min(t1, times[i + 1])
            if b > a:
                yield a, b, ci

    def fair_share_bytes(self, t0: float, t1: float, flow_start: float = 0.1) -> float:
        """Bytes one RL flow would deliver at its fair share over [t0,t1].

        Fair share = (capacity - UDP load) / (number of active TCP flows).
        """
        total = 0.0
        cuts = sorted({t0, t1} | {x for x in self._event_times() if t0 < x < t1})
        for a, b in zip(cuts[:-1], cuts[1:]):
            mid = 0.5 * (a + b)
            for (sa, sb, c) in self._segments(a, b):
                avail = c
                if self.udp_rate_bps > 0 and self.udp_on[0] <= mid < self.udp_on[1]:
                    avail -= self.udp_rate_bps
                n = sum(1 for s in self.rl_starts if s <= mid)
                if self.competitor and self.competitor[0] <= mid < self.competitor[1]:
                    n += 1
                n = max(n, 1)
                total += max(avail, 0.05 * c) / n * (sb - sa) / 8.0
        return total

    def _event_times(self):
        ev = [t for (t, _) in self.capacity]
        if self.udp_rate_bps > 0:
            ev += list(self.udp_on)
        if self.competitor:
            ev += list(self.competitor)
        ev += list(self.rl_starts)
        return ev

    # ---------------------------------------------------------------
    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self))

    @staticmethod
    def from_json(s: str) -> "Scenario":
        d = json.loads(s)
        d["capacity"] = [tuple(x) for x in d["capacity"]]
        d["udp_on"] = tuple(d["udp_on"])
        d["rl_starts"] = tuple(d["rl_starts"])
        if d["competitor"] is not None:
            d["competitor"] = tuple(d["competitor"])
        return Scenario(**d)


# -------------------------------------------------------------------
# samplers
# -------------------------------------------------------------------

def _loguniform(rng, lo, hi):
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def sample_train(seed: int, duration: float = 20.0,
                 cap_range=(4e6, 60e6), cap_clip=(2e6, 80e6),
                 delay_range=(0.005, 0.050), change_rate=(0.05, 1.0),
                 p_udp=0.4, p_competitor=0.3, label="train") -> Scenario:
    rng = np.random.default_rng(seed)
    d = float(rng.uniform(*delay_range))
    c0 = _loguniform(rng, *cap_range)
    lam = _loguniform(rng, *change_rate)
    cap = [(0.0, c0)]
    t, c = 0.0, c0
    while True:
        t += float(rng.exponential(1.0 / lam))
        if t >= duration - 0.5:
            break
        c = float(np.clip(c * _loguniform(rng, 0.25, 4.0), *cap_clip))
        cap.append((round(t, 4), c))

    base_rtt = 2 * (d + 2 * ACCESS_DELAY_S)
    bdp_pkts = c0 * base_rtt / (8 * PKT_BYTES)
    buf = int(max(16, round(rng.uniform(0.5, 2.0) * bdp_pkts)))

    udp_rate, udp_on = 0.0, (0.0, 0.0)
    if rng.random() < p_udp:
        cmin = min(x for (_, x) in cap)
        udp_rate = float(rng.uniform(0.05, 0.3) * cmin)
        a = float(rng.uniform(0.0, duration * 0.5))
        udp_on = (a, float(min(duration, a + rng.uniform(5.0, duration))))

    comp = None
    if rng.random() < p_competitor:
        a = float(rng.uniform(3.0, 10.0))
        comp = (a, float(min(duration, a + rng.uniform(5.0, 10.0))))

    return Scenario(seed=seed, duration=duration, bottleneck_delay=d, capacity=cap,
                    buffer_pkts=buf, udp_rate_bps=udp_rate, udp_on=udp_on,
                    competitor=comp, label=label)


def sample_change_rate(seed: int, lam: float, duration: float = 30.0) -> Scenario:
    """Fixed regime-change rate, no cross traffic: validates the cost model."""
    sc = sample_train(seed, duration=duration, change_rate=(lam, lam * 1.0000001),
                      p_udp=0.0, p_competitor=0.0, label=f"lambda={lam}")
    return sc


def step_scenario(seed: int = 0) -> Scenario:
    """Deterministic capacity steps used for the time-series figure."""
    return Scenario(seed=seed, duration=30.0, bottleneck_delay=0.026,
                    capacity=[(0.0, 40e6), (10.0, 10e6), (20.0, 60e6)],
                    buffer_pkts=150, label="step")


def sample_ood(seed: int, duration: float = 20.0) -> Scenario:
    """Outside the training ranges: higher capacity and longer RTT."""
    return sample_train(seed, duration=duration, cap_range=(60e6, 120e6),
                        cap_clip=(10e6, 150e6), delay_range=(0.050, 0.100),
                        label="ood")


# -------------------------------------------------------------------
# OMNeT++ files
# -------------------------------------------------------------------

def write_run_files(sc: Scenario, workdir: str, record: bool = False,
                    rl_class: str = "StTcp", other_class: str = "TcpNewReno",
                    num_rl: int = 1) -> str:
    """Write episode ini + scenario xml; returns path of the ini file."""
    os.makedirs(workdir, exist_ok=True)
    xml_path = os.path.join(workdir, f"scen_{sc.seed}.xml")
    ini_path = os.path.join(workdir, f"ep_{sc.seed}.ini")

    lines = ["<scenario>"]
    for (t, c) in sc.capacity[1:]:
        lines.append(f'  <at t="{t}"><set-channel-param src-module="r1" src-gate="pppg$o[0]" '
                     f'par="datarate" value="{c:.0f}bps"/></at>')
    if sc.competitor:
        idx = num_rl
        lines.append(f'  <at t="{sc.competitor[1]}"><set-channel-param src-module="sender[{idx}]" '
                     f'src-gate="pppg$o[0]" par="disabled" value="true"/></at>')
    lines.append("</scenario>")
    with open(xml_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_senders = num_rl + (1 if sc.competitor else 0)
    o = ["[General]",
         f"seed-set = {sc.seed % 2_000_000_000}",
         f"sim-time-limit = {sc.duration}s",
         f"*.numSenders = {max(n_senders, 1)}",
         f"*.bottleneckDatarate = {sc.capacity[0][1]:.0f}bps",
         f"*.bottleneckDelay = {sc.bottleneck_delay:.6f}s",
         f"*.r1.ppp[0].queue.packetCapacity = {sc.buffer_pkts}",
         f'*.scenarioManager.script = xmldoc("{xml_path}")']
    for i in range(num_rl):
        start = sc.rl_starts[i] if i < len(sc.rl_starts) else sc.rl_starts[-1]
        o += [f'*.sender[{i}].tcp.tcpAlgorithmClass = "{rl_class}"',
              f"*.sender[{i}].app[0].tOpen = {start}s",
              f"*.sender[{i}].app[0].tSend = {start}s"]
    if sc.competitor:
        i = num_rl
        o += [f'*.sender[{i}].tcp.tcpAlgorithmClass = "{other_class}"',
              f"*.sender[{i}].app[0].tOpen = {sc.competitor[0]}s",
              f"*.sender[{i}].app[0].tSend = {sc.competitor[0]}s"]
    if sc.udp_rate_bps > 0:
        mean_iv = 1200 * 8 / sc.udp_rate_bps
        o += [f"*.udpSrc.app[0].startTime = {sc.udp_on[0]}s",
              f"*.udpSrc.app[0].stopTime = {sc.udp_on[1]}s",
              f"*.udpSrc.app[0].sendInterval = exponential({mean_iv:.9f}s)"]
    if record:
        o += ["**.receiver[*].app[0].packetReceived:vector(packetBytes).vector-recording = true",
              "*.r1.ppp[0].queue.queueingTime:vector.vector-recording = true",
              "*.r1.ppp[0].queue.droppedPacketsQueueOverflow:count.scalar-recording = true",
              "**.receiver[*].app[0].packetReceived.statistic-recording = true",
              "*.r1.ppp[0].queue.queueingTime.statistic-recording = true",
              "*.r1.ppp[0].queue.droppedPacketsQueueOverflow.statistic-recording = true",
              f"output-vector-file = {workdir}/ep_{sc.seed}.vec",
              f"output-scalar-file = {workdir}/ep_{sc.seed}.sca"]
    # episode settings come first so they take precedence over the base file
    o.append(f"include {BASE_INI}")
    with open(ini_path, "w") as f:
        f.write("\n".join(o) + "\n")
    return ini_path


def sample_steps(seed: int, duration: float = 40.0) -> Scenario:
    """Four capacity steps (alternating up/down by 2.5-5x) at fixed times; no cross traffic."""
    rng = np.random.default_rng(seed)
    d = float(rng.uniform(0.005, 0.050))
    lo = _loguniform(rng, 4e6, 15e6)
    hi = lo * float(rng.uniform(2.5, 5.0))
    start_high = rng.random() < 0.5
    cap = []
    for k, t in enumerate([0.0, 8.0, 16.0, 24.0, 32.0]):
        c = hi if (k % 2 == 0) == start_high else lo
        cap.append((t, c))
    base_rtt = 2 * (d + 2 * ACCESS_DELAY_S)
    buf = int(max(16, round(float(rng.uniform(0.5, 2.0)) * hi * base_rtt / (8 * PKT_BYTES))))
    return Scenario(seed=seed, duration=duration, bottleneck_delay=d, capacity=cap, buffer_pkts=buf,
                    label="steps")
