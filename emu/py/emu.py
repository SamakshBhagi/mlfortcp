"""Emulation counterpart of runtime.Episode / evaluate.run_one.

Each scenario of the paper is replayed on real Linux TCP through Mahimahi:
  * mm-delay   one-way delay = base RTT / 2 (rounded to 1 ms)
  * mm-link    uplink trace = the scenario's piecewise-constant capacity,
               drop-tail buffer of the scenario's size (packets)
  * flows      learned flow(s) with the bpf_stcc executor, or Linux CUBIC /
               BBR / Reno; NewReno competitor -> Linux Reno; UDP cross
               traffic -> constant-bit-rate UDP
The supervisor is the trained PPO policy (train.Supervisor), reached by the
user-space link (stcc_link) with the simulator's 16/6-double protocol.
"""
from __future__ import annotations

import json
import math
import os
import selectors
import shutil
import socket
import struct
import subprocess
import sys
import threading
import time
from typing import Dict

import numpy as np

SIMPY = os.path.expanduser("~/sim/stcc/py")
sys.path.insert(0, SIMPY)
import torch  # noqa: E402

torch.set_num_threads(1)
from agent import PPO  # noqa: E402
from runtime import FlowCtx, features, interval_reward, STATE_LEN, ACTION_LEN  # noqa: E402
from scenario import Scenario, ACCESS_DELAY_S  # noqa: E402
from train import Supervisor  # noqa: E402

EMU = os.path.expanduser("~/emu")
LINK_BIN = os.path.join(EMU, "bpf", "stcc_link")
SENDER = os.path.join(EMU, "py", "emu_sender.py")
RUNS = os.path.expanduser("~/sim/runs")
PKT = 1504                 # Mahimahi delivery opportunity (bytes)
ACK_TRACE_MBPS = 100       # reverse path
USER = "mm"


# ---------------------------------------------------------------------------
# scenario -> Mahimahi
# ---------------------------------------------------------------------------

def owd_ms(sc: Scenario) -> int:
    return max(1, int(round(sc.base_rtt * 1000 / 2)))


def emu_scenario(sc: Scenario) -> Scenario:
    """Same scenario with the propagation delay rounded to Mahimahi's 1 ms."""
    d = owd_ms(sc) / 1000.0 - 2 * ACCESS_DELAY_S
    j = json.loads(sc.to_json())
    j["bottleneck_delay"] = d
    return Scenario.from_json(json.dumps(j))


def write_trace(path: str, sc: Scenario, extra: float = 3.0):
    T = int((sc.duration + extra) * 1000)
    out, acc = [], 0.0
    caps = sc.capacity
    k = 0
    for ms in range(1, T + 1):
        t = (ms - 1) / 1000.0
        while k + 1 < len(caps) and caps[k + 1][0] <= t:
            k += 1
        acc += caps[k][1] / 8.0 / 1000.0
        while acc >= PKT:
            out.append(ms)
            acc -= PKT
    if not out:
        out = [T]
    with open(path, "w") as f:
        f.write("\n".join(map(str, out)) + "\n")


def write_const_trace(path: str, mbps: float, seconds: float = 2.0):
    per_ms = mbps * 1e6 / 8 / 1000 / PKT
    out, acc = [], 0.0
    for ms in range(1, int(seconds * 1000) + 1):
        acc += per_ms
        while acc >= 1:
            out.append(ms)
            acc -= 1
    with open(path, "w") as f:
        f.write("\n".join(map(str, out)) + "\n")


# ---------------------------------------------------------------------------
# receiver (host side)
# ---------------------------------------------------------------------------

class Receiver:
    def __init__(self, tcp_ports, udp_port=None):
        self.events: Dict[int, list] = {p: [] for p in tcp_ports}
        self.stop = False
        self.socks = []
        self.threads = []
        for p in tcp_ports:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 << 20)
            s.bind(("0.0.0.0", p))
            s.listen(4)
            s.settimeout(0.2)
            self.socks.append(s)
            th = threading.Thread(target=self._serve, args=(s, p), daemon=True)
            th.start()
            self.threads.append(th)
        if udp_port:
            u = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            u.bind(("0.0.0.0", udp_port))
            u.settimeout(0.2)
            self.socks.append(u)
            th = threading.Thread(target=self._sink, args=(u,), daemon=True)
            th.start()
            self.threads.append(th)

    def _serve(self, s, port):
        while not self.stop:
            try:
                c, _ = s.accept()
            except socket.timeout:
                continue
            c.settimeout(0.2)
            ev = self.events[port]
            while not self.stop:
                try:
                    d = c.recv(1 << 20)
                except socket.timeout:
                    continue
                except OSError:
                    break
                if not d:
                    break
                ev.append((time.monotonic_ns(), len(d)))
            c.close()

    def _sink(self, u):
        while not self.stop:
            try:
                u.recv(4096)
            except (socket.timeout, OSError):
                pass

    def close(self):
        self.stop = True
        for t in self.threads:
            t.join(timeout=2)
        for s in self.socks:
            s.close()


# ---------------------------------------------------------------------------
# supervisor server (same protocol as runtime.Episode)
# ---------------------------------------------------------------------------

class SupServer:
    def __init__(self, port: int, sup: Supervisor, sc: Scenario):
        self.sup, self.sc = sup, sc
        self.flows: Dict[int, FlowCtx] = {}
        self.srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.srv.bind(("127.0.0.1", port))
        self.srv.listen(16)
        self.srv.setblocking(False)
        self.stop = False
        self.th = threading.Thread(target=self._loop, daemon=True)
        self.th.start()

    def _loop(self):
        sel = selectors.DefaultSelector()
        sel.register(self.srv, selectors.EVENT_READ, None)
        bufs = {}
        n = STATE_LEN * 8
        while not self.stop:
            for key, _ in sel.select(timeout=0.2):
                if key.data is None:
                    c, _ = self.srv.accept()
                    c.setblocking(True)
                    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                    sel.register(c, selectors.EVENT_READ, "c")
                    bufs[c] = b""
                    continue
                c = key.fileobj
                try:
                    d = c.recv(65536)
                except OSError:
                    d = b""
                if not d:
                    sel.unregister(c)
                    c.close()
                    continue
                bufs[c] += d
                while len(bufs[c]) >= n:
                    raw, bufs[c] = bufs[c][:n], bufs[c][n:]
                    m = np.array(struct.unpack(f"<{STATE_LEN}d", raw))
                    fid = int(m[0])
                    ctx = self.flows.setdefault(fid, FlowCtx(flow_id=fid, last_t=m[1]))
                    final = int(m[14]) == 4
                    if ctx.decisions > 0 or ctx.watcher_calls > 0:
                        rw = interval_reward(m, self.sc, ctx.last_t)
                        ctx.log.append(dict(t=float(m[1]), **rw, cwnd=float(m[10]),
                                            rate=float(m[3] / max(rw["dur"], 1e-4)), reason=int(m[14])))
                        self.sup.on_reward(ctx, rw, final, m)
                    ctx.last_t = float(m[1])
                    if final:
                        continue
                    obs = features(m, ctx)
                    a = self.sup.act(obs, ctx, m)
                    c.sendall(struct.pack(f"<{ACTION_LEN}d", *a))
        sel.close()

    def close(self):
        self.stop = True
        self.th.join(timeout=2)
        self.srv.close()


# ---------------------------------------------------------------------------
# one episode
# ---------------------------------------------------------------------------

_agents = {}


def load_agent(run):
    if run not in _agents:
        _agents[run] = PPO.load(os.path.join(RUNS, run, "final.pt"))
    return _agents[run]


def run_episode(method: dict, sc0: Scenario, work: str, sup_port: int = 7700, keep=False) -> dict:
    """method: {"name", "kind": "rl"|"tcp"|"static", "run", "mode", "hold", "tripwire", "cc"}"""
    sc = emu_scenario(sc0)
    os.makedirs(work, exist_ok=True)
    os.chmod(work, 0o777)
    up, down = os.path.join(work, "up.trace"), os.path.join(work, "down.trace")
    write_trace(up, sc)
    write_const_trace(down, ACK_TRACE_MBPS)
    uplog = os.path.join(work, "up.log")
    t0file = os.path.join(work, "t0")
    declog = os.path.join(work, "decisions.csv")
    for p in (uplog, t0file, declog):
        if os.path.exists(p):
            os.remove(p)

    n_rl = sc.rl_flows
    rl_ports = [9001 + i for i in range(n_rl)]
    cc = method.get("cc", "bpf_stcc") if method["kind"] == "tcp" else "bpf_stcc"
    flows = [dict(port=p, cc=cc, start=sc.rl_starts[i] if i < len(sc.rl_starts) else 0.1,
                  stop=sc.duration) for i, p in enumerate(rl_ports)]
    comp_port = None
    if sc.competitor:
        comp_port = 9100
        flows.append(dict(port=comp_port, cc=method.get("comp_cc", "reno"),
                          start=sc.competitor[0], stop=sc.competitor[1]))
    udp = None
    if sc.udp_rate_bps > 0:
        udp = dict(port=9200, rate_bps=sc.udp_rate_bps, start=sc.udp_on[0], stop=sc.udp_on[1])
    spec = dict(uplog=uplog, t0file=t0file, duration=sc.duration, flows=flows, udp=udp)
    spec_path = os.path.join(work, "spec.json")
    json.dump(spec, open(spec_path, "w"))

    rx = Receiver([f["port"] for f in flows], 9200 if udp else None)
    srv, link, sup = None, None, None
    comp_log = []
    # unprivileged sockets may only pick algorithms on this list (reset by reboots)
    with open("/proc/sys/net/ipv4/tcp_allowed_congestion_control", "w") as f:
        f.write("reno cubic bbr")
    try:
        if cc == "bpf_stcc":
            args = [LINK_BIN, "--t0file", t0file, "--log", declog]
            if method["kind"] == "rl":
                agent = load_agent(method["run"])
                sup = Supervisor(agent, method["mode"], method.get("cost", 0.1), method.get("hold", 0.1),
                                 deterministic=True, train=False)
                srv = SupServer(sup_port, sup, sc)
                args += ["--sup", str(sup_port), "--tripwire", "1" if method.get("tripwire") else "0"]
            else:
                th = method.get("theta", [0, 0, 0, 0])
                args += ["--static", ",".join(f"{x:.6f}" for x in th), "--tripwire", "0"]
            if method.get("comp"):
                args += ["--comp", "1"]
            link = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            line = link.stdout.readline()
            if "READY" not in line:
                raise RuntimeError(f"stcc_link failed: {line}{link.stdout.read()}")
            # a freshly registered algorithm is not in the unprivileged allow-list
            with open("/proc/sys/net/ipv4/tcp_allowed_congestion_control", "w") as f:
                f.write("reno cubic bbr bpf_stcc")
        shell = (f"mm-delay {owd_ms(sc)} mm-link {up} {down} --uplink-queue=droptail "
                 f"--uplink-queue-args=packets={sc.buffer_pkts} --uplink-log={uplog} -- "
                 f"python3 {SENDER} {spec_path}")
        t_start = time.time()
        r = subprocess.run(["su", USER, "-c", shell], capture_output=True, text=True,
                           timeout=sc.duration + 60)
        wall = time.time() - t_start
        if r.returncode != 0:
            raise RuntimeError(f"mahimahi failed: {r.stderr[-1500:]}")
        time.sleep(0.3)
    finally:
        rx.close()
        if link is not None:
            link.terminate()
            try:
                link.wait(timeout=5)
            except subprocess.TimeoutExpired:
                link.kill()
            try:
                for ln in open(declog).read().splitlines():
                    if ln.startswith("#comp,"):
                        _, port, en, ep = ln.split(",")[:4]
                        comp_log.append((int(port), int(en), int(ep)))
            except Exception:
                pass
        if srv is not None:
            srv.close()

    row = summarize(sc, rx, uplog, t0file, rl_ports, comp_port)
    if row["thr_mbps"] <= 0 or (comp_port and row.get("comp_thr_mbps", 1) <= 0):
        raise RuntimeError("a flow delivered no data (congestion control not selectable?)")
    row.update(method=method["name"], seed=sc0.seed, label=sc0.label, wall=wall,
               owd_ms=owd_ms(sc), C0=sc.capacity[0][1] / 1e6, n_changes=len(sc.capacity) - 1,
               base_rtt_ms=sc.base_rtt * 1e3, udp=sc.udp_rate_bps > 0, comp=sc.competitor is not None)
    # supervisor statistics (first learned flow)
    T = sc.duration - 0.1
    if sup is not None and srv.flows:
        ctx = srv.flows[min(srv.flows)]
        row["gen_rate"] = ctx.decisions / T
        row["watch_rate"] = ctx.watcher_calls / T
        row["early_rate"] = sum(1 for x in ctx.log if x.get("decision_reason") in (1, 2)) / T
        holds = [x["hold"] for x in ctx.log if "hold" in x]
        row["mean_hold"] = float(np.mean(holds)) if holds else np.nan
    else:
        row["gen_rate"] = row["watch_rate"] = row["early_rate"] = 0.0
        row["mean_hold"] = np.nan
    row["calls"] = row["gen_rate"] + row["watch_rate"]
    if comp_log:
        row["comp_enters"] = sum(x[1] for x in comp_log)
        row["comp_epochs"] = sum(x[2] for x in comp_log)
    if not keep:
        shutil.rmtree(work, ignore_errors=True)
    return row


def summarize(sc: Scenario, rx: Receiver, uplog: str, t0file: str, rl_ports, comp_port,
              t_from: float = 1.0) -> dict:
    t0 = int(open(t0file).read().strip())
    out = {}
    span = sc.duration - t_from

    def delivered(port, a=t_from, b=None):
        b = sc.duration if b is None else b
        tot = 0
        for (ts, n) in rx.events.get(port, []):
            t = (ts - t0) / 1e9
            if a <= t < b:
                tot += n
        return tot

    d0 = delivered(rl_ports[0])
    out["thr_mbps"] = d0 * 8 / span / 1e6
    fair = sc.fair_share_bytes(t_from, sc.duration)
    out["util_fair"] = d0 / max(fair, 1.0)
    for i, p in enumerate(rl_ports[1:], start=1):
        out[f"thr{i}_mbps"] = delivered(p) * 8 / span / 1e6
    if comp_port:
        out["comp_thr_mbps"] = delivered(comp_port) * 8 / span / 1e6

    # shares over the window in which every flow is active (starts + 1 s settling)
    starts = list(getattr(sc, "rl_starts", None) or [0.0])[:len(rl_ports)]
    if comp_port and getattr(sc, "competitor", None):
        starts.append(sc.competitor[0])
    ov_a = max(starts) + 1.0
    ports = list(rl_ports) + ([comp_port] if comp_port else [])
    if len(ports) > 1 and ov_a < sc.duration - 1.0:
        x = np.array([delivered(p, ov_a, sc.duration) for p in ports], float)
        out["ov_from"] = ov_a
        out["ov_share0"] = float(x[0] / max(x.sum(), 1.0))
        out["ov_jain"] = float(x.sum() ** 2 / (len(x) * max((x ** 2).sum(), 1.0)))
        out["ov_rl_jain"] = float(x[:len(rl_ports)].sum() ** 2 /
                                  (len(rl_ports) * max((x[:len(rl_ports)] ** 2).sum(), 1.0)))

    # queueing delay of every packet leaving the bottleneck (all flows)
    qs, drops, base_ms = [], 0, None
    with open(uplog) as f:
        for line in f:
            if line.startswith("#"):
                if line.startswith("# base timestamp:"):
                    base_ms = int(line.split(":")[1])
                continue
            p = line.split()
            if len(p) < 3:
                continue
            t = int(p[0]) / 1000.0   # ms since mm-link init (= scenario t=0)
            if t < t_from:
                continue
            if p[1] == "-" and len(p) >= 4:
                qs.append(int(p[3]))
            elif p[1] == "d":
                drops += 1
    qs = np.array(qs if qs else [0.0], float)
    out["qdelay_mean_ms"] = float(qs.mean())
    out["qdelay_p95_ms"] = float(np.percentile(qs, 95))
    out["rtt_mean_ms"] = sc.base_rtt * 1e3 + out["qdelay_mean_ms"]
    out["infl_mean"] = out["qdelay_mean_ms"] / (sc.base_rtt * 1e3)
    out["drops"] = drops
    tot = sum(delivered(p) for p in rx.events)
    avail = sum(c * (b - a) / 8 for a, b, c in sc._segments(t_from, sc.duration))
    out["link_util"] = tot / max(avail, 1.0)
    return out
