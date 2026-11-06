"""Runs one OMNeT++ episode and serves StTcp decision requests.

The executor (C++) sends a 16-double state at every decision point and
blocks until it receives a 6-double action. All learning-related logic
(observation features, reward, discounting) lives here.
"""
from __future__ import annotations

import math
import os
import selectors
import socket
import struct
import subprocess
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

import numpy as np

from scenario import Scenario, write_run_files, SIM_DIR, MSS

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INET_DIR = os.environ.get("INET_DIR", os.path.expanduser("~/sim/inet4.5"))
OPP_BIN = os.environ.get("OPP_BIN", os.path.expanduser("~/sim/omnetpp-6.0.3/bin"))

STATE_LEN, ACTION_LEN = 16, 6
HOLDS = np.array([0.05, 0.1, 0.2, 0.4, 0.8])   # seconds (mode 'st')
HOLDS_LONG = np.array([0.1, 0.2, 0.4, 0.8, 1.6])  # seconds (mode 'stl')
TIME_UNIT = 0.1                                # reward/discount unit (s)

# reward weights
W_DELAY = 0.5
W_LOSS_EVENT = 0.2
W_RTO = 1.0

OBS_DIM = 24


@dataclass
class FlowCtx:
    flow_id: int
    prev_obs_core: np.ndarray = field(default_factory=lambda: np.zeros(3, np.float32))
    prev_theta: np.ndarray = field(default_factory=lambda: np.zeros(4, np.float32))
    prev_hold_idx: int = 1
    last_t: float = 0.0
    pending: Optional[dict] = None      # transition waiting for its reward
    decisions: int = 0                  # supervisor (generator) invocations
    watcher_calls: int = 0              # watcher-only invocations
    log: List[dict] = field(default_factory=list)


def features(m: np.ndarray, ctx: FlowCtx) -> np.ndarray:
    dur = max(m[2], 1e-3)
    rate = m[3] / dur
    base = max(m[7], 1e-4)
    maxd = max(m[12], 1.0)
    reason = int(m[14])
    core = np.array([
        np.clip(math.log((rate + 1.0) / maxd), -5, 0.5),
        np.clip(m[4] / base - 1.0, 0, 5),
        np.clip(m[8] / (dur / TIME_UNIT), 0, 5),
    ], np.float32)
    bdp = maxd * base + 2 * MSS
    f = np.array([
        core[0],
        math.log1p(maxd * 8 / 1e6) / 5.0,
        core[1],
        np.clip(m[5] / base - 1.0, 0, 5),
        np.clip(m[6] / base - 1.0, 0, 5),
        math.log(base / 0.05),
        core[2],
        1.0 if m[9] > 0 else 0.0,
        np.clip(m[10] / bdp, 0, 5),
        np.clip(m[11] / max(m[10], 1.0), 0, 3),
        math.log(dur / TIME_UNIT),
        float(reason == 0), float(reason == 1), float(reason == 2), float(reason == 3),
        np.clip(math.log((m[13] + 1.0) / maxd), -5, 0.5),
        *ctx.prev_theta,
        ctx.prev_hold_idx / (len(HOLDS) - 1),
        *ctx.prev_obs_core,
    ], np.float32)
    ctx.prev_obs_core = core
    assert f.shape[0] == OBS_DIM, f.shape
    return f


def interval_reward(m: np.ndarray, sc: Scenario, t0: float) -> Dict[str, float]:
    """Reward for the interval that just ended at m[1]; uses privileged path
    knowledge (capacity schedule, true base RTT) which is never observed."""
    t1 = m[1]
    dur = max(t1 - t0, 1e-4)
    fair = sc.fair_share_bytes(t0, t1)
    util = min(m[3] / max(fair, 1.0), 1.0)
    rtt = m[4] if m[4] > 0 else m[15]
    infl = float(np.clip(rtt / sc.base_rtt - 1.0, 0.0, 4.0)) if rtt > 0 else 0.0
    rate_term = util - W_DELAY * infl
    r = rate_term * dur / TIME_UNIT - W_LOSS_EVENT * m[8] - W_RTO * m[9]
    return dict(r=float(r), util=util, infl=infl, dur=dur, loss=m[8], rto=m[9])


# -------------------------------------------------------------------------

class Episode:
    """One simulation; `act(obs, ctx, msg)` returns (action6, extra)."""

    def __init__(self, sc: Scenario, workdir: str, port: int, *, tripwire: bool,
                 num_rl: int = 1, record: bool = False, rl_class: str = "StTcp",
                 other_class: str = "TcpNewReno", static_theta=None,
                 ph_hq: float = None, ph_hr: float = None, timeout: float = 900):
        self.sc, self.workdir, self.port = sc, workdir, port
        self.num_rl, self.record = num_rl, record
        self.ini = write_run_files(sc, workdir, record=record, rl_class=rl_class,
                                   other_class=other_class, num_rl=num_rl)
        self.env = dict(os.environ)
        self.env["STCC_PORT"] = str(port)
        self.env["STCC_TRIPWIRE"] = "1" if tripwire else "0"
        if static_theta is not None:
            self.env["STCC_STATIC"] = ",".join(f"{x:.6f}" for x in static_theta)
        if ph_hq is not None:
            self.env["STCC_PH_HQ"] = str(ph_hq)
        if ph_hr is not None:
            self.env["STCC_PH_HR"] = str(ph_hr)
        self.timeout = timeout
        self.needs_link = (rl_class == "StTcp" and static_theta is None)

    def _cmd(self):
        return [os.path.join(OPP_BIN, "opp_run_release"), "-m", "-u", "Cmdenv",
                "-n", f"{ROOT}:{INET_DIR}/src",
                "-l", f"{INET_DIR}/src/INET", "-l", f"{ROOT}/src/stcc",
                "-f", self.ini, "-c", "General"]

    def run(self, act: Callable, on_reward: Callable = None) -> Dict[int, FlowCtx]:
        flows: Dict[int, FlowCtx] = {}
        log_path = os.path.join(self.workdir, f"ep_{self.sc.seed}.log")
        srv = None
        if self.needs_link:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(("127.0.0.1", self.port))
            srv.listen(16)
            srv.setblocking(False)
        with open(log_path, "w") as lf:
            proc = subprocess.Popen(self._cmd(), cwd=SIM_DIR, env=self.env,
                                    stdout=lf, stderr=subprocess.STDOUT)
        t_start = time.time()
        try:
            if srv is None:
                proc.wait(timeout=self.timeout)
            else:
                sel = selectors.DefaultSelector()
                sel.register(srv, selectors.EVENT_READ, None)
                bufs: Dict[socket.socket, bytes] = {}
                open_conns = 0
                while True:
                    if time.time() - t_start > self.timeout:
                        raise TimeoutError("episode timeout")
                    events = sel.select(timeout=0.5)
                    if not events and proc.poll() is not None and open_conns == 0:
                        break
                    for key, _ in events:
                        if key.data is None:
                            conn, _ = srv.accept()
                            conn.setblocking(True)
                            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                            sel.register(conn, selectors.EVENT_READ, "c")
                            bufs[conn] = b""
                            open_conns += 1
                            continue
                        conn = key.fileobj
                        data = conn.recv(65536)
                        if not data:
                            sel.unregister(conn)
                            conn.close()
                            open_conns -= 1
                            continue
                        bufs[conn] += data
                        while len(bufs[conn]) >= STATE_LEN * 8:
                            raw = bufs[conn][:STATE_LEN * 8]
                            bufs[conn] = bufs[conn][STATE_LEN * 8:]
                            m = np.array(struct.unpack(f"<{STATE_LEN}d", raw))
                            fid = int(m[0])
                            ctx = flows.setdefault(fid, FlowCtx(flow_id=fid, last_t=m[1]))
                            final = int(m[14]) == 4
                            if ctx.decisions > 0 or ctx.watcher_calls > 0:
                                rw = interval_reward(m, self.sc, ctx.last_t)
                                ctx.log.append(dict(t=float(m[1]), **rw, cwnd=float(m[10]),
                                                    rate=float(m[3] / max(rw["dur"], 1e-4)),
                                                    reason=int(m[14])))
                                if on_reward is not None:
                                    on_reward(ctx, rw, final, m)
                            ctx.last_t = float(m[1])
                            if final:
                                continue
                            obs = features(m, ctx)
                            a = act(obs, ctx, m)
                            conn.sendall(struct.pack(f"<{ACTION_LEN}d", *a))
                if proc.poll() is None:
                    proc.wait(timeout=60)
                sel.close()
        finally:
            if proc.poll() is None:
                proc.kill()
            if srv is not None:
                srv.close()
        if proc.returncode not in (0, None):
            tail = open(log_path).read()[-2000:]
            raise RuntimeError(f"OMNeT++ exited with {proc.returncode}:\n{tail}")
        self.wall = time.time() - t_start
        return flows
