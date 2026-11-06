"""Runs inside the Mahimahi shell. Starts the flows of one scenario at the
right times relative to the emulated link's own clock.

    python3 emu_sender.py spec.json
spec: {"uplog": path, "t0file": path, "duration": s,
       "flows": [{"port": p, "cc": "bpf_stcc", "start": s, "stop": s}, ...],
       "udp": {"port": p, "rate_bps": r, "start": s, "stop": s} | null}
"""
import json
import os
import socket
import sys
import threading
import time

spec = json.load(open(sys.argv[1]))
host = os.environ["MAHIMAHI_BASE"]

# the link's clock starts at mm-link's "init timestamp" (ms, wall clock)
t0_real = None
for _ in range(400):
    try:
        for line in open(spec["uplog"]):
            if line.startswith("# init timestamp:"):
                t0_real = int(line.split(":")[1]) / 1000.0
                break
    except FileNotFoundError:
        pass
    if t0_real:
        break
    time.sleep(0.005)
if t0_real is None:
    t0_real = time.time()
t0_mono = time.monotonic_ns() - int((time.time() - t0_real) * 1e9)
with open(spec["t0file"], "w") as f:
    f.write(str(t0_mono))


def at(t):
    d = t0_real + t - time.time()
    if d > 0:
        time.sleep(d)


def tcp_flow(fl):
    at(fl["start"])
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_CONGESTION, fl["cc"].encode())
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 << 20)
    s.connect((host, fl["port"]))
    buf = b"\0" * 65536
    s.settimeout(0.2)
    end = t0_real + fl["stop"]
    while time.time() < end:
        try:
            s.send(buf)
        except socket.timeout:
            pass
        except OSError:
            break
    s.close()


def udp_flow(u):
    at(u["start"])
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    pkt = b"\0" * 1472
    gap = len(pkt) * 8 / u["rate_bps"]
    nxt = time.time()
    end = t0_real + u["stop"]
    while nxt < end:
        try:
            s.sendto(pkt, (host, u["port"]))
        except OSError:
            pass
        nxt += gap
        d = nxt - time.time()
        if d > 0:
            time.sleep(d)


th = [threading.Thread(target=tcp_flow, args=(f,), daemon=True) for f in spec["flows"]]
if spec.get("udp"):
    th.append(threading.Thread(target=udp_flow, args=(spec["udp"],), daemon=True))
for t in th:
    t.start()
at(spec["duration"] + 0.3)
