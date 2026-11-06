import sys, json, time
sys.path.insert(0, __import__('os').path.expanduser('~/emu/py'))
from emu import run_episode
from scenario import sample_train
sc = sample_train(9_000_000 + 0, label="test")
print("scenario", sc.capacity[:4], "buf", sc.buffer_pkts, "rtt", round(sc.base_rtt*1e3,1), "udp", sc.udp_rate_bps, "comp", sc.competitor)
for m in [dict(name="ST#s1", kind="rl", run="stl_c0.1_s1", mode="stl", hold=0.1, tripwire=True),
          dict(name="CUBIC", kind="tcp", cc="cubic")]:
    t=time.time(); r = run_episode(m, sc, __import__('os').path.expanduser('~/emu/work/smoke'), keep=True)
    print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in r.items()}), round(time.time()-t,1))
