// SPDX-License-Identifier: GPL-2.0
//
// stcc.bpf.c -- the delay-tolerance executor of the paper as a Linux TCP
// congestion-control module (BPF struct_ops). Port of StTcp.cc (OMNeT++/INET).
//
//   * theta = (tolerance, incGain, decGain, lossBeta) is written by the
//     user-space link (stcc_link.c) into `params`, keyed by local port;
//   * window rules run once per round-trip epoch, as in StTcp::onEpochEnd;
//   * a two-sided Page-Hinkley detector on queueing ratio and log delivery
//     rate (fixed point, milli-units) raises an early-decision event through
//     a ring buffer; an RTO raises one too;
//   * cumulative interval statistics are exported in `stats` for the link to
//     difference at each decision.
// Loss detection and recovery (SACK, RACK-TLP, PRR, F-RTO) are Linux's own.
//
#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

char _license[] SEC("license") = "GPL";

#define TCP_CA_Open 0
#define TCP_CA_Recovery 3
#define TCP_CA_Loss 4
#define TCP_INFINITE_SSTHRESH 0x7fffffff
#define NBUCKETS 16      /* array size (power of two) */
#define WINDOW_S 10      /* min-RTT window, seconds   */
#define NSEC_PER_SEC 1000000000ULL

extern void tcp_reno_cong_avoid(struct sock *sk, __u32 ack, __u32 acked) __ksym;
extern __u32 tcp_reno_undo_cwnd(struct sock *sk) __ksym;
extern __u32 tcp_slow_start(struct tcp_sock *tp, __u32 acked) __ksym;

struct stcc_params {            /* written by user space */
	__u32 tol_milli;       /* allowed queueing delay, x base RTT, milli   */
	__u32 inc_milli;       /* additive increase, milli-MSS per RTT        */
	__u32 dec_milli;       /* proportional decrease gain, milli           */
	__u32 beta_milli;      /* loss back-off factor, milli                 */
	__u32 gen;             /* decision generation (bumped at each decision)*/
	__u32 tripwire;        /* 1 = change detector may trigger decisions   */
};

struct stcc_stats {             /* read by user space; cumulative unless noted */
	__u64 bytes_acked;
	__u64 rtt_sum_us;
	__u64 rtt_cnt;
	__u64 max_rate;        /* bytes/s, slowly decaying maximum            */
	__u64 last_rate;       /* bytes/s, last epoch                         */
	__u32 rtt_min_us;      /* since last decision                         */
	__u32 rtt_max_us;      /* since last decision                         */
	__u32 loss_events;
	__u32 rtos;
	__u32 base_us;
	__u32 last_rtt_us;
	__u32 cwnd_pkts;
	__u32 mss;
	__u32 inflight;        /* bytes                                       */
	__u32 gen_seen;
	__u32 comp_enters;     /* times competition mode was entered          */
	__u32 comp_epochs;     /* epochs spent in competition mode            */
};

struct stcc_ph {                /* Page-Hinkley, milli-units */
	__s64 mean, n, up, down;
};

struct stcc_state {
	__s64 bucket_id[NBUCKETS];
	__u32 bucket_min[NBUCKETS];
	__u64 epoch_start_ns;
	__u64 epoch_bytes0;
	__u64 gen_ns;
	__u64 cwnd_milli;
	__u32 epoch_end_seq;
	__u32 epoch_min_us;
	__u32 epoch_active;
	__u32 gen;
	__u32 fired;
	__u32 ca_state;
	__u32 over;            /* consecutive epochs with queue above tolerance */
	__u32 comp;            /* 1 = competition mode                          */
	__u32 probe;           /* 0 none, 1-2 epochs into a drain probe         */
	__u32 pcount;          /* epochs in competition mode since last probe   */
	__u64 q_ref;           /* queue before the probe (us)                   */
	__u32 cwnd_saved;
	__u64 q_prev;
	__u16 port;
	struct stcc_ph phq, phr;
};

struct stcc_event {
	__u32 type;            /* 1 start, 2 trigger, 3 end */
	__u32 port;
	__u32 reason;          /* 1 change detector, 2 RTO */
	__u32 pad;
	__u64 ts_ns;
};

struct {
	__uint(type, BPF_MAP_TYPE_SK_STORAGE);
	__uint(map_flags, BPF_F_NO_PREALLOC);
	__type(key, int);
	__type(value, struct stcc_state);
} st_map SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, __u32);
	__type(value, struct stcc_params);
} params SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_HASH);
	__uint(max_entries, 1024);
	__type(key, __u32);
	__type(value, struct stcc_stats);
} stats SEC(".maps");

struct {
	__uint(type, BPF_MAP_TYPE_RINGBUF);
	__uint(max_entries, 1 << 16);
} events SEC(".maps");

/* PH thresholds (milli-units), same values as StTcp.h */
const volatile __s64 PH_DQ = 100, PH_HQ = 1500, PH_DR = 50, PH_HR = 800;
const volatile __s64 PH_WARMUP = 3;
const volatile __u64 MIN_GAP_NS = 50000000ULL;
/* competition mode: entered after COMP_K consecutive epochs whose queueing
 * delay exceeds the tolerance even though the window keeps shrinking (a
 * queue the flow cannot drain is someone else's); left once an epoch sees
 * the queue nearly empty (q < base / COMP_EXIT_DIV). While on, the window
 * follows AIMD with Reno's back-off so that the queue drains on loss when
 * the flow is alone. */
const volatile __u32 COMP_ENABLE = 0;
const volatile __u32 COMP_PROBE = 16;
const volatile __u32 COMP_DROP_DIV = 4;
const volatile __u32 COMP_K = 10;
const volatile __u32 COMP_EXIT_DIV = 10;
const volatile __u32 COMP_BETA_MILLI = 500;

static __always_inline struct tcp_sock *tcp_sk(const struct sock *sk)
{
	return (struct tcp_sock *)sk;
}

static __always_inline __s64 sdiv(__s64 a, __s64 b)
{
	if (b <= 0)
		return 0;
	if (a < 0)
		return -(__s64)((__u64)(-a) / (__u64)b);
	return (__s64)((__u64)a / (__u64)b);
}

/* natural log in milli-units, |error| < 0.005 */
static __always_inline __s64 ln_milli(__u64 v)
{
	__u32 k = 0;
	__u64 x;
	__s64 l2;
	if (v == 0)
		return 0;
	for (int i = 0; i < 63; i++) {
		if ((v >> (k + 1)) == 0)
			break;
		k++;
	}
	if (k >= 20)
		x = (v - (1ULL << k)) >> (k - 10);      /* fraction * 1024 */
	else
		x = ((v - (1ULL << k)) << 10) >> k;
	x = x * 1000 / 1024;                        /* fraction, milli */
	/* log2(1+x) ~ x (1.3466 - 0.3466 x) */
	l2 = (__s64)k * 1000 + (__s64)(x * (1347 - 347 * x / 1000) / 1000);
	return l2 * 693 / 1000;
}

static __always_inline bool ph_update(struct stcc_ph *p, __s64 x, __s64 delta, __s64 h)
{
	__s64 t;
	p->n += 1;
	p->mean += sdiv(x - p->mean, p->n);
	if (p->n <= PH_WARMUP)
		return false;
	t = p->up + x - p->mean - delta;
	p->up = t > 0 ? t : 0;
	t = p->down + p->mean - x - delta;
	p->down = t > 0 ? t : 0;
	return p->up > h || p->down > h;
}

static __always_inline void emit(__u32 type, __u32 port, __u32 reason)
{
	struct stcc_event *e = bpf_ringbuf_reserve(&events, sizeof(*e), 0);
	if (!e)
		return;
	e->type = type;
	e->port = port;
	e->reason = reason;
	e->pad = 0;
	e->ts_ns = bpf_ktime_get_ns();
	bpf_ringbuf_submit(e, 0);
}

static __always_inline __u32 base_rtt(struct stcc_state *s, struct stcc_stats *x)
{
	__u32 m = 0xffffffff;
	for (int i = 0; i < NBUCKETS; i++)
		if (s->bucket_id[i] >= 0 && s->bucket_min[i] < m)
			m = s->bucket_min[i];
	if (m == 0xffffffff)
		m = x->last_rtt_us ? x->last_rtt_us : 100000;
	return m;
}

static __always_inline struct stcc_params *get_params(struct stcc_state *s)
{
	__u32 key = s->port;
	return bpf_map_lookup_elem(&params, &key);
}

static __always_inline struct stcc_stats *get_stats(struct stcc_state *s)
{
	__u32 key = s->port;
	return bpf_map_lookup_elem(&stats, &key);
}

SEC("struct_ops")
void BPF_PROG(stcc_init, struct sock *sk)
{
	struct stcc_state *s = bpf_sk_storage_get(&st_map, sk, 0, BPF_SK_STORAGE_GET_F_CREATE);
	struct stcc_params p = { .tol_milli = 250, .inc_milli = 1000, .dec_milli = 250,
			    .beta_milli = 700, .gen = 0, .tripwire = 1 };
	struct stcc_stats z = {};
	__u32 key;
	if (!s)
		return;
	__builtin_memset(s, 0, sizeof(*s));
	for (int i = 0; i < NBUCKETS; i++)
		s->bucket_id[i] = -1;
	s->port = sk->__sk_common.skc_num;
	s->epoch_min_us = 0xffffffff;
	s->gen_ns = bpf_ktime_get_ns();
	key = s->port;
	z.rtt_min_us = 0xffffffff;
	z.mss = tcp_sk(sk)->mss_cache;
	bpf_map_update_elem(&params, &key, &p, BPF_NOEXIST);
	bpf_map_update_elem(&stats, &key, &z, BPF_ANY);
	emit(1, key, 3);
}

SEC("struct_ops")
void BPF_PROG(stcc_release, struct sock *sk)
{
	struct stcc_state *s = bpf_sk_storage_get(&st_map, sk, 0, 0);
	if (s)
		emit(3, s->port, 4);
}

SEC("struct_ops")
__u32 BPF_PROG(stcc_ssthresh, struct sock *sk)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct stcc_state *s = bpf_sk_storage_get(&st_map, sk, 0, 0);
	__u32 beta = 700, w;
	if (s) {
		struct stcc_params *p = get_params(s);
		if (p)
			beta = p->beta_milli;
		if (s->comp)
			beta = COMP_BETA_MILLI;
	}
	w = (__u32)((__u64)tp->snd_cwnd * beta / 1000);
	return w > 2 ? w : 2;
}

SEC("struct_ops")
__u32 BPF_PROG(stcc_undo, struct sock *sk)
{
	return tcp_reno_undo_cwnd(sk);
}

SEC("struct_ops")
void BPF_PROG(stcc_set_state, struct sock *sk, __u8 new_state)
{
	struct stcc_state *s = bpf_sk_storage_get(&st_map, sk, 0, 0);
	struct stcc_stats *x;
	struct stcc_params *p;
	if (!s)
		return;
	x = get_stats(s);
	p = get_params(s);
	if (x && new_state == TCP_CA_Recovery && s->ca_state != TCP_CA_Recovery)
		x->loss_events++;
	if (x && new_state == TCP_CA_Loss && s->ca_state != TCP_CA_Loss) {
		x->rtos++;
		s->epoch_active = 0;
		if (p && p->tripwire &&
		    bpf_ktime_get_ns() - s->gen_ns >= MIN_GAP_NS && !s->fired) {
			s->fired = 1;
			emit(2, s->port, 2);
		}
	}
	s->ca_state = new_state;
}

/* slow start only; congestion avoidance runs per epoch in pkts_acked */
SEC("struct_ops")
void BPF_PROG(stcc_cong_avoid, struct sock *sk, __u32 ack, __u32 acked)
{
	struct tcp_sock *tp = tcp_sk(sk);
	if (tp->snd_cwnd < tp->snd_ssthresh)
		tcp_slow_start(tp, acked);
}

static __always_inline void epoch_end(struct sock *sk, struct stcc_state *s, struct stcc_stats *x,
				      struct stcc_params *p, __u64 now)
{
	struct tcp_sock *tp = tcp_sk(sk);
	__u64 dt = now - s->epoch_start_ns;
	__u32 base = base_rtt(s, x);
	__u64 rate = 0;

	if (dt > 0) {
		rate = (tp->bytes_acked - s->epoch_bytes0) * NSEC_PER_SEC / dt;
		x->last_rate = rate;
		/* exp(-dt/10 s) ~ 1 - dt/10 s for dt << 10 s */
		__u64 dec = dt < 10 * NSEC_PER_SEC ? x->max_rate * (dt / 1000000) / 10000 : x->max_rate;
		__u64 m = x->max_rate - (dec < x->max_rate ? dec : x->max_rate);
		x->max_rate = rate > m ? rate : m;
	}

	if (s->epoch_min_us != 0xffffffff && base > 0) {
		__u64 q = s->epoch_min_us > base ? s->epoch_min_us - base : 0;
		__u64 allowed = (__u64)p->tol_milli * base / 1000;

		if (COMP_ENABLE) {
			if (s->comp) {
				x->comp_epochs++;
				if (q * COMP_EXIT_DIV < base || (s->probe && q * COMP_DROP_DIV < s->q_ref)) {
					/* queue drained, or our own drain removed most of it:
					   no elastic competitor, return to the delay rule */
					s->comp = 0;
					s->over = 0;
					s->probe = 0;
					s->pcount = 0;
				} else if (s->probe == 1) {
					s->probe = 2;
				} else if (s->probe == 2) {
					/* the queue stayed: a competitor holds it; restore */
					s->probe = 0;
					tp->snd_cwnd = s->cwnd_saved;
					s->cwnd_milli = (__u64)s->cwnd_saved * 1000;
				} else if (COMP_PROBE && ++s->pcount >= COMP_PROBE && s->epoch_min_us > 0) {
					/* drain probe: cut to the window that would leave no queue */
					__u64 w = (__u64)tp->snd_cwnd * base * 9 / 10 / s->epoch_min_us;
					s->pcount = 0;
					s->probe = 1;
					s->q_ref = q;
					s->cwnd_saved = tp->snd_cwnd;
					if (w < 2)
						w = 2;
					tp->snd_cwnd = (__u32)w;
					s->cwnd_milli = w * 1000;
				}
			} else if (q > allowed && q * COMP_EXIT_DIV >= base && tp->snd_cwnd >= tp->snd_ssthresh) {
				/* count only epochs in which the queue did not shrink although
				   the delay rule is backing off */
				if (q * 10 >= s->q_prev * 9 && ++s->over >= COMP_K) {
					s->comp = 1;
					x->comp_enters++;
					s->cwnd_milli = (__u64)tp->snd_cwnd * 1000;
				}
			} else {
				s->over = 0;
			}
		}

		s->q_prev = q;
		if (s->ca_state == TCP_CA_Open && s->comp) {
			/* AIMD: +1 MSS per epoch regardless of delay */
			__u64 cm = s->cwnd_milli;
			if (cm / 1000 != tp->snd_cwnd)
				cm = (__u64)tp->snd_cwnd * 1000;
			if (tp->snd_cwnd >= tp->snd_ssthresh && !s->probe)
				cm += 1000;
			s->cwnd_milli = cm;
			tp->snd_cwnd = (__u32)(cm / 1000);
		} else if (s->ca_state == TCP_CA_Open) {
			if (tp->snd_cwnd < tp->snd_ssthresh) {
				if (q > allowed)
					tp->snd_ssthresh = tp->snd_cwnd;
			} else {
				__u64 cm = s->cwnd_milli;
				if (cm / 1000 != tp->snd_cwnd)
					cm = (__u64)tp->snd_cwnd * 1000;
				if (q <= allowed) {
					cm += p->inc_milli;
				} else {
					__u64 frac = (__u64)p->dec_milli * (q - allowed) / (s->epoch_min_us ? s->epoch_min_us : 1);
					if (frac > 500)
						frac = 500;
					cm = cm * (1000 - frac) / 1000;
				}
				if (cm < 2000)
					cm = 2000;
				if (cm > 100000000)
					cm = 100000000;
				s->cwnd_milli = cm;
				tp->snd_cwnd = (__u32)(cm / 1000);
				if (tp->snd_ssthresh > tp->snd_cwnd)
					tp->snd_ssthresh = tp->snd_cwnd;
			}
		}

		if (rate > 0) {
			__s64 xq = (__s64)(q * 1000 / base);
			__s64 xr = ln_milli(rate);
			bool fq = ph_update(&s->phq, xq, PH_DQ, PH_HQ);
			bool fr = ph_update(&s->phr, xr, PH_DR, PH_HR);
			if ((fq || fr) && p->tripwire && !s->fired &&
			    now - s->gen_ns >= MIN_GAP_NS) {
				s->fired = 1;
				emit(2, s->port, 1);
			}
		}
	}
	s->epoch_end_seq = tp->snd_nxt;
	s->epoch_start_ns = now;
	s->epoch_min_us = 0xffffffff;
	s->epoch_bytes0 = tp->bytes_acked;
}

SEC("struct_ops")
void BPF_PROG(stcc_pkts_acked, struct sock *sk, const struct ack_sample *sample)
{
	struct tcp_sock *tp = tcp_sk(sk);
	struct stcc_state *s = bpf_sk_storage_get(&st_map, sk, 0, 0);
	struct stcc_params *p;
	struct stcc_stats *x;
	__u64 now = bpf_ktime_get_ns();
	__s32 rtt;
	__u32 base;

	if (!s)
		return;
	p = get_params(s);
	x = get_stats(s);
	if (!p || !x)
		return;

	if (p->gen != s->gen) {            /* new decision: reset interval state */
		s->gen = p->gen;
		s->gen_ns = now;
		s->fired = 0;
		__builtin_memset(&s->phq, 0, sizeof(s->phq));
		__builtin_memset(&s->phr, 0, sizeof(s->phr));
		x->rtt_min_us = 0xffffffff;
		x->rtt_max_us = 0;
		x->gen_seen = p->gen;
	}

	rtt = sample->rtt_us;
	if (rtt > 0) {
		__s64 id = (__s64)(now / NSEC_PER_SEC);
		__u32 slot = (__u32)id & (NBUCKETS - 1);
		{
			if (s->bucket_id[slot] != id) {
				s->bucket_id[slot] = id;
				s->bucket_min[slot] = rtt;
			} else if ((__u32)rtt < s->bucket_min[slot]) {
				s->bucket_min[slot] = rtt;
			}
		}
		for (int i = 0; i < NBUCKETS; i++)
			if (s->bucket_id[i] >= 0 && id - s->bucket_id[i] >= WINDOW_S)
				s->bucket_id[i] = -1;
		x->last_rtt_us = rtt;
		x->rtt_sum_us += rtt;
		x->rtt_cnt++;
		if ((__u32)rtt < x->rtt_min_us)
			x->rtt_min_us = rtt;
		if ((__u32)rtt > x->rtt_max_us)
			x->rtt_max_us = rtt;
		if ((__u32)rtt < s->epoch_min_us)
			s->epoch_min_us = rtt;
		/* per-sample delay-based slow-start exit */
		base = base_rtt(s, x);
		if (s->ca_state == TCP_CA_Open && tp->snd_cwnd < tp->snd_ssthresh &&
		    (__u64)(rtt - base) * 1000 > (__u64)p->tol_milli * base && (__u32)rtt > base)
			tp->snd_ssthresh = tp->snd_cwnd;
	}

	if (!s->epoch_active) {
		s->epoch_active = 1;
		s->epoch_end_seq = tp->snd_nxt;
		s->epoch_start_ns = now;
		s->epoch_min_us = 0xffffffff;
		s->epoch_bytes0 = tp->bytes_acked;
	} else {
		__u32 lr = x->last_rtt_us;
		__u32 b = base_rtt(s, x);
		__u64 cap = 2ULL * (lr > b ? lr : b) * 1000;
		if ((__s32)(tp->snd_una - s->epoch_end_seq) >= 0 || now - s->epoch_start_ns > cap)
			epoch_end(sk, s, x, p, now);
	}

	x->bytes_acked = tp->bytes_acked;
	x->base_us = base_rtt(s, x);
	x->cwnd_pkts = tp->snd_cwnd;
	x->mss = tp->mss_cache;
	x->inflight = tp->snd_nxt - tp->snd_una;
}

SEC(".struct_ops.link")
struct tcp_congestion_ops stcc = {
	.init = (void *)stcc_init,
	.release = (void *)stcc_release,
	.ssthresh = (void *)stcc_ssthresh,
	.cong_avoid = (void *)stcc_cong_avoid,
	.undo_cwnd = (void *)stcc_undo,
	.set_state = (void *)stcc_set_state,
	.pkts_acked = (void *)stcc_pkts_acked,
	.name = "bpf_stcc",
};
