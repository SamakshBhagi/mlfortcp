// SPDX-License-Identifier: MIT
// stcc_link.c -- user-space side of the Linux executor.
//
// Loads bpf_stcc (stcc.bpf.c) and plays the role of StTcp::decide() in the
// simulator: at every decision point it sends the supervisor the same
// 16-double state over a local TCP socket, waits for the 6-double action,
// and writes the new theta into the kernel. Decision points are the
// expiry of the hold time chosen by the supervisor and early-decision events
// raised by the in-kernel change detector or an RTO.
//
//   stcc_link --t0file F [--sup PORT] [--tripwire 0|1] [--static a,b,c,d] [--log F]
//
#include <errno.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>
#include "stcc.skel.h"

#define STATE_LEN 16
#define ACTION_LEN 6
#define MAXF 64
#define MIN_GAP_NS 50000000ULL

struct stcc_params { __u32 tol_milli, inc_milli, dec_milli, beta_milli, gen, tripwire; };
struct stcc_stats {
	__u64 bytes_acked, rtt_sum_us, rtt_cnt, max_rate, last_rate;
	__u32 rtt_min_us, rtt_max_us, loss_events, rtos, base_us, last_rtt_us, cwnd_pkts, mss, inflight, gen_seen;
	__u32 comp_enters, comp_epochs;
};
struct stcc_event { __u32 type, port, reason, pad; __u64 ts_ns; };

struct flow {
	int used, port, sock, done, clog;
	__u64 deadline, last_dec, int_start;
	__u64 p_bytes, p_rsum, p_rcnt;
	__u32 p_loss, p_rto, gen;
};

static struct flow F[MAXF];
static struct stcc *skel;
static int sup_port = 0, tripwire = 1, have_static = 0, comp = 0;
static double static_theta[4];
static __u64 t0_ns = 0;
static const char *t0file = NULL;
static FILE *dlog = NULL;
static volatile int stop = 0;

static __u64 now_ns(void)
{
	struct timespec ts;
	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (__u64)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

static double tsec(__u64 ns)
{
	if (!t0_ns && t0file) {
		FILE *f = fopen(t0file, "r");
		if (f) {
			unsigned long long v = 0;
			if (fscanf(f, "%llu", &v) == 1)
				t0_ns = v;
			fclose(f);
		}
	}
	return t0_ns ? (double)(ns - t0_ns) / 1e9 : (double)ns / 1e9;
}

static double clamp(double x, double a, double b) { return x < a ? a : (x > b ? b : x); }

static void write_theta(struct flow *f, const double *a, int bump)
{
	struct stcc_params p;
	__u32 key = f->port;
	double u[4];
	for (int i = 0; i < 4; i++)
		u[i] = (clamp(a[i], -1, 1) + 1.0) / 2.0;
	p.tol_milli = (__u32)llround(1000.0 * 0.05 * pow(40.0, u[0]));
	p.inc_milli = (__u32)llround(1000.0 * 0.25 * pow(64.0, u[1]));
	p.dec_milli = (__u32)llround(1000.0 * (0.05 + 0.45 * u[2]));
	p.beta_milli = (__u32)llround(1000.0 * (0.50 + 0.45 * u[3]));
	if (bump)
		f->gen++;
	p.gen = f->gen;
	p.tripwire = tripwire;
	bpf_map__update_elem(skel->maps.params, &key, sizeof(key), &p, sizeof(p), BPF_ANY);
}

static int read_stats(struct flow *f, struct stcc_stats *s)
{
	__u32 key = f->port;
	return bpf_map__lookup_elem(skel->maps.stats, &key, sizeof(key), s, sizeof(*s), 0);
}

static int xfer(int fd, void *buf, size_t len, int send_)
{
	char *p = buf;
	while (len) {
		ssize_t n = send_ ? send(fd, p, len, MSG_NOSIGNAL) : recv(fd, p, len, 0);
		if (n <= 0) {
			if (n < 0 && errno == EINTR)
				continue;
			return -1;
		}
		p += n;
		len -= n;
	}
	return 0;
}

static void build_msg(struct flow *f, __u64 now, int reason, double *m)
{
	struct stcc_stats s;
	memset(&s, 0, sizeof(s));
	read_stats(f, &s);
	double last = s.last_rtt_us / 1e6;
	__u64 dcnt = s.rtt_cnt - f->p_rcnt;
	m[0] = f->port;
	m[1] = tsec(now);
	m[2] = (now - f->int_start) / 1e9;
	m[3] = (double)(s.bytes_acked - f->p_bytes);
	m[4] = dcnt ? (double)(s.rtt_sum_us - f->p_rsum) / dcnt / 1e6 : last;
	m[5] = (dcnt && s.rtt_min_us != 0xffffffff) ? s.rtt_min_us / 1e6 : last;
	m[6] = (dcnt && s.rtt_max_us) ? s.rtt_max_us / 1e6 : last;
	m[7] = s.base_us ? s.base_us / 1e6 : (last > 0 ? last : 0.1);
	m[8] = s.loss_events - f->p_loss;
	m[9] = s.rtos - f->p_rto;
	m[10] = (double)s.cwnd_pkts * (s.mss ? s.mss : 1448);
	m[11] = s.inflight;
	m[12] = (double)s.max_rate;
	m[13] = (double)s.last_rate;
	m[14] = reason;
	m[15] = last;
	f->p_bytes = s.bytes_acked;
	f->p_rsum = s.rtt_sum_us;
	f->p_rcnt = s.rtt_cnt;
	f->p_loss = s.loss_events;
	f->p_rto = s.rtos;
}

static void decide(struct flow *f, int reason)
{
	__u64 now = now_ns();
	double m[STATE_LEN], a[ACTION_LEN];
	build_msg(f, now, reason, m);
	if (xfer(f->sock, m, sizeof(m), 1) || xfer(f->sock, a, sizeof(a), 0)) {
		fprintf(stderr, "stcc_link: supervisor link lost (port %d)\n", f->port);
		f->done = 1;
		return;
	}
	write_theta(f, a, 1);
	double hold = clamp(a[4], 0.01, 5.0);
	f->int_start = now;
	f->last_dec = now;
	f->deadline = now + (__u64)(hold * 1e9);
	if (dlog)
		fprintf(dlog, "%.4f,%d,%d,%.3f,%.4f,%.4f,%.4f,%.4f\n", m[1], f->port, reason, hold, a[0], a[1], a[2], a[3]);
}

static struct flow *find(int port)
{
	for (int i = 0; i < MAXF; i++)
		if (F[i].used && F[i].port == port)
			return &F[i];
	return NULL;
}

static int connect_sup(void)
{
	int s = socket(AF_INET, SOCK_STREAM, 0), one = 1;
	struct sockaddr_in a = { .sin_family = AF_INET, .sin_port = htons(sup_port) };
	inet_pton(AF_INET, "127.0.0.1", &a.sin_addr);
	if (connect(s, (struct sockaddr *)&a, sizeof(a)) < 0) {
		close(s);
		return -1;
	}
	setsockopt(s, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
	return s;
}

static int on_event(void *ctx, void *data, size_t len)
{
	struct stcc_event *e = data;
	struct flow *f = find(e->port);
	__u64 now = now_ns();
	if (e->type == 1) {                       /* new connection */
		if (!f) {
			for (int i = 0; i < MAXF; i++)
				if (!F[i].used) { f = &F[i]; break; }
			if (!f)
				return 0;
		}
		memset(f, 0, sizeof(*f));
		f->used = 1;
		f->port = e->port;
		f->int_start = now;
		f->sock = -1;
		if (have_static || sup_port == 0) {
			double a[4] = {0, 0, 0, 0};
			if (have_static)
				memcpy(a, static_theta, sizeof(a));
			write_theta(f, a, 0);
			f->deadline = ~0ULL;
			return 0;
		}
		f->sock = connect_sup();
		if (f->sock < 0) {
			fprintf(stderr, "stcc_link: cannot reach supervisor\n");
			f->done = 1;
			return 0;
		}
		decide(f, 3);
	} else if (e->type == 2) {                /* early-decision request */
		if (f && !f->done && f->sock >= 0 && now - f->last_dec >= MIN_GAP_NS)
			decide(f, e->reason);
	} else if (e->type == 3) {                /* connection closed */
		if (f && dlog && !f->clog) {
			struct stcc_stats st;
			memset(&st, 0, sizeof(st));
			if (!read_stats(f, &st)) {
				fprintf(dlog, "#comp,%d,%u,%u\n", f->port, st.comp_enters, st.comp_epochs);
				f->clog = 1;
			}
		}
		if (f && !f->done && f->sock >= 0) {
			double m[STATE_LEN];
			build_msg(f, now, 4, m);
			xfer(f->sock, m, sizeof(m), 1);
			close(f->sock);
		}
		if (f) {
			__u32 key = f->port;
			bpf_map__delete_elem(skel->maps.params, &key, sizeof(key), 0);
			bpf_map__delete_elem(skel->maps.stats, &key, sizeof(key), 0);
			f->used = 0;
		}
	}
	return 0;
}

static void on_sig(int s) { (void)s; stop = 1; }

int main(int argc, char **argv)
{
	for (int i = 1; i < argc; i++) {
		if (!strcmp(argv[i], "--sup") && i + 1 < argc) sup_port = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--tripwire") && i + 1 < argc) tripwire = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--comp") && i + 1 < argc) comp = atoi(argv[++i]);
		else if (!strcmp(argv[i], "--t0file") && i + 1 < argc) t0file = argv[++i];
		else if (!strcmp(argv[i], "--log") && i + 1 < argc) dlog = fopen(argv[++i], "w");
		else if (!strcmp(argv[i], "--static") && i + 1 < argc) {
			have_static = sscanf(argv[++i], "%lf,%lf,%lf,%lf", &static_theta[0], &static_theta[1],
					     &static_theta[2], &static_theta[3]) == 4;
		}
	}
	if (dlog) {
		setvbuf(dlog, NULL, _IOLBF, 0);
		fprintf(dlog, "t,port,reason,hold,a0,a1,a2,a3\n");
	}
	signal(SIGINT, on_sig);
	signal(SIGTERM, on_sig);

	skel = stcc__open();
	if (skel) {
		skel->rodata->COMP_ENABLE = comp ? 1 : 0;
		if (stcc__load(skel)) {
			stcc__destroy(skel);
			skel = NULL;
		}
	}
	if (!skel) {
		fprintf(stderr, "stcc_link: load failed\n");
		return 1;
	}
	struct bpf_link *link = bpf_map__attach_struct_ops(skel->maps.stcc);
	if (!link) {
		fprintf(stderr, "stcc_link: attach failed: %s\n", strerror(errno));
		return 1;
	}
	struct ring_buffer *rb = ring_buffer__new(bpf_map__fd(skel->maps.events), on_event, NULL, NULL);
	printf("READY\n");
	fflush(stdout);

	while (!stop) {
		__u64 now = now_ns(), next = now + 200000000ULL;
		for (int i = 0; i < MAXF; i++)
			if (F[i].used && !F[i].done && F[i].deadline < next)
				next = F[i].deadline;
		int ms = next > now ? (int)((next - now + 999999) / 1000000) : 0;
		ring_buffer__poll(rb, ms);
		now = now_ns();
		for (int i = 0; i < MAXF; i++)
			if (F[i].used && !F[i].done && F[i].sock >= 0 && F[i].deadline <= now)
				decide(&F[i], 0);
	}
	for (int i = 0; i < MAXF; i++)
		if (F[i].used && !F[i].clog && dlog) {
			struct stcc_stats st;
			memset(&st, 0, sizeof(st));
			if (!read_stats(&F[i], &st))
				fprintf(dlog, "#comp,%d,%u,%u\n", F[i].port, st.comp_enters, st.comp_epochs);
			F[i].clog = 1;
		}
	if (dlog)
		fflush(dlog);
	ring_buffer__free(rb);
	bpf_link__destroy(link);
	stcc__destroy(skel);
	return 0;
}
