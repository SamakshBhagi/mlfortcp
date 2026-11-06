#include "StTcp.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iostream>

#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <unistd.h>

#include "inet/transportlayer/tcp/Tcp.h"

namespace inet {
namespace tcp {

Register_Class(StTcp);

// Wire format (all little-endian doubles)
//   executor -> supervisor : STATE_LEN values, see decide()
//   supervisor -> executor : ACTION_LEN values [a0 a1 a2 a3 hold_s reserved]
static constexpr int STATE_LEN = 16;
static constexpr int ACTION_LEN = 6;

static double envDouble(const char *name, double def)
{
    const char *v = getenv(name);
    return (v && *v) ? atof(v) : def;
}

StTcp::StTcp() : TcpNewReno()
{
    for (int i = 0; i < MINRTT_BUCKETS; i++) {
        bucketMin[i] = 1e9;
        bucketId[i] = -1;
    }
}

StTcp::~StTcp()
{
    if (decisionTimer)
        delete cancelEvent(decisionTimer);
    if (sock >= 0) {
        // final report without waiting for a reply, so the supervisor can
        // account for the last (truncated) interval
        if (linkUp) {
            triggerReason = 4;
            double msg[STATE_LEN] = {0};
            double dur = (simTime() - intervalStart).dbl();
            msg[0] = flowId; msg[1] = simTime().dbl(); msg[2] = dur;
            msg[3] = (double)intervalDelivered;
            msg[4] = intervalRttCount ? intervalRttSum / intervalRttCount : lastRtt;
            msg[5] = intervalRttCount ? intervalRttMin : lastRtt;
            msg[6] = intervalRttCount ? intervalRttMax : lastRtt;
            msg[7] = baseRtt(); msg[8] = intervalLossEvents; msg[9] = intervalRtos;
            msg[10] = 0; msg[11] = 0;
            msg[12] = maxDeliveryRate; msg[13] = lastEpochRate; msg[14] = triggerReason; msg[15] = lastRtt;
            sendAll(msg, sizeof(msg));
        }
        close(sock);
        sock = -1;
    }
}

void StTcp::initialize()
{
    TcpNewReno::initialize();
    decisionTimer = new cMessage("STCC_DECISION");
    decisionTimer->setContextPointer(conn);

    tripwireEnabled = envDouble("STCC_TRIPWIRE", 1.0) > 0.5;
    phHQ = envDouble("STCC_PH_HQ", phHQ);
    phHR = envDouble("STCC_PH_HR", phHR);

    const char *st = getenv("STCC_STATIC");
    if (st && *st) {
        staticMode = true;
        std::stringstream ss(st);
        std::string tok;
        int i = 0;
        while (std::getline(ss, tok, ',') && i < 4)
            staticTheta[i++] = atof(tok.c_str());
    }
}

// ---------------------------------------------------------------------------
// supervisor link
// ---------------------------------------------------------------------------

bool StTcp::sendAll(const void *buf, size_t len)
{
    const char *p = static_cast<const char *>(buf);
    while (len > 0) {
        ssize_t n = ::send(sock, p, len, MSG_NOSIGNAL);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

bool StTcp::recvAll(void *buf, size_t len)
{
    char *p = static_cast<char *>(buf);
    while (len > 0) {
        ssize_t n = ::recv(sock, p, len, 0);
        if (n <= 0) {
            if (n < 0 && errno == EINTR)
                continue;
            return false;
        }
        p += n;
        len -= n;
    }
    return true;
}

void StTcp::connectSupervisor()
{
    int port = (int)envDouble("STCC_PORT", 5000);
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0)
        throw cRuntimeError("StTcp: socket() failed: %s", strerror(errno));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(port);
    inet_pton(AF_INET, "127.0.0.1", &addr.sin_addr);
    if (::connect(sock, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0)
        throw cRuntimeError("StTcp: cannot reach supervisor on port %d: %s", port, strerror(errno));
    int one = 1;
    setsockopt(sock, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    // the supervisor may be busy with a gradient update; wait generously
    timeval tv{};
    tv.tv_sec = 3600;
    setsockopt(sock, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    linkUp = true;
}

void StTcp::applyTheta(const double *a)
{
    auto u = [](double x) { return (std::clamp(x, -1.0, 1.0) + 1.0) / 2.0; };
    tolerance = 0.05 * std::pow(40.0, u(a[0]));   // [0.05, 2.0] x base RTT
    incGain = 0.25 * std::pow(64.0, u(a[1]));     // [0.25, 16] MSS / RTT
    decGain = 0.05 + 0.45 * u(a[2]);              // [0.05, 0.50]
    lossBeta = 0.50 + 0.45 * u(a[3]);             // [0.50, 0.95]
}

void StTcp::decide()
{
    simtime_t now = simTime();
    double dur = (now - intervalStart).dbl();

    double msg[STATE_LEN];
    msg[0] = flowId;
    msg[1] = now.dbl();
    msg[2] = dur;
    msg[3] = (double)intervalDelivered;
    msg[4] = intervalRttCount ? intervalRttSum / intervalRttCount : lastRtt;
    msg[5] = intervalRttCount ? intervalRttMin : lastRtt;
    msg[6] = intervalRttCount ? intervalRttMax : lastRtt;
    msg[7] = baseRtt();
    msg[8] = intervalLossEvents;
    msg[9] = intervalRtos;
    msg[10] = state->snd_cwnd;
    msg[11] = state->snd_max - state->snd_una;
    msg[12] = maxDeliveryRate;
    msg[13] = lastEpochRate;
    msg[14] = triggerReason;
    msg[15] = lastRtt;

    double act[ACTION_LEN];
    if (!sendAll(msg, sizeof(msg)) || !recvAll(act, sizeof(act)))
        throw cRuntimeError("StTcp: supervisor link lost at t=%s", now.str().c_str());

    applyTheta(act);
    double hold = std::clamp(act[4], 0.01, 5.0);

    // reset the decision interval
    intervalStart = now;
    intervalDelivered = 0;
    intervalRttSum = 0.0;
    intervalRttCount = 0;
    intervalRttMin = 1e9;
    intervalRttMax = 0.0;
    intervalLossEvents = 0;
    intervalRtos = 0;
    phQueue.reset();
    phRate.reset();
    lastDecision = now;
    triggerReason = 0;

    if (decisionTimer->isScheduled())
        cancelEvent(decisionTimer);
    conn->scheduleAfter(hold, decisionTimer);
}

void StTcp::requestEarlyDecision(int reason)
{
    if (staticMode || !linkUp || !tripwireEnabled)
        return;
    if ((simTime() - lastDecision).dbl() < minGap)
        return;
    triggerReason = reason;
    decide();
}

void StTcp::established(bool active)
{
    TcpNewReno::established(active);
    flowId = conn->localPort;
    intervalStart = simTime();
    lastDecision = simTime();
    if (staticMode) {
        applyTheta(staticTheta);
        return;
    }
    connectSupervisor();
    triggerReason = 3;
    decide();
}

void StTcp::connectionClosed()
{
    if (decisionTimer && decisionTimer->isScheduled())
        cancelEvent(decisionTimer);
    TcpNewReno::connectionClosed();
}

void StTcp::processTimer(cMessage *timer, TcpEventCode& event)
{
    if (timer == decisionTimer) {
        triggerReason = 0;
        decide();
        return;
    }
    TcpNewReno::processTimer(timer, event);
}

// ---------------------------------------------------------------------------
// RTT sampling (one sample per transmitted burst of new data, Karn's rule)
// ---------------------------------------------------------------------------

void StTcp::dataSent(uint32_t fromseq)
{
    TcpNewReno::dataSent(fromseq);
    uint32_t end = state->snd_nxt;
    // only bursts that start at or beyond the highest byte ever sent are new
    // data; anything else is a retransmission and is not timed
    if (!haveHighestSent || !seqLess(fromseq, highestSentEnd)) {
        if (seqGreater(end, fromseq)) {
            sendTimes.emplace_back(end, simTime());
            highestSentEnd = end;
            haveHighestSent = true;
        }
    }
    else if (seqGreater(end, highestSentEnd)) {
        highestSentEnd = end;
    }
    if (!epochActive) {
        epochActive = true;
        epochEndSeq = state->snd_max;
        epochStart = simTime();
        epochMinRtt = 1e9;
        epochDelivered = 0;
    }
}

void StTcp::segmentRetransmitted(uint32_t fromseq, uint32_t toseq)
{
    TcpNewReno::segmentRetransmitted(fromseq, toseq);
    // invalidate samples whose bytes may have been retransmitted
    for (auto it = sendTimes.begin(); it != sendTimes.end();) {
        if (seqGreater(it->first, fromseq))
            it = sendTimes.erase(it);
        else
            ++it;
    }
}

double StTcp::baseRtt() const
{
    double m = 1e9;
    for (int i = 0; i < MINRTT_BUCKETS; i++)
        if (bucketId[i] >= 0)
            m = std::min(m, bucketMin[i]);
    return m < 1e8 ? m : (lastRtt > 0 ? lastRtt : 0.1);
}

void StTcp::onRttSample(double rtt)
{
    lastRtt = rtt;
    long id = (long)std::floor(simTime().dbl());
    int slot = (int)(id % MINRTT_BUCKETS);
    if (bucketId[slot] != id) {
        bucketId[slot] = id;
        bucketMin[slot] = rtt;
    }
    else
        bucketMin[slot] = std::min(bucketMin[slot], rtt);
    // drop buckets older than the window
    for (int i = 0; i < MINRTT_BUCKETS; i++)
        if (bucketId[i] >= 0 && id - bucketId[i] >= MINRTT_BUCKETS)
            bucketId[i] = -1;

    intervalRttSum += rtt;
    intervalRttCount++;
    intervalRttMin = std::min(intervalRttMin, rtt);
    intervalRttMax = std::max(intervalRttMax, rtt);
    epochMinRtt = std::min(epochMinRtt, rtt);

    // HyStart-style delay-based slow-start exit, checked on every sample:
    // within a slow-start round the queue drains between bursts, so the
    // per-round minimum would hide it.
    if (!state->lossRecovery && state->snd_cwnd < state->ssthresh) {
        double base = baseRtt();
        if (rtt - base > tolerance * base) {
            state->ssthresh = state->snd_cwnd;
            conn->emit(ssthreshSignal, state->ssthresh);
        }
    }
}

// ---------------------------------------------------------------------------
// executor: once per round-trip epoch
// ---------------------------------------------------------------------------

void StTcp::onEpochEnd()
{
    double dt = (simTime() - epochStart).dbl();
    double base = baseRtt();
    static bool dbg = getenv("STCC_DEBUG") != nullptr;
    if (dbg)
        std::cout << "EPOCH t=" << simTime() << " dt=" << dt << " base=" << base << " emin=" << epochMinRtt
                  << " last=" << lastRtt << " cwnd=" << state->snd_cwnd / state->snd_mss << " ssth=" << std::min<uint32_t>(state->ssthresh, 1u << 30) / state->snd_mss
                  << " rec=" << state->lossRecovery << " deliv=" << epochDelivered << " samples=" << sendTimes.size() << std::endl;
    if (dt > 0) {
        lastEpochRate = epochDelivered / dt;
        double decay = std::exp(-dt / 10.0);
        maxDeliveryRate = std::max(lastEpochRate, maxDeliveryRate * decay);
    }

    if (epochMinRtt < 1e8 && base > 0) {
        double q = std::max(0.0, epochMinRtt - base);
        double allowed = tolerance * base;
        uint32_t mss = state->snd_mss;

        if (!state->lossRecovery) {
            if (state->snd_cwnd < state->ssthresh) {
                // delay-based slow-start exit
                if (q > allowed) {
                    state->ssthresh = state->snd_cwnd;
                    conn->emit(ssthreshSignal, state->ssthresh);
                }
            }
            else {
                double cwnd = state->snd_cwnd;
                if (q <= allowed)
                    cwnd += incGain * mss;
                else {
                    double frac = decGain * (q - allowed) / std::max(epochMinRtt, 1e-6);
                    cwnd *= 1.0 - std::min(frac, 0.5);
                }
                cwnd = std::clamp(cwnd, 2.0 * mss, 16.0 * 1024 * 1024);
                state->snd_cwnd = (uint32_t)cwnd;
                state->ssthresh = std::min<uint32_t>(state->ssthresh, state->snd_cwnd);
                conn->emit(cwndSignal, state->snd_cwnd);
            }
        }

        // change detector on the epoch-level signals
        if (base > 0 && lastEpochRate > 0) {
            bool fq = phQueue.update(q / base, phDeltaQ, phHQ, phWarmup);
            bool fr = phRate.update(std::log(lastEpochRate), phDeltaR, phHR, phWarmup);
            if (fq || fr) {
                // start the next epoch before handing control to the supervisor
                epochEndSeq = state->snd_max;
                epochStart = simTime();
                epochMinRtt = 1e9;
                epochDelivered = 0;
                requestEarlyDecision(1);
                return;
            }
        }
    }

    epochEndSeq = state->snd_max;
    epochStart = simTime();
    epochMinRtt = 1e9;
    epochDelivered = 0;
}

// ---------------------------------------------------------------------------
// ACK processing (NewReno recovery, executor window rules)
// ---------------------------------------------------------------------------

void StTcp::receivedDataAck(uint32_t firstSeqAcked)
{
    // RTT sample from the newest fully acknowledged timed burst
    bool have = false;
    simtime_t sentAt;
    while (!sendTimes.empty() && !seqGreater(sendTimes.front().first, state->snd_una)) {
        sentAt = sendTimes.front().second;
        have = true;
        sendTimes.pop_front();
    }
    if (have) {
        double rtt = (simTime() - sentAt).dbl();
        if (rtt > 0)
            onRttSample(rtt);
    }

    uint32_t newly = state->snd_una - firstSeqAcked;
    intervalDelivered += newly;
    epochDelivered += newly;

    TcpBaseAlg::receivedDataAck(firstSeqAcked);

    if (state->lossRecovery) {
        if (seqGE(state->snd_una - 1, state->recover)) {
            uint32_t flight = state->snd_max - state->snd_una;
            state->snd_cwnd = std::min(state->ssthresh, flight + state->snd_mss);
            conn->emit(cwndSignal, state->snd_cwnd);
            state->lossRecovery = false;
            state->firstPartialACK = false;
        }
        else {
            conn->retransmitOneSegment(false);
            state->snd_cwnd -= state->snd_una - firstSeqAcked;
            if (state->snd_una - firstSeqAcked >= state->snd_mss)
                state->snd_cwnd += state->snd_mss;
            conn->emit(cwndSignal, state->snd_cwnd);
            sendData(false);
            if (state->lossRecovery && !state->firstPartialACK) {
                state->firstPartialACK = true;
                restartRexmitTimer();
            }
        }
    }
    else {
        if (state->snd_cwnd < state->ssthresh) {
            state->snd_cwnd += state->snd_mss;   // slow start
            conn->emit(cwndSignal, state->snd_cwnd);
        }
        // congestion avoidance is handled once per epoch in onEpochEnd()
        state->recover = state->snd_una - 2;
    }

    if (epochActive) {
        double cap = 2.0 * std::max(lastRtt, baseRtt());
        if (!seqLess(state->snd_una, epochEndSeq) || (simTime() - epochStart).dbl() > cap)
            onEpochEnd();
    }

    sendData(false);
}

void StTcp::receivedDuplicateAck()
{
    bool before = state->lossRecovery;
    TcpNewReno::receivedDuplicateAck();
    if (!before && state->lossRecovery)
        intervalLossEvents++;
}

void StTcp::recalculateSlowStartThreshold()
{
    uint32_t flight = std::min(state->snd_cwnd, state->snd_wnd);
    state->ssthresh = std::max((uint32_t)(lossBeta * flight), 2 * state->snd_mss);
    conn->emit(ssthreshSignal, state->ssthresh);
}

void StTcp::processRexmitTimer(TcpEventCode& event)
{
    TcpNewReno::processRexmitTimer(event);
    if (event == TCP_E_ABORT)
        return;
    intervalRtos++;
    sendTimes.clear();
    epochActive = false;
    requestEarlyDecision(2);
}

} // namespace tcp
} // namespace inet
