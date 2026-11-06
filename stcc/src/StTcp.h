//
// StTcp: delay-tolerance executor with a self-triggered learned supervisor.
//
// The executor runs inside the INET TCP model on every ACK and once per
// round-trip epoch. Its behaviour is fixed by a four-parameter vector theta
// that is supplied by an external supervisor (Python/PPO) over a local TCP
// socket. The supervisor is only contacted at "decision points":
//   * when the hold time chosen at the previous decision expires, or
//   * when the in-executor change detector (Page-Hinkley on queueing ratio
//     and log delivery rate) or a retransmission timeout fires an early
//     trigger (if enabled).
//
// Loss recovery (fast retransmit, NewReno partial-ACK recovery, RTO) is
// inherited from INET's TcpNewReno; only the window growth/decrease rules
// and the loss back-off factor are replaced.
//
#ifndef STCC_STTCP_H_
#define STCC_STTCP_H_

#include <deque>
#include <utility>
#include <string>

#include "inet/transportlayer/tcp/flavours/TcpNewReno.h"

namespace inet {
namespace tcp {

class INET_API StTcp : public TcpNewReno
{
  public:
    StTcp();
    virtual ~StTcp();

    virtual void initialize() override;
    virtual void established(bool active) override;
    virtual void connectionClosed() override;
    virtual void processTimer(cMessage *timer, TcpEventCode& event) override;
    virtual void receivedDataAck(uint32_t firstSeqAcked) override;
    virtual void receivedDuplicateAck() override;
    virtual void dataSent(uint32_t fromseq) override;
    virtual void segmentRetransmitted(uint32_t fromseq, uint32_t toseq) override;

  protected:
    virtual void recalculateSlowStartThreshold() override;
    virtual void processRexmitTimer(TcpEventCode& event) override;

    // ---- executor -------------------------------------------------------
    void onRttSample(double rtt);
    void onEpochEnd();
    void applyTheta(const double *a);
    double baseRtt() const;

    // theta in physical units (mapped from [-1,1] actions)
    double tolerance = 0.25;   // allowed queueing delay as a fraction of base RTT
    double incGain = 1.0;      // additive increase, MSS per RTT
    double decGain = 0.25;     // proportional decrease gain per RTT
    double lossBeta = 0.7;     // multiplicative back-off on loss

    // ---- RTT sampling (Karn-safe, per burst) -----------------------------
    std::deque<std::pair<uint32_t, simtime_t>> sendTimes; // (end seq, time)
    uint32_t highestSentEnd = 0;
    bool haveHighestSent = false;
    double lastRtt = 0.0;

    // windowed minimum (10 buckets of 1 s)
    static constexpr int MINRTT_BUCKETS = 10;
    double bucketMin[MINRTT_BUCKETS];
    long bucketId[MINRTT_BUCKETS];

    // ---- epoch bookkeeping ------------------------------------------------
    uint32_t epochEndSeq = 0;
    bool epochActive = false;
    double epochMinRtt = 1e9;
    simtime_t epochStart;
    uint64_t epochDelivered = 0;

    // ---- decision-interval statistics ----------------------------------
    simtime_t intervalStart;
    uint64_t intervalDelivered = 0;
    double intervalRttSum = 0.0;
    long intervalRttCount = 0;
    double intervalRttMin = 1e9;
    double intervalRttMax = 0.0;
    int intervalLossEvents = 0;
    int intervalRtos = 0;
    double maxDeliveryRate = 0.0;   // bytes/s, decays slowly
    double lastEpochRate = 0.0;

    // ---- change detector (Page-Hinkley, two-sided, two signals) ---------
    struct PageHinkley {
        double mean = 0.0, n = 0.0, up = 0.0, down = 0.0;
        void reset() { mean = n = up = down = 0.0; }
        // returns true if a change is declared
        bool update(double x, double delta, double h, int warmup) {
            n += 1.0;
            mean += (x - mean) / n;
            if (n <= warmup) return false;
            up = std::max(0.0, up + x - mean - delta);
            down = std::max(0.0, down + mean - x - delta);
            return up > h || down > h;
        }
    };
    PageHinkley phQueue, phRate;
    bool tripwireEnabled = true;
    double phDeltaQ = 0.10, phHQ = 1.5;
    double phDeltaR = 0.05, phHR = 0.8;
    int phWarmup = 3;
    double minGap = 0.05;           // s, minimum spacing between decisions
    int triggerReason = 3;          // 0 hold expired, 1 change, 2 rto, 3 start

    // ---- supervisor link ------------------------------------------------
    cMessage *decisionTimer = nullptr;
    simtime_t lastDecision;
    int sock = -1;
    bool linkUp = false;
    bool staticMode = false;        // fixed theta, never contacts supervisor
    double staticTheta[4] = {0, 0, 0, 0};
    double staticHold = 1.0;
    int flowId = 0;

    void connectSupervisor();
    void decide();
    bool sendAll(const void *buf, size_t len);
    bool recvAll(void *buf, size_t len);
    void requestEarlyDecision(int reason);
};

} // namespace tcp
} // namespace inet

#endif
