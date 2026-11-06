package stcc.simulations;

import inet.common.scenario.ScenarioManager;
import inet.networklayer.configurator.ipv4.Ipv4NetworkConfigurator;
import inet.node.inet.Router;
import inet.node.inet.StandardHost;
import ned.DatarateChannel;

//
// Dumbbell: numSenders TCP senders and one UDP cross-traffic source share a
// single bottleneck r1 -> r2. The bottleneck capacity is changed at run
// time by the ScenarioManager script generated for each episode.
//
network Dumbbell
{
    parameters:
        int numSenders = default(2);
        double bottleneckDatarate @unit(bps) = default(20Mbps);
        double bottleneckDelay @unit(s) = default(20ms);
        double accessDelay @unit(s) = default(2ms);

    types:
        channel Access extends DatarateChannel
        {
            datarate = 1Gbps;
            delay = parent.accessDelay;
        }
        channel Bottleneck extends DatarateChannel
        {
            datarate = parent.bottleneckDatarate;
            delay = parent.bottleneckDelay;
        }

    submodules:
        configurator: Ipv4NetworkConfigurator;
        scenarioManager: ScenarioManager;
        r1: Router;
        r2: Router;
        sender[numSenders]: StandardHost;
        receiver[numSenders]: StandardHost;
        udpSrc: StandardHost;
        udpSink: StandardHost;

    connections:
        // keep this first: it becomes pppg[0] on both routers
        r1.pppg++ <--> Bottleneck <--> r2.pppg++;
        for i=0..numSenders-1 {
            sender[i].pppg++ <--> Access <--> r1.pppg++;
            r2.pppg++ <--> Access <--> receiver[i].pppg++;
        }
        udpSrc.pppg++ <--> Access <--> r1.pppg++;
        r2.pppg++ <--> Access <--> udpSink.pppg++;
}
