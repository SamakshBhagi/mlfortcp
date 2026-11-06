#!/bin/bash
# usage: run_train.sh <name> <port> [train.py args...]
name=$1; port=$2; shift 2
export PATH=$HOME/sim/omnetpp-6.0.3/bin:$PATH
export LD_LIBRARY_PATH=$HOME/sim/omnetpp-6.0.3/lib:$HOME/sim/inet4.5/src:$HOME/sim/stcc/src
cd $HOME/sim/stcc/py
mkdir -p $HOME/sim/runs
exec python3 train.py --name "$name" --port "$port" "$@" > $HOME/sim/runs/$name.out 2>&1
