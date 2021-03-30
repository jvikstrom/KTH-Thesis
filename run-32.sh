#!/bin/bash

# Script for running the ML on the GPU server.

export LC_ALL=C.UTF-8
export LANG=C.UTF-8


N=32
Runs=5
Batches=1
Iters=5000
export DATA_DIR="data-32-1"
cd dfl

mkdir data
CUDA_VISIBLE_DEVICES="0" python3 main.py emnist centralized $N $Runs $Batches $Iters 0.01 > centralized.stdout 2>&1 &
centralized_PID=$!
CUDA_VISIBLE_DEVICES="0" python3 main.py emnist fls $N $Runs $Batches $Iters 0.06 > fls.stdout 2>&1 &
fls_PID=$!
CUDA_VISIBLE_DEVICES="1" python3 main.py emnist exchange-cycle-adam $N $Runs $Batches $Iters 0.001 > exchange-cycle-adam.stdout 2>&1 &
exchange_cycle_adam_PID=$!
CUDA_VISIBLE_DEVICES="1" python3 main.py emnist exchange-cycle $N $Runs $Batches $Iters 0.01 > exchange-cycle.stdout 2>&1 &
exchange_cycle_PID=$!
CUDA_VISIBLE_DEVICES="2" python3 main.py emnist none-gossip $N $Runs $Batches $Iters 0.01 > none-gossip.stdout 2>&1 &
none_gossip_PID=$!
CUDA_VISIBLE_DEVICES="2" python3 main.py emnist exchange $N $Runs $Batches $Iters 0.01 > exchange.stdout 2>&1 &
exchange_PID=$!
CUDA_VISIBLE_DEVICES="3" python3 main.py emnist agg-hypercube $N $Runs $Batches $Iters 0.01 > agg-hypercube.stdout 2>&1 &
agg_hypercube_PID=$!
cd ..

echo "pids: " $centralized_PID $exchange_cycle_adam_PID $exchange_cycle_PID $none_gossip_PID $exchange_PID $agg_hypercube_PID $fls_PID

wait $centralized_PID
wait $exchange_cycle_adam_PID
wait $exchange_cycle_PID
wait $none_gossip_PID
wait $exchange_PID
wait $agg_hypercube_PID
wait $fls_PID

