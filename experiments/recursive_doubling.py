from typing import List, Tuple
import numpy as np
from decentralized import Model, NextPeerStrategy, Worker, Runner

class MeanRDModel(Model):
    def __init__(self, w: np.array):
        self.w = w
    def recv_model(self, data: np.array, other):
        avg = (self.w + other.w) / 2
        return MeanRDModel(avg)
    def model(self, data) -> float:
        return self.w
    def train(self, data: np.array) -> Model:
        lr = 0.9
        delta = np.zeros_like(self.w)
        delta = self.w - data
        w = self.w - lr * delta
        return MeanRDModel(w)

class RDPeerStrategy(NextPeerStrategy):
    def __init__(self, peers: List[Worker] = []):
        self.init(peers)

    def init(self, peers: List[Worker]):
        self.max_round = 0
        self.round = 0
        self.targets = []
        if len(peers) == 0:
            return
        nrounds = np.log2(len(peers))
        # Peers must be a power of two.
        for i in range(len(peers)):
            itar = []
            for j in range((int(nrounds))):
                xor = 0
                xor |= (1 << j)
                xor != xor
                trgt = int(i ^ xor)
                itar.append(trgt)
            self.targets.append(itar)
            self.max_round = max(len(itar), self.max_round)

    def next_peer(self, worker: Worker) -> int:
        return self.targets[worker.id][self.round]
    def step(self):
        self.round += 1
        if self.round == self.max_round:
            # Start a new "training" round.
            self.round = 0
            return True
        return False

def run_rd(n:int, feature_size:int, n_rounds:int, datas, datas_mean, msg_drop) -> Tuple[List[float], List[float], List[float], List[float]]:
    peers = []
    for i in range(n):
        g_peers = list(range(0, i)) + list(range(i+1, n))
        peers.append(g_peers)
    workers = [Worker(i, x[0], x[1], MeanRDModel(np.zeros(feature_size))) for (i,x) in enumerate(zip(peers, datas))]
    strat = RDPeerStrategy(workers)
    strat.init(workers)
    rdfl = Runner(strat, msg_drop, workers)

    rdfl_errors = []
    rdfl_50_perc = []
    rdfl_90_perc = []
    rdfl_95_perc = []
    for pidx in range(len(rdfl.peers)):
        rdfl.peers[pidx].model = rdfl.peers[pidx].model.train(rdfl.peers[pidx].data)
    for i in range(n_rounds):
        if rdfl.run_step():
            # Check number of different models.
            for pidx in range(len(rdfl.peers)):
                rdfl.peers[pidx].model = rdfl.peers[pidx].model.train(rdfl.peers[pidx].data)
        fitt = rdfl.average_model()
        rdfl_50_perc.append(np.linalg.norm(rdfl.percentile_model(0.5)-datas_mean))
        rdfl_90_perc.append(np.linalg.norm(rdfl.percentile_model(0.9)-datas_mean))
        rdfl_95_perc.append(np.linalg.norm(rdfl.percentile_model(0.95)-datas_mean))
        rdfl_errors.append(np.linalg.norm(fitt - datas_mean))
    while not rdfl.run_step():
        pass
    return (rdfl_errors, rdfl_50_perc, rdfl_90_perc, rdfl_95_perc)
