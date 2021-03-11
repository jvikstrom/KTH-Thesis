import numpy as np
from decentralized import Model, NextPeerStrategy, Worker, Runner
from typing import List, Tuple

class GossipModel(Model):
    def __init__(self, t, w):
        self.t = t
        self.w = w
    # By default this simply tries to calculate the mean.
    def recv_model(self, data: np.array, other):
        model = self._merge(other)
        model = self._update(model, other, data)
        return model
    def _merge(self, other):
        a = other.t / (self.t + other.t)
        t = max(self.t, other.t)
        w = (1 - a) * self.w + a * other.w
        return GossipModel(t, w)
    def _update(self, current, other, data: np.array):
        t = self.t + len(data)
        nk = len(data) / t
        delta = np.zeros_like(current.w)
        delta = current.w - data
        w = current.w - nk * delta
        return GossipModel(t, w)
    def model(self, data) -> float:
        return self.w

class RandomPeerStrategy(NextPeerStrategy):
    def next_peer(self, worker: Worker) -> int:
        return np.random.choice(worker.peers)

def run_gossip(n:int, feature_size:int, n_rounds:int, datas, datas_mean, msg_drop, peer_strategy = RandomPeerStrategy()) -> Tuple[List[float], List[float], List[float], List[float]]:
    peers = []
    for i in range(n):
        g_peers = list(range(0, i)) + list(range(i+1, n))
        peers.append(g_peers)
    workers = [Worker(i, x[0], x[1], GossipModel(1, np.zeros(feature_size))) for (i,x) in enumerate(zip(peers, datas))]
    peer_strategy.init(workers)

    gossip = Runner(peer_strategy, msg_drop, workers)
    gossip_errors = []
    gossip_50_perc = []
    gossip_90_perc = []
    gossip_95_perc = []
    for i in range(n_rounds):
        gossip.run_step()
        fitt = np.linalg.norm(gossip.average_model() - datas_mean)
        gossip_50_perc.append(np.linalg.norm(gossip.percentile_model(0.5)-datas_mean))
        gossip_90_perc.append(np.linalg.norm(gossip.percentile_model(0.9)-datas_mean))
        gossip_95_perc.append(np.linalg.norm(gossip.percentile_model(0.95)-datas_mean))
        gossip_errors.append(fitt)
    return (gossip_errors, gossip_50_perc, gossip_90_perc, gossip_95_perc)
