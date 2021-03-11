import numpy as np
from typing import List

class MessageDropStrategy:
    def __init__(self, drop_percentage: int = 0.0):
        self.drop_percentage = drop_percentage
    def should_drop(self):
        return self.drop_percentage > np.random.uniform()

class Model:
    def recv_model(self, data: np.array, other):
        # Do any training/merging here,.
        pass
    def model(self, data) -> float:
        # Return the fit to the data.
        pass

class Worker:
    def __init__(self, id: int, peers: List[int], data: np.array, model: Model):
        self.id = id
        self.peers = peers
        self.data = data
        self.model = model
    def recv_model(self, other: Model):
        new_model = self.model.recv_model(self.data, other)
        return Worker(self.id, self.peers, self.data, new_model)

class NextPeerStrategy:
    def init(self, workers: List[Worker]):
        pass
    def next_peer(self, worker: Worker) -> int:
        pass
    def step(self) -> bool:
        return False

class Runner:
    def __init__(self, next_peer: NextPeerStrategy, msg_drop: MessageDropStrategy, peers: List[Worker]):
        self.next_peer = next_peer
        self.msg_drop = msg_drop
        self.peers = peers
    def run_step(self) -> bool:
        old = self.peers.copy()
        for peer in old:
            nxt_idx = self.next_peer.next_peer(peer)
            if not self.msg_drop.should_drop():
                self.peers[nxt_idx] = self.peers[nxt_idx].recv_model(peer.model)
        return self.next_peer.step()
    def average_model(self) -> np.array:
        fit = 0.0
        for peer in self.peers:
            fit += peer.model.model(peer.data)
        return fit / len(self.peers)
    def percentile_model(self, percentile: float) -> np.array:
        assert percentile >= 0 and percentile <= 1.0
        avg = self.average_model()
        model_diffs = []
        for peer in self.peers:
            model = peer.model.model(peer.data)
            diff = np.linalg.norm(model - avg)
            model_diffs.append((diff, model))
        model_diffs.sort(key=lambda x:x[0])
        return model_diffs[int(len(self.peers) * percentile)][1]
    def model_variance(self) -> np.array:
        variance = 0.0
        mean = self.average_model()
        for peer in self.peers:
            variance += np.square(peer.model.model(peer.data) - mean)
        return variance / len(self.peers)
