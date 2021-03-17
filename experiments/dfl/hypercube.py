import numpy as np
from typing import List
from tqdm import tqdm
from client import Client, combine
from train import Trainer


def preprocess_client_targets(clients: List[Client]):
    n_rounds = np.log2(len(clients))
    rounds = []
    for i in range(int(n_rounds)):
        targets = []
        for j in range(len(clients)):
            xor = 0
            xor |= (1 << i)
            target = int(j ^ xor)
            if target > j:
                targets.append((j, target))
        rounds.append(targets)
    return rounds


class Hypercube(Trainer):
    def __init__(self, clients: List[Client], unused):
        Trainer.__init__(self, clients)
        for client in self.clients:
            client.model.set_weights(clients[0].model.get_weights())
        self.rounds = preprocess_client_targets(clients)

    def run(self, batches: int = 1, iterations: int = 100):
        for i in range(iterations):
            for client in tqdm(self.clients, desc="train"):
                client.train(batches=batches)

            for round in self.rounds:
                for a, b in round:
                    combine(self.clients[a], self.clients[b])
            if i % 100 == 0 and i != 0:
                self.eval_train(i)
            self.eval_test(i)
            #a_weights = self.clients[0].model.get_weights()
            #for a_weight in a_weights:
            #    print(f"mean:{a_weight.mean()}, max:{a_weight.max()}, min:{a_weight.min()}, std:{a_weight.std()}")
