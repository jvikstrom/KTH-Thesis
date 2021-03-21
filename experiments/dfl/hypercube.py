import numpy as np
from typing import List
from tqdm import tqdm
from client import Client, Guider
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


def combine(a: Client, b: Client):
    weights = [model.get_weights() for model in [a.model, b.model]]
    new_weights = list()
    for a_weight, b_weight in zip(*weights):
        new_weights.append((a_weight + b_weight) / 2.0)
    a.model.set_weights([weight.copy() for weight in new_weights])
    b.model.set_weights([weight.copy() for weight in new_weights])


class Hypercube(Trainer):
    def __init__(self, clients: List[Client], unused):
        Trainer.__init__(self, clients)
        for client in self.clients:
            client.model.set_weights([weight.copy() for weight in clients[0].model.get_weights()])
        self.rounds = preprocess_client_targets(clients)

    def run(self, batches: int = 1, iterations: int = 100):
#        vals = [[i] for i in range(len(self.clients))]
#        for round in self.rounds:
#            for a, b in round:
#                print(f"{a}<-{b}, {vals[a]}, {vals[b]}")
#                vals[a] += vals[b].copy()
#                vals[b] = vals[a].copy()
#        print(vals[0], sum([i for i in range(len(self.clients))]))

        for i in range(iterations):
            for client in tqdm(self.clients, desc="train"):
                client.train(batches=batches)

            aggregated_weights = [np.zeros_like(weight) for weight in self.clients[0].model.get_weights()]
            client_weights = [client.model.get_weights() for client in self.clients]
            for round in self.rounds:
                for a, b in round:
#                    combine(self.clients[a], self.clients[b])
                    new_weights = [(a+b).copy() for a,b in zip(client_weights[a], client_weights[b])]
                    client_weights[a] = [weight.copy() / 2.0 for weight in new_weights]
                    client_weights[b] = [weight.copy() / 2.0 for weight in new_weights]
            correct_weights = [weight.copy() for weight in client_weights[0]]
            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in correct_weights])
            if i % 100 == 0 and i != 0:
                self.eval_train(i)
            if i % 1 == 0:
                self.eval_test(i)
            #a_weights = self.clients[0].model.get_weights()
            #for a_weight in a_weights:
            #    print(f"mean:{a_weight.mean()}, max:{a_weight.max()}, min:{a_weight.min()}, std:{a_weight.std()}")
        Trainer.step(self)


class HamiltonCycleGuider(Guider):
    def __init__(self, clients: List[Client]):
        Guider.__init__(self, clients)
        self.round = [0 for _ in self.clients]

    def next(self, client_idx: int) -> int:
        if self.round[client_idx] % 2 == client_idx % 2:
            add = 1
        else:
            add = -1
        self.round[client_idx] += 1
        return (client_idx + add) % len(self.clients)
