import numpy as np
from typing import List
from tqdm import tqdm
from client import Client, Guider
from train import Trainer, TrainerConfig
from pydantic import BaseModel


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


class HypercubeConfig(BaseModel):
    trainer_config: TrainerConfig


class Hypercube(Trainer):
    def __init__(self, trainer_input, clients: List[Client], cfg: HypercubeConfig, all_train, all_test):
        Trainer.__init__(self, trainer_input, clients, cfg.trainer_config, all_train, all_test)
        for client in self.clients:
            client.model.set_weights([weight.copy() for weight in clients[0].model.get_weights()])
        self.rounds = preprocess_client_targets(clients)

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i)
            for client in tqdm(self.clients, desc="train", disable=self.disable_tqdm):
                client.train(batches=self.trainer_config.batches)

            aggregated_weights = [np.zeros_like(weight) for weight in self.clients[0].model.get_weights()]
            client_weights = [client.model.get_weights() for client in self.clients]
            for round in self.rounds:
                for a, b in round:
                    new_weights = [(a+b).copy() for a,b in zip(client_weights[a], client_weights[b])]
                    client_weights[a] = [weight.copy() / 2.0 for weight in new_weights]
                    client_weights[b] = [weight.copy() / 2.0 for weight in new_weights]
            correct_weights = [weight.copy() for weight in client_weights[0]]
            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in correct_weights])
        Trainer.step_and_eval(self, self.trainer_config.iterations)


class HamiltonCycleGuider(Guider):
    def __init__(self, clients: List[Client]):
        Guider.__init__(self, clients)
        self.round = [0 for _ in self.all_clients]

    def next(self, clients, index_idx: int) -> int:
        if len(clients) == 1:
            return -1
        client_idx = clients[index_idx]
        if self.round[client_idx] % 2 == client_idx % 2:
            add = 1
        else:
            add = -1
        nxt_idx = (index_idx + add) % len(clients)
        nxt = clients[nxt_idx]
        self.round[client_idx] += 1
        self.round[nxt] += 1
        #print(f"Client {client_idx} to {nxt}: {self.round[client_idx]}, {add} avail: {clients}")
        return nxt_idx
