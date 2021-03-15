from typing import List, Tuple, Callable
from tqdm import tqdm
import numpy as np
from client import Client, combine
from train import Trainer


class Guider:
    def __init__(self, clients: List[Client]):
        self.clients = clients

    def next(self, client_idx: int) -> int:
        while True:
            nxt = np.random.randint(0, len(self.clients))
            if nxt != client_idx:
                break
        return nxt


def recv_model(sender: Tuple[int, Client], receiver: Tuple[int, Client]) -> int:
    # Implements the MergeAverage.
    # Merge the model to receiver.
#    a = receiver[0] / (sender[0] + receiver[0])
    t = max(sender[0], receiver[0])
    weights = [model.get_weights() for model in [sender[1].model, receiver[1].model]]
    new_weights = list()
    for send_weight, recv_weight in zip(*weights):
        new_weights.append(send_weight)
#        new_weights.append((1 - a) * send_weight + a * recv_weight)

    # Do the update step.
    receiver[1].model.set_weights(new_weights)
    receiver[1].train()
    # Return the new version of receiver.
    return t + 1


class Gossip(Trainer):
    def __init__(self, clients: List[Client], guider: Callable[[List[Client]], Guider]):
        Trainer.__init__(self, clients)
#        for client in self.clients:
#            client.model.set_weights(self.clients[0].model.get_weights())
#            client.model.set_weights([np.zeros_like(w) for w in client.model.get_weights()])
        self.versions = [1 for _ in range(len(clients))]
        self.guider = guider(clients)

    def run(self, epochs: int = 1, iterations: int = 100):
        for i in range(iterations):
            for client_idx in tqdm(range(len(self.clients))):
                nxt = self.guider.next(client_idx)
                self.versions[nxt] = recv_model((self.versions[client_idx], self.clients[client_idx]),
                           (self.versions[nxt], self.clients[nxt]))
            if i % 50 == 0 and i != 0:
                self.eval_train(i)
            self.eval_test(i)
            a_weights = self.clients[0].model.get_weights()
            for a_weight in a_weights:
                print(f"mean:{a_weight.mean()}, max:{a_weight.max()}, min:{a_weight.min()}, std:{a_weight.std()}")
