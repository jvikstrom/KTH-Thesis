from typing import List, Tuple, Callable
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from client import Client, Guider
from train import Trainer


def recv_model(sender: Tuple[int, Client], receiver: Tuple[int, Client], batches=1) -> int:
    # Implements the MergeAverage.
    # Merge the model to receiver.
    #    a = receiver[0] / (sender[0] + receiver[0])
    t = max(sender[0], receiver[0])
    # weights = [model.get_weights() for model in [sender[1].model, receiver[1].model]]
    # new_weights = list()
    # for send_weight, recv_weight in zip(*weights):
    #    new_weights.append(send_weight.copy())
    #        new_weights.append((1 - a) * send_weight + a * recv_weight)

    # Do the update step.
    receiver[1].model.set_weights([weight.copy() for weight in sender[1].model.get_weights()])
    receiver[1].train(batches)
    # Return the new version of receiver.
    return t + 1


def exchange_recv_model(sender: Client, receiver: Client, failure: Tuple[bool, bool] = (False, False), batches=1) -> \
        Tuple[tf.keras.Model, tf.keras.Model]:
    # TODO: This should really happen in "parallel"
    # TODO: Figure out how we should deal with failures with the recursive doubling.
    # Really want to do a hamiltonian cycle through the hypercube.
    old = sender.model
    if not failure[0]:
        sender.model = receiver.model
        sender.train(batches=batches)
    if not failure[1]:
        receiver.model = old
        receiver.train(batches=batches)
    return sender.model, receiver.model


class Gossip(Trainer):
    def __init__(self, clients: List[Client], guider: Callable[[List[Client]], Guider]):
        Trainer.__init__(self, clients)
        self.versions = [1 for _ in range(len(clients))]
        self.guider = guider(clients)
        self.recv_model = recv_model

    def run(self, batches: int = 1, iterations: int = 100):
        for i in range(iterations):
            old_weights = [client.model.get_weights() for client in self.clients]
            for client_idx in tqdm(range(len(self.clients))):
                nxt = self.guider.next(client_idx)
                send_weights = old_weights[client_idx]
                self.clients[nxt].model.set_weights([weight.copy() for weight in send_weights])
                self.clients[nxt].train(batches)
#                self.versions[nxt] = self.recv_model((self.versions[client_idx], self.clients[client_idx]),
#                                                     (self.versions[nxt], self.clients[nxt]), batches=batches)
            # if i % 50 == 0 and i != 0:
            #    self.eval_train(i)
            if i % 1 == 0 and i != 0:
                self.eval_test(i)
            # a_weights = self.clients[0].model.get_weights()
            # for a_weight in a_weights:
            #    print(f"mean:{a_weight.mean()}, max:{a_weight.max()}, min:{a_weight.min()}, std:{a_weight.std()}")
            Trainer.step(self)


class ExchangeGossip(Trainer):
    def __init__(self, clients: List[Client], guider: Callable[[List[Client]], Guider]):
        Trainer.__init__(self, clients)
        self.guider = guider(clients)
        self.recv_model = exchange_recv_model

    def run(self, batches: int = 1, iterations: int = 100):
        for i in range(iterations):
            exchanged = []
            for client_idx in range(len(self.clients)):
                nxt = self.guider.next(client_idx)
                exchange_pair = (client_idx, nxt)
                if client_idx > nxt:
                    exchange_pair = (nxt, client_idx)
                if exchange_pair not in exchanged:
                    #    self.recv_model(self.clients[client_idx], self.clients[nxt], batches=batches)
                    exchanged.append(exchange_pair)
            # Do the actual exchanges.
            model_weights = [client.model.get_weights() for client in self.clients]
            optimizer_configs = [client.model.optimizer.get_config() for client in self.clients]
            for p1, p2 in exchanged:
                # print(f"{p1} -> {p2}")
                # Need to clone the model as otherwise the exchanges might cause multiple clients to have the same model
                # pointer.
                self.clients[p1].model.set_weights([weight.copy() for weight in model_weights[p2]])
                # self.clients[p1].model.optimizer = tf.keras.optimizers.Adam.from_config(optimizer_configs[p2])
                self.clients[p2].model.set_weights([weight.copy() for weight in model_weights[p1]])
                # self.clients[p2].model.optimizer = tf.keras.optimizers.Adam.from_config(optimizer_configs[p1])

            #                self.clients[p2].model = tf.keras.models.clone_model(models[p1])
            for client in tqdm(self.clients):
                client.train(batches)

            if i % 20 == 0 and i != 0:
                self.eval_train(i)
            if i % 1 == 0 and i != 0:
                self.eval_test(i)
            Trainer.step(self)
# 4323
