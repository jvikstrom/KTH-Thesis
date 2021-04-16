from typing import List, Tuple, Callable
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from client import Client, Guider
from train import Trainer, TrainerConfig
from pydantic import BaseModel


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


class BaseGossipConfig(BaseModel):
    trainer_config: TrainerConfig
    guider: Callable[[List[Client]], Guider]


class GossipConfig(BaseModel):
    base_config: BaseGossipConfig
    average: bool


class Gossip(Trainer):
    def __init__(self, trainer_input, clients: List[Client], cfg: GossipConfig, all_train, all_test):
        Trainer.__init__(self, trainer_input, clients, cfg.base_config.trainer_config, all_train, all_test)
        self.versions = [1 for _ in range(len(clients))]
        self.guider = cfg.base_config.guider(clients)
        self.recv_model = recv_model
        self.config = cfg

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i)

            old_weights = [client.model.get_weights() for client in self.clients]
            for client_idx in tqdm(range(len(self.clients)), disable=self.disable_tqdm):
                nxt = self.guider.next(self.clients, client_idx)
                if nxt == -1:
                    continue
                send_weights = old_weights[client_idx]
                # Does the preprocessing for averaging.
                both_weights = zip(send_weights, old_weights[nxt])
                a = self.versions[nxt] / (self.versions[nxt] + self.versions[client_idx])
                if self.config.average:
                    # Only do the averaging if the average flag is set.
                    send_weights = []
                    for a_w, r_w in both_weights:
                        # TODO: Try shifting a_w and r_w
                        send_weights.append((1-a)*a_w.copy() + a * r_w.copy())

                t = max(self.versions[nxt], self.versions[client_idx])
                self.versions[nxt] = t+1
                self.clients[nxt].model.set_weights([weight.copy() for weight in send_weights])
                self.clients[nxt].train(self.trainer_config.batches)
        Trainer.step_and_eval(self, self.trainer_config.iterations)


class ExchangeConfig(BaseModel):
    base_config: BaseGossipConfig
    # TODO: Use this for setting optimizers as well.
    swap_optimizer: bool


class ExchangeGossip(Trainer):
    def __init__(self, trainer_input, clients: List[Client], cfg: ExchangeConfig, all_train, all_test):
        Trainer.__init__(self, trainer_input, clients, cfg.base_config.trainer_config, all_train, all_test)
        self.guider = cfg.base_config.guider(clients)
        self.recv_model = exchange_recv_model
        self.config = cfg

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i)
            exchanged = []
            for client_idx in range(len(self.clients)):
                nxt = self.guider.next(self.clients, client_idx)
                if nxt == -1:
                    continue
                exchange_pair = (client_idx, nxt)
                if client_idx > nxt:
                    exchange_pair = (nxt, client_idx)
                if exchange_pair not in exchanged:
                    exchanged.append(exchange_pair)
            print(f"Exchanged: {exchanged}")
            # Do the actual exchanges.
            model_weights = [client.model.get_weights() for client in self.clients]
            optimizer_weights = [client.model.optimizer.get_weights() for client in self.clients]
            old_models = [client.model for client in self.clients]
            optimizer_configs = [client.model.optimizer.get_config() for client in self.clients]
            for p1, p2 in exchanged:
                # TODO: This will fuck up and have multiple references in case switch with same (can happen with exchange-gossip).
                if self.config.swap_optimizer:
                    self.clients[p1].model = old_models[p2]
                    self.clients[p2].model = old_models[p1]
                    # TODO: For this to work we probably need to send the slots as well..
#                    self.clients[p1].model.set_weights([weight.copy() for weight in model_weights[p2]])
#                    self.clients[p2].model.set_weights([weight.copy() for weight in model_weights[p1]])
#                    self.clients[p1].model.optimizer.set_weights([weight.copy() for weight in optimizer_weights[p2]])
#                    self.clients[p2].model.optimizer.set_weights([weight.copy() for weight in optimizer_weights[p1]])
                else:
                    self.clients[p1].model.set_weights([weight.copy() for weight in model_weights[p2]])
                    self.clients[p2].model.set_weights([weight.copy() for weight in model_weights[p1]])

#                self.clients[p1].model.optimizer.set_weights([weight.copy() for weight in old_models[p2].optimizer.get_weights()])
#                self.clients[p2].model.optimizer.set_weights([weight.copy() for weight in old_models[p1].optimizer.get_weights()])

            for client in tqdm(self.clients, disable=self.disable_tqdm):
                client.train(self.trainer_config.batches)

        Trainer.step_and_eval(self, self.trainer_config.iterations)

# 4323
