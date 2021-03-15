import numpy as np
from typing import List
from tqdm import tqdm
from client import Client, combine
from dataset import concat_data


class Trainer:
    def __init__(self, clients: List[Client]):
        self.clients = clients
        self.test_concated = concat_data([client.get_test_data() for client in self.clients])
        self.train_concated = concat_data([client.get_train_data() for client in self.clients])

    def __eval_data(self, data_set, epoch, data):
        losses, accuracies = [], []
        for client in tqdm(self.clients, desc="eval"):
            loss, accuracy = client.model.evaluate(*data, verbose=0, batch_size=32)
            losses.append(loss)
            accuracies.append(accuracy)
        print(f"{data_set} {epoch} ::: loss: {np.mean(losses)}   ----   accuracy: {np.mean(accuracies)}")

    def eval_test(self, epoch):
        return self.__eval_data("TEST", epoch, self.test_concated)

    def eval_train(self, epoch):
        return self.__eval_data("TRAIN", epoch, self.train_concated)

    def run(self, epochs: int = 1, iterations: int = 100):
        pass
