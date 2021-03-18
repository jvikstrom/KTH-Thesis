import tensorflow as tf
from typing import List
import numpy as np


class Client:
    def __init__(self, train_data, test_data, model_fn):
        """
        :param id: Numerical id
        :param data_source: Tf data
        :param model_fn: Returns a keras model.
        """
        self.model: tf.keras.models.Model = model_fn()
        self.train_data = train_data
        self.test_data = test_data
        self.indexes = np.arange(0, len(train_data[0]))

    def train(self, batches: int = 1, batch_size: int = 32):
#        for i in range(batches):
#            batch = np.random.choice(self.indexes, batch_size)
            #print(f"batch: {batch}")
#            x,y = self.train_data
#            x_batch = x[batch]
#            y_batch = y[batch]
#            self.model.train_on_batch(x_batch, y_batch)

        self.model.fit(*self.train_data, batch_size=32, epochs=1, verbose=0)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


def combine(a: Client, b: Client):
    weights = [model.get_weights() for model in [a.model, b.model]]
    new_weights = list()
    for a_weight, b_weight in zip(*weights):
        new_weights.append((a_weight + b_weight) / 2.0)
    a.model.set_weights([weight.copy() for weight in new_weights])
    b.model.set_weights([weight.copy() for weight in new_weights])


class Guider:
    def __init__(self, clients: List[Client]):
        self.clients = clients

    def next(self, client_idx: int) -> int:
        while True:
            nxt = np.random.randint(0, len(self.clients))
            if nxt != client_idx:
                break
        return nxt
