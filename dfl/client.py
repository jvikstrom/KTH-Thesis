import tensorflow as tf
from typing import List
import numpy as np
import random


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
        #self.model.fit(*self.train_data, batch_size=32, epochs=1, verbose=0)

        shuffled_indexes = self.indexes.copy()
        random.shuffle(shuffled_indexes)
        number_batches = len(self.indexes) // batch_size
        if number_batches == 0:
            print("There is a client with 0 batches!")
            return
        for i in range(batches):
            #batch = np.random.choice(self.indexes, batch_size)
            #print(f"batch: {batch}")
            batch = i % number_batches
            start = batch * batch_size
            end = (batch + 1) * batch_size
            indexes = shuffled_indexes[start:end]
            x,y = self.train_data
            x_batch = x[indexes]
            y_batch = y[indexes]
            self.model.train_on_batch(x_batch, y_batch)

        #for i in range(batches):
        #    batch = np.random.randint(0, len(self.train_data[0]), size=batch_size)
        #    x, y = self.train_data
        #    self.model.train_on_batch(x[batch], y[batch])

    def set_data(self, train, test):
        self.test_data = test
        self.train_data = train
        self.indexes = np.arange(0, len(self.train_data[0]))

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


class Guider:
    def __init__(self, clients: List[Client]):
        self.all_clients = clients

    def next(self, clients: List[Client], client_idx: int) -> int:
        self.clients = clients
        if len(self.clients) <= 1:
            return -1
        while True:
            nxt = np.random.randint(0, len(self.clients))
            if nxt != client_idx:
                break
        return nxt
