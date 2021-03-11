import sys
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from typing import List
from tqdm import tqdm


class Client:
    def __init__(self, id, data_source, model_fn):
        """
        :param id: Numerical id
        :param data_source: Tf data
        :param model_fn: Returns a keras model.
        """
        self.model: tf.keras.models.Model = model_fn()
        self.id = id
        self.data = data_source.create_tf_dataset_for_client(data_source.client_ids[id]).map(
            lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
        ).repeat(10).batch(20)

    def train(self, epochs: int = 1):
        self.model.fit(self.data, epochs=epochs, verbose=0)

    def get_data(self):
        return self.data


def combine(a: Client, b: Client):
    weights = [model.get_weights() for model in [a.model, b.model]]
    new_weights = list()
    for a_weight, b_weight in zip(*weights):
        new_weights.append((a_weight + b_weight) / 2.0)
    a.model.set_weights(new_weights)
    b.model.set_weights(new_weights)


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


class Hypercube:
    def __init__(self, clients: List[Client]):
        self.clients = clients
        self.rounds = preprocess_client_targets(clients)

    def eval(self, epoch):
        data = [client.get_data() for client in self.clients]

        losses, accuracies = [], []
        for client in self.clients:
            loss_sum, accuracy_sum = 0, 0
            for d in data:
                loss, accuracy = client.model.evaluate(d, verbose=0)
                loss_sum += loss
                accuracy_sum += accuracy
            losses.append(loss_sum / len(self.clients))
            accuracies.append(accuracy_sum / len(self.clients))
        print(f"{epoch} ::: loss: {np.mean(losses)}   ----   accuracy: {np.mean(accuracies)}")

    def run(self, epochs: int = 1, iterations: int = 100):
        for i in range(iterations):
            for client in tqdm(self.clients):
                client.train(epochs=epochs)

            for round in self.rounds:
                for a, b in round:
                    combine(self.clients[a], self.clients[b])
            self.eval(i)


if __name__ == "__main__":
    # Load simulation data.
    source, _ = tff.simulation.datasets.emnist.load_data()


    def model_fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(10, tf.nn.softmax, input_shape=(784,),
                                  kernel_initializer='zeros')
        ])
        model.compile(optimizer='adam',
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model


    N = 4
    clients = [Client(i, source, model_fn) for i in range(N)]

    hyper = Hypercube(clients)
    hyper.run(epochs=5, iterations=100)
