import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_emnist, concat_data
from client import Client, Guider
from hypercube import Hypercube
from gossip_impl import Gossip, ExchangeGossip
import numpy as np
import pandas as pd
import storage

tf.compat.v1.enable_eager_execution()


def model_fn():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(36),
    ])
    model.compile(optimizer=tf.optimizers.Adam(0.001),
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    return model


def run_emnist(data_dir: str, name: str, N, strategy, batches=1, iterations=100):
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    clients = [Client(load_from_emnist(train, i), load_from_emnist(test, i), model_fn) for i in range(N)]

    hyper = strategy(clients, Guider)
    hyper.run(batches=batches, iterations=iterations)
    df = pd.DataFrame()
    for i in range(len(hyper.test_evals)):
        loss, accuracy = hyper.test_evals[i]
        df.append({
            'name': name,
            'N': N,
            'batches': batches,
            'iterations': iterations,
            'current_iteration': i,
            'loss': loss,
            'accuracy': accuracy,
        })

    storage.append(data_dir, name+".csv", df)


def run_emnist_fls(data_dir, name, N, batches=1, iterations=100):
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data(only_digits=False)

    clients = [Client(load_from_emnist(train, i), load_from_emnist(test, i), model_fn) for i in range(N)]
    test_concated = concat_data([client.get_test_data() for client in clients])
    df = pd.DataFrame()
    metrics = []
    for i in range(iterations):
        for client in clients:
            client.model.set_weights([weights for weights in clients[0].model.get_weights()])
        for client in clients:
            client.train(batches)
        all_weights = [client.model.get_weights() for client in clients]
        avg_weights = all_weights[0]
        for j in range(1, len(all_weights)):
            for k in range(len(avg_weights)):
                avg_weights[k] += all_weights[j][k]
        for k in range(len(avg_weights)):
            avg_weights[k] = avg_weights[k] / len(clients)

        for client in clients:
            client.model.set_weights(avg_weights)
        loss, accuracy = clients[0].model.evaluate(*test_concated, verbose=0, batch_size=32)
        df.append({
            'name': name,
            'N': N,
            'batches': batches,
            'iterations': iterations,
            'current_iteration': i,
            'loss': loss,
            'accuracy': accuracy,
        })
        print(f"Round {i} ::: loss: {loss}, accuracy: {accuracy}")
        metrics.append((loss, accuracy))

        storage.append(data_dir, name+".csv", df)


if __name__ == "__main__":
    N = 128
    iterations = 101
    batches = 10
    data_dir = "./data"
    run_emnist(data_dir, "exchange-gossip", N, ExchangeGossip, batches=batches, iterations=iterations)
    run_emnist(data_dir, "gossip", N, Gossip, batches=batches, iterations=iterations)
    run_emnist_fls(data_dir, "fls", N, batches=batches, iterations=iterations)
