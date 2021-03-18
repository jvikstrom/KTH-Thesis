import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_emnist, concat_data
from client import Client, Guider
from hypercube import Hypercube, HamiltonCycleGuider
from gossip_impl import Gossip, ExchangeGossip
from fls import FLS
import numpy as np
import pandas as pd
import storage
import gc


def model_fn_factory(learning_rate):
    def fn():
        """            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(62),
"""
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=(1,1), input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=(1,1), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2,2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(62, activation='softmax'),
        ])

        model.compile(optimizer=tf.optimizers.SGD(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model
    return fn


def run_emnist(data_dir: str, name: str, N, strategy, learning_rate, guider=Guider, batches=1, iterations=100):
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    clients = [Client(load_from_emnist(train, i), load_from_emnist(test, i), model_fn_factory(learning_rate)) for i in range(N)]

    hyper = strategy(clients, guider)
    hyper.run(batches=batches, iterations=iterations)
    df = pd.DataFrame()
    for i in range(len(hyper.test_evals)):
        loss, accuracy = hyper.test_evals[i]
        df = df.append({
            'name': name,
            'N': N,
            'batches': batches,
            'iterations': iterations,
            'current_iteration': i,
            'loss': loss,
            'accuracy': accuracy,
        }, ignore_index=True)

    print(f"Writing: {len(df)} records to {name}")
    #storage.append(data_dir, name+".csv", df)


if __name__ == "__main__":
    #gc.set_debug(gc.DEBUG_LEAK)
    N = 8
    iterations = 50
    batches = 4
    # learning_rate = 0.001 <-- Adam learning rate
    learning_rate = 0.05
    data_dir = "./data"
#    run_emnist(data_dir, "exchange-gossip", N, ExchangeGossip, learning_rate=learning_rate, batches=batches, iterations=iterations)
#    run_emnist(data_dir, "exchange-cycle", N, ExchangeGossip, learning_rate=learning_rate, guider=HamiltonCycleGuider, batches=batches, iterations=iterations)
    run_emnist(data_dir, "agg-gossip", N, Gossip, learning_rate=learning_rate, batches=batches, iterations=iterations)
    run_emnist(data_dir, "agg-hypercube", N, Hypercube, learning_rate=learning_rate, batches=batches, iterations=iterations)
    client_learning_rate = 0.1
    run_emnist(data_dir, "agg-fls", N, FLS, learning_rate=client_learning_rate, batches=batches, iterations=iterations)
    server_learning_rate = 1.0

