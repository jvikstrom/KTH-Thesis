import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_emnist, concat_data
from client import Client, Guider
from hypercube import Hypercube, HamiltonCycleGuider, HypercubeConfig
from gossip_impl import Gossip, ExchangeGossip, GossipConfig, BaseGossipConfig, ExchangeConfig
from centralized import Centralized
from fls import FLS, FLSConfig
import numpy as np
import pandas as pd
import storage
import gc
from tqdm import tqdm
from pydantic import BaseModel
from typing import Any, Callable
from train import TrainerConfig, TrainerInput
from configs import Config, none_gossip_config, exchange_cycle_config, exchange_config, aggregate_hypercube_config, fls_config, centralized_config


def model_fn_factory(learning_rate, optimizer):
    def fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 3, strides=(1, 1), input_shape=(28, 28, 1), activation='relu'),
            tf.keras.layers.Conv2D(64, 3, strides=(1, 1), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(62, activation='softmax'),
        ])

        model.compile(optimizer=optimizer(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    return fn

# Exchange cycle adam 1 at 100: (train, test): (0.347, 0.293)
# Exchange cycle adam 2 at 100: (train, test): (0.463, 0.393)
# Exchange cycle adam 3 at 100: (train, test): (0.498, 0.415)
# Exchange cycle adam 4 at 100: (train, test): (0.402, 0.345)


# Exchange cycle adam weights 1 at 100: (train, test): (0.431, 0.373)
# Exchange cycle adam weights 2 at 100: (train, test): (0.612, 0.516)
# Exchange cycle adam weights 3 at 100: (train, test): (0.277, 0.243)
# Exchange cycle adam weights 4 at 100: (train, test): (0.671, 0.599)


def run_emnist(data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1, failure_schedule=None):
    data_fac = os.getenv("PERC_DATA")
    if data_fac is not None:
        data_fac = float(data_fac)
    if data_fac is None:
        data_fac = 0.1
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    print("Loading data into memory.")
    all_train_data, all_test_data = [], []
    for i in tqdm(range(int(len(train.client_ids) * data_fac)), desc="Loading data", disable=cfg.disable_tqdm):
        all_train_data.append(load_from_emnist(train, i))
        all_test_data.append(load_from_emnist(test, i))

    print("Loaded all data.")
    clients = [Client(
        all_train_data[i],
        all_test_data[i],
        model_fn_factory(learning_rate, cfg.optimizer)
    ) for i in range(N)]

    hyper = strategy(TrainerInput(
        name=name,
        version=version,
        data_dir=data_dir,
        eval_test_gap=10,
        eval_train_gap=50,
        disable_tqdm=cfg.disable_tqdm),
        clients, cfg.extra_config, all_train_data, all_test_data, failure_schedule=failure_schedule)
    print(f"Start running {name}...")
    hyper.run()


def run(cfg: Config, version: int, failure_schedule=None):
    run_emnist(cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg, learning_rate=cfg.learning_rate,
               version=version, failure_schedule=failure_schedule)


if __name__ == "__main__":
    number = 10
    N = 16
    iterations = 1500
    batches = 12
    # learning_rate = 0.001 <-- Adam learning rate
    learning_rate = 0.05
    data_dir = "./data"
    none_gossip_cfg = none_gossip_config(n=N, data_dir=data_dir, learning_rate=learning_rate, batches=batches,
                                         iterations=iterations)
    exchange_cycle_config = exchange_cycle_config(n=N, data_dir=data_dir, learning_rate=learning_rate, batches=batches,
                                                  iterations=iterations)
    exchange_config = exchange_config(n=N, data_dir=data_dir, learning_rate=learning_rate, batches=batches,
                                      iterations=iterations)
    aggregate_hypercube_config = aggregate_hypercube_config(n=N, data_dir=data_dir, learning_rate=learning_rate,
                                                            batches=batches,
                                                            iterations=iterations)
    fls_config = fls_config(n=N, data_dir=data_dir, learning_rate=learning_rate, batches=batches, iterations=iterations)
    centralized_config = centralized_config(n=N, data_dir=data_dir, learning_rate=learning_rate, batches=batches,
                                            iterations=iterations)
    for i in range(number):
        print(f"\n\n\n\n-------------------\nIteration {i}\n---------------------------\n\n\n\n\n")
        #        run(none_gossip_cfg, i)
        #        run(exchange_config, i)
        run(exchange_cycle_config, i)
        #        run(aggregate_hypercube_config, i)
        #        run(fls_config, i)
        #        run(centralized_config, i)

        #        run_emnist(data_dir, f"agg-gossip", N, Gossip, learning_rate=learning_rate, batches=batches,
        #                   iterations=iterations, version=i)
"""        run_emnist(data_dir, f"exchange-gossip", N, ExchangeGossip, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"exchange-cycle", N, ExchangeGossip, learning_rate=learning_rate, guider=HamiltonCycleGuider, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"agg-gossip", N, Gossip, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"agg-hypercube", N, Hypercube, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        client_learning_rate = 0.05
        run_emnist(data_dir, f"agg-fls", N, FLS, learning_rate=client_learning_rate, batches=batches, iterations=iterations, version=i)
        server_learning_rate = 1.0
        run_emnist(data_dir, f"centralized", N, Centralized, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
"""
