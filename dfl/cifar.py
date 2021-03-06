import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_cifar, concat_data
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
from train import TrainerConfig
from configs import Config, none_gossip_config, exchange_cycle_config, exchange_config, aggregate_hypercube_config, \
    fls_config, centralized_config
from resnet import ResnetBuilder

def model_fn_factory(learning_rate, optimizer):
    def fn():
#        model = tf.keras.models.Sequential([
#            tf.keras.layers.Conv1D(64, 7, strides=2, input_shape=(112, 112, 1), activation='relu'),
#            tf.keras.layers.Conv2D()
            # tf.keras.layers.Conv2D(32, 3, strides=(1, 1), input_shape=(28, 28, 1), activation='relu'),
            # tf.keras.layers.Conv2D(64, 3, strides=(1, 1), activation='relu'),
            # tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            # tf.keras.layers.Dropout(0.25),
            # tf.keras.layers.Flatten(),
            # tf.keras.layers.Dense(128, activation='relu'),
            # tf.keras.layers.Dropout(0.5),
            # tf.keras.layers.Dense(62, activation='softmax'),
#        ])
        model = ResnetBuilder.build_resnet_18((32,32,3), 100)

        model.compile(optimizer=optimizer(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    return fn


def run_cifar(data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1):
    # Load simulation data.
    train, test = tff.simulation.datasets.cifar100.load_data()
    clients = [Client(
        load_from_cifar(train, i),
        load_from_cifar(test, i),
        model_fn_factory(learning_rate, cfg.optimizer)
    ) for i in range(N)]

    hyper = strategy(clients, cfg.extra_config)
    hyper.run()
    df = pd.DataFrame()
    for i in range(len(hyper.test_evals)):
        iter, loss, accuracy = hyper.test_evals[i]
        df = df.append({
            'name': f"{name}-{version}",
            'version': version,
            'N': N,
            #            'batches': batches,
            #            'iterations': iterations,
            'current_iteration': iter,
            'loss': loss,
            'accuracy': accuracy,
        }, ignore_index=True)

    print(f"Writing: {len(df)} records to {name}")
    storage.append(data_dir, name + ".csv", df)

    df = pd.DataFrame()
    for i in range(len(hyper.train_evals)):
        iter, loss, accuracy = hyper.train_evals[i]
        df = df.append({
            'name': f"{name}-{version}",
            'version': version,
            'N': N,
            #            'batches': batches,
            #            'iterations': iterations,
            'current_iteration': iter,
            'loss': loss,
            'accuracy': accuracy,
        }, ignore_index=True)

    print(f"Writing: {len(df)} records to {name}-train")
    storage.append(data_dir, name + "-train.csv", df)

    df = pd.DataFrame()
    for i in range(len(hyper.test_model_stats)):
        iter, accuracies, losses = hyper.test_model_stats[i]
        di = {
            'name': f"{name}-{version}",
            'version': version,
            'N': N,
            'current_iteration': iter,
        }
        for j in range(len(accuracies)):
            di[f"{name}-accuracy-{j}"] = accuracies[j]
            di[f"{name}-loss-{j}"] = losses[j]
        df = df.append(di, ignore_index=True)
    storage.append(data_dir, f"{name}-{version}-models.csv", df)


def run(cfg: Config, version: int):
    run_cifar(cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg, learning_rate=cfg.learning_rate,
               version=version)


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
