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
from typing import Any
from train import TrainerConfig


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


class Config(BaseModel):
    N: int
    data_dir: str
    name: str
    learning_rate: float
    extra_config: Any
    strategy: Any


def run_emnist(data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1):
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data(only_digits=False)
    clients = [Client(load_from_emnist(train, i), load_from_emnist(test, i), model_fn_factory(learning_rate)) for i in range(N)]

    hyper = strategy(clients, cfg)
    hyper.run()
    df = pd.DataFrame()
    for i in range(len(hyper.test_evals)):
        loss, accuracy = hyper.test_evals[i]
        df = df.append({
            'name': f"{name}-{version}",
            'version': version,
            'N': N,
            'batches': batches,
            'iterations': iterations,
            'current_iteration': i,
            'loss': loss,
            'accuracy': accuracy,
        }, ignore_index=True)

    print(f"Writing: {len(df)} records to {name}")
    storage.append(data_dir, name+".csv", df)


def run(cfg: Config, version: int):
    run_emnist(cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg.extra_config, learning_rate=cfg.learning_rate, version=version)


if __name__ == "__main__":
    number = 10
    N = 16
    iterations = 1500
    batches = 12
    # learning_rate = 0.001 <-- Adam learning rate
    learning_rate = 0.05
    data_dir = "./data"
    none_gossip_cfg = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="none-gossip",
        extra_config=GossipConfig(
            average=False,
            base_config=BaseGossipConfig(
                trainer_config=TrainerConfig(
                    batches=batches,
                    iterations=iterations,
                ),
                guider=Guider,
            )
        ),
        strategy=Gossip,
    )
    exchange_cycle_config = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="exchange-cycle-gossip",
        extra_config=ExchangeConfig(
            base_config=BaseGossipConfig(
                trainer_config=TrainerConfig(
                    batches=batches,
                    iterations=iterations,
                ),
                guider=HamiltonCycleGuider,
            )
        ),
        strategy=ExchangeGossip,
    )
    exchange_config = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="exchange-gossip",
        extra_config=ExchangeConfig(
            base_config=BaseGossipConfig(
                trainer_config=TrainerConfig(
                    batches=batches,
                    iterations=iterations,
                ),
                guider=Guider,
            )
        ),
        strategy=ExchangeGossip,
    )
    aggregate_hypercube_config = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="agg-hypercube",
        extra_config=HypercubeConfig(
            trainer_config=TrainerConfig(
                batches=batches,
                iterations=iterations,
            ),
        ),
        strategy=Hypercube,
    )
    fls_config = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="fls",
        extra_config=FLSConfig(
            trainer_config=TrainerConfig(
                batches=batches,
                iterations=iterations,
            ),
        ),
        strategy=FLS,
    )
    centralized_config = Config(
        N=N,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="fls",
        extra_config=TrainerConfig(
                batches=batches,
                iterations=iterations,
        ),
        strategy=Centralized,
    )
    for i in range(number):
#        run(none_gossip_cfg, i)
#        run(exchange_config, i)
        run(exchange_cycle_config, i)
#        run(aggregate_hypercube_config, i)
#        run(fls_config, i)
#        run(centralized_config, i)

        #        run_emnist(data_dir, f"agg-gossip", N, Gossip, learning_rate=learning_rate, batches=batches,
#                   iterations=iterations, version=i)
        print(f"\n\n\n\n-------------------\nIteration {i}\n---------------------------\n\n\n\n\n")
"""        run_emnist(data_dir, f"exchange-gossip", N, ExchangeGossip, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"exchange-cycle", N, ExchangeGossip, learning_rate=learning_rate, guider=HamiltonCycleGuider, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"agg-gossip", N, Gossip, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        run_emnist(data_dir, f"agg-hypercube", N, Hypercube, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
        client_learning_rate = 0.05
        run_emnist(data_dir, f"agg-fls", N, FLS, learning_rate=client_learning_rate, batches=batches, iterations=iterations, version=i)
        server_learning_rate = 1.0
        run_emnist(data_dir, f"centralized", N, Centralized, learning_rate=learning_rate, batches=batches, iterations=iterations, version=i)
"""
