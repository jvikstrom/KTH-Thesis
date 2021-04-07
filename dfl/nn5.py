import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_nn5, concat_data, NN5Source
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
from configs import Config, none_gossip_config, exchange_cycle_config, exchange_config, aggregate_hypercube_config, fls_config, centralized_config


def model_fn_factory(learning_rate, optimizer):
    def fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.LSTM(256, activation='relu', input_shape=(70, 1)),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(56),
        ])

        model.compile(optimizer=optimizer(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    return fn


def run_nn5(nn5_file_path: str, data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1):
    # Load simulation data.
    nn5source = NN5Source(nn5_file_path)
    clients = [Client(
        load_from_nn5(nn5source, i),
        load_from_nn5(nn5source, i, test=True),
        model_fn_factory(learning_rate, cfg.optimizer)
    ) for i in range(N)]

    return
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
        iter, losses, accuracies = hyper.test_model_stats[i]
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


def run(nn5_file_path: str, cfg: Config, version: int):
    run_nn5(nn5_file_path, cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg, learning_rate=cfg.learning_rate,
               version=version)


if __name__ == "__main__":
    run_nn5(os.getenv("NN5"), os.getenv("DATA_DIR"), "fls", 2, FLS, fls_config(2,"",0.06,32,1000), 0.06)