import sys
import os
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_shakespeare, concat_data
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

#You step up to the wall and take a closer look. Indeed it is covered with various sex toys of different shape, size and material. From the floor to the ceiling almost every tile has one. You try to pull one off but it´s either glued or screwed right on there tightly.
#You try to discern a pattern in them and not much comes up first. But after a while you notice that only the middle of the wall contains red colored sex toys. All of them in fact, and they seem to write something out.
#V-I-IX-III - You are smart enough to know that these are Roman numerals. This must be the passcode then.

def model_fn_factory(learning_rate, optimizer):
    def fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Embedding(90, 8, input_shape=(80,)),
#            tf.keras.layers.Reshape([80, 8]),
            #tf.keras.layers.Flatten(),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.LSTM(256, return_sequences=True),
            tf.keras.layers.Dense(90, activation='softmax')
#            tf.keras.layers.Conv2D(32, 3, strides=(1, 1), input_shape=(28, 28, 1), activation='relu'),
#            tf.keras.layers.Conv2D(64, 3, strides=(1, 1), activation='relu'),
#            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
#            tf.keras.layers.Dropout(0.25),
#            tf.keras.layers.Flatten(),
#            tf.keras.layers.Dense(128, activation='relu'),
#            tf.keras.layers.Dropout(0.5),
#            tf.keras.layers.Dense(62, activation='softmax'),
        ])

        #model.summary()
        model.compile(optimizer=optimizer(learning_rate),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    return fn


def run_shakespeare(data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1):
    # ^ is beginging of line token
    #
    vocab = list('dhlptx@DHLPTX $(,048cgkoswCGKOSW[_#\'/37;?bfjnrvzBFJNRVZ"&*.26:\naeimquyAEIMQUY]!%)-159\r^~<>')
    print("Vocab length:",len(vocab))

    # Create chars to index lookup table.
    table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=vocab, values=tf.constant(list(range(len(vocab))),
                                           dtype=tf.int64)),
        default_value=0)

    # Load simulation data.
    train, test = tff.simulation.datasets.shakespeare.load_data()
    clients = [Client(
        load_from_shakespeare(train, i, table),
        load_from_shakespeare(test, i, table),
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


def run(cfg: Config, version: int):
    run_shakespeare(cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg, learning_rate=cfg.learning_rate,
                    version=version)
