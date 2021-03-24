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
from train import TrainerConfig


class Config(BaseModel):
    N: int
    data_dir: str
    name: str
    learning_rate: float
    extra_config: Any
    strategy: Any
    optimizer: Callable = tf.optimizers.SGD


def none_gossip_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
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


def exchange_cycle_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
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


def exchange_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
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


def aggregate_hypercube_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
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


def fls_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
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


def centralized_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="fls",
        extra_config=TrainerConfig(
            batches=batches,
            iterations=iterations,
        ),
        strategy=Centralized,
    )
