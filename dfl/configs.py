import sys
import os
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_federated as tff
from dataset import load_from_emnist, concat_data
from client import Guider
from hypercube import Hypercube, HamiltonCycleGuider, HypercubeConfig
from gossip_impl import Gossip, ExchangeGossip, GossipConfig, BaseGossipConfig, ExchangeConfig
from centralized import Centralized
from fls import FLS, FLSConfig
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
    optimizer: Callable


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
        optimizer=tf.optimizers.SGD,
    )


def exchange_cycle_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float) -> Config:
    return Config(
        N=n,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="exchange-cycle",
        extra_config=ExchangeConfig(
            base_config=BaseGossipConfig(
                trainer_config=TrainerConfig(
                    batches=batches,
                    iterations=iterations,
                ),
                guider=HamiltonCycleGuider,
            ),
            swap_optimizer=True,
        ),
        strategy=ExchangeGossip,
        optimizer=tf.optimizers.SGD,
    )


def exchange_cycle_adam_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    cfg = exchange_cycle_config(n, data_dir, learning_rate, batches, iterations)
#    cfg.optimizer = tfa.optimizers.Yogi
    cfg.optimizer = tf.optimizers.Adam
    cfg.name += "-adam"
    return cfg


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
            ),
            swap_optimizer=False,
        ),
        strategy=ExchangeGossip,
        optimizer=tf.optimizers.SGD,
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
        optimizer=tf.optimizers.SGD,
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
        optimizer=tf.optimizers.SGD,
    )


def centralized_config(n: int, data_dir: str, learning_rate: float, batches: float, iterations: float):
    return Config(
        N=n,
        learning_rate=learning_rate,
        data_dir=data_dir,
        name="centralized",
        extra_config=TrainerConfig(
            batches=batches,
            iterations=iterations,
        ),
        strategy=Centralized,
        optimizer=tf.optimizers.SGD,
    )
