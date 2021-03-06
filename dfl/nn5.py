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
from train import TrainerConfig, TrainerInput
from configs import Config, none_gossip_config, exchange_cycle_config, exchange_config, aggregate_hypercube_config, fls_config, centralized_config


def smape(y_p, y_t):
    return 2 * tf.reduce_mean((tf.abs(y_p - y_t) / (1e-8 + tf.abs(y_t) + tf.abs(y_p)))) * 100.0


def model_fn_factory(learning_rate, optimizer):
    def fn():
        num_encoder_tokens = 1
        num_decoder_tokens = 1
        latent_dim = 256
        # random_normal
        # random_uniform
        # truncated_normal
        # zeros
        # ones
        # glorot_normal
        # glorot_uniform
        # identity
        # orthogonal
        # constant
        # variance_scaling
        #ki = 'glorot_uniform'
        ki = tf.keras.initializers.RandomUniform(minval=0.1, maxval=0.2)
        ri = 'orthogonal'
        bi = 'zeros'
#        kid = 'glorot_uniform'
        kid = tf.keras.initializers.RandomUniform(minval=0.1, maxval=0.2)
        bid = 'zeros'

        encoder_inputs = tf.keras.layers.Input(shape=(None, num_encoder_tokens))
        encoder = tf.keras.layers.LSTM(latent_dim, return_state=True, activation='tanh',
                                       kernel_initializer=ki,
                                       recurrent_initializer=ri,
                                       bias_initializer=bi)
        encoder_outputs, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        #decoder_inputs = tf.keras.layers.Input(shape=(None, num_decoder_tokens))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = tf.keras.layers.LSTM(latent_dim, return_sequences=True, return_state=True, activation='tanh',
                                            kernel_initializer=ki,
                                            recurrent_initializer=ri,
                                            bias_initializer=bi)
        decoder_outputs, _, _ = decoder_lstm(encoder_inputs,
                                             initial_state=encoder_states)
        decoder_dense = tf.keras.layers.Dense(num_decoder_tokens, activation='relu',
                                              kernel_initializer=kid,
                                              bias_initializer=bid)
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = tf.keras.Model(encoder_inputs, decoder_outputs)

        model.compile(optimizer=optimizer(learning_rate),
                      loss=smape,#tf.keras.losses.MeanAbsoluteError(),#smape,
                      metrics=[smape])
        return model

    return fn


def run_nn5(nn5_file_path: str, data_dir: str, name: str, N, strategy, cfg: Config, learning_rate, version=1):
    # Load simulation data.
    nn5source = NN5Source(nn5_file_path)
    all_train_data = []
    all_test_data = []
    for i in range(nn5source.n_clients()):
        all_train_data.append(load_from_nn5(nn5source, i))
        all_test_data.append(load_from_nn5(nn5source, i, test=True))
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
            eval_train_gap=10,
            learning_rate=cfg.learning_rate,
            disable_tqdm=cfg.disable_tqdm), clients, cfg.extra_config, np.array(all_train_data), np.array(all_test_data))
    hyper.run()


def run(nn5_file_path: str, cfg: Config, version: int):
    run_nn5(nn5_file_path, cfg.data_dir, cfg.name, cfg.N, cfg.strategy, cfg, learning_rate=cfg.learning_rate,
               version=version)


if __name__ == "__main__":
    run_nn5(os.getenv("NN5"), os.getenv("DATA_DIR"), "fls", 2, FLS, fls_config(2,"",0.06,32,1000), 0.06)