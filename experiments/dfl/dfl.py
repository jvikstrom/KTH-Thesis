import sys
import tensorflow as tf
import tensorflow_federated as tff
from dataset import load_from_emnist
from client import Client
from hypercube import Hypercube
from gossip_impl import Gossip, Guider
tf.compat.v1.enable_eager_execution()


def run_emnist(N, epochs=1, iterations=100):
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data()

    def model_fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),
        ])
        model.compile(optimizer=tf.optimizers.Adam(0.0005),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model

    clients = [Client(load_from_emnist(train, i), load_from_emnist(test, i), model_fn) for i in range(N)]

    hyper = Gossip(clients, Guider)
    hyper.run(epochs=epochs, iterations=iterations)


if __name__ == "__main__":
    run_emnist(1024, 3, 200)
