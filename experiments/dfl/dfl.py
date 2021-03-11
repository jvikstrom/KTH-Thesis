import sys
import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from typing import List
from tqdm import tqdm

tf.compat.v1.enable_eager_execution()


def load_data(source, id):
    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    )

    images, labels = [], []
    for image, label in data.as_numpy_iterator():
        images.append(image.reshape(28,28,1))
        labels.append(label)

    return np.array(images), np.array(labels)


class Client:
    def __init__(self, id, train_source, test_source, model_fn):
        """
        :param id: Numerical id
        :param data_source: Tf data
        :param model_fn: Returns a keras model.
        """
        self.model: tf.keras.models.Model = model_fn()
        self.id = id
        self.train_data = load_data(train_source, id)
        self.test_data = load_data(test_source, id)

    def train(self, epochs: int = 1):
        self.model.fit(*self.train_data, batch_size=1, epochs=epochs, verbose=0)

    def get_train_data(self):
        return self.train_data

    def get_test_data(self):
        return self.test_data


def combine(a: Client, b: Client):
    weights = [model.get_weights() for model in [a.model, b.model]]
    new_weights = list()
    for a_weight, b_weight in zip(*weights):
        new_weights.append((a_weight + b_weight) / 2.0)
        #print("a",a_weight.shape, "b", b_weight.shape)
    #print("done")
    a.model.set_weights(new_weights)
    b.model.set_weights(new_weights)


def preprocess_client_targets(clients: List[Client]):
    n_rounds = np.log2(len(clients))
    rounds = []
    for i in range(int(n_rounds)):
        targets = []
        for j in range(len(clients)):
            xor = 0
            xor |= (1 << i)
            target = int(j ^ xor)
            if target > j:
                targets.append((j, target))
        rounds.append(targets)
    return rounds


def concat_data(data):
    all_images, all_labels = [], []
    for images, labels in data:
        all_images.extend(images)
        all_labels.extend(labels)
    return np.array(all_images), np.array(all_labels)


class Hypercube:
    def __init__(self, clients: List[Client]):
        self.clients = clients
        for client in clients:
            client.model.set_weights(clients[0].model.get_weights())
        self.rounds = preprocess_client_targets(clients)
        self.test_concated = concat_data([client.get_test_data() for client in self.clients])
        self.train_concated = concat_data([client.get_train_data() for client in self.clients])

    def __eval_data(self, data_set, epoch, data):
        losses, accuracies = [], []
        for client in tqdm(self.clients, desc="eval"):
            loss, accuracy = client.model.evaluate(*data, verbose=0, batch_size=32)
            losses.append(loss)
            accuracies.append(accuracy)
        print(f"{data_set} {epoch} ::: loss: {np.mean(losses)}   ----   accuracy: {np.mean(accuracies)}")

    def eval_test(self, epoch):
        return self.__eval_data("TEST", epoch, self.test_concated)

    def eval_train(self, epoch):
        return self.__eval_data("TRAIN", epoch, self.train_concated)

    def run(self, epochs: int = 1, iterations: int = 100):
        for i in range(iterations):
            for client in tqdm(self.clients, desc="train"):
                client.train(epochs=epochs)

            for round in self.rounds:
                for a, b in round:
                    combine(self.clients[a], self.clients[b])
            if i % 10 == 0:
                self.eval_train(i)
            self.eval_test(i)
            a_weights = self.clients[0].model.get_weights()
            for a_weight in a_weights:
                print(f"mean:{a_weight.mean()}, max:{a_weight.max()}, min:{a_weight.min()}, std:{a_weight.std()}")


if __name__ == "__main__":
    # Load simulation data.
    train, test = tff.simulation.datasets.emnist.load_data()
    def model_fn():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28,28,1)),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(10),


            #            tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
#            tf.keras.layers.Dense(10),
        ])
        model.compile(optimizer=tf.optimizers.SGD(0.001),
                      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
        return model


    N = 128
    clients = [Client(i, train, test, model_fn) for i in range(N)]

    hyper = Hypercube(clients)
    hyper.run(epochs=5, iterations=200)
