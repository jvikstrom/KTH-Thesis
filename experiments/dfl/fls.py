from train import Trainer


class FLS(Trainer):
    def __init__(self, clients, unused):
        Trainer.__init__(self, clients)

    def run(self, batches: int = 1, iterations: int = 100):
        for i in range(iterations):
            old_weights = [weights for weights in self.clients[0].model.get_weights()]
            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in old_weights])
            # Does server SGD aggregation with server learning rate = 1.0
            for client in self.clients:
                client.train(batches)
            all_weights = [client.model.get_weights() for client in self.clients]

            aggregated_weights = all_weights[0]
            for j in range(1, len(all_weights)):
                for k in range(len(aggregated_weights)):
                    aggregated_weights[k] += all_weights[j][k]
                    # aggregated_weights[k] += aggregated_weights[j][k] - old_model[k]
            for k in range(len(aggregated_weights)):
                aggregated_weights[k] = aggregated_weights[k] / len(self.clients)
                # aggregated_weights[k] = old_model[k] - server_learning_rate * aggregated_weights[k] / len(self.clients)

            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in aggregated_weights])
            loss, accuracy = self.clients[0].model.evaluate(*self.test_concated, verbose=0, batch_size=32)
            self.test_evals.append((loss, accuracy))
            print(f"FLS {i} ::: loss: {loss}   ----   accuracy: {accuracy}")
            Trainer.step(self)

"""
import nest_asyncio
nest_asyncio.apply()
import collections

import numpy as np
import tensorflow as tf
import tensorflow_federated as tff

np.random.seed(0)

tff.federated_computation(lambda: 'Hello, World!')()
emnist_train, emnist_test = tff.simulation.datasets.emnist.load_data()
example_dataset = emnist_train.create_tf_dataset_for_client(
    emnist_train.client_ids[0])

example_element = next(iter(example_dataset))

example_element['label'].numpy()

NUM_CLIENTS = 64
NUM_EPOCHS = 3
BATCH_SIZE = 128
SHUFFLE_BUFFER = 100
PREFETCH_BUFFER = 10

def preprocess(dataset):

  def batch_format_fn(element):
    # Flatten a batch `pixels` and return the features as an `OrderedDict`.
    return collections.OrderedDict(
        x=tf.reshape(element['pixels'], [-1, 28, 28, 1]),
        y=tf.reshape(element['label'], [-1, 1]))

  return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
      BATCH_SIZE).map(batch_format_fn).prefetch(PREFETCH_BUFFER)
preprocessed_example_dataset = preprocess(example_dataset)

sample_batch = tf.nest.map_structure(lambda x: x.numpy(),
                                     next(iter(preprocessed_example_dataset)))

def make_federated_data(client_data, client_ids):
  return [
      preprocess(client_data.create_tf_dataset_for_client(x))
      for x in client_ids
  ]
sample_clients = emnist_train.client_ids[0:NUM_CLIENTS]

federated_train_data = make_federated_data(emnist_train, sample_clients)

print('Number of client datasets: {l}'.format(l=len(federated_train_data)))
print('First dataset: {d}'.format(d=federated_train_data[0]))
def keras_model_fn():
    return tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, 5, padding='same', activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Conv2D(64, 5, padding='same', activation='relu'),
        tf.keras.layers.MaxPooling2D(pool_size=2, strides=2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10),
    ])

def model_fn():
  # We _must_ create a new model here, and _not_ capture it from an external
  # scope. TFF will call this within different graph contexts.
  keras_model = keras_model_fn()
  return tff.learning.from_keras_model(
      keras_model,
      input_spec=preprocessed_example_dataset.element_spec,
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
iterative_process = tff.learning.build_federated_averaging_process(
    model_fn,
    client_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=0.02),
    server_optimizer_fn=lambda: tf.keras.optimizers.SGD(learning_rate=1.0))
state = iterative_process.initialize()
NUM_ROUNDS = 150
for round_num in range(NUM_ROUNDS):
  state, metrics = iterative_process.next(state, federated_train_data)
  print('round {:2d}, metrics={}'.format(round_num, metrics))

#state, metrics = iterative_process.next(state, federated_train_data)
#print('round  1, metrics={}'.format(metrics))
"""