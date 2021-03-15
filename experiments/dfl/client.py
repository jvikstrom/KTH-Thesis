import tensorflow as tf


class Client:
    def __init__(self, train_data, test_data, model_fn):
        """
        :param id: Numerical id
        :param data_source: Tf data
        :param model_fn: Returns a keras model.
        """
        self.model: tf.keras.models.Model = model_fn()
        self.train_data = train_data
        self.test_data = test_data

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
    a.model.set_weights(new_weights)
    b.model.set_weights(new_weights)
