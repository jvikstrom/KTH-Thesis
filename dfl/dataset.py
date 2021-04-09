import numpy as np
import tensorflow as tf


class NN5Source:
    def __init__(self, file_path: str):
        raw_data = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                values = [v for v in line.split(',')]
                for i in range(len(values)):
                    try:
                        values[i] = float(values[i])
                    except ValueError:
                        values[i] = 0.0
                        continue
                if len(values) < 10:
                    continue
                raw_data.append(values)

        # Normalize the data. Didn't really work that well...
#        mean = np.mean(raw_data)
#        raw_data = np.array(raw_data) - mean
#        raw_data = raw_data / np.std(raw_data)

        time_series_train = [[] for _ in range(len(raw_data[0]))]
        time_series_test = [[] for _ in range(len(raw_data[0]))]
        for j in range(len(raw_data)):
            nn5 = raw_data[j]
            for i in range(len(nn5)):
                if j < len(raw_data) - 56:
                    time_series_train[i].append(nn5[i])
                else:
                    time_series_test[i].append(nn5[i])

        self.time_series_train = [np.array(v) for v in time_series_train]
        self.time_series_test = [np.array(v) for v in time_series_test]


def load_from_nn5(source: NN5Source, id, test=False, input_size=56):
    output_size = 56
    train = source.time_series_train[id]
    inputs = []
    for i in range(len(train)-input_size+1):
        inputs.append(train[i:i+input_size])
    if test:
        return np.array([inputs[len(inputs)-1]]).reshape((-1,input_size, 1)), np.array([source.time_series_test[id]]).reshape((-1,output_size, 1))

    outputs = []
    for i in range(input_size, len(train)-output_size+1):
        outputs.append(train[i:i+output_size])
    train_inputs = []
    labels = []
    for i in range(len(outputs)):
        train_inputs.append(inputs[i])
        labels.append(outputs[i])

    return np.array(train_inputs).reshape((-1,input_size, 1)), np.array(labels).reshape((-1,output_size, 1))


def load_from_emnist(source, id):
    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    )

    images, labels = [], []
    for image, label in data.as_numpy_iterator():
        images.append(image.reshape(28, 28, 1))
        labels.append(label)

    return np.array(images), np.array(labels)


def load_from_cifar(source, id):
    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id]).map(
        lambda e: (tf.reshape(e['image'], [-1]), e['label'])
    )

    images, labels = [], []
    for image, label in data.as_numpy_iterator():
        images.append(image.reshape(32, 3, 32, 1))
        labels.append(label)

    return np.array(images), np.array(labels)


SHAKESPEARE_LINE_BEGIN = '^'
SHAKESPEARE_OUT_OF_VOCAB = '~'
SHAKESPEARE_BOL = '<'
SHAKESPEARE_EOL = '>'


# Table is a mapping from string chars to indicies.
def load_from_shakespeare(source, id, table):
    def to_ids(x):
        s = tf.reshape(x['snippets'], shape=[1])
        chars = tf.strings.bytes_split(s).values
        ids = table.lookup(chars)
        return ids

    def seqs_of_len(arr, l):
        seqs = []
        for start in range(len(arr)):
            end = start + l
            if end < len(arr):
                seqs.append(arr[start:end])
        return seqs

    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id])

    inputs = []
    labels = []
    FEATURE_LEN = 80

    for d in data:
        seqs = seqs_of_len(to_ids(d), FEATURE_LEN)
        for i in range(len(seqs)-1):
            inputs.append(seqs[i])
            labels.append(seqs[i+1])
    return np.array(inputs), np.array(labels)


def concat_data(data):
    # Concat an array of tuple data into a pair of arrays.
    all_images, all_labels = [], []
    for images, labels in data:
        all_images.extend(images)
        all_labels.extend(labels)
    return np.array(all_images), np.array(all_labels)
