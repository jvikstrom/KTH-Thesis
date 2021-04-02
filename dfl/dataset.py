import numpy as np
import tensorflow as tf


def load_from_emnist(source, id):
    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id]).map(
        lambda e: (tf.reshape(e['pixels'], [-1]), e['label'])
    )

    images, labels = [], []
    for image, label in data.as_numpy_iterator():
        images.append(image.reshape(28,28,1))
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
        #print(x, ids)
        return ids

    def seqs_of_len(arr, l):
        seqs = []
        for start in range(len(arr)):
            end = start + l
            if end < len(arr):
                seqs.append(arr[start:end])
        return seqs
    #def split_input_target(chunk):
    #    print(chunk)
    #    input_text = tf.map_fn(lambda x: x[:-1], chunk)
    #    target_text = tf.map_fn(lambda x: x[1:], chunk)
    #    return (input_text, target_text)

    data: tf.data.Dataset = source.create_tf_dataset_for_client(source.client_ids[id])

    #mapped = data.map(to_ids).unbatch().map(split_input_target)
    #print(mapped)
    #print(data)
    inputs = []
    labels = []
    FEATURE_LEN = 80

    for d in data:
#        print(d, to_ids(d))
        # TODO: Also add the other 4 characters.
        seqs = seqs_of_len(to_ids(d), FEATURE_LEN)
        for i in range(len(seqs)-1):
            inputs.append(seqs[i])
            labels.append(seqs[i+1])
#    print("inp, labels:", inputs[0], labels[0])
    return np.array(inputs), np.array(labels)

#    raise AssertionError("lol")


def concat_data(data):
    # Concat an array of tuple data into a pair of arrays.
    all_images, all_labels = [], []
    for images, labels in data:
        all_images.extend(images)
        all_labels.extend(labels)
    return np.array(all_images), np.array(all_labels)
