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

def concat_data(data):
    # Concat an array of tuple data into a pair of arrays.
    all_images, all_labels = [], []
    for images, labels in data:
        all_images.extend(images)
        all_labels.extend(labels)
    return np.array(all_images), np.array(all_labels)
