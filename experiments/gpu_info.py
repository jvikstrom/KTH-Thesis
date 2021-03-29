import tensorflow as tf
print(f"Tensorflow version:  {tf.__version__}")
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))
print("GPU info: ", physical_devices)
