import numpy as np
from typing import List
from tqdm import tqdm
from client import Client
from dataset import concat_data
import gc
from pydantic import BaseModel


class TrainerConfig(BaseModel):
    batches: int
    iterations: int


class Trainer:
    def __init__(self, clients: List[Client], cfg: TrainerConfig):
        self.clients = clients
        self.test_concated = concat_data([client.get_test_data() for client in self.clients])
        self.train_concated = concat_data([client.get_train_data() for client in self.clients])
        self.test_evals = []
        self.trainer_config = cfg

    def __eval_data(self, data_set, epoch, data):
        losses, accuracies = [], []
        for client in tqdm(self.clients, desc="eval"):
            """
            Convert numpy array to tf.tensor: input_tensor = tf.convert_to_tensor(input_ndarray)
            Use the tensor directly as an argument to the model. output_tensor = model(input_tensor)
            Convert the output tensor to numpy back if needed. output_array = output_tensor.numpy()
            """
            loss, accuracy = client.model.evaluate(*data, verbose=0, batch_size=32)
            losses.append(loss)
            accuracies.append(accuracy)
        loss = np.mean(losses)
        accuracy = np.mean(accuracies)
        #diff = np.max(accuracies) - np.min(accuracies)
        print(f"{data_set} {epoch} ::: loss: {loss}   ----   accuracy: {accuracy}")
        #print(f"{data_set} {epoch} ::: accuracy diff: {diff}")
        return loss, accuracy

    def eval_test(self, epoch):
        loss, accuracy = self.__eval_data("TEST", epoch, self.test_concated)
        self.test_evals.append((epoch, loss, accuracy))

    def eval_train(self, epoch):
        return self.__eval_data("TRAIN", epoch, self.train_concated)

    def run(self):
        pass

    def step(self):
        # Python doesn't do a full collect otherwise, causing us to eventually run out of memory.
        gc.collect()
