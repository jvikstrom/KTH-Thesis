import numpy as np
from typing import List, Callable
from tqdm import tqdm
from client import Client
from dataset import concat_data
import gc
from pydantic import BaseModel
from storage import append


class TrainerConfig(BaseModel):
    batches: int
    iterations: int


class Trainer:
    def __init__(self, clients: List[Client], cfg: TrainerConfig):
        self.clients = clients
        self.test_concated = concat_data([client.get_test_data() for client in self.clients])
        self.train_concated = concat_data([client.get_train_data() for client in self.clients])
        self.test_evals = []
        self.test_model_stats = []
        self.train_evals = []
        self.trainer_config = cfg
        print(f"{len(self.train_concated[0])} number of train samples, {len(self.test_concated[0])} number of test samples")

    def __eval_data(self, data_set, epoch, data, model=None):
        losses, accuracies = [], []
        if model is not None:
            loss, accuracy = model.evaluate(*data, verbose=0, batch_size=32)
            losses.append(loss)
            accuracies.append(accuracy)
        else:
            for client in tqdm(self.clients, desc="eval"):
                """
                Convert numpy array to tf.tensor: input_tensor = tf.convert_to_tensor(input_ndarray)
                Use the tensor directly as an argument to the model. output_tensor = model(input_tensor)
                Convert the output tensor to numpy back if needed. output_array = output_tensor.numpy()
                """
                loss, accuracy = client.model.evaluate(*data, verbose=0, batch_size=32)
                losses.append(loss)
                accuracies.append(accuracy)
        print(f"{data_set} {epoch} ::: loss: {np.mean(losses)}   ----   accuracy: {np.mean(accuracies)}")
        return losses, accuracies

    def eval_test(self, epoch, model=None):
        losses, accuracies = self.__eval_data("TEST", epoch, self.test_concated, model=model)
        self.test_model_stats.append((epoch, losses, accuracies))
        self.test_evals.append((epoch, np.mean(losses), np.mean(accuracies)))

    def eval_train(self, epoch, model=None):
        losses, accuracies = self.__eval_data("TRAIN", epoch, self.train_concated, model=model)
        loss, accuracy = np.mean(losses), np.mean(accuracies)
        self.train_evals.append((epoch, loss, accuracy))
        return loss, accuracy

    def run(self):
        pass

    def step(self):
        # Python doesn't do a full collect otherwise, causing us to eventually run out of memory.
        gc.collect()

    def step_and_eval(self, epoch, model=None):
        if epoch % 10 == 0:
            self.eval_test(epoch, model=model)
        if epoch % 50 == 0:
            self.eval_train(epoch, model=model)
        self.step()
