import numpy as np
from typing import List, Callable
from tqdm import tqdm
from client import Client
from dataset import concat_data
import gc
from pydantic import BaseModel
from storage import append
import pandas as pd
import storage


class TrainerConfig(BaseModel):
    batches: int
    iterations: int


class TrainerInput(BaseModel):
    name: str
    version: int
    data_dir: str
    eval_test_gap: int
    eval_train_gap: int
    disable_tqdm: bool


class Trainer:
    def __init__(self, trainer_input: TrainerInput, clients: List[Client], cfg: TrainerConfig,
                 all_train_data, all_test_data, failure_schedule=None):
        self.name = trainer_input.name
        self.version = trainer_input.version
        self.data_dir = trainer_input.data_dir
        self.eval_test_gap = trainer_input.eval_test_gap
        self.eval_train_gap = trainer_input.eval_train_gap
        self.max_iter = cfg.iterations
        self.disable_tqdm = trainer_input.disable_tqdm
        print(f"Running with disabled tqdm {self.disable_tqdm}")

        self.clients = clients
        self.all_test_data = all_test_data
        self.all_train_data = all_train_data
        self.test_concated = concat_data(all_test_data)
        #        self.test_concated = concat_data([client.get_test_data() for client in self.clients])

        self.train_concated = concat_data([client.get_train_data() for client in self.clients])
        self.test_evals = []
        self.test_model_stats = []
        self.train_evals = []
        self.trainer_config = cfg
        print(
            f"{len(self.train_concated[0])} number of train samples, {len(self.test_concated[0])} number of test samples")
        if failure_schedule is not None:
            self._fail_per_iter = failure_schedule['fails']
            self._alive_per_iter = failure_schedule['alive']
            self._join_per_iter = failure_schedule['join']
        else:
            self._fail_per_iter = []
            self._alive_per_iter = []
            self._join_per_iter = []
        self.currently_alive = list(range(len(clients)))
        self.last_write = -1

    def __eval_data(self, data_set, epoch, data, model=None):
        losses, accuracies = [], []
        if model is not None:
            loss, accuracy = model.evaluate(*data, verbose=0, batch_size=32)
            losses.append(loss)
            accuracies.append(accuracy)
        else:
            for client in tqdm(self.clients, desc="eval", disable=self.disable_tqdm):
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

    def step(self, epoch):
        # Python doesn't do a full collect otherwise, causing us to eventually run out of memory.
        gc.collect()
        self._write_incremental(epoch)
        # Recalculate what's alive.
        if epoch >= len(self._fail_per_iter) - 1:
            return
        for die_idx, new_client_idx, join_by in zip(self._fail_per_iter[epoch], self._alive_per_iter[epoch],
                                                    self._join_per_iter[epoch]):
            self.clients[die_idx].set_data(self.all_train_data[new_client_idx], self.all_test_data[new_client_idx])
            self.clients[die_idx].model.set_weights(
                [weights.copy() for weights in self.clients[join_by].model.get_weights()])

    def step_and_eval(self, epoch, model=None):
        # Eval every iteration the last 100 iterations
        if epoch % self.eval_test_gap == 0 or self.max_iter - epoch < 100:
            self.eval_test(epoch, model=model)
        if epoch % self.eval_train_gap == 0:
            self.eval_train(epoch, model=model)
        self.step(epoch)

    def _write_incremental(self, epoch):
        df = pd.DataFrame()
        for i in range(len(self.test_evals)):
            iter, loss, accuracy = self.test_evals[i]
            if iter > self.last_write:
                df = df.append({
                    'name': f"{self.name}-{self.version}",
                    'version': self.version,
                    'N': len(self.clients),
                    #            'batches': batches,
                    #            'iterations': iterations,
                    'current_iteration': iter,
                    'loss': loss,
                    'accuracy': accuracy,
                }, ignore_index=True)
        if len(df) > 0:
            print(f"Writing: {len(df)} records to {self.name}")
        storage.append(self.data_dir, self.name + ".csv", df)

        df = pd.DataFrame()
        for i in range(len(self.train_evals)):
            iter, loss, accuracy = self.train_evals[i]
            if iter > self.last_write:
                df = df.append({
                    'name': f"{self.name}-{self.version}",
                    'version': self.version,
                    'N': len(self.clients),
                    #            'batches': batches,
                    #            'iterations': iterations,
                    'current_iteration': iter,
                    'loss': loss,
                    'accuracy': accuracy,
                }, ignore_index=True)

        if len(df) > 0:
            print(f"Writing: {len(df)} records to {self.name}")
        storage.append(self.data_dir, self.name + "-train.csv", df)

        df = pd.DataFrame()
        for i in range(len(self.test_model_stats)):
            iter, losses, accuracies = self.test_model_stats[i]
            if iter > self.last_write:
                di = {
                    'name': f"{self.name}-{self.version}",
                    'version': self.version,
                    'N': len(self.clients),
                    'current_iteration': iter,
                }
                for j in range(len(accuracies)):
                    di[f"{self.name}-accuracy-{j}"] = accuracies[j]
                    di[f"{self.name}-loss-{j}"] = losses[j]
                df = df.append(di, ignore_index=True)
        if len(df) > 0:
            print(f"Writing: {len(df)} records to {self.name}-{self.version}-models.csv")

        storage.append(self.data_dir, f"{self.name}-{self.version}-models.csv", df)
        self.last_write = epoch
        self.test_model_stats = list(filter(lambda iter: iter[0] < self.last_write, self.test_model_stats))
        self.test_evals = list(filter(lambda iter: iter[0] < self.last_write, self.test_evals))
        self.train_evals = list(filter(lambda iter: iter[0] < self.last_write, self.train_evals))
