from train import Trainer, TrainerConfig
from pydantic import BaseModel
from tqdm import tqdm


class FLSConfig(BaseModel):
    trainer_config: TrainerConfig
    # TODO: Might want to include a server learning rate here.


class FLS(Trainer):
    def __init__(self, trainer_input, clients, cfg: FLSConfig, all_train, all_test):
        Trainer.__init__(self, trainer_input, clients, cfg.trainer_config, all_train, all_test)
        self.model = self.clients[0].model
        for client in self.clients:
            client.model.set_weights([weight.copy() for weight in self.model.get_weights()])

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i, model=self.model)
            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in self.model.get_weights()])
            # Does server SGD aggregation with server learning rate = 1.0
            for client in tqdm(self.clients):
                client.train(self.trainer_config.batches)

            all_weights = [client.model.get_weights() for client in self.clients]

            aggregated_weights = [weight.copy() for weight in all_weights[0]]
            for j in range(1, len(all_weights)):
                for k in range(len(aggregated_weights)):
                    aggregated_weights[k] += all_weights[j][k]
                    # aggregated_weights[k] += aggregated_weights[j][k] - old_model[k]
            for k in range(len(aggregated_weights)):
                aggregated_weights[k] = aggregated_weights[k] / len(self.clients)
                # aggregated_weights[k] = old_model[k] - server_learning_rate * aggregated_weights[k] / len(self.clients)

            for client in self.clients:
                client.model.set_weights([weight.copy() for weight in aggregated_weights])
            self.model.set_weights([weight.copy() for weight in aggregated_weights])
# TODO: Move everything to step_and_eval
        Trainer.step_and_eval(self, self.trainer_config.iterations, model=self.clients[0].model)
