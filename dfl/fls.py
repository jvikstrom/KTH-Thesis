from train import Trainer, TrainerConfig
from pydantic import BaseModel


class FLSConfig(BaseModel):
    trainer_config: TrainerConfig
    # TODO: Might want to include a server learning rate here.


class FLS(Trainer):
    def __init__(self, clients, cfg: FLSConfig):
        Trainer.__init__(self, clients, cfg.trainer_config)
        for client in self.clients:
            client.model.set_weights([weight.copy() for weight in self.clients[0].model.get_weights()])

    def run(self):
        loss, accuracy = self.clients[0].model.evaluate(*self.test_concated, verbose=0, batch_size=32)
        self.test_evals.append((-1, loss, accuracy))
        print(f"FLS {-1} ::: loss: {loss}   ----   accuracy: {accuracy}")

        for i in range(self.trainer_config.iterations):
            # Does server SGD aggregation with server learning rate = 1.0
            for client in self.clients:
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
            loss, accuracy = self.clients[0].model.evaluate(*self.test_concated, verbose=0, batch_size=32)
            self.test_evals.append((i, loss, accuracy))
            print(f"FLS {i} ::: loss: {loss}   ----   accuracy: {accuracy}")
            Trainer.step(self)
