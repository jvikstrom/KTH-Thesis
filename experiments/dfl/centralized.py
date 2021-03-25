from train import Trainer


class Centralized(Trainer):
    def __init__(self, clients, cfg):
        Trainer.__init__(self, clients, cfg)
        self.model = self.clients[0].model

    def run(self):
        for i in range(self.trainer_config.iterations):
            self.model.fit(*self.train_concated, verbose=0, batch_size=32)
            loss, accuracy = self.model.evaluate(*self.test_concated, verbose=0, batch_size=32)
            print(f"CENTRALIZED ::: ({i}) loss: {loss} ---- accuracy: {accuracy}")
            self.test_evals.append((i, loss, accuracy))
            Trainer.step(self)
