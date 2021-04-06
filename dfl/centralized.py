from train import Trainer


class Centralized(Trainer):
    def __init__(self, clients, cfg):
        Trainer.__init__(self, clients, cfg)
        self.model = self.clients[0].model

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i, model=self.model)
            self.model.fit(*self.train_concated, verbose=0, batch_size=32)
        Trainer.step_and_eval(self, self.trainer_config.iterations, model=self.model)
