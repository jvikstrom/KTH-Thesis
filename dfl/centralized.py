from train import Trainer


class Centralized(Trainer):
    def __init__(self, trainer_input, clients, cfg, all_train, all_test):
        Trainer.__init__(self, trainer_input, clients, cfg, all_train, all_test)
        self.model = self.clients[0].model

    def run(self):
        for i in range(self.trainer_config.iterations):
            Trainer.step_and_eval(self, i, model=self.model)
            #for client in self.clients:
                #client.model = self.model
                #client.train(0, 32)
            self.model.fit(*self.train_concated, verbose=0, batch_size=32)
        Trainer.step_and_eval(self, self.trainer_config.iterations, model=self.model)
