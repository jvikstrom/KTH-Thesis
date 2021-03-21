from train import Trainer


class Centralized(Trainer):
    def __init__(self, clients, unused):
        Trainer.__init__(self, clients)
        self.model = self.clients[0].model

    def run(self, batches: int = 1, iterations: int = 100):
        for i in range(iterations):
            self.model.fit(*self.train_concated, verbose=0, batch_size=32)
            loss, accuracy = self.model.evaluate(*self.test_concated, verbose=0, batch_size=32)
            print(f"CENTRALIZED ::: ({i}) loss: {loss} ---- accuracy: {accuracy}")
            self.test_evals.append((loss, accuracy))
            Trainer.step(self)
