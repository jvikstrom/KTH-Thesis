import numpy as np
from typing import List
from tqdm import tqdm
from client import Client, Guider
from train import Trainer


class Braintorrent(Trainer):
    def __init__(self, clients: List[Client], cfg):
        Trainer.__init__(self, clients, cfg)
        self.versions = [0 for _ in self.clients]

    def run(self):
        for i in range(self.trainer_config.iterations):
            for client_idx in range(len(self.clients)):
#                client_idx = self.guider.next(client_idx)
                v_old = self.versions[client_idx]
                v_new = self.versions
                for j in range(len(self.clients)):
                    # Receive updated Wj from clients[j]
                    pass

