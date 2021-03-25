import os
import typer
from typing import Optional
from configs import Config, none_gossip_config, exchange_cycle_config, exchange_config, aggregate_hypercube_config, fls_config, centralized_config, exchange_cycle_adam_config
from emnist import run as run_emnist

data_dir = os.getenv("DATA_DIR")
if data_dir is None:
    raise AssertionError("Please set DATA_DIR env variable")

app = typer.Typer()


def load_config(strategy: str, n: int, data_dir: str, learning_rate: float, batches: int, iterations: int):
    cfgs = {
        "none-gossip": none_gossip_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-cycle": exchange_cycle_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-cycle-adam": exchange_cycle_adam_config(n, data_dir, learning_rate, batches, iterations),
        "exchange": exchange_config(n, data_dir, learning_rate, batches, iterations),
        "agg-hypercube": aggregate_hypercube_config(n, data_dir, learning_rate, batches, iterations),
        "fls": fls_config(n, data_dir, learning_rate, batches, iterations),
        "centralized": centralized_config(n, data_dir, learning_rate, batches, iterations),
    }
    if strategy not in cfgs:
        raise AssertionError(f"'{strategy}' is not a valid strategy, must be one of: {cfgs.keys()}" )

    return cfgs[strategy]


@app.command()
def emnist(strategy: str, n: int, runs: int, batches: Optional[int] = typer.Argument(1),
           iterations: Optional[int] = typer.Argument(100), learning_rate: Optional[float] = typer.Argument(0.001)):
    cfg = load_config(strategy, n, data_dir, learning_rate, batches, iterations)
    for i in range(runs):
        run_emnist(cfg, i)


@app.command()
def cifar100():
    pass


if __name__ == "__main__":
    app()
