import os
import typer
import tensorflow as tf
from typing import Optional
from configs import \
    Config,\
    none_gossip_config,\
    exchange_cycle_config,\
    exchange_config,\
    aggregate_hypercube_config,\
    fls_config,\
    centralized_config,\
    exchange_cycle_adam_config, \
    exchange_cycle_yogi_config, \
    centralized_yogi_config,\
    average_gossip_config,\
    exchange_adam_config,\
    exchange_yogi_config
from emnist import run as run_emnist
from shakespeare import run as run_shakespeare
from cifar import run as run_cifar
from nn5 import run as run_nn5
import json

data_dir = os.getenv("DATA_DIR")
if data_dir is None:
    raise AssertionError("Please set DATA_DIR env variable")

for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)

alive_segment_path = os.getenv("ALIVE_SEG_PATH")

disable_tqdm = os.getenv("DISABLE_TQDM")
app = typer.Typer()


def load_config(strategy: str, n: int, data_dir: str, learning_rate: float, batches: int, iterations: int):
    cfgs = {
        "none-gossip": none_gossip_config(n, data_dir, learning_rate, batches, iterations),
        "average-gossip": average_gossip_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-cycle": exchange_cycle_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-cycle-adam": exchange_cycle_adam_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-cycle-yogi": exchange_cycle_yogi_config(n, data_dir, learning_rate, batches, iterations),
        "exchange": exchange_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-adam": exchange_adam_config(n, data_dir, learning_rate, batches, iterations),
        "exchange-yogi": exchange_yogi_config(n, data_dir, learning_rate, batches, iterations),
        "agg-hypercube": aggregate_hypercube_config(n, data_dir, learning_rate, batches, iterations),
        "fls": fls_config(n, data_dir, learning_rate, batches, iterations),
        "centralized": centralized_config(n, data_dir, learning_rate, batches, iterations),
        "centralized-yogi": centralized_yogi_config(n, data_dir, learning_rate, batches, iterations),
    }
    if strategy not in cfgs:
        raise AssertionError(f"'{strategy}' is not a valid strategy, must be one of: {cfgs.keys()}" )

    return cfgs[strategy]


@app.command()
def emnist(strategy: str, n: int, runs: int, batches: Optional[int] = typer.Argument(1),
           iterations: Optional[int] = typer.Argument(100), learning_rate: Optional[float] = typer.Argument(0.001)):
    cfg = load_config(strategy, n, data_dir, learning_rate, batches, iterations)
    cfg.disable_tqdm = disable_tqdm is not None
    if alive_segment_path is not None:
        print("Loading churn schedule...")
        with open(alive_segment_path, 'r') as f:
            alive_segments = json.loads(f.read())['segments']
    else:
        alive_segments = [[(0, iterations)] for _ in range(n)]
    for i in range(runs):
        run_emnist(cfg, i, alive_segments)


@app.command()
def nn5(strategy: str, n: int, runs: int, batches: Optional[int] = typer.Argument(1),
           iterations: Optional[int] = typer.Argument(100), learning_rate: Optional[float] = typer.Argument(0.001)):
    nn5_path = os.getenv("NN5_PATH")
    if nn5_path is None:
        raise AssertionError("Must specify the 'NN5_PATH' env variable to the nn5 data csv.")
    cfg = load_config(strategy, n, data_dir, learning_rate, batches, iterations)
    cfg.disable_tqdm = disable_tqdm is not None
    for i in range(runs):
        run_nn5(nn5_path, cfg, i)


@app.command()
def cifar(strategy: str, n: int, runs: int, batches: Optional[int] = typer.Argument(1),
           iterations: Optional[int] = typer.Argument(100), learning_rate: Optional[float] = typer.Argument(0.001)):
    cfg = load_config(strategy, n, data_dir, learning_rate, batches, iterations)
    for i in range(runs):
        run_cifar(cfg, i)


@app.command()
def shakespeare(strategy: str, n: int, runs: int, batches: Optional[int] = typer.Argument(1),
           iterations: Optional[int] = typer.Argument(100), learning_rate: Optional[float] = typer.Argument(0.001)):
    cfg = load_config(strategy, n, data_dir, learning_rate, batches, iterations)
    for i in range(runs):
        run_shakespeare(cfg, i)


@app.command()
def cifar100():
    pass


if __name__ == "__main__":
    app()
