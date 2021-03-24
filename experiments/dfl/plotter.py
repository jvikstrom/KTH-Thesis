import os
from storage import read
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np

sb.set_style('whitegrid')

directory = os.getenv("DATA_DIR")

if directory is None:
    raise AssertionError("Must set DATA_DIR environment variable.")

types = ["exchange-gossip", "exchange-cycle", "none-gossip", "agg-hypercube", "agg-fls", "centralized"]


if __name__ == "__main__":
    dfs = [read(directory, name + ".csv") for name in types]

    accuracies = {}
    losses = {}
    for name, df in zip(types, dfs):
        grouped = df.groupby(["current_iteration"])
        loss_means = grouped["loss"].mean()
        accuracy_means = grouped["accuracy"].mean()
        accuracies[name] = (accuracy_means.to_numpy(), grouped['accuracy'].std().to_numpy())
        losses[name] = loss_means.to_numpy()

    plt.figure(figsize=(10, 10))
    for name, (accuracy, stds) in accuracies.items():
        x = np.arange(accuracy.size)
        sb.lineplot(y=accuracy, x=x, label=name)
        plt.fill_between(x, accuracy-stds, accuracy+stds, alpha=0.3)

    plt.legend()
    plt.savefig("plot.png", bbox_inches='tight')
    plt.show()

