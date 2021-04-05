import os
from storage import read
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Plot the data")
parser.add_argument("--train", dest="train", action="store_const", const=True, default=False, help="Plot the training accuracy")
parser.add_argument("--models", dest="models", action="store_const", const=True, default=False, help="Plot the model losses")
parser.add_argument("--max-iter", dest="max_iter")
parser.add_argument("--title", dest="title", default="")

args = parser.parse_args()

sb.set_style('whitegrid')

directory = os.getenv("DATA_DIR")

if directory is None:
    raise AssertionError("Must set DATA_DIR environment variable.")

types = ["exchange-gossip", "exchange-cycle", "none-gossip", "agg-hypercube", "agg-fls", "centralized"]

files = os.listdir(directory)

types = list(map(lambda x: x[:-4], files))

def plot_test():
    dfs = [read(directory, name + ".csv") for name in filter(lambda x: "-train" not in x and "-models" not in x, types)]

    accuracies = {}
    losses = {}
    for name, df in zip(types, dfs):
        grouped = df.groupby(["current_iteration"])
        if args.max_iter is not None:
            grouped = df[df.current_iteration < int(args.max_iter)]
            grouped = grouped.groupby(["current_iteration"])
        loss_means = grouped["loss"].mean()
        accuracy_means = grouped["accuracy"].mean()
        accuracies[name] = (grouped["current_iteration"].mean().to_numpy(), accuracy_means.to_numpy(), grouped['accuracy'].std().to_numpy())
        losses[name] = loss_means.to_numpy()

    plt.figure(figsize=(10, 10))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(args.title)
    for name, (i, accuracy, stds) in accuracies.items():
        #x = np.arange(accuracy.size)
        x = i
        sb.lineplot(y=accuracy, x=x, label=name)
        plt.fill_between(x, accuracy-stds, accuracy+stds, alpha=0.3)
# Plot contains the mean accuracy of the node's models. Each model

if __name__ == "__main__":
    plot_test()
    plt.legend()
    plt.savefig("plot.png", bbox_inches='tight')
    plt.show()

