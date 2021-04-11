import os
from storage import read
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import argparse
# KDEplot: https://seaborn.pydata.org/generated/seaborn.kdeplot.html

parser = argparse.ArgumentParser(description="Plot the data")
parser.add_argument("--train", dest="plot_train", action="store_const", const=True, default=False, help="Plot the training accuracy")
parser.add_argument("--models", dest="models", action="store_const", const=True, default=False, help="Plot the model losses")
parser.add_argument("--iter", dest="iter", help="What iter to print model accuracies at")
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

print('read: ', [name for name in filter(lambda x: "-train" not in x and "-models" not in x, types)])


def plot_accuracy():
    if not args.plot_train:
        names = [name for name in filter(lambda x: "-train" not in x and "-models" not in x, types)]
    else:
        names = [name for name in filter(lambda x: "-train" in x and "-models" not in x, types)]
    dfs = [read(directory, name + ".csv") for name in names]

    max_iter = 0
    accuracies = {}
    losses = {}
    for name, df in zip(names, dfs):
        grouped = df.groupby(["current_iteration"])
        max_iter = max(df.current_iteration)
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

    if args.max_iter is not None:
        max_iter = int(args.max_iter)
    end_accuracies = {}
    for name, (i, accuracy, stds) in accuracies.items():
        #x = np.arange(accuracy.size)
        x = i
        sb.lineplot(y=accuracy, x=x, label=name)
        plt.fill_between(x, accuracy-stds, accuracy+stds, alpha=0.3)
        accum = 0.0
        t = 0
        for j, ac in zip(i, accuracy):
            if max_iter < j + 100:
                accum += ac
                t += 1
        if t == 0:
            print(f"{name} has no entries!")
            continue
        end_accuracies[name] = accum / t
    print(end_accuracies)
# Plot contains the mean accuracy of the node's models. Each model


def plot_models():
    at_iter = args.iter
    names = [name for name in filter(lambda x: "-models" in x, types)]
    dfs = [read(directory, name + ".csv") for name in names]
    accuracies = {}
    xs = {}
    plt.ylabel("N models")
    plt.xlabel("Accuracy")

    for name, df in zip(names, dfs):
        acs = df[df.current_iteration == int(at_iter)].filter(regex='loss').to_numpy()[0]
        accuracies[name] = acs
        xs[name] = list(range(len(acs)))

    for key in xs.keys():
        y = accuracies[key]
        plt.hist(y)


if __name__ == "__main__":
    if not args.models:
        plot_accuracy()
    else:
        plot_models()
    plt.legend()
    plt.savefig("plot.png", bbox_inches='tight')
    plt.show()

