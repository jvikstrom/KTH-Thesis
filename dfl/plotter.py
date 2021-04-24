import os
from storage import read
import matplotlib.pyplot as plt
import seaborn as sb
import numpy as np
import argparse
import re
# KDEplot: https://seaborn.pydata.org/generated/seaborn.kdeplot.html

parser = argparse.ArgumentParser(description="Plot the data")
parser.add_argument("--train", dest="plot_train", action="store_const", const=True, default=False, help="Plot the training accuracy")
parser.add_argument("--models", dest="models", action="store_const", const=True, default=False, help="Plot the model losses")
parser.add_argument("--iter", dest="iter", help="What iter to print model accuracies at")
parser.add_argument("--max-iter", dest="max_iter")
parser.add_argument("--title", dest="title", default="")
parser.add_argument("--filter", dest="filter", default=None)
args = parser.parse_args()

sb.set_style('whitegrid')

directory = os.getenv("DATA_DIR")

if directory is None:
    raise AssertionError("Must set DATA_DIR environment variable.")

types = ["exchange-gossip", "exchange-cycle", "none-gossip", "agg-hypercube", "agg-fls", "centralized"]

files = os.listdir(directory)

types = list(map(lambda x: x[:-4], files))

print('read: ', [name for name in filter(lambda x: "-train" not in x and "-models" not in x, types)])


def passes_filter(filter: str, name: str) -> bool:
    if len(filter) == 0:
        return True

    if filter[len(filter)-1] == '$':
        # Match needs to be at the end.
        if len(name) < len(filter[:-1]):
            return False
        start = len(name) - len(filter) + 1
        print(f'name: {name[start:]}, filter {filter[:-1]}')
        return name[start:] == filter[:-1]


def plot_stdouts():
    base = "data/stdouts"
    files = ["agg-hypercube.stdout", "exchange-cycle.stdout", "exchange-no-fail.stdout", "exchange.stdout", "none-gossip.stdout"]
    plt.figure(figsize=(10, 10))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    for file in files:
        path = f"{base}/{file}"
        with open(path, "r") as f:
            xs = []
            ys = []
            lines = f.readlines()
            for line in lines:
                m = re.search(r"TEST (\d+) .+ accuracy: (\d+\.\d+)", line)
                if m is None:
                    continue
                iter = m.group(1)
                accuracy = m.group(2)
                xs.append(int(iter))
                ys.append(float(accuracy))
#                print(f"Accuracy: {accuracy}")
            sb.lineplot(y=ys, x=xs, label=file)



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
        if args.filter is not None:
            if not passes_filter(args.filter, name):
                print(f"{name} does not pass filter {args.filter}, skipping...")
                continue
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
        if args.filter is not None:
            if not passes_filter(args.filter, name):
                print(f"{name} does not pass filter {args.filter}, skipping...")
                continue
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

