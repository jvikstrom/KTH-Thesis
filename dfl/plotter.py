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
parser.add_argument('--no-ignored', dest='no_ignore', action='store_const', const=True, default=False, help='If we should ignore worse variants')
parser.add_argument("--plot-max-min", dest="plot_max_min", action="store_const", const=True, default=False, help="Plot the max and min model losses")
parser.add_argument("--loss-regex", dest="loss_regex", default='loss', help="Regex to use when looking for loss")
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


color_map = {
    'fls': 'red',
    'centralized': 'pink',
    'centralized-yogi': 'purple',
    'none-gossip': 'orange',
    'average-gossip': '#3b67c4', # Lightblue/blue color
    'exchange-gossip': 'gray',
    'exchange-gossip-yogi': 'brown',
    'exchange-gossip-adam': 'black',
    'exchange-cycle': 'green',
    'exchange-cycle-yogi': '#1FAA06', # Lime
    'exchange-cycle-adam': 'darkgreen',
}


def get_color(l):
    if l in color_map:
        return color_map[l]
    if l.replace('-train', '') in color_map:
        return color_map[l.replace('-train', '')]
    return None
#    raise AssertionError(f"Color does not exist for label {l}")


label_transform = {
    'exchange-gossip': 'exchange',
    'exchange-gossip-adam': 'exchange-adam',
    'exchange-gossip-yogi': 'exchange-yogi',
}


included_methods = [
    'centralized-yogi',
    'exchange-yogi',
    'exchange-cycle-yogi',
    'none-gossip',
    'average-gossip',
    'fls',
]

def should_include(l: str):
    if args.no_ignore:
        return True
    if len(included_methods) == 0:
        return True
    if transform_label(l) in included_methods:
        return True
    return False


def transform_label(l):
    if l in label_transform:
        return label_transform[l]
    return l


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

    return name.find(filter) != -1


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

    plt.figure(figsize=(7, 7))
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(args.title)

    if args.max_iter is not None:
        max_iter = int(args.max_iter)
    end_accuracies = {}
    for name, (i, accuracy, stds) in accuracies.items():
        if len(list(filter(lambda x: x > 0.2, accuracy))) == 0:
            continue
        if not should_include(name):
            continue
        x = i
        sb.lineplot(x=x, y=accuracy, label=transform_label(name), color=get_color(name))#, color=get_color(name))

#        sb.lineplot(y=accuracy, x=x, label=transform_label(name), color=get_color(name))
        #plt.fill_between(x, accuracy-stds, accuracy+stds, alpha=0.3, color=get_color(name))
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

def model_name_cut(name: str):
    return name[:-len("-0-models")]


def plot_models():
    names = [name for name in filter(lambda x: "-models" in x, types)]
    dfs = [read(directory, name + ".csv") for name in names]
    plt.figure(figsize=(10, 10))

    plt.ylabel("Loss")
    plt.xlabel("Epoch")

    datas = []
    for name, df in zip(names, dfs):
        if args.filter is not None:
            if not passes_filter(args.filter, name):
                print(f"{name} does not pass filter {args.filter}, skipping...")
                continue
        acs = df.filter(regex=args.loss_regex).to_numpy()
        means = np.array([np.mean(ac) for ac in acs])
        stds = np.array([np.std(ac) for ac in acs])
        mins = np.array([np.min(ac) for ac in acs])
        maxs = np.array([np.max(ac) for ac in acs])

        iters = df.current_iteration.to_numpy()
        datas.append((name, iters, means, stds, mins, maxs))
        print(acs)
        print(iters)
        print(len(acs), len(iters))
        print(f"means: {means}\nstds: {stds}")

    grouped_data = {}
    for (name, iters, means, stds, mins, maxs) in datas:
        if name not in grouped_data:
            grouped_data[model_name_cut(name)] = (model_name_cut(name), iters, means, stds, mins, maxs, 1)
        else:
            o_name, o_iters, o_means, o_stds, o_mins, o_maxs, count = grouped_data[model_name_cut(name)]
            if len(o_iters) > len(iters):
                o_name, name = o_name, name
                o_iters, iters = iters, o_iters
                o_means, means = means, count * o_means
                o_stds, stds = stds, count * o_stds
                o_mins, mins = mins, o_mins
                o_maxs, maxs = maxs, o_maxs
            for i in range(len(o_iters)):
                means[i] = (means[i] + o_means[i]) / (count + 1.0)
                stds[i] = (stds[i] + o_stds[i]) / (count + 1.0)
                mins[i] = min(mins[i], o_mins[i])
                maxs[i] = max(maxs[i], o_maxs[i])

            grouped_data[model_name_cut(name)] = (name, iters, means, stds, mins, maxs, count + 1.0)

    for name, x, y, stds, mins, maxs, count in grouped_data.values():
        sb.lineplot(x=x, y=y, label=transform_label(name))
        #if args.plot_max_min:
        #    sb.lineplot(x=x, y=mins, label=transform_label(name), color=get_color(name))
        #    sb.lineplot(x=x, y=maxs, label=transform_label(name), color=get_color(name))
        #plt.fill_between(x, y-stds, y+stds, alpha=0.3)
        plt.fill_between(x, mins, maxs, alpha=0.3)#, color=get_color(name))


if __name__ == "__main__":
    if not args.models:
        plot_accuracy()
    else:
        plot_models()
    plt.legend()
    plt.savefig("plot.pdf", bbox_inches='tight')
    plt.show()

