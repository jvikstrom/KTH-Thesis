import os
import typer
import random
import json
import matplotlib.pyplot as plt
import numpy as np

prefix = os.getenv("PREFIX")
if prefix is None:
    prefix = ""

ALIVE_PERC = 0.2
ALIVE_TIME = 10
DEAD_TIME = 40


def main(n: int, iterations: int):
    cut_away = 500

    came_alive_at = [0 for _ in range(n)]
    segments = [[] for _ in range(n)]
    alive = np.random.uniform(size=n) < ALIVE_PERC
    alive_iter = np.random.poisson(DEAD_TIME, size=n)
    die_iter = np.random.poisson(ALIVE_TIME, size=n)
    #print(alive)
    #print(die_iter)
    n_alive_at_iter = []
    for t in range(iterations+cut_away):
        n_alive_at_iter.append(np.sum(alive) / n)
        for i in range(n):
            if alive[i] and t >= die_iter[i]:
                alive[i] = not alive[i]
                alive_iter[i] = np.random.poisson(DEAD_TIME) + t
                segments[i].append((came_alive_at[i], t))
            elif not alive[i] and t >= alive_iter[i]:
                alive[i] = not alive[i]
                die_iter[i] = np.random.poisson(ALIVE_TIME) + t
                came_alive_at[i] = t
    # Close any open segments
    for i in range(n):
        if alive[i]:
            segments[i].append((came_alive_at[i], iterations+cut_away))

    # Cut away first 200 iterations.
    n_alive_at_iter = n_alive_at_iter[cut_away:]
    for i in range(n):
        new_segs = []
        for start, end in segments[i]:
            if start > cut_away:
                # Safe to just append.
                new_segs.append((start - cut_away, end - cut_away))
            elif end > cut_away:
                new_segs.append((0, end - cut_away))
        segments[i] = new_segs

    # Show number distribution of alive times.
    segs_len = []
    for i in range(n):
        segs_len += [end-start for start, end in segments[i]]
    plt.hist(segs_len)
#    plt.show()

    plt.plot(n_alive_at_iter)
#    plt.show()
    file_name = prefix + str(n) + "-" + str(iterations) + ".json"
    with open(file_name, "w+") as f:
        f.write(json.dumps({"segments": segments}))
    print(f'Saved churn file at path: {file_name}')



if __name__ == '__main__':
    typer.run(main)
