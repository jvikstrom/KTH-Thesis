import os
import typer
import random
import json
import matplotlib.pyplot as plt
import numpy as np

prefix = os.getenv("PREFIX")
if prefix is None:
    prefix = ""

#GROUP_ALIVE_TIME = [60,30,10,10,10,10,120,90]
#0,6,12,18
IPH = 10
#GROUP_ALIVE_TIME = [ITERS_PER_HOUR*3,ITERS_PER_HOUR,ITERS_PER_HOUR,ITERS_PER_HOUR*6]
#0,3,6,9,12,15,18,21
#GROUP_DEAD_TIME = [10,30,150,120,90,40,40,10]
#0,6,12,18
#GROUP_DEAD_TIME = [ITERS_PER_HOUR*1.5,ITERS_PER_HOUR*3.5,ITERS_PER_HOUR*2.5,ITERS_PER_HOUR]
#GROUP_DEAD_TIME = [40,60,50,40]
#ITERATIONS_PER_GROUP = ITERS_PER_HOUR*5
IPG = IPH * 6
#GROUP_ALIVE_TIME = [2*IPH,0.5*IPH,0.8*IPH,4*IPH]
#GROUP_DEAD_TIME = [5*IPH,18*IPH,15*IPH,5*IPH]

GROUP_ALIVE_TIME = [3*IPH,0.1*IPH,0.3*IPH,6*IPH]
GROUP_DEAD_TIME = [10*IPH,18*IPH,5*IPH,15*IPH]
#GROUP_ALIVE_TIME = [IPH]
#GROUP_DEAD_TIME = [5*IPH]

ALIVE_PERC = 0.2
#ALIVE_TIME = 10
#DEAD_TIME = 40


def main(n: int, iterations: int):
    cut_away = 1000

    came_alive_at = [0 for _ in range(n)]
    segments = [[] for _ in range(n)]
    alive = np.random.uniform(size=n) < ALIVE_PERC
    ALIVE_TIME = GROUP_ALIVE_TIME[(iterations // 1) % len(GROUP_ALIVE_TIME)]
    DEAD_TIME = GROUP_DEAD_TIME[(iterations // 1) % len(GROUP_DEAD_TIME)]
    alive_iter = np.random.exponential(DEAD_TIME, size=n)
    die_iter = np.random.poisson(ALIVE_TIME, size=n)
    #print(alive)
    #print(die_iter)
    n_alive_at_iter = []
    for t in range(iterations+cut_away):
        ALIVE_TIME = GROUP_ALIVE_TIME[(iterations // 1+t) % len(GROUP_ALIVE_TIME)]
        DEAD_TIME = GROUP_DEAD_TIME[(iterations // 1+t) % len(GROUP_DEAD_TIME)]

        n_alive_at_iter.append(np.sum(alive) / n)
        for i in range(n):
            if alive[i] and t >= die_iter[i]:
                alive[i] = not alive[i]
                alive_iter[i] = np.random.poisson(DEAD_TIME) + t
                segments[i].append((came_alive_at[i], t))
            elif not alive[i] and t >= alive_iter[i]:
                alive[i] = not alive[i]
                die_iter[i] = np.random.exponential(ALIVE_TIME) + t
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

    for i in range(n):
        if n_alive_at_iter[i] == 0:
            print(f"None alive at {i}")

    # Show number distribution of alive times.
    segs_len = []
    for i in range(n):
        segs_len += [end-start for start, end in segments[i]]
    plt.hist(segs_len)
    plt.show()

    plt.hist([len(seg) for seg in segments])
    plt.show()

    plt.plot(n_alive_at_iter)
    plt.show()
    file_name = prefix + str(n) + "-" + str(iterations) + ".json"
#    with open(file_name, "w+") as f:
#        f.write(json.dumps({"segments": segments}))
    print(f'Saved churn file at path: {file_name}')



if __name__ == '__main__':
    typer.run(main)
