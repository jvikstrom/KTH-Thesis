import os
import typer
import random
import json
#import matplotlib.pyplot as plt

prefix = os.getenv("PREFIX")
if prefix is None:
    prefix = ""


def main(n: int, max_n: int, iterations: int, fail: float):
    print(f"Generating schedule for {n} nodes over {iterations} iterations with failure chance of {fail}")
    current_alive = set(list(range(n)))
    current_ids = list(range(n))
    available = set(list(range(n, max_n)))
    fail_per_iter = []
    alive_per_iter = []
    joining_by = []  # The peer that we dial to join the learning place.
    for i in range(iterations):
        #print(f"-----------------\nITER {i}\n-----------------")
        fail_now = []
        new_available = []

        # Add which node will churn.
        for j in range(len(current_ids)):
            f = random.uniform(0, 1)
            if f < fail:
                fail_now.append(j)
                new_available.append(current_ids[j])
        # Remove the node from currently alive nodes.
        for f in fail_now:
            #print(f"fail: {f} i.e. : {current_ids[f]}, ids ( {current_ids} ), actually alive: {current_alive}")
            current_alive.remove(current_ids[f])
        fail_per_iter.append(fail_now)
        alive_now = []
        join_by = []
        # Add which node the joinee will join by.
        for k in range(len(fail_now)):
            # Joins are done by index.
            c = random.choice(list(current_alive))
            #print(f"ids: {current_ids}, chose: {c}")
            join_by.append(current_ids.index(c))
        joining_by.append(join_by)

        # Add what new node is joining.
        for k in fail_now:
            if len(available) == 0:
                available = available.union(new_available)
                new_available = set([])
            j = random.choice(list(available))
            available.remove(j)
            current_alive.add(j)
            alive_now.append(j)
            #print(f"Failed at {k} replace by {j}")
            current_ids[k] = j
        available = available.union(new_available)
        alive_per_iter.append(alive_now)

    for i in range(iterations):
        assert len(fail_per_iter[i]) == len(alive_per_iter[i])

    max_fail_n = 0.0
    for i in range(iterations):
        max_fail_n = max(max_fail_n, len(fail_per_iter[i]))
    print(f"Max fail per iter percentage is: {max_fail_n / n}, max number of failures per iteration is {max_fail_n}")
    # TODO: Write the schedule.
    file_name = prefix + str(n) + "-" + str(iterations) + "-" + str(fail) + ".json"
    with open(file_name, "w+") as f:
        f.write(json.dumps({"fails": fail_per_iter, "alive": alive_per_iter, "join": joining_by}))

    f_x = []
    for f in fail_per_iter:
        f_x.append(len(f))
#    plt.plot(f_x)


#    plt.show()
if __name__ == '__main__':
    typer.run(main)
