from rrbc import RRBC
from time import time
from itertools import product
import yaml
import csv
import os


def make_environment(robot_num, sampling_m, edge_distance):
    with open("environment.yaml", 'r') as f:
        environment = yaml.load(f, Loader=yaml.FullLoader)
    environment["robot_num"] = robot_num
    environment["sampling_m"] = sampling_m
    environment["edge_distance"] = edge_distance

    new_environment_file = f"test_data/environment_{robot_num}_{sampling_m}_{int(edge_distance * 10)}.yaml"
    with open(new_environment_file, 'w') as f:
        yaml.dump(environment, f, default_flow_style=False)

    return new_environment_file


if __name__ == '__main__':
    robot_nums = [2, 4, 6]
    sampling_ms = [4, 8, 12]
    edge_distances = [1, 1.2, 1.4]
    test_count = 5

    if not os.path.exists(f"test_data"):
        os.mkdir(f"test_data")

    for robot_num, sampling_m, edge_distance in list(product(*[robot_nums, sampling_ms, edge_distances])):
        environment_file = make_environment(robot_num, sampling_m, edge_distance)

        with open(f"test_data/test_result.csv", 'a') as f:
            wr = csv.writer(f)
            wr.writerow([robot_num, sampling_m, edge_distance])

        for _ in range(test_count):
            start_time = time()

            rrbc = RRBC(environment_file)
            last_state = rrbc.search()

            end_time = time()

            computation_time = end_time - start_time

            re_state_paths = rrbc.reconstruct_paths(last_state)
            sum_of_cost, max_cost = rrbc.calculate_cost(re_state_paths)

            with open(f"test_data/test_result.csv", 'a') as f:
                wr = csv.writer(f)
                wr.writerow([computation_time, sum_of_cost, max_cost])





