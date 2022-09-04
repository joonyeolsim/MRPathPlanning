import copy
import heapq
import math
import random
import time
import yaml

import matplotlib.pyplot as plt

from a_star import AStar
from rrt_modify import RRT


class Environment:
    def __init__(self):
        with open("environment.yaml") as f:
            environment = yaml.load(f, Loader=yaml.FullLoader)
        self.robot_num = environment["robot_num"]
        self.robot_radius = environment["robot_radius"]

        self.map_width = environment["map_width"]
        self.map_height = environment["map_height"]
        self.obstacle_list = environment["obstacle_list"]

        self.rrt_rand_area = environment["rrt_rand_area"]
        self.rrt_expand_distance = environment["rrt_expand_distance"]
        self.rrt_path_resolution = environment["rrt_path_resolution"]
        self.rrt_goal_sample_rate = environment["rrt_goal_sample_rate"]
        self.rrt_max_iter = environment["rrt_max_iter"]
        self.rrt_play_area = environment["rrt_play_area"]

        self.sampling_m = environment["sampling_m"]
        self.expand_distance = environment["expand_distance"]
        self.threshold_distance = environment["threshold_distance"]

        self.colors = environment["colors"]


class SSSP:
    def __init__(self):
        self.env = Environment()
        self.a_star = AStar()

        self.roadmaps = []
        self.start_node = []
        self.goal_node = []
        self.paths = [[] for _ in range(self.env.robot_num)]

        self.rrt = RRT(
            obstacle_list=self.env.obstacle_list,
            rand_area=self.env.rrt_rand_area,
            expand_dis=self.env.rrt_expand_distance,
            path_resolution=self.env.rrt_path_resolution,
            goal_sample_rate=self.env.rrt_goal_sample_rate,
            max_iter=self.env.rrt_max_iter,
            play_area=self.env.rrt_play_area,
            robot_radius=self.env.robot_radius,
        )

        self.sample_time = 0
        self.search_time = 0

    def collide(self, cur_q):
        for robot1 in cur_q:
            for robot2 in cur_q:
                if robot1 == robot2:
                    continue

                distance = math.sqrt((robot1.x - robot2.x) ** 2 + (robot1.y - robot2.y) ** 2)
                if 0 <= distance < self.env.robot_radius * 2:
                    return True
        return False

    def calculate_score(self, q):  # bfs -> A* 알고리즘.
        cost_sum = 0
        for j, q_state in enumerate(q):
            cost_sum += self.a_star.search(self.roadmaps[j], q_state, self.goal_node[j])
        return cost_sum

    @staticmethod
    def edge_connect(roadmap, q_state_new, max_weight):
        # 자기 자신으로 가는 엣지 추가
        roadmap[q_state_new].append((0, q_state_new))

        # 로드맵을 확장하는 기준이 계속 바뀔 수 있다. => 일관적으로 유지해보자 (입실론을 기준으로)
        for node in roadmap:
            dx = abs(node.x - q_state_new.x)
            dy = abs(node.y - q_state_new.y)
            weight = math.sqrt(dx ** 2 + dy ** 2)
            if weight <= max_weight:
                roadmap[q_state_new].append((weight, node))
                roadmap[node].append((weight, q_state_new))

    def get_min_q_distance(self, q_state_new, roadmap):
        min_distance = math.inf
        for q_state in roadmap:
            x_diff = abs(q_state.x - q_state_new.x)
            y_diff = abs(q_state.y - q_state_new.y)
            distance = math.sqrt(x_diff ** 2 + y_diff ** 2)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    def search(self):
        start_list = sssp.get_random_pos_list(self.env.robot_num, self.env.map_width, self.env.map_height)
        goal_list = sssp.get_random_pos_list(self.env.robot_num, self.env.map_width, self.env.map_height)
        for start, goal in zip(start_list, goal_list):
            self.rrt.set_position(start, goal)
            roadmap = self.rrt.planning()
            self.start_node.append(list(roadmap)[0])
            self.goal_node.append(list(roadmap)[-1])
            self.roadmaps.append(roadmap)

        while True:
            frontier = list()
            q_init = [start_q for start_q in self.start_node]
            parent = None
            # score와 로봇들의 현재 위치, next, parent
            heapq.heappush(frontier, (0, q_init, parent))  # score, Q, next, parent
            explored = list()
            explored.append(q_init)

            while frontier:
                s = heapq.heappop(frontier)
                # check all q is at goal
                s_q = s[1]
                self.draw_graph(s_q)
                for j, q in enumerate(s_q):
                    if q != self.goal_node[j]:
                        break
                else:
                    return s

                start_time = time.time()
                # 모든 로봇들에 대해서 sampling
                for i in range(self.env.robot_num):
                    q_state_from = s_q[i]

                    # vertex expansion via sampling
                    for _ in range(self.env.sampling_m):
                        # 랜덤 노드 생성 후 steer로 새로운 new node 생성
                        q_state_rand = self.rrt.get_random_node()
                        weight, q_state_new = self.rrt.steer(q_state_from, q_state_rand, self.env.expand_distance)

                        # 가장 가까운 q로부터 threadhold보다 멀리 있다면
                        if self.get_min_q_distance(q_state_new, self.roadmaps[i]) > self.env.threshold_distance:
                            # 장애물에 걸리는지 확인 후 로드맵에 추가 및 edge 연결
                            if self.rrt.check_if_outside_play_area(q_state_new, self.rrt.play_area) and \
                                    self.rrt.check_collision(q_state_new, self.rrt.obstacle_list,
                                                             self.rrt.robot_radius):
                                self.roadmaps[i][q_state_new] = list()
                                self.edge_connect(self.roadmaps[i], q_state_new, self.env.expand_distance)

                end_time = time.time()
                self.sample_time += end_time - start_time

                start_time = time.time()
                # 모든 로봇들의 움직임의 조합
                # search node expansion
                q_prime = s_q[:]

                def combination_samples(robot):
                    for _, q_state_to in self.roadmaps[robot][s_q[robot]]:
                        q_prime[robot] = q_state_to

                        if robot == self.env.robot_num - 1:
                            if q_prime not in explored:
                                copy_q_prime = q_prime[:]

                                if not self.collide(copy_q_prime):
                                    score = self.calculate_score(copy_q_prime)  # a*로 최소 거리 계산
                                    heapq.heappush(frontier, (score, copy_q_prime, s))  # frontier에 push
                                    explored.append(copy_q_prime)  # explored에 추가

                        else:
                            combination_samples(robot + 1)

                combination_samples(0)
                end_time = time.time()
                self.search_time += end_time - start_time

            self.env.threshold_distance *= 0.5

    def get_random_pos_list(self, robot_n, max_x, max_y):
        pos_list = []
        while len(pos_list) < robot_n:
            pos = (random.randint(0, max_x), random.randint(0, max_y))
            if pos not in pos_list:
                for (ox, oy, size) in self.rrt.obstacle_list:
                    dx = ox - pos[0]
                    dy = oy - pos[1]
                    if dx * dx + dy * dy <= (size + self.rrt.robot_radius) ** 2:
                        break
                else:
                    pos_list.append(pos)
        return pos_list

    def draw_graph(self, s_q=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if s_q is not None:
            for q_state in s_q:
                plt.plot(q_state.x, q_state.y, "^k")
                if self.rrt.robot_radius > 0.0:
                    self.rrt.plot_circle(q_state.x, q_state.y, self.rrt.robot_radius, '-r')

        edge_x = []
        edge_y = []
        for i, roadmap in enumerate(self.roadmaps):
            for node in roadmap:
                for weight, next_node in roadmap[node]:
                    if [next_node.x, node.x] not in edge_x or [next_node.y, node.y] not in edge_y:
                        plt.plot([node.x, next_node.x], [node.y, next_node.y], self.env.colors[i])
                        edge_x.append([node.x, next_node.x])
                        edge_y.append([node.y, next_node.y])
            plt.plot(self.start_node[i].x, self.start_node[i].y, marker='x', color=self.env.colors[i])
            plt.plot(self.goal_node[i].x, self.goal_node[i].y, marker='o', color=self.env.colors[i])

        for (ox, oy, size) in self.rrt.obstacle_list:
            self.rrt.plot_circle(ox, oy, size)

        if self.rrt.play_area is not None:
            plt.plot([self.rrt.play_area.xmin, self.rrt.play_area.xmax,
                      self.rrt.play_area.xmax, self.rrt.play_area.xmin,
                      self.rrt.play_area.xmin],
                     [self.rrt.play_area.ymin, self.rrt.play_area.ymin,
                      self.rrt.play_area.ymax, self.rrt.play_area.ymax,
                      self.rrt.play_area.ymin],
                     "-k")

        plt.axis("equal")
        plt.axis([-2, self.env.map_width, -2, self.env.map_height])
        plt.grid(True)
        plt.pause(0.01)

    def draw_result(self, s_):
        next_s = copy.deepcopy(s_)
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        while next_s:
            q = next_s[1]

            q_paths = []
            step = 5
            for j, q_state in enumerate(q):
                d, theta = self.rrt.calc_distance_and_angle(q_state, next_s[2][1][j])

                current_x = q_state.x
                current_y = q_state.y
                q_path = [(current_x, current_y)]

                small_walk = d / step

                for _ in range(step):
                    current_x += math.cos(theta) * small_walk
                    current_y += math.sin(theta) * small_walk
                    q_path.append((current_x, current_y))
                q_paths.append(q_path)

            for k in range(step + 1):
                plt.clf()
                for i, path in enumerate(self.paths):
                    for state in path:
                        plt.plot(state[0], state[1], color=self.env.colors[i])

                for q_path in q_paths:
                    self.rrt.plot_circle(q_path[k][0], q_path[k][1], self.rrt.robot_radius, '-r')

                # for q_state in q:
                #     self.rrt.plot_circle(q_state.x, q_state.y, self.rrt.robot_radius, '-r')

                for i, roadmap in enumerate(self.roadmaps):
                    plt.plot(self.start_node[i].x, self.start_node[i].y, marker='x', color=self.env.colors[i])
                    plt.plot(self.goal_node[i].x, self.goal_node[i].y, marker='o', color=self.env.colors[i])

                for (ox, oy, size) in self.rrt.obstacle_list:
                    self.rrt.plot_circle(ox, oy, size)

                if self.rrt.play_area is not None:
                    plt.plot([self.rrt.play_area.xmin, self.rrt.play_area.xmax,
                              self.rrt.play_area.xmax, self.rrt.play_area.xmin,
                              self.rrt.play_area.xmin],
                             [self.rrt.play_area.ymin, self.rrt.play_area.ymin,
                              self.rrt.play_area.ymax, self.rrt.play_area.ymax,
                              self.rrt.play_area.ymin],
                             "-k")

                plt.axis("equal")
                plt.axis([-2, self.env.map_width, -2, self.env.map_height])
                plt.grid(True)
                plt.pause(0.001)

            next_s = next_s[2]

    def reconstruct_paths(self, s_):
        next_s = copy.deepcopy(s_)
        # path 그리기
        while next_s[2]:
            for i, node in enumerate(next_s[1]):
                self.paths[i].append([[node.x, next_s[2][1][i].x], [node.y, next_s[2][1][i].y]])
            next_s = next_s[2]


if __name__ == '__main__':
    sssp = SSSP()
    start_time = time.time()

    s = sssp.search()

    end_time = time.time()
    print(f"Sample time: {sssp.sample_time}")
    print(f"Search time: {sssp.search_time}")
    print(f"All time: {end_time - start_time}")

    # path 만들기
    sssp.reconstruct_paths(s)

    # 결과 그리기
    sssp.draw_result(s)
