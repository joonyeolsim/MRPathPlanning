import copy
import heapq
import math
import random
import time
from itertools import combinations

import matplotlib.pyplot as plt

from astar import AStar
from environment import Environment
from rrt import RRT
from snapshot import SnapShot


class SSSP:
    def __init__(self):
        self.env = Environment()
        self.a_star = AStar()

        self.roadmaps = []
        self.start_states = []
        self.goal_states = []

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

    @staticmethod
    def ccw(x1, y1, x2, y2, x3, y3):
        return (x1 * y2 + x2 * y3 + x3 * y1) - (x2 * y1 + x3 * y2 + x1 * y3)

    @staticmethod
    def line_segment_intersects(p1, p2, p3, p4):
        # 외적으로 방향을 구함.
        ab = SSSP.ccw(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) * SSSP.ccw(p1.x, p1.y, p2.x, p2.y, p4.x, p4.y)
        cd = SSSP.ccw(p3.x, p3.y, p4.x, p4.y, p1.x, p1.y) * SSSP.ccw(p3.x, p3.y, p4.x, p4.y, p2.x, p2.y)

        return ab < 0 and cd < 0

    def collision_check(self, prev_states, next_states):
        collisions = []

        if prev_states == next_states:
            return collisions

        # 각 로봇들의 이동 경로 구하기
        for prev_state, next_state in zip(prev_states, next_states):
            distance, theta = self.rrt.calc_distance_and_angle(prev_state, next_state)

            prev_state.path_x = [prev_state.x]
            prev_state.path_y = [prev_state.y]

            local_path_steps = math.floor(distance / self.env.step_resolution)
            local_x = prev_state.x
            local_y = prev_state.y

            for _ in range(local_path_steps):
                local_x += self.env.step_resolution * math.cos(theta)
                local_y += self.env.step_resolution * math.sin(theta)
                prev_state.path_x.append(local_x)
                prev_state.path_y.append(local_y)

            prev_state.path_x.append(next_state.x)
            prev_state.path_y.append(next_state.y)

        for robot1, robot2 in combinations(range(0, self.env.robot_num), 2):
            prev_robot_state1 = prev_states[robot1]
            next_robot_state1 = next_states[robot1]

            prev_robot_state2 = prev_states[robot2]
            next_robot_state2 = next_states[robot2]

            prev_robot_state1_len = len(prev_robot_state1.path_x)
            prev_robot_state2_len = len(prev_robot_state2.path_x)
            max_path_len = max(prev_robot_state1_len, prev_robot_state2_len)

            if prev_robot_state1_len < max_path_len:
                prev_robot_state1.path_x.extend(
                    [prev_robot_state1.path_x[-1] for _ in range(max_path_len - prev_robot_state1_len)])
                prev_robot_state1.path_y.extend(
                    [prev_robot_state1.path_y[-1] for _ in range(max_path_len - prev_robot_state1_len)])

            if prev_robot_state2_len < max_path_len:
                prev_robot_state2.path_x.extend(
                    [prev_robot_state2.path_x[-1] for _ in range(max_path_len - prev_robot_state2_len)])
                prev_robot_state2.path_y.extend(
                    [prev_robot_state2.path_y[-1] for _ in range(max_path_len - prev_robot_state2_len)])

            for local_x1, local_y1, local_x2, local_y2 in zip(prev_robot_state1.path_x,
                                                              prev_robot_state1.path_y,
                                                              prev_robot_state2.path_x,
                                                              prev_robot_state2.path_y):
                distance = self.get_euclidean_distance_by_pose(local_x1, local_y1, local_x2, local_y2)
                if distance < self.env.robot_radius * 2:
                    collisions.append(((robot1, next_robot_state1), (robot2, next_robot_state2)))
                    break

        return collisions

    def calculate_score_path(self, q):  # A* 알고리즘.
        cost_sum = 0
        paths = []
        for j, q_state in enumerate(q):
            path, cost = self.a_star.search(self.roadmaps[j], q_state, self.goal_states[j])
            paths.append(path)
            cost_sum += cost
        return paths, cost_sum

    def edge_connect(self, roadmap, new_state, max_distance):
        # 자기 자신으로 가는 엣지 추가
        roadmap[new_state].append((0, new_state))

        # 로드맵을 확장하는 기준이 계속 바뀔 수 있다. => 일관적으로 유지해보자 (입실론을 기준으로)
        for state in roadmap:
            distance = self.get_euclidean_distance(state, new_state)
            if distance <= max_distance * 1.5:
                roadmap[new_state].append((distance, state))
                roadmap[state].append((distance, new_state))

    def get_min_q_distance(self, new_state, roadmap):
        min_distance = math.inf
        for state in roadmap:
            distance = self.get_euclidean_distance(state, new_state)
            if distance < min_distance:
                min_distance = distance
        return min_distance

    @staticmethod
    def get_euclidean_distance(state1, state2):
        return math.sqrt(math.pow(state1.x - state2.x, 2) + math.pow(state1.y - state2.y, 2))

    @staticmethod
    def get_euclidean_distance_by_pose(x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    def search(self):
        # 랜덤으로 시작과 끝 지점들을 생성
        start_positions = sssp.get_random_positions(self.env.robot_num, self.env.map_width, self.env.map_height)
        goal_positions = sssp.get_random_positions(self.env.robot_num, self.env.map_width, self.env.map_height)

        # 각 시작 지점과 끝 지점에 대해서 roadmap을 생성함.
        for start_position, goal_position in zip(start_positions, goal_positions):
            self.rrt.set_position(start_position, goal_position)
            roadmap = self.rrt.planning()
            self.start_states.append(list(roadmap)[0])
            self.goal_states.append(list(roadmap)[-1])
            self.roadmaps.append(roadmap)

        while True:
            frontier = list()
            explored = list()

            # 처음 snapshot을 만들고 frontier에 삽입.
            init_states = [start_state for start_state in self.start_states]
            parent = None
            paths, score = self.calculate_score_path(init_states)
            timestep = 1

            heapq.heappush(frontier, SnapShot(score, init_states, parent, paths, timestep))

            while frontier:
                snapshot = heapq.heappop(frontier)
                explored.append(snapshot.robot_states)

                next_score = snapshot.score
                next_robot_states = snapshot.robot_states[:]
                next_parent = snapshot
                next_paths = snapshot.paths
                next_timestep = snapshot.timestep + 1

                if self.env.draw_graph:
                    self.draw_graph(snapshot.robot_states)

                for robot, robot_state in enumerate(snapshot.robot_states):
                    if robot_state != self.goal_states[robot]:
                        break
                else:
                    return snapshot

                for robot, path in enumerate(next_paths):
                    if len(path) > snapshot.timestep:
                        next_robot_states[robot] = path[snapshot.timestep]
                        minus_score = self.get_euclidean_distance(snapshot.robot_states[robot], next_robot_states[robot])
                        next_score -= minus_score

                collisions_list = self.collision_check(snapshot.robot_states, next_robot_states)
                if not collisions_list:
                    if next_robot_states not in explored:
                        heapq.heappush(frontier, SnapShot(next_score, next_robot_states, next_parent, next_paths, next_timestep))
                    else:
                        print("Explored!")

                else:
                    # 충돌한 로봇들은 각각 Sampling 후 충돌하지 않는 지점을 frontier 큐에 넣음.
                    for collisions in collisions_list:
                        for collision in collisions:
                            # 충돌한 로봇과 충돌한 위치
                            robot, collision_state = collision

                            # 충돌 하기 전 위치
                            if len(snapshot.paths[robot]) > snapshot.timestep - 1:
                                last_state = snapshot.paths[robot][snapshot.timestep - 1]

                                # 충돌 하기 전 위치에서 vertex expansion
                                self.vertex_expand(robot, last_state)

                                next_state_list = []
                                # 가장 a* 방향과 근접하고 충돌하지 않는 지점을 찾음.
                                for _, edge_state in self.roadmaps[robot][last_state]:
                                    distance = self.get_euclidean_distance(edge_state, collision_state)
                                    next_state_list.append((distance, edge_state))
                                next_state_list.sort()

                                # 편의를 위해서 collision check 함수를 썼지만 모든 로봇이 아닌 하나의 로봇만 바꿨기 때문에
                                # 하나의 로봇만 확인할 수 있는 간소화된 함수를 사용하면 더 빨라질 수 있음.
                                copy_next_robot_states = next_robot_states[:]
                                for _, next_state in next_state_list:
                                    copy_next_robot_states[robot] = next_state
                                    if not self.collision_check(snapshot.robot_states, copy_next_robot_states):
                                        if copy_next_robot_states not in explored:
                                            next_paths, next_score = self.calculate_score_path(copy_next_robot_states)
                                            next_timestep = 1
                                            heapq.heappush(frontier, SnapShot(next_score, copy_next_robot_states, next_parent, next_paths, next_timestep))
                                            break
                                        else:
                                            print("Collision Explored!")

            self.env.threshold_distance *= 0.5

    def vertex_expand(self, robot, q_state_from):
        for _ in range(self.env.sampling_m):
            # 랜덤 노드 생성 후 steer로 새로운 new node 생성
            q_state_rand = self.rrt.get_random_node()
            weight, q_state_new = self.rrt.steer(q_state_from, q_state_rand, self.env.expand_distance)

            # 가장 가까운 q로부터 threadhold보다 멀리 있다면
            if self.get_min_q_distance(q_state_new, self.roadmaps[robot]) > self.env.threshold_distance:
                # 장애물에 걸리는지 확인 후 로드맵에 추가 및 edge 연결
                if self.rrt.check_if_outside_play_area(q_state_new, self.rrt.play_area) and \
                        self.rrt.check_collision(q_state_new, self.rrt.obstacle_list, self.rrt.robot_radius):
                    self.roadmaps[robot][q_state_new] = list()
                    self.edge_connect(self.roadmaps[robot], q_state_new, self.env.expand_distance)

    def get_random_positions(self, robot_n, max_x, max_y):
        positions = []
        while len(positions) < robot_n:
            pos = (random.randint(0, max_x), random.randint(0, max_y))
            if pos not in positions:
                for (ox, oy, size) in self.rrt.obstacle_list:
                    dx = ox - pos[0]
                    dy = oy - pos[1]
                    if dx * dx + dy * dy <= (size + self.rrt.robot_radius) ** 2:
                        break
                else:
                    for other_pos in positions:
                        dx = other_pos[0] - pos[0]
                        dy = other_pos[1] - pos[1]
                        if math.sqrt(dx * dx + dy * dy) <= self.rrt.robot_radius * 2:
                            break
                    else:
                        positions.append(pos)
        return positions

    def draw_graph(self, snapshotq=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if snapshotq is not None:
            for i, q_state in enumerate(snapshotq):
                if self.rrt.robot_radius > 0.0:
                    self.rrt.plot_circle(q_state.x, q_state.y, self.rrt.robot_radius, '-r')
                    plt.text(q_state.x - 0.5, q_state.y - 0.5, str(i), color="red", fontsize=12)

        edge_x = []
        edge_y = []
        for i, roadmap in enumerate(self.roadmaps):
            for node in roadmap:
                for weight, next_node in roadmap[node]:
                    if [next_node.x, node.x] not in edge_x or [next_node.y, node.y] not in edge_y:
                        plt.plot([node.x, next_node.x], [node.y, next_node.y], self.env.colors[i])
                        edge_x.append([node.x, next_node.x])
                        edge_y.append([node.y, next_node.y])
            plt.plot(self.start_states[i].x, self.start_states[i].y, marker='x', color=self.env.colors[i])
            plt.plot(self.goal_states[i].x, self.goal_states[i].y, marker='o', color=self.env.colors[i])

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

    def draw_result(self, state_paths):
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])

        # roadmap 그리기를 위한 좌표 배열 생성
        state_x_paths = [[] for _ in range(self.env.robot_num)]
        state_y_paths = [[] for _ in range(self.env.robot_num)]

        for robot, state_path in enumerate(state_paths):
            for state in state_path:
                state_x_paths[robot].append(state.x)
                state_y_paths[robot].append(state.y)

        # 경로를 step으로 만들어주기
        step_x_paths = [[] for _ in range(self.env.robot_num)]
        step_y_paths = [[] for _ in range(self.env.robot_num)]

        for robot, state_path in enumerate(state_paths):
            for i, state in enumerate(state_path[:-1]):
                distance, theta = self.rrt.calc_distance_and_angle(state_path[i], state_path[i + 1])

                local_x = state.x
                local_y = state.y

                step_x_paths[robot].append(local_x)
                step_y_paths[robot].append(local_y)

                step_distance = distance / self.env.step_num

                for _ in range(self.env.step_num):
                    local_x += math.cos(theta) * step_distance
                    local_y += math.sin(theta) * step_distance
                    step_x_paths[robot].append(local_x)
                    step_y_paths[robot].append(local_y)

        for step in range(len(step_x_paths[0])):
            plt.clf()
            for robot, (state_x_path, state_y_path) in enumerate(zip(state_x_paths, state_y_paths)):
                # path 그리기
                plt.plot(state_x_path, state_y_path, color=self.env.colors[robot])

                # robot 그리기
                self.rrt.plot_circle(step_x_paths[robot][step], step_y_paths[robot][step], self.rrt.robot_radius, '-r')
                plt.text(step_x_paths[robot][step] - 0.5, step_y_paths[robot][step] - 0.5, str(robot), color="red", fontsize=12)

            # 시작, 종료 지점 그리기
            for i, roadmap in enumerate(self.roadmaps):
                plt.plot(self.start_states[i].x, self.start_states[i].y, marker='x', color=self.env.colors[i])
                plt.plot(self.goal_states[i].x, self.goal_states[i].y, marker='o', color=self.env.colors[i])

            # 장애물 그리기
            for (ox, oy, size) in self.rrt.obstacle_list:
                self.rrt.plot_circle(ox, oy, size)

            # 좌표 공간 그리기
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
            plt.pause(self.env.pause_time)

    def reconstruct_paths(self, snapshot):
        next_snapshot = copy.deepcopy(snapshot)
        state_paths = [[] for _ in range(self.env.robot_num)]

        # snapshot을 거꾸로 올라가면서 path 만들기
        while next_snapshot:
            for robot, state in enumerate(next_snapshot.robot_states):
                state_paths[robot].append(state)
            next_snapshot = next_snapshot.parent

        for state_path in state_paths:
            state_path.reverse()

        return state_paths


if __name__ == '__main__':
    sssp = SSSP()
    start_time = time.time()

    last_snapshot = sssp.search()

    end_time = time.time()
    print(f"All time: {end_time - start_time}")

    # path 만들기
    re_state_paths = sssp.reconstruct_paths(last_snapshot)

    # 결과 그리기
    sssp.draw_result(re_state_paths)
