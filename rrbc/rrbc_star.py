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
from state import State


class RRBC:
    def __init__(self, environment_file):
        self.env = Environment(environment_file)
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
        ab = RRBC.ccw(p1.x, p1.y, p2.x, p2.y, p3.x, p3.y) * RRBC.ccw(p1.x, p1.y, p2.x, p2.y, p4.x, p4.y)
        cd = RRBC.ccw(p3.x, p3.y, p4.x, p4.y, p1.x, p1.y) * RRBC.ccw(p3.x, p3.y, p4.x, p4.y, p2.x, p2.y)

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
        start_positions = self.get_random_positions(self.env.robot_num, self.env.map_width, self.env.map_height)
        goal_positions = self.get_random_positions(self.env.robot_num, self.env.map_width, self.env.map_height)

        # 각 시작 지점과 끝 지점에 대해서 roadmap을 생성함.
        for start_position, goal_position in zip(start_positions, goal_positions):
            self.rrt.set_position(start_position, goal_position)
            roadmap = self.rrt.planning()
            self.start_states.append(list(roadmap)[0])
            self.goal_states.append(list(roadmap)[-1])
            self.roadmaps.append(roadmap)

        while self.env.iteration:
            frontier = list()
            explored = list()

            # 처음 state을 만들고 frontier에 삽입.
            init_states = [start_state for start_state in self.start_states]
            parent = None
            paths, score = self.calculate_score_path(init_states)
            timestep = 1
            passed_score = 0

            heapq.heappush(frontier, State(score, init_states, parent, paths, timestep, passed_score))

            while frontier:
                state = heapq.heappop(frontier)
                explored.append(state.robot_states)

                next_score = state.score
                next_robot_states = state.robot_states[:]
                next_parent = state
                next_paths = state.paths
                next_timestep = state.timestep + 1
                next_passed_score = state.passed_score

                if self.env.draw_graph:
                    self.draw_graph(state.robot_states)

                for robot, robot_state in enumerate(state.robot_states):
                    if robot_state != self.goal_states[robot]:
                        break
                else:
                    return state

                pass_scores = [0 for _ in range(self.env.robot_num)]
                for robot, path in enumerate(next_paths):
                    if len(path) > state.timestep:
                        next_robot_states[robot] = path[state.timestep]
                        pass_score = self.get_euclidean_distance(state.robot_states[robot], next_robot_states[robot])
                        pass_scores[robot] = pass_score

                collisions_list = self.collision_check(state.robot_states, next_robot_states)
                if not collisions_list:
                    if next_robot_states not in explored:
                        next_passed_score += sum(pass_scores)
                        heapq.heappush(frontier, State(next_score, next_robot_states, next_parent, next_paths, next_timestep, next_passed_score))
                    else:
                        print("Explored!")

                else:
                    # 충돌한 로봇들은 각각 Sampling 후 충돌하지 않는 지점을 frontier 큐에 넣음.
                    for collisions in collisions_list:
                        for collision in collisions:
                            # 충돌한 로봇과 충돌한 위치
                            robot, collision_state = collision
                            copy_next_passed_score = next_passed_score
                            copy_pass_scores = pass_scores[:]
                            copy_pass_scores[robot] = 0

                            # 충돌 하기 전 위치
                            if len(state.paths[robot]) > state.timestep - 1:
                                last_state = state.paths[robot][state.timestep - 1]

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
                                    if not self.collision_check(state.robot_states, copy_next_robot_states):
                                        if copy_next_robot_states not in explored:
                                            copy_pass_scores[robot] = self.get_euclidean_distance(state.robot_states[robot], copy_next_robot_states[robot])
                                            copy_next_passed_score += sum(copy_pass_scores)
                                            next_paths, next_score = self.calculate_score_path(copy_next_robot_states)
                                            next_timestep = 1
                                            heapq.heappush(frontier, State(next_score + copy_next_passed_score, copy_next_robot_states, next_parent, next_paths, next_timestep, copy_next_passed_score))
                                            break
                                        else:
                                            print("Collision Explored!")

            self.env.threshold_distance *= 0.5
            self.env.iteration -= 1
            print(f"Iteration Count: {self.env.iteration}")
        return None

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

    def draw_graph(self, stateq=None):
        plt.clf()
        # for stopping simulation with the esc key.
        plt.gcf().canvas.mpl_connect(
            'key_release_event',
            lambda event: [exit(0) if event.key == 'escape' else None])
        if stateq is not None:
            for i, q_state in enumerate(stateq):
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

    def reconstruct_paths(self, state):
        next_state = copy.deepcopy(state)
        state_paths = [[] for _ in range(self.env.robot_num)]

        # state을 거꾸로 올라가면서 path 만들기
        while next_state:
            for robot, state in enumerate(next_state.robot_states):
                state_paths[robot].append(state)
            next_state = next_state.parent

        for state_path in state_paths:
            state_path.reverse()

        return state_paths


if __name__ == '__main__':
    rrbc = RRBC("environment.yaml")
    start_time = time.time()

    last_state = rrbc.search()

    end_time = time.time()
    if last_state:
        print(f"All time: {end_time - start_time}")
        print(f"Cost: {last_state.passed_score}")

        # path 만들기
        re_state_paths = rrbc.reconstruct_paths(last_state)

        # 결과 그리기
        rrbc.draw_result(re_state_paths)
    else:
        print("Search Fail...")
