import math
import heapq


def get_heuristic_score(current_node, goal_node):
    dx = abs(current_node.x - goal_node.x)
    dy = abs(current_node.y - goal_node.y)
    return math.sqrt(dx ** 2 + dy ** 2)


class ANode:
    def __init__(self, f_score, g_score, h_score, node, parent_node):
        self.f_score = f_score
        self.g_score = g_score
        self.h_score = h_score
        self.node = node
        self.parent_node = parent_node

    def __eq__(self, other):
        return self.node == other.node

    def __lt__(self, other):
        return self.f_score < other.f_score


class AStar:
    def reconstruct_path(self, node):
        path = []

        current_node = node
        while current_node:
            path.append(current_node.node)
            current_node = current_node.parent_node
        return path[::-1]

    def search(self, roadmap, start_node, goal_node):
        open_list = list()
        closed_list = list()

        heapq.heappush(open_list, ANode(0, 0, 0, start_node, None))

        while open_list:
            a_node = heapq.heappop(open_list)
            closed_list.append(a_node)

            if a_node.node == goal_node:
                path = self.reconstruct_path(a_node)
                return path, a_node.g_score

            for next_weight, next_node in roadmap[a_node.node]:
                next_g_score = a_node.g_score + next_weight
                next_h_score = get_heuristic_score(next_node, goal_node)
                next_f_score = next_g_score + next_h_score
                next_a_node = ANode(next_f_score, next_g_score, next_h_score, next_node, a_node)

                if next_a_node not in closed_list:
                    if next_a_node not in open_list:
                        heapq.heappush(open_list, next_a_node)

                    else:
                        if next_a_node.g_score > next_g_score:
                            next_index = open_list.index(next_a_node)
                            open_list[next_index] = next_a_node
