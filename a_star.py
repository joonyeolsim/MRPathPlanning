import math


def get_heuristic_score(current_node, goal_node):
    dx = abs(current_node.x - goal_node.x)
    dy = abs(current_node.y - goal_node.y)
    return math.sqrt(dx ** 2 + dy ** 2)


class AStar():
    def search(self, roadmap, start_node, goal_node):
        open_set = set()
        closed_set = set()
        f_score = dict()
        g_score = dict()

        open_set |= {start_node}  # f_score, g_score, h_score, vertex
        f_score[start_node] = 0
        g_score[start_node] = 0

        while open_set:
            current_node = min(f_score, key=f_score.get)
            open_set -= {current_node}
            closed_set |= {current_node}

            if current_node == goal_node:
                return g_score[current_node]

            for next_weight, next_node in roadmap[current_node]:
                next_g_score = g_score[current_node] + next_weight
                next_h_score = get_heuristic_score(next_node, goal_node)
                next_f_score = next_g_score + next_h_score

                if next_node not in closed_set:
                    # open_set에 있는 노드이고
                    if next_node in open_set:
                        # g_score가 더 작다면
                        if g_score[next_node] > next_g_score:
                            g_score[next_node] = next_g_score
                            f_score[next_node] = next_f_score

                    # open_set에 없고 closed_set에도 없다면
                    else:
                        open_set |= {next_node}
                        g_score[next_node] = next_g_score
                        f_score[next_node] = next_f_score

            f_score.pop(current_node, None)
            g_score.pop(current_node, None)