class State:
    def __init__(self, score, robot_states, parent, paths, timestep, passed_score=0):
        self.score = score
        self.robot_states = robot_states
        self.parent = parent
        self.paths = paths
        self.timestep = timestep
        self.passed_score = passed_score

    def __lt__(self, other):
        return self.score < other.score
