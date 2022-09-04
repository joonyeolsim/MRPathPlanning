class SnapShot:
    def __init__(self, score, robot_states, parent, paths, timestep):
        self.score = score
        self.robot_states = robot_states
        self.parent = parent
        self.paths = paths
        self.timestep = timestep

    def __lt__(self, other):
        return self.score < other.score
