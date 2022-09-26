import yaml


class Environment:
    def __init__(self, environment_file):
        with open(environment_file) as f:
            environment = yaml.load(f, Loader=yaml.FullLoader)
        for key in environment.keys():
            setattr(self, key, environment[key])
