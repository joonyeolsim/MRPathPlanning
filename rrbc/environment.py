import yaml


class Environment:
    def __init__(self):
        with open("environment.yaml") as f:
            environment = yaml.load(f, Loader=yaml.FullLoader)
        for key in environment.keys():
            setattr(self, key, environment[key])
