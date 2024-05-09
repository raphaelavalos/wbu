from belief_learner.networks.model_architecture import ModelArchitecture
import os

REGISTERY = {}
network_dir = os.path.join(
    os.path.dirname(__file__),
    "networks"
)

for dir_path, dir_names, file_names in os.walk(network_dir):
    base = dir_path.removeprefix(network_dir)
    if len(base) > 0 and base[0] == '/':
        base = base[1:] + '/'
    for file_name in file_names:
        if file_name.endswith('.toml'):
            REGISTERY[f"{base}{file_name.removesuffix('.toml')}"] = \
                ModelArchitecture.read_from_toml(os.path.join(dir_path, file_name))


def get_from_config_register(name):
    if name not in REGISTERY:
        raise ValueError(f"{name} not in registery.")
    return REGISTERY[name]


if __name__ == '__main__':
    print(REGISTERY)
