import yaml


def load_config(
        **kwargs
):
    config_path = kwargs['config_path']

    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config