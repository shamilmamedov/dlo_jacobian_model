import yaml

def load_config_params():
    path = 'config/params_sim.yaml'
    with open(path, 'r') as file:
        params = yaml.safe_load(file)
    return params