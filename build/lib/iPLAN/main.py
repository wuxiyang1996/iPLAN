import sys
import os
from os.path import dirname, abspath
import collections
from copy import deepcopy
import yaml

import numpy as np
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import torch as th

from utils.logging import get_logger
from run_ippo import run as run_ippo

SETTINGS['CONFIG']['READ_ONLY_CONFIG'] = False
SETTINGS['CAPTURE_MODE'] = 'sys'
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
logger = get_logger()

ex = Experiment("Highway_iPLAN")
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = "results"

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    config = config_copy(_config)

    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    config['env_args']['seed'] = config["seed"] # overrides env_args seed
    # run the framework
    run_ippo(_run, config, _log)


def _get_config(config_name, subfolder):
    if config_name is not None:
        with open(os.path.join(os.path.dirname(__file__), "config", subfolder, "{}.yaml".format(config_name)), "r") as f:
            try:
                config_dict = yaml.safe_load(f)
            except yaml.YAMLError as exc:
                assert False, "{}.yaml error: {}".format(config_name, exc)
        return config_dict

def set_arg(config_dict, params, arg_name, arg_type):
    for _i, _v in enumerate(params):
        if _v.split("=")[0] == arg_name:
            arg_name = _v.split("=")[0].replace("--", "")
            arg_value = _v.split("=")[1]
            config_dict[arg_name] = arg_type(arg_value)
            del params[_i]
            return config_dict


def recursive_dict_update(d, u):
    '''update dict d with items in u recursively. if key present in both d and u, 
    value in d takes precedence. 
    '''
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            if k not in d.keys():
                d[k] = v                
    return d


def config_copy(config):
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)


if __name__ == '__main__':
    # Get the defaults from default.yaml
    config_dict = _get_config("default",  "")

    # Load env base configs when considering difficulties
    if config_dict["env"] == "highway":
        env_config = _get_config("highway", "envs")
    elif config_dict["env"] == "MPE":
        if config_dict["difficulty"] == "easy":
            env_config = _get_config("simple_spread_Hetero", "envs")
        else:
            env_config = _get_config("simple_spread_Hetero_H", "envs")

    # Load algorithm configs
    alg_config = _get_config("ippo", "algs")

    # update env_config and alg_config with values in config dict 
    # copy modified env args for logging purpose 
    config_dict = recursive_dict_update(config_dict, env_config)
    config_dict = recursive_dict_update(config_dict, alg_config)

    # add config to sacred
    ex.add_config(config_dict)

    # Save to disk by default for sacred
    logger.info("Saving to FileStorageObserver in results/sacred.")
    file_obs_path = os.path.join(results_path, "sacred")
    ex.observers.append(FileStorageObserver.create(file_obs_path))

    ex.run_commandline()

