import numpy as np
import os
import collections
from os.path import dirname, abspath
from copy import deepcopy
from sacred import Experiment, SETTINGS
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds
import sys
import torch
import logging
import random
from run import run
import yaml

def get_logger():
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s %(asctime)s] %(name)s %(message)s', '%H:%M:%S')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.setLevel('DEBUG')
    return logger

# set to "no" if you want to see stdout/stderr in console
SETTINGS['CAPTURE_MODE'] = "fd"
logger = get_logger()

ex = Experiment()
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds

results_path = os.path.join(dirname(abspath(__file__)), "results")

@ex.main
def my_main(_run, _config, _log):
    # Setting the random seed throughout the modules
    random.seed(_config["seed"])
    np.random.seed(_config["seed"])
    torch.manual_seed(_config["seed"])
    # run the framework
    run(_run, _config, _log)

def parse_config_file(params):
    config_file='grid7'
    for i, v in enumerate(params):
        if v.split("=")[0] == "--config":
            config_file=v.split("=")[1]
            del params[i]
            break
    return config_file

if __name__ == '__main__':
    params = deepcopy(sys.argv)
    config_file=parse_config_file(params)

    ex.add_config(f'config/{config_file}.yaml')

    logger.info(
        f"Saving to FileStorageObserver in results/sacred/{config_file}.")
    file_obs_path = os.path.join(results_path, "sacred", config_file)
    ex.add_config(name=config_file)
    ex.add_config(ex_results_path=file_obs_path)
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run_commandline(params)
