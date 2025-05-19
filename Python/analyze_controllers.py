#%%
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import *
from util.rw import load_object
from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]
log_path = './logs/exercise5/'  # path for logging the simulation data
#%%
controller = load_object('{}controller{}'.format(log_path, 0))  
#%%