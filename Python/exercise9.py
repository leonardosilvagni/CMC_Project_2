
from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import *

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise9(**kwargs):
    """
    Impairment: modify the parameter impairment, or witha for loop over the values of 1,2,3 to see the effect asked
    """
    n_iterations= 5001
    w=0.25 # weight of the feedback
    impairment = 1# 0=No impairment, 1=ipsilateral cutoff, 2=contralateral cutoff, 3=both ipsi and contralateral cutoff
        
    impairment_map = {
        0: "No impairment",
        1: "Ipsilateral cutoff",
        2: "Contralateral cutoff",
        3: "Ipsi and Contralateral cutoff"
    }
    impairment_string = impairment_map.get(impairment)
    pylog.info("Ex 9")
    pylog.info("Implement exercise 9")
    log_path = './logs/exercise9/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)
    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )
    all_pars = SimulationParameters(
        n_iterations       = n_iterations,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        return_network     = True,
        drive              = 10,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        feedback_weights_ipsi = w*ws_ref,
        feedback_weights_contra = -w*ws_ref,
        impairment = impairment, # 0=No impairment, 1=ipsilateral cutoff, 2=contralateral cutoff, 3=both ipsi and contralateral cutoff
        **kwargs
    )
    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )
    # Plots, either as in exercise 5 or only what is more interesting
    plot_trajectory(
        controller,
        title=f"Trajectory_{impairment_map.get(impairment)}"
    )
    plt.show()
if __name__ == '__main__':

    exercise9()

