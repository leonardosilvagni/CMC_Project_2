
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from simulation_parameters import SimulationParameters
from util.run_open_loop import run_single, run_multiple
from util.entraining_signals import define_entraining_signals
from plotting_common import plot_time_histories

from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise7(**kwargs):

    pylog.info("Ex 7")
    log_path = './logs/exercise7/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)
    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )
    n_iterations= 5001
    all_pars = [SimulationParameters(
        n_iterations       = n_iterations,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        return_network     = True,
        drive              = 10,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        feedback_weights_ipsi = 0.25*ws_ref,
        feedback_weights_contra = -0.25*ws_ref,
        entraining_signals=entraining,
        **kwargs
    ) for entraining in [None, define_entraining_signals(n_iterations,8,45) ]]
    pylog.info("Running the simulation")
    controllers = run_multiple(
        all_pars
    )
    # Retrieve neural frequencies from both controllers
    neur_freq_no_entraining = controllers[0].metrics["neur_frequency"]
    neur_freq_entraining = controllers[1].metrics["neur_frequency"]
    # Prepare the output text
    result_text = (
        "Neural frequency (no entraining signals): " + str(neur_freq_no_entraining) + "\n" +
        "Neural frequency (with entraining signals): " + str(neur_freq_entraining) + "\n" +
        "Difference: " + str(neur_freq_entraining - neur_freq_no_entraining) + "\n\n" +
        "Metrics for controller without entraining signals:\n" + str(controllers[0].metrics) + "\n\n" +
        "Metrics for controller with entraining signals:\n" + str(controllers[1].metrics)
    )

    # Print the output
    print(result_text)

    # Save the results to a text file in log_path named results.txt
    results_filepath = os.path.join(log_path, 'results.txt')
    with open(results_filepath, 'w') as file:
        file.write(result_text)

if __name__ == '__main__':

    exercise7()

