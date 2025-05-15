#%%
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt

from util.rw import load_object
from util.run_closed_loop import run_multiple
from simulation_parameters import SimulationParameters
from plotting_common import plot_1d, save_figures
from plotting_common import *
from util.zebrafish_hyperparameters import define_hyperparameters
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise6(**kwargs):

    pylog.info("Ex 6")
    pylog.info("Implement exercise 6")
    log_path = './logs/exercise6/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )

    # Define range for testing, scaled by ws_ref
    ipsi_strengths  = np.linspace(-1, 1, 6) 
    contra_strengths = np.linspace(-1, 1, 6) 

    # Test ipsilateral feedback variation (contralateral fixed to 0)
    all_pars_list_ipsi = [SimulationParameters(
            n_iterations       = 5001,
            controller         = "abstract oscillator",
            log_path           = log_path,
            compute_metrics    = 'all',
            print_metrics      = True,
            return_network     = True,
            drive              = 10,
            headless           = True,
            cpg_amplitude_gain = motor_output_gains,
            entraining_signals = ws_ref,
            feedback_weights_ipsi = w_ipsi,
            feedback_weights_contra = 0,
            **kwargs
    ) for w_ipsi in ipsi_strengths]
    controllers_ipsi = run_multiple(all_pars_list_ipsi)

    # Test contralateral feedback variation (ipsilateral fixed to 0)
    all_pars_list_contra = [SimulationParameters(
            n_iterations       = 5001,
            controller         = "abstract oscillator",
            log_path           = log_path,
            compute_metrics    = 'all',
            print_metrics      = True,
            return_network     = True,
            drive              = 10,
            headless           = True,
            cpg_amplitude_gain = motor_output_gains,
            entraining_signals = ws_ref,
            feedback_weights_ipsi = 0,
            feedback_weights_contra = w_contra,
            **kwargs
    ) for w_contra in contra_strengths]
    controllers_contra = run_multiple(all_pars_list_contra)

    # Extract and plot metrics for ipsilateral variation
    ipsi_vals = ipsi_strengths
    neur_frequency_ipsi = [ctrl.metrics["neur_frequency"] for ctrl in controllers_ipsi]
    neur_twl_ipsi       = [ctrl.metrics["neur_twl"]       for ctrl in controllers_ipsi]
    mech_cot_ipsi       = [ctrl.metrics["mech_cot"]       for ctrl in controllers_ipsi]
    mech_energy_ipsi    = [ctrl.metrics["mech_energy"]    for ctrl in controllers_ipsi]
    mech_speed_ipsi     = [ctrl.metrics["mech_speed_fwd"]   for ctrl in controllers_ipsi]
    mech_torque_ipsi    = [ctrl.metrics["mech_torque"]      for ctrl in controllers_ipsi]
    # Plot neural frequency, total wave lag, cost of transport, energy consumption, forward speed, and torque for ipsi variation
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0,0].plot(ipsi_vals, neur_frequency_ipsi, marker='o')
    axs[0,0].set_title("Neural Frequency (ipsi)")
    axs[0,1].plot(ipsi_vals, neur_twl_ipsi, marker='o')
    axs[0,1].set_title("Total Wave Lag (ipsi)")
    axs[0,2].plot(ipsi_vals, mech_cot_ipsi, marker='o')
    axs[0,2].set_title("Cost of Transport (ipsi)")
    axs[1,0].plot(ipsi_vals, mech_energy_ipsi, marker='o')
    axs[1,0].set_title("Energy Consumption (ipsi)")
    axs[1,1].plot(ipsi_vals, mech_speed_ipsi, marker='o')
    axs[1,1].set_title("Forward Speed (ipsi)")
    axs[1,2].plot(ipsi_vals, mech_torque_ipsi, marker='o')
    axs[1,2].set_title("Sum of Torques (ipsi)")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path,"metrics_ipsi_variation"))
    plt.close(fig)

    # Extract and plot metrics for contralateral variation
    contra_vals = contra_strengths
    neur_frequency_contra = [ctrl.metrics["neur_frequency"] for ctrl in controllers_contra]
    neur_twl_contra       = [ctrl.metrics["neur_twl"]       for ctrl in controllers_contra]
    mech_cot_contra       = [ctrl.metrics["mech_cot"]       for ctrl in controllers_contra]
    mech_energy_contra    = [ctrl.metrics["mech_energy"]    for ctrl in controllers_contra]
    mech_speed_contra     = [ctrl.metrics["mech_speed_fwd"]   for ctrl in controllers_contra]
    mech_torque_contra    = [ctrl.metrics["mech_torque"]      for ctrl in controllers_contra]
    # Plot neural frequency, total wave lag, cost of transport, energy consumption, forward speed, and torque for contra variation
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    axs[0,0].plot(contra_vals, neur_frequency_contra, marker='o')
    axs[0,0].set_title("Neural Frequency (contra)")
    axs[0,1].plot(contra_vals, neur_twl_contra, marker='o')
    axs[0,1].set_title("Total Wave Lag (contra)")
    axs[0,2].plot(contra_vals, mech_cot_contra, marker='o')
    axs[0,2].set_title("Cost of Transport (contra)")
    axs[1,0].plot(contra_vals, mech_energy_contra, marker='o')
    axs[1,0].set_title("Energy Consumption (contra)")
    axs[1,1].plot(contra_vals, mech_speed_contra, marker='o')
    axs[1,1].set_title("Forward Speed (contra)")
    axs[1,2].plot(contra_vals, mech_torque_contra, marker='o')
    axs[1,2].set_title("Sum of Torques (contra)")
    plt.tight_layout()
    plt.savefig(os.path.join(log_path,"metrics_contra_variation"))
    plt.close(fig)

    
if __name__ == '__main__':

    exercise6()

