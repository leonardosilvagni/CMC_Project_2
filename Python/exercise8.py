
import os
import numpy as np
import matplotlib.pyplot as plt
import farms_pylog as pylog

from util.rw import load_object
from util.run_open_loop import run_multiple
from util.entraining_signals import define_entraining_signals
from simulation_parameters import SimulationParameters
from plotting_common import plot_2d, save_figures

from util.zebrafish_hyperparameters import define_hyperparameters
import pickle
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise8(**kwargs):

    pylog.info("Ex 8")
    pylog.info("Implement exercise 8")
    log_path = './logs/exercise8bis/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)
    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )

    # Define range for testing, scaled by ws_ref
    w_strengths  = np.linspace(0, 2.5, 6) * ws_ref
    frequencies = np.linspace(6.6,20,20)
    n_iterations = 5001
    # Create a 2D grid for testing feedback strengths vs. entrainment frequencies
    all_pars_grid = [
        [
            SimulationParameters(
                n_iterations       = n_iterations,
                controller         = "abstract oscillator",
                log_path           = log_path,
                compute_metrics    = 'all',
                print_metrics      = True,
                return_network     = True,
                drive              = 10,
                headless           = True,
                cpg_amplitude_gain = motor_output_gains,
                feedback_weights_ipsi = w,
                feedback_weights_contra = -w, 
                entraining_signals = define_entraining_signals(n_iterations, freq, 45),
                **kwargs
            )
            for freq in frequencies
        ]
        for w in w_strengths
    ]

    # Run the simulation for each grid point
    controllers = []
    for pars_row in all_pars_grid:
        result_row = run_multiple(pars_row)
        controllers.append(result_row)
    # Compute the baseline neural frequencies (with w=0) for the different entrainment frequencies.
    ref_results = controllers[0]

    # For each other value of w, compute differences relative to the baseline.
    for i, row in enumerate(controllers[1:], start=1):
        # The x-axis: difference between the entraining (target) frequency and the baseline neural frequency.
        diff_x = [freq - ref_results[j].metrics["neur_frequency"] for j, freq in enumerate(frequencies)]
        # The y-axis: difference between the actual neural frequency and the baseline neural frequency.
        diff_y = [row[j].metrics["neur_frequency"] - ref_results[j].metrics["neur_frequency"] for j in range(len(row))]
        plt.plot(diff_x, diff_y, marker='o', linestyle='-', label=f"w = {w_strengths[i]/ws_ref:.2f}")

    plt.xlabel("Entraining frequency difference (Hz)")
    plt.ylabel("Neural frequency difference (Hz)")
    plt.title("Neural vs. Entraining Frequency Difference")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(log_path,"exercise8_frequency_diff"))
    plt.show()

if __name__ == '__main__':

    exercise8()

