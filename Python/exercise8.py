#%%
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
#%%

def exercise8(**kwargs):
    test=kwargs.get("test", False)
    pylog.info("Ex 8")
    log_path = './logs/exercise8/'  # path for logging the simulation data
    if test:
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
    w_strengths  = np.linspace(0, 2, 5) * ws_ref
    frequencies = np.linspace(3.5,10,10)
    if test:
        w_strengths  = np.linspace(0, 6, 60) * ws_ref
        frequencies = np.linspace(3,30,60)
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
    f_actual_matrix = np.zeros((len(w_strengths) - 1, len(frequencies)))


    for idx, pars_row in enumerate(all_pars_grid):
        result_row = run_multiple(pars_row)
        if idx == 0:
            # The first row is the baseline; save it for later use.
            ref_results = result_row
        else:
            # For subsequent rows, populate the preallocated matrix.
            for j, res in enumerate(result_row):
                f_actual_matrix[idx - 1, j] = res.metrics["neur_frequency"]

    f_ref_array = np.array([ref_results[j].metrics["neur_frequency"] for j in range(len(frequencies))])
    # Compute the x-axis differences once using the baseline neural frequencies.
    diff_y = f_actual_matrix - f_ref_array[np.newaxis, :]
    diff_x = frequencies - f_ref_array
    # For each subsequent w value, compute the neural frequency difference relative to the baseline.
    for idx, diff_y_row in enumerate(diff_y):
        plt.plot(diff_x, diff_y_row, marker='o', linestyle='-', label=f"w = {w_strengths[idx+1]/ws_ref:.2f}")

    plt.xlabel("Entraining frequency difference (Hz)")
    plt.ylabel("Neural frequency difference (Hz)")
    plt.title("Neural vs. Entraining Frequency Difference")
    plt.legend()
    plt.grid(True)
    filename = "exercise8_frequency_diff"
    if test:
        filename = "exercise8bis_frequency_diff"
    plt.savefig(os.path.join(log_path,filename))
    if test:
        import pickle
        from scipy.interpolate import RegularGridInterpolator
        # Prepare raw data storage
        # Save raw data for later recomputation
        data_structure = {
            "f_actual_matrix": f_actual_matrix,
            "frequencies": frequencies,
            "w_strengths": w_strengths[1:],  # exclude the zero-coupling baseline
            "f_ref": f_ref_array
        }

        with open(os.path.join(log_path, "arnold_tongue_raw_data.pkl"), "wb") as f:
            pickle.dump(data_structure, f)

        # Compute fitness metric

        # Compute fitness metric
        #epsilon = 1e-6
        #fitness_map = 1 - np.abs(f_actual_matrix - frequencies[np.newaxis,:]) / (np.abs(f_actual_matrix - f_ref_array[np.newaxis, :]) + epsilon)
        #fitness_map = np.clip(fitness_map, 0, 1)
        diff_ratio = (data_structure["f_actual_matrix"] - data_structure["f_ref"][np.newaxis, :]) / (data_structure["frequencies"] - data_structure["f_ref"])
        angles = np.arctan(diff_ratio)
        fitness_map=angles/(np.pi/4)
        grid_freq = frequencies
        grid_w = w_strengths[1:] / ws_ref
        interp_func = RegularGridInterpolator(
        (grid_w, grid_freq),
        fitness_map,
        method='linear',
        bounds_error=False,
        fill_value=0
        )

        # Create finer meshgrid
        fine_freq = np.linspace(frequencies[0], frequencies[-1], 100)
        fine_w = np.linspace(grid_w[0], grid_w[-1], 100)
        X_fine, Y_fine = np.meshgrid(fine_freq, fine_w)

        # Evaluate interpolator on the fine grid
        points = np.array([Y_fine.ravel(), X_fine.ravel()]).T
        fitness_map_interp = interp_func(points).reshape(X_fine.shape)
        # Plot the interpolated fitness map
        X_fine, Y_fine = np.meshgrid(fine_freq, fine_w)
        plt.figure(figsize=(12, 8))
        cmap = plt.cm.Reds  # white = low fitness, red = high
        im = plt.pcolormesh(X_fine, Y_fine, fitness_map_interp, cmap=cmap, shading='auto', vmin=0, vmax=1)
        cbar = plt.colorbar(im)
        cbar.set_label("Coupling Fitness (1 = full entrainment)")

        plt.xlabel("Entraining Frequency (Hz)")
        plt.ylabel("Entraining Gain (w)")
        plt.tight_layout()

        plt.savefig(os.path.join(log_path, "exercise8_arnold_tongue_fitness_interpolated"))
    pylog.info("Close figures to continue")
    plt.show()

if __name__ == '__main__':

    exercise8(test=False)

