
from util.run_closed_loop import run_single
from simulation_parameters import SimulationParameters
import matplotlib.pyplot as plt
import os
import farms_pylog as pylog
import numpy as np

def exercise_single(**kwargs):
    """
    Exercise example, running a single simulation and plotting the results
    """
    log_path = './logs/example_single/' # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # the optimized motor output gains from exercise 4 in Project 1
    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )

    all_pars = SimulationParameters(
        n_iterations       = 5001,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        return_network     = True,
        drive              = 10,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        **kwargs
    )


    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )




if __name__ == '__main__':
    exercise_single()
    plt.show()



