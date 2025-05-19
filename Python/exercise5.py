
from util.run_closed_loop import run_single
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


def exercise5(**kwargs):
 
    pylog.info("Ex 5")
    pylog.info("Implement exercise 5")
    log_path = './logs/exercise5/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)
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
        feedback_weights_ipsi = 0.25*ws_ref,
        feedback_weights_contra = -0.25*ws_ref,
        #video_record = True,
        #video_name = "exercise5",
        #camera_id=3,
        
        **kwargs
    )


    pylog.info("Running the simulation")
    controller = run_single(
        all_pars
    )

        # --- Save plots ---
    tag = "ex5"
    plot_phases_ampl(
        controller.times, controller.state,
        title=f"Oscillator_States_{tag}", legend=True
    )
    plot_lateral_difference(
        controller.times, controller.motor_out,
        controller.motor_l, controller.motor_r,
        title=f"Lateral_Diff_{tag}"
    )

    plot_left_right(
        controller.times, controller.motor_out,
        controller.motor_l, controller.motor_r,
        title=f"Motor_{tag}"
    )

    plot_joint_angles(
        controller, joints_to_plot=[0, 3, 8, 11, 14],
        amplitudes=controller.amplitudes if hasattr(controller, "amplitudes") else None,
        title=f"Joint_Angles_{tag}"
    )
    
    plot_trajectory(
        controller,
        title=f"Trajectory_{tag}"
    )
    plt.show()
    
if __name__ == '__main__':

    exercise5()

