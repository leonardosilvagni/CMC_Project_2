from util.run_closed_loop import run_single, run_multiple
from simulation_parameters import SimulationParameters
import os
import farms_pylog as pylog
import numpy as np
import matplotlib.pyplot as plt
from plotting_common import *

from util.zebrafish_hyperparameters import define_hyperparameters
from matplotlib.ticker import FormatStrFormatter
hyperparameters = define_hyperparameters()
REF_JOINT_AMP = hyperparameters["REF_JOINT_AMP"]
ws_ref = hyperparameters["ws_ref"]


def exercise9():
    """
    Impairment: modify the parameter impairment, or witha for loop over the values of 1,2,3 to see the effect asked
    """
    n_iterations= 10001
    ws = np.arange(0.5,3.01,0.5)
    impairments = [0, 1, 2, 3]
    initial_phases = np.random.random(26)*2*np.pi 
        
    impairment_map = {
        0: "No impairment",
        1: "Ipsilateral cutoff",
        2: "Contralateral cutoff",
        3: "Ipsi and Contralateral cutoff"
    }

    motor_output_gains = np.array(
        [
            0.00824, 0.00328, 0.00328, 0.00370, 0.00451,
            0.00534, 0.00628, 0.00680, 0.00803, 0.01084,
            0.01115, 0.01149, 0.01655,
        ]
    )

    pylog.info("Ex 9")
    pylog.info("Implement exercise 9")
    log_path = './logs/exercise9/'  # path for logging the simulation data
    os.makedirs(log_path, exist_ok=True)

    # Default phases

    all_pars_list = [SimulationParameters(
        n_iterations       = n_iterations,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        drive              = 10,
        return_network     = True,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        entraining_signals = ws_ref,
        feedback_weights_ipsi = w,
        feedback_weights_contra = -w,
        impairment         = impairment,
    ) for impairment in impairments for w in ws]
    pylog.info("Running the simulation")
    controllers = run_multiple(
        all_pars_list
    )
    
    controllers = np.array(controllers)

    colors = cm.viridis(np.linspace(0, 1, len(ws)))
    os.makedirs("figures", exist_ok=True)
    for imp_idx, impairment in enumerate(impairments):
        plt.figure()
        for w_idx in reversed(range(len(ws))):
            plot_trajectory(
                controllers[w_idx + len(ws) * imp_idx],
                label=f'w = {ws[w_idx]}',
                color=colors[w_idx],
                new_fig=False
            )
        plt.legend(fontsize=14)
        plt.title(f'{impairment_map[impairment]} (default phases)')
        plt.ylim(0, 0.40)
        plt.xlim(-0.01, 0.01)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        filepath = os.path.join("figures", f'{impairment_map[impairment]}_default')
        plt.savefig(filepath, dpi=300)
    plt.show()

    # Random phases

    controllers = None

    all_pars_list = [SimulationParameters(
        n_iterations       = n_iterations,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        drive              = 10,
        return_network     = True,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        initial_phases     = initial_phases,
        entraining_signals = ws_ref,
        feedback_weights_ipsi = w,
        feedback_weights_contra = -w,
        impairment         = impairment,
    ) for impairment in impairments for w in ws]
    pylog.info("Running the simulation")
    controllers = run_multiple(
        all_pars_list
    )
    
    controllers = np.array(controllers)

    colors = cm.viridis(np.linspace(0, 1, len(ws)))
    os.makedirs("figures", exist_ok=True)
    for imp_idx, impairment in enumerate(impairments):
        plt.figure()
        for w_idx in reversed(range(len(ws))):
            plot_trajectory(
                controllers[w_idx + len(ws) * imp_idx],
                label=f'w = {ws[w_idx]}',
                color=colors[w_idx],
                new_fig=False
            )
        plt.legend(fontsize=14)
        plt.title(f'{impairment_map[impairment]} (Random phases)')
        plt.ylim(0, 0.20)
        plt.xlim(-0.01, 0.01)
        ax = plt.gca()
        ax.xaxis.set_major_locator(plt.MaxNLocator(6))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        filepath = os.path.join("figures", f'{impairment_map[impairment]}_random')
        plt.savefig(filepath, dpi=300)
    plt.show()

    # minimum CPG contralateral connection

    controllers = None
    w = 2.0
    wbbcs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    all_pars_list = [SimulationParameters(
        n_iterations       = n_iterations,
        controller         = "abstract oscillator",
        log_path           = log_path,
        compute_metrics    = 'all',
        print_metrics      = True,
        drive              = 10,
        return_network     = True,
        headless           = True,
        cpg_amplitude_gain = motor_output_gains,
        initial_phases     = initial_phases,
        entraining_signals = ws_ref,
        feedback_weights_ipsi = w,
        feedback_weights_contra = -w,
        weights_body2body_contralateral = wbbc,
    ) for wbbc in wbbcs]

    controllers = run_multiple(all_pars_list)
    controllers = np.array(controllers)

    colors = cm.viridis(np.linspace(0, 1, len(wbbcs)))

    plt.figure(figsize=(7, 6))
    for wbbc_idx, wbbc in enumerate(wbbcs):
        plot_trajectory(
            controllers[wbbc_idx],
            label=f'wbbc = {wbbc}',
            color=colors[wbbc_idx],
            new_fig=False
        )

    plt.legend(fontsize=8)
    plt.title('Minimum CPG contralateral connection (trajectory)', fontsize=14)
    plt.ylim(0, 0.20)
    plt.xlim(-0.01, 0.01)
    ax = plt.gca()
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    filepath = os.path.join("figures", "min_contralateral_connections_trajectory.png")
    plt.savefig(filepath, dpi=300)

    mech_speed = [ctrl.metrics["mech_speed_fwd"] for ctrl in controllers]

    plt.figure(figsize=(7, 6))
    plt.bar(range(len(wbbcs)), mech_speed, color=colors)
    plt.xticks(range(len(wbbcs)), [f'{w:.2f}' for w in wbbcs], rotation=45)
    plt.ylabel("Mech Speed (m/s)")
    plt.xlabel("Weights Body-to-Body Contralateral")
    plt.title("Minimum CPG contralateral connection (forward speed)", fontsize=14)

    filepath = os.path.join("figures", "min_contralateral_connections_speed.png")
    plt.savefig(filepath, dpi=300)
    plt.show()

    
if __name__ == '__main__':
    np.random.seed(0)
    exercise9()

